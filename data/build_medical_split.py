#!/usr/bin/env python3
"""Build the medicine training / held-out sets (RubricHub medical) and the two HealthBench-300 sets.

Inputs (local files, or --*-url to download them first):
  --rubrichub   RubricHub_v1 RuRL/rurbichub_v1_Medical.parquet (29,681 rows), e.g.
                https://huggingface.co/datasets/sojuL/RubricHub_v1/resolve/3837d55971473a872e84879c88f708b8da3ec2ef/RuRL/rurbichub_v1_Medical.parquet
                used fields: prompt (chat list), Rubrics [{criterion, points}], data_source / ability (domain filter)
  --healthbench HealthBench oss_eval JSONL (5,000 rows), e.g.
                https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl
                used fields: prompt (chat list), rubrics [{criterion, points}], prompt_id

Outputs (--output):
  train.parquet                 12,519 medical prompts
  heldout.parquet               300 medical prompts, disjoint from train
  healthbench_heldout.parquet   300 HealthBench prompts, seed 42 (in-training validation set)
  healthbench_disjoint.parquet  300 HealthBench prompts, seed 43, disjoint from healthbench_heldout
  manifest.json

Selection (seed 42 unless noted; `random.Random(seed)` instances, never the global generator):
  medical rows  rows whose data_source / ability / domain (top level or extra_info) equals "medical"; all rows when
                no row carries such a marker. No deduplication and no item-count filter.
  heldout       pool = Random(42).sample(range(N), 2,300); heldout = the last 300 of pool, in pool order,
                ids "rubrichub_medical:2000" .. ":2299". rubric_permuted is the next row's rubric within the full
                2,300-row pool (so the last held-out row points to pool row 0, not to held-out row 0).
  train         Random(42).sample(indices not in heldout, 12,519), in sample order, ids
                "rubrichub_medical:train:<source row index>"; rubric_permuted cyclic within train.
  healthbench_heldout   Random(42).sample(all rows, 300); ids "healthbench:<prompt_id>"; data_source "healthbench".
  healthbench_disjoint  Random(43).sample(rows whose prompt_id is not in healthbench_heldout, source order, 300);
                        ids "healthbench_disjoint:<prompt_id>"; data_source "healthbench_disjoint".

Row contract (every file):
  prompt       chat messages [{role, content}] (non-string content JSON-encoded)
  data_source  "rubrichub_medical" / "healthbench" / "healthbench_disjoint"
  extra_info   schema_version, id, problem (the "problem"/"query"/"question" field if present, else the last user
               message), rubric [{criterion, weight = points}] (at least one positive weight), rubric_correct and
               rubric_permuted ("i. [weight=w] criterion" lines), self_golden "", opsd_context_kind "rubric_repro_v1"
Criterion tags are not carried over. Negative-weight criteria (HealthBench) keep their sign.

    python3 build_medical_split.py --rubrichub rurbichub_v1_Medical.parquet \\
        --healthbench 2025-05-07-06-14-12_oss_eval.jsonl --output release/medical-atomic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import sha256_file  # noqa: E402

SEED = 42
DISJOINT_SEED = 43
TRAIN_ROWS = 12_519
HELDOUT_ROWS = 300
POOL_HEAD_ROWS = 2_000
POOL_TAIL_ROWS = 300
HEALTHBENCH_ROWS = 300
MEDICAL_SOURCE = "rubrichub_medical"
HB_SOURCE = "healthbench"
HB_DISJOINT_SOURCE = "healthbench_disjoint"
SCHEMA_VERSION = "opd-rubric-repro-dataset/v1"
OPSD_CONTEXT_KIND = "rubric_repro_v1"  # the OPSD context profile in ../distill checks this value


# ---------------------------------------------------------------- field extraction (shared contract helpers)

def _json_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_prompt(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return [{"role": "user", "content": value}]
    messages = []
    for message in value:
        role = str(message.get("role") or "user").strip()
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = _json_text(content)
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("prompt must contain at least one message")
    return messages


def _candidates(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out = [row]
    for key in ("extra_info", "reward_model"):
        if isinstance(row.get(key), Mapping):
            out.append(row[key])
    return out


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for candidate in _candidates(row):
        for key in keys:
            if key in candidate and candidate[key] is not None:
                return candidate[key]
    return None


def _extract_prompt(row: Mapping[str, Any]) -> list[dict[str, str]]:
    value = _first_value(row, ("prompt", "messages", "conversation", "query", "question"))
    if value is None:
        raise ValueError("source row has no prompt/messages/query/question field")
    return _normalize_prompt(value)


def _extract_problem(row: Mapping[str, Any], prompt: Sequence[Mapping[str, Any]]) -> str:
    value = _first_value(row, ("problem", "query", "question"))
    if value is not None and str(value).strip():
        return str(value).strip()
    for message in reversed(prompt):
        if str(message.get("role", "")).lower() == "user" and str(message.get("content", "")).strip():
            return str(message["content"]).strip()
    content = str(prompt[-1].get("content", "")).strip()
    if not content:
        raise ValueError("could not derive a non-empty problem from prompt")
    return content


def _criterion(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    for key in ("criterion", "criteria", "description", "text", "rubric", "content"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("rubric item has no criterion text")


def _weight(item: Any) -> float:
    if not isinstance(item, Mapping):
        return 1.0
    for key in ("weight", "points", "score", "importance"):
        if item.get(key) is not None:
            weight = float(item[key])
            if not math.isfinite(weight):
                raise ValueError(f"rubric weight is not finite: {item[key]!r}")
            return weight
    return 1.0


def _extract_rubric(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = None
    for candidate in _candidates(row):
        for key in ("rubric", "rubrics", "Rubrics", "criteria"):
            if candidate.get(key):
                source = candidate[key]
                break
        if source is not None:
            break
    if source is None:
        raise ValueError("source row has no rubric/rubrics/Rubrics/criteria field")
    if isinstance(source, Mapping):
        source = [source]
    rubric = [{"criterion": _criterion(item), "weight": _weight(item)} for item in source]
    if not rubric:
        raise ValueError("rubric must contain at least one criterion")
    if all(item["weight"] <= 0 for item in rubric):
        raise ValueError("rubric must contain at least one positive weight")
    return rubric


def _sample_id(row: Mapping[str, Any], index: int, prefix: str) -> str:
    for candidate in _candidates(row):
        for key in ("id", "sample_id", "prompt_id", "uid", "tid"):
            value = candidate.get(key)
            if value is not None and str(value).strip():
                return f"{prefix}:{value}"
    return f"{prefix}:{index}"


def _render_rubric(rubric: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f"{i}. [weight={float(it['weight']):g}] {str(it['criterion']).strip()}"
                     for i, it in enumerate(rubric, start=1))


def _contract_row(prompt, problem, rubric, permuted_rubric, data_source, row_id) -> dict[str, Any]:
    rubric = [{"criterion": str(it["criterion"]).strip(), "weight": float(it["weight"])} for it in rubric]
    permuted = [{"criterion": str(it["criterion"]).strip(), "weight": float(it["weight"])} for it in permuted_rubric]
    return {
        "prompt": [dict(m) for m in prompt],
        "data_source": data_source,
        "extra_info": {
            "schema_version": SCHEMA_VERSION,
            "id": row_id,
            "problem": problem,
            "rubric": rubric,
            "rubric_correct": _render_rubric(rubric),
            "rubric_permuted": _render_rubric(permuted),
            "self_golden": "",
            "opsd_context_kind": OPSD_CONTEXT_KIND,
        },
    }


def _build_rows(normalized, data_source) -> list[dict[str, Any]]:
    """normalized: [(prompt, problem, rubric, row_id)]; rubric_permuted = next row's rubric (cyclic)."""
    return [_contract_row(p, q, r, normalized[(i + 1) % len(normalized)][2], data_source, rid)
            for i, (p, q, r, rid) in enumerate(normalized)]


def _normalize(source_rows, selected, id_for) -> list[tuple]:
    out = []
    for position, row in enumerate(selected):
        prompt = _extract_prompt(row)
        out.append((prompt, _extract_problem(row, prompt), _extract_rubric(row), id_for(position, row)))
    return out


# ---------------------------------------------------------------- medical split

def _medical_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    marked = []
    for row in rows:
        values = [str(row[k]).strip().lower() for k in ("data_source", "ability", "domain") if row.get(k) is not None]
        extra = row.get("extra_info")
        if isinstance(extra, Mapping):
            values += [str(extra[k]).strip().lower() for k in ("data_source", "ability", "domain")
                       if extra.get(k) is not None]
        marked.append(values)
    if not any(marked):
        return [dict(r) for r in rows]
    medical = [dict(r) for r, v in zip(rows, marked) if "medical" in v]
    if not medical:
        raise ValueError("RubricHub source contains no medical-domain rows")
    return medical


def medical_split(source_rows: Sequence[Mapping[str, Any]], seed: int = SEED):
    rows = _medical_rows(source_rows)
    if len(rows) < TRAIN_ROWS + HELDOUT_ROWS:
        raise ValueError(f"RubricHub medical source has {len(rows)} rows; {TRAIN_ROWS + HELDOUT_ROWS} are required")
    pool = random.Random(seed).sample(range(len(rows)), min(POOL_HEAD_ROWS + POOL_TAIL_ROWS, len(rows)))
    n_held = min(POOL_TAIL_ROWS, HELDOUT_ROWS, len(pool))
    held_idx = pool[-n_held:]
    held_set = set(held_idx)
    train_idx = random.Random(seed).sample([i for i in range(len(rows)) if i not in held_set], TRAIN_ROWS)

    train = _build_rows(_normalize(rows, [rows[i] for i in train_idx],
                                   lambda pos, r, idx=train_idx: f"{MEDICAL_SOURCE}:train:{idx[pos]}"),
                        MEDICAL_SOURCE)
    # Held-out rows are built inside the 2,300-row pool (this fixes their rubric_permuted), then cut.
    pool_rows = _build_rows(_normalize(rows, [rows[i] for i in pool], lambda pos, r: ""), MEDICAL_SOURCE)
    heldout = pool_rows[-n_held:]
    for position, row in enumerate(heldout, start=POOL_HEAD_ROWS):
        row["extra_info"]["id"] = f"{MEDICAL_SOURCE}:{position}"
    return rows, train, heldout


# ---------------------------------------------------------------- HealthBench sets

def healthbench_sets(source_rows: Sequence[Mapping[str, Any]]):
    if len(source_rows) < HEALTHBENCH_ROWS:
        raise ValueError(f"HealthBench source has {len(source_rows)} rows; {HEALTHBENCH_ROWS} are required")
    selected = random.Random(SEED).sample(list(source_rows), HEALTHBENCH_ROWS)
    held = _build_rows(_normalize(source_rows, selected, lambda pos, r: _sample_id(r, pos, HB_SOURCE)), HB_SOURCE)
    held_keys = {_sample_id(r, 0, "") for r in selected}
    rest = [r for r in source_rows if _sample_id(r, 0, "") not in held_keys]
    disjoint_sel = random.Random(DISJOINT_SEED).sample(rest, HEALTHBENCH_ROWS)
    disjoint = _build_rows(_normalize(source_rows, disjoint_sel,
                                      lambda pos, r: _sample_id(r, pos, HB_DISJOINT_SOURCE)), HB_DISJOINT_SOURCE)
    return held, disjoint


# ---------------------------------------------------------------- I/O

def _fetch(url: str, destination: Path) -> Path:
    with urlopen(Request(url, headers={"User-Agent": "prorubric-data/1"}), timeout=120) as response, \
            destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    return destination


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return pq.read_table(path).to_pylist()


def _negative(rows) -> dict[str, int]:
    neg = [sum(float(it["weight"]) < 0 for it in r["extra_info"]["rubric"]) for r in rows]
    return {"criteria": sum(neg), "rows": sum(n > 0 for n in neg)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rubrichub", type=Path, help="local rurbichub_v1_Medical.parquet (or JSONL)")
    ap.add_argument("--rubrichub-url", help="download the RubricHub medical parquet from this URL instead")
    ap.add_argument("--healthbench", type=Path, help="local HealthBench oss_eval JSONL (or parquet)")
    ap.add_argument("--healthbench-url", help="download the HealthBench JSONL from this URL instead")
    ap.add_argument("--output", required=True, type=Path, help="release directory to write")
    a = ap.parse_args(argv)
    if (a.rubrichub is None) == (a.rubrichub_url is None) or (a.healthbench is None) == (a.healthbench_url is None):
        ap.error("give exactly one of --rubrichub/--rubrichub-url and one of --healthbench/--healthbench-url")

    with tempfile.TemporaryDirectory(prefix="medical-split-") as tmp:
        rh_path = a.rubrichub or _fetch(a.rubrichub_url, Path(tmp) / "medical.parquet")
        hb_path = a.healthbench or _fetch(a.healthbench_url, Path(tmp) / "healthbench.jsonl")
        source = {"rubrichub": a.rubrichub_url or f"local:{rh_path.name}", "rubrichub_sha256": sha256_file(rh_path),
                  "healthbench": a.healthbench_url or f"local:{hb_path.name}", "healthbench_sha256": sha256_file(hb_path)}
        med_rows, train, heldout = medical_split(_read_rows(rh_path))
        hb_rows = _read_rows(hb_path)
        hb_held, hb_disjoint = healthbench_sets(hb_rows)

    a.output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, rows in (("train.parquet", train), ("heldout.parquet", heldout),
                       ("healthbench_heldout.parquet", hb_held), ("healthbench_disjoint.parquet", hb_disjoint)):
        dst = a.output / name
        pq.write_table(pa.Table.from_pylist(rows), dst, compression="zstd")
        outputs[name] = {"rows": len(rows), "sha256": sha256_file(dst)}
        print(name, len(rows))
    manifest = {
        "schema_version": "opd-rubric-repro-release/v1",
        "artifact_id": a.output.name,
        "source": source,
        "source_rows": {"rubrichub_medical": len(med_rows), "healthbench": len(hb_rows)},
        "members": {
            "train.parquet": f"{TRAIN_ROWS} medical prompts, seed {SEED}, disjoint from heldout",
            "heldout.parquet": f"{HELDOUT_ROWS} medical prompts (last {POOL_TAIL_ROWS} of a seed-{SEED} "
                               f"sample of {POOL_HEAD_ROWS + POOL_TAIL_ROWS})",
            "healthbench_heldout.parquet": f"{HEALTHBENCH_ROWS} HealthBench prompts, seed {SEED}",
            "healthbench_disjoint.parquet": f"{HEALTHBENCH_ROWS} HealthBench prompts, seed {DISJOINT_SEED}, "
                                            "disjoint from healthbench_heldout",
        },
        "negative_weight_criteria": {k: _negative(v) for k, v in
                                     (("train", train), ("heldout", heldout), ("healthbench_heldout", hb_held),
                                      ("healthbench_disjoint", hb_disjoint))},
        "contract": "prompt(list[chat]) / data_source / extra_info{schema_version, id, problem, rubric[list{criterion,weight}], "
                    "rubric_correct, rubric_permuted, self_golden, opsd_context_kind}",
        "outputs": outputs,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
