#!/usr/bin/env python3
"""Build the science (RaR-Science) training and held-out sets from the public ScaleAI/RaR-Science release.

Input: the public RaR-Science parquet files (one per split), either given locally with --train-local /
--heldout-local or downloaded from huggingface.co (dataset API -> data/<split>-*.parquet). Each source row has
  question (str), reference_answer (str, optional), question_source (optional),
  rubric: list of {title, description, weight}, where description reads "<Category> Criteria: ..." with
  Category in Essential / Important / Optional / Pitfall and weight is the generation-time integer.

Output directory (--output):
  train.parquet       the full train split (18,333 prompts), source order
  heldout.parquet     the full held-out split (default: val), source order
  heldout300.parquet  random.seed(42); random.sample(heldout rows, 300) -- the science held-out evaluation set
  manifest.json       counts, category weights, source description, sha256 of every parquet

Row contract (every file):
  prompt       [{"role": "user", "content": question}]
  data_source  "rar_science"
  extra_info   id               "rar_science:<split>:<row index in the split>"
               problem          question
               rubric           [{criterion, weight}] with criterion = "<title>: <description>"
                                (i.e. "<title>: <Category> Criteria: ...", the prefix that
                                ../generate/rewrite_atomic.py and build_atomic_rw.py rely on) and the RaR
                                categorical weight: Essential 1.0 / Important 0.7 / Optional 0.3 / Pitfall -0.9
               rubric_correct   the rubric rendered as "i. [weight=w] criterion" lines
               rubric_permuted  the next row's rubric rendered the same way (cyclic, within the split)
               self_golden "", schema_version, opsd_context_kind "rubric_repro_v1"
               reference_answer, question_source (when present), rar_split,
               rar_category_weight_map (category -> magnitude), rar_source_rubric (per-criterion provenance:
               title, description, category, paper_weight, source_weight)

Pitfall criteria describe undesirable behaviour, so the categorical magnitude 0.9 is applied with a negative sign
(polarity carried by sign, as in RubricHub / HealthBench). A category is read from the "<Category> Criteria:"
prefix of the description; for the few malformed rows without a recognisable prefix it is inferred from the
generation-time integer weight (essential = 5, important = 3/4, optional = 1/2, pitfall = -1/-2).

    python3 build_science_split.py --output release/science-atomic
    python3 build_science_split.py --output release/science-atomic \\
        --train-local train-00000-of-00001.parquet --heldout-local val-00000-of-00001.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import sha256_file  # noqa: E402

DEFAULT_DATASET_ID = "ScaleAI/RaR-Science"
DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
DATA_SOURCE = "rar_science"
HELDOUT_SAMPLE_SEED = 42
HELDOUT_SAMPLE_SIZE = 300
SCHEMA_VERSION = "opd-rubric-repro-dataset/v1"
OPSD_CONTEXT_KIND = "rubric_repro_v1"  # the OPSD context profile in ../distill checks this value

# RaR paper Section 4.4 categorical weights (magnitudes; pitfall is applied as -0.9).
RAR_CATEGORY_WEIGHTS = {
    "essential": 1.0,
    "important": 0.7,
    "optional": 0.3,
    "pitfall": 0.9,
}

_CATEGORY_ALIASES = {
    "importance": "important",
    "esssential": "essential",
    "mandatory": "essential",
    "option": "optional",
}


# ---------------------------------------------------------------- download / read

def _download(url: str, destination: Path, *, max_bytes: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "prorubric-data/1"})
    try:
        with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
            raw_length = response.headers.get("Content-Length")
            if raw_length and raw_length.isdigit() and int(raw_length) > max_bytes:
                raise ValueError(f"refusing to download {raw_length} bytes from {url}; cap is {max_bytes}")
            written = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(f"refusing to download more than {max_bytes} bytes from {url}")
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _load_dataset_info(dataset_id: str, endpoint: str, revision: str) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/api/datasets/{dataset_id}"
    if revision != "main":
        url += f"/revision/{quote(revision, safe='')}"
    with urlopen(Request(url, headers={"User-Agent": "prorubric-data/1"}), timeout=60) as response:
        return dict(json.loads(response.read().decode("utf-8")))


def _split_file(info: Mapping[str, Any], split: str) -> str:
    prefix = f"data/{split}-"
    for sibling in info.get("siblings") or []:
        filename = str(sibling.get("rfilename", ""))
        if filename.startswith(prefix) and filename.endswith(".parquet"):
            return filename
    raise ValueError(f"dataset has no parquet sibling for split {split!r}")


def _download_split(*, dataset_id: str, split: str, revision: str, output_dir: Path, endpoint: str,
                    max_bytes: int) -> tuple[Path, str | None]:
    info = _load_dataset_info(dataset_id, endpoint, revision)
    filename = _split_file(info, split)
    url = f"{endpoint.rstrip('/')}/datasets/{quote(dataset_id, safe='/')}/resolve/{quote(revision, safe='')}/{quote(filename)}"
    return _download(url, output_dir / filename.replace("/", "_"), max_bytes=max_bytes), info.get("sha")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        return pq.read_table(path).to_pylist()
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            return [dict(row) for row in json.load(handle)]
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------- rubric conversion

def _category_from_source_weight(weight: Any) -> str | None:
    """Fallback for the ~20 malformed rows in the public release (integer weights are category-exact)."""
    try:
        value = float(weight)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return "pitfall"
    if value >= 5:
        return "essential"
    if value >= 3:
        return "important"
    if value >= 1:
        return "optional"
    return None


def _category(description: str, source_weight: Any = None) -> str:
    prefix, sep, _ = description.partition(":")
    normalized = prefix.strip().lower().removesuffix(" criteria").strip() if sep else ""
    normalized = _CATEGORY_ALIASES.get(normalized, normalized)
    if normalized in RAR_CATEGORY_WEIGHTS:
        return normalized
    inferred = _category_from_source_weight(source_weight)
    if inferred is not None:
        return inferred
    if not sep:
        raise ValueError(f"RaR rubric description has no category prefix: {description!r}")
    raise ValueError(f"unknown RaR rubric category {prefix!r}")


def _rar_rubric(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rubric = row.get("rubric")
    if not isinstance(raw_rubric, Sequence) or isinstance(raw_rubric, (str, bytes)) or not raw_rubric:
        raise ValueError("RaR row must contain a non-empty rubric list")
    rubric, provenance = [], []
    for index, item in enumerate(raw_rubric):
        description = str(item.get("description") or "").strip()
        if not description:
            raise ValueError(f"RaR rubric[{index}] has no description")
        category = _category(description, item.get("weight"))
        paper_weight = RAR_CATEGORY_WEIGHTS[category]
        if category == "pitfall":
            paper_weight = -paper_weight
        title = str(item.get("title") or "").strip()
        criterion = description if not title else f"{title}: {description}"
        rubric.append({"criterion": criterion, "weight": paper_weight})
        provenance.append({"title": title, "description": description, "category": category,
                           "paper_weight": paper_weight, "source_weight": item.get("weight")})
    return rubric, provenance


def _render_rubric(rubric: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f"{i}. [weight={float(it['weight']):g}] {str(it['criterion']).strip()}"
                     for i, it in enumerate(rubric, start=1))


def _contract_rows(source_rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(source_rows):
        question = str(row.get("question") or "").strip()
        if not question:
            raise ValueError(f"RaR {split} row {index} has no question")
        rubric, provenance = _rar_rubric(row)
        rubric = [{"criterion": str(it["criterion"]).strip(), "weight": float(it["weight"])} for it in rubric]
        normalized.append((question, rubric, provenance, row, f"{DATA_SOURCE}:{split}:{index}"))

    rows = []
    for index, (question, rubric, provenance, source_row, row_id) in enumerate(normalized):
        next_rubric = normalized[(index + 1) % len(normalized)][1]
        extra: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "id": row_id,
            "problem": question,
            "rubric": rubric,
            "rubric_correct": _render_rubric(rubric),
            "rubric_permuted": _render_rubric(next_rubric),
            "self_golden": "",
            "opsd_context_kind": OPSD_CONTEXT_KIND,
        }
        reference = source_row.get("reference_answer")
        if reference is not None and str(reference).strip():
            extra["reference_answer"] = str(reference)
        if source_row.get("question_source") is not None:
            extra["question_source"] = str(source_row["question_source"])
        extra["rar_split"] = split
        extra["rar_category_weight_map"] = dict(RAR_CATEGORY_WEIGHTS)
        extra["rar_source_rubric"] = provenance
        rows.append({"prompt": [{"role": "user", "content": question}], "data_source": DATA_SOURCE,
                     "extra_info": extra})
    return rows


_CRITERION = pa.struct([("criterion", pa.string()), ("weight", pa.float64())])
# Column and field order of the released files (absent optional keys are written as null).
SCHEMA = pa.schema([
    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("data_source", pa.string()),
    ("extra_info", pa.struct([
        ("schema_version", pa.string()), ("id", pa.string()), ("problem", pa.string()),
        ("rubric", pa.list_(_CRITERION)),
        ("rubric_correct", pa.string()), ("rubric_permuted", pa.string()), ("self_golden", pa.string()),
        ("opsd_context_kind", pa.string()), ("reference_answer", pa.string()), ("question_source", pa.string()),
        ("rar_split", pa.string()),
        ("rar_category_weight_map", pa.struct([(k, pa.float64()) for k in RAR_CATEGORY_WEIGHTS])),
        ("rar_source_rubric", pa.list_(pa.struct([
            ("title", pa.string()), ("description", pa.string()), ("category", pa.string()),
            ("paper_weight", pa.float64()), ("source_weight", pa.int64())]))),
    ])),
])


def _negative_source_weight_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    criteria = rows_with_negative = 0
    for row in rows:
        negative = [it for it in row["extra_info"].get("rar_source_rubric") or []
                    if isinstance(it.get("source_weight"), (int, float)) and float(it["source_weight"]) < 0]
        criteria += len(negative)
        rows_with_negative += bool(negative)
    return {"criteria": criteria, "rows": rows_with_negative}


# ---------------------------------------------------------------- main

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path, help="release directory to write")
    ap.add_argument("--train-local", type=Path, help="local RaR-Science train parquet/JSONL (else downloaded)")
    ap.add_argument("--heldout-local", type=Path, help="local RaR-Science held-out split parquet/JSONL (else downloaded)")
    ap.add_argument("--heldout-split", default="val", choices=("val", "test"),
                    help="public split the held-out set comes from (paper: val)")
    ap.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    ap.add_argument("--revision", default="main", help="dataset revision for downloads")
    ap.add_argument("--hf-endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--max-download-bytes", type=int, default=DEFAULT_MAX_DOWNLOAD_BYTES)
    a = ap.parse_args(argv)

    source: dict[str, Any] = {"dataset_id": a.dataset_id}
    with tempfile.TemporaryDirectory(prefix="rar-science-") as tmp:
        paths = {}
        for key, local, split in (("train", a.train_local, "train"), ("heldout", a.heldout_local, a.heldout_split)):
            if local is None:
                paths[key], sha = _download_split(dataset_id=a.dataset_id, split=split, revision=a.revision,
                                                  output_dir=Path(tmp), endpoint=a.hf_endpoint,
                                                  max_bytes=a.max_download_bytes)
                source[key] = f"{a.dataset_id}:{split}@{sha or a.revision}"
            else:
                paths[key] = local
                source[key] = f"local:{local.name}"
            source[key + "_sha256"] = sha256_file(paths[key])
        train = _contract_rows(_read_rows(paths["train"]), split="train")
        held = _contract_rows(_read_rows(paths["heldout"]), split=a.heldout_split)

    random.seed(HELDOUT_SAMPLE_SEED)
    held300 = random.sample(held, HELDOUT_SAMPLE_SIZE)

    a.output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, rows in (("train.parquet", train), ("heldout.parquet", held), ("heldout300.parquet", held300)):
        dst = a.output / name
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), dst)
        outputs[name] = sha256_file(dst)
        print(name, len(rows))
    ex = train[0]
    print("sample id:", ex["extra_info"]["id"], "| n_rubric:", len(ex["extra_info"]["rubric"]))
    print("sample rubric[0]:", str(ex["extra_info"]["rubric"][0])[:160])

    tn, hn = _negative_source_weight_stats(train), _negative_source_weight_stats(held)
    manifest = {
        "schema_version": "opd-rubric-repro-release/v1",
        "artifact_id": a.output.name,
        "data_source": DATA_SOURCE,
        "source": source,
        "heldout_split": a.heldout_split,
        "members": {"train.parquet": len(train), "heldout.parquet": len(held),
                    "heldout300.parquet": f"{HELDOUT_SAMPLE_SIZE} of heldout.parquet, "
                                          f"random.seed({HELDOUT_SAMPLE_SEED}); random.sample"},
        "paper_category_weights": dict(RAR_CATEGORY_WEIGHTS),
        "pitfall_sign": -1,
        "source_negative_weight_criteria": {"train": tn["criteria"], "heldout": hn["criteria"],
                                            "total": tn["criteria"] + hn["criteria"]},
        "source_negative_weight_rows": {"train": tn["rows"], "heldout": hn["rows"]},
        "contract": "prompt(list[chat]) / data_source=rar_science / extra_info{id, problem, rubric[list{criterion,weight}], "
                    "rubric_correct, rubric_permuted, reference_answer, question_source, rar_split, "
                    "rar_category_weight_map, rar_source_rubric, ...}",
        "outputs": outputs,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
