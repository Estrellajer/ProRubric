#!/usr/bin/env python3
"""Score an evaluation responses.jsonl offline: multiple-choice accuracy and
per-criterion official HealthBench grading.

    python3 score_records.py \
        --questions healthbench_full.parquet --questions healthbench_consensus.parquet \
        --questions medqa_usmle_4opt.parquet --questions gpqa_diamond.parquet \
        --responses responses.jsonl --scores scores.jsonl [--judge-role LITE] [--workers 32]

Each response row is scored against the question row with the same id:

* rows whose ``extra_info.rubric`` is a non-empty list (HealthBench-full,
  HealthBench-consensus, other rubric sets) are graded one judge call per
  criterion with the official HealthBench ``GRADER_TEMPLATE``
  (``../healthbench_judge.py``), temperature 0; the per-question score is the
  official HealthBench score (``healthbench_judge.score``);
* all other rows (MedQA, GPQA-Diamond, ...) are multiple-choice: the answer
  letter is extracted with the evaluation harness's parser
  (``parse_mcq_answer_letter``) and compared with ``extra_info.gold_letter``
  (or ``answer_idx`` / ``answer_letter``); no judge call.

The judge is the package client (``../../reward/judge_client.py``) built with
``judge_from_env(--judge-role)``. In the paper this script is run with role
``LITE`` (Doubao-lite, the second evaluator); its output provides the
``c_lite`` / ``g_lite`` columns. Thinking is disabled in the request and
``max_tokens`` is 4096, as in the paper's runs.

Failures that are not the response's fault are retried: HTTP 429 with jittered
exponential backoff (capped at 600 s per wait) until ``--budget-429`` seconds
have passed, and an unparseable judge reply (``JSONDecodeError``) up to
``--parse-attempts`` more times. A retry regrades the whole question. Any other
failure becomes an error row.

Inputs
------
``--questions`` (repeatable) parquet, one row per question::

    prompt       string or [{"role", "content"}] messages
    data_source  suite name, e.g. healthbench_full / healthbench_consensus / medqa / gpqa_diamond
    extra_info   {"id": "<suite>:<n>",
                  "rubric": [{"criterion", "weight", "tags"?}, ...]          (rubric sets)
                  "gold_letter": "A".."J", "options": {...} or [...]         (multiple choice)
                  "example_tags"?, "eval_kind"?}

Ids are unique across all given parquets (``healthbench_full:*``,
``healthbench_consensus:*``, ``medqa:*``, ``gpqa_diamond:*``, ...). ``extra_info``
may be a struct or a JSON string.

``--responses`` JSONL ``{"id", "response"}`` as written by ``generate.py``.

Output
------
``--scores`` JSONL, appended; one row per scored question. Rubric rows::

    {"id", "data_source", "eval_kind": "rubric", "response", "satisfied": [bool per criterion],
     "score", "weighted_score", "n_criteria", "tag_scores", "raw", "usage", "error": null, ...}

multiple-choice rows::

    {"id", "data_source", "eval_kind": "mcq", "response", "predicted_letter", "gold_letter",
     "accuracy": 0.0 | 1.0, "score", "error": null, ...}

failed rows ``{"id", "data_source", "error": "<message>", "response"}``. The run is
resumable: ids that already have a row without ``error`` are skipped, and a
later clean row for an id supersedes an earlier error row (the readers in
``../../analysis/tables/`` keep the last clean row per id).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "reward"))
sys.path.insert(0, os.path.join(HERE, ".."))

from judge_client import _content_from_response, judge_from_env  # noqa: E402
from healthbench_judge import grader_prompt, score, strip_fence  # noqa: E402

MCQ_LETTERS = tuple("ABCDEFGHIJ")
GENERATION_SKIPPED_PREFIX = "[[GENERATION_SKIPPED"


# ---------------------------------------------------------------------------
# Official per-criterion grading. The template is healthbench_judge.GRADER_TEMPLATE
# (byte-identical to the harness's vendored copy, sha256 2adffd51fd259554...).
# The conversation rendering and verdict check below are the harness's own.
# ---------------------------------------------------------------------------


def render_conversation(prompt, response: str) -> str:
    """Flatten a chat prompt plus the assistant response the way the official
    grader expects: one ``role: content`` block per turn, blank-line separated."""

    if isinstance(prompt, str):
        turns = [{"role": "user", "content": prompt}]
    else:
        turns = [dict(turn) for turn in prompt]
    turns = turns + [{"role": "assistant", "content": response}]
    return "\n\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)


@dataclass(frozen=True)
class OfficialResult:
    satisfied: list[bool]
    weighted_score: float
    raw: str
    usage: dict[str, Any] = field(default_factory=dict)
    verdict_mode: str = "official"
    verdict_adapter: str = "simple-evals"


def score_official(judge, prompt, response: str, rubric: list[dict[str, Any]], *, temperature: float = 0.0) -> OfficialResult:
    """One call per rubric item with the verbatim upstream template.

    The request goes straight to the client's transport (no client-side retry);
    retries are the caller's (``score_with_retry``)."""

    if not rubric:
        raise ValueError("rubric must contain at least one criterion")
    conversation = render_conversation(prompt, response)
    satisfied: list[bool] = []
    raws: list[str] = []
    usage_total: dict[str, Any] = {}
    for item in rubric:
        if not isinstance(item, Mapping) or "criterion" not in item or "weight" not in item:
            raise ValueError("each rubric item needs criterion and weight")
        rendered = grader_prompt(conversation, item)  # "[{weight}] {criterion}" in the template
        raw_response = judge._call([{"role": "user", "content": rendered}], temperature=temperature)
        raw, usage, _ = _content_from_response(raw_response)
        payload = json.loads(strip_fence(raw))
        met = payload.get("criteria_met") if isinstance(payload, Mapping) else None
        if met is not True and met is not False:
            raise ValueError(f"official grader returned a non-boolean criteria_met: {met!r}")
        satisfied.append(bool(met))
        raws.append(raw)
        for key, value in (usage or {}).items():
            if isinstance(value, (int, float)):
                usage_total[key] = usage_total.get(key, 0) + value
    return OfficialResult(
        satisfied=satisfied,
        weighted_score=score(satisfied, rubric),
        raw="\n".join(raws),
        usage=usage_total,
    )


_RETRY = {"429": 0, "parse": 0, "gaveup_429": 0, "gaveup_parse": 0}
_RETRY_LOCK = threading.Lock()


def _bump(key):
    with _RETRY_LOCK:
        _RETRY[key] += 1


def retry_counts():
    with _RETRY_LOCK:
        return dict(_RETRY)


def score_with_retry(judge, prompt, resp, rubric, budget_429=3600.0, parse_attempts=3):
    """Retry the two failures that are not the answer's fault: rate limiting and
    an unparseable reply.

      429              -- exponential backoff capped at 10 min per wait, until
                          ``budget_429`` seconds have gone by.
      JSONDecodeError  -- the judge replied with something that is not the
                          expected JSON; retry a few times with a short pause.

    Anything else is raised immediately."""
    last = None
    parse_left = parse_attempts
    t_end = time.time() + budget_429
    a = -1
    while True:
        a += 1
        try:
            return score_official(judge, prompt, resp, rubric)
        except Exception as exc:
            last = exc
            is429 = "429" in str(exc) or getattr(exc, "status_code", None) == 429
            isparse = isinstance(exc, json.JSONDecodeError) or "JSONDecodeError" in type(exc).__name__
            if is429:
                if time.time() >= t_end:
                    break
                _bump("429")
                time.sleep(min(600.0, 1.0 * (2 ** min(a, 10))) * (0.75 + 0.5 * random.random()))
                continue
            if isparse and parse_left > 0:
                _bump("parse")
                parse_left -= 1
                time.sleep(0.5 + random.random())
                continue
            if isparse:
                _bump("gaveup_parse")
            raise
    _bump("gaveup_429")
    raise last


# ---------------------------------------------------------------------------
# Evaluation-harness record builders and multiple-choice parser (verbatim).
# ---------------------------------------------------------------------------


def _extra_info(row: Mapping[str, Any]) -> Mapping[str, Any]:
    extra = row.get("extra_info")
    return extra if isinstance(extra, Mapping) else {}


def _rubric(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    extra = _extra_info(row)
    if not isinstance(extra, Mapping) or not isinstance(extra.get("rubric"), list):
        raise ValueError("parquet row is missing extra_info.rubric")
    rubric = []
    for item in extra["rubric"]:
        if not isinstance(item, Mapping) or "criterion" not in item or "weight" not in item:
            raise ValueError("every extra_info.rubric item needs criterion and weight")
        normalized = {"criterion": str(item["criterion"]), "weight": float(item["weight"])}
        tags = item.get("tags")
        if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes, bytearray)):
            normalized["tags"] = [str(tag) for tag in tags]
        rubric.append(normalized)
    if not rubric:
        raise ValueError("extra_info.rubric must not be empty")
    return rubric


def _example_tags(row: Mapping[str, Any]) -> list[str]:
    tags = _extra_info(row).get("example_tags")
    if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes, bytearray)):
        return []
    return [str(tag) for tag in tags]


def _score_from_weighted_bools(
    values: Sequence[bool],
    rubric: Sequence[Mapping[str, Any]],
) -> float:
    positive_total = sum(max(0.0, float(item["weight"])) for item in rubric)
    if positive_total <= 0:
        return math.nan
    score = sum(float(item["weight"]) * int(value) for item, value in zip(rubric, values))
    return max(0.0, min(1.0, score / positive_total))


def _rubric_tag_scores(
    rubric: Sequence[Mapping[str, Any]],
    result: OfficialResult,
) -> dict[str, float]:
    tags: dict[str, list[tuple[Mapping[str, Any], bool]]] = {}
    for item, satisfied in zip(rubric, result.satisfied):
        raw_tags = item.get("tags")
        if not isinstance(raw_tags, Sequence) or isinstance(raw_tags, (str, bytes, bytearray)):
            continue
        for tag in raw_tags:
            tags.setdefault(str(tag), []).append((item, bool(satisfied)))
    scores: dict[str, float] = {}
    for tag, items in tags.items():
        tagged_rubric = [item for item, _ in items]
        tagged_satisfied = [satisfied for _, satisfied in items]
        score = _score_from_weighted_bools(tagged_satisfied, tagged_rubric)
        if math.isfinite(score):
            scores[tag] = score
    return scores


def _response_tokens(response: str) -> int:
    if not response:
        return 0
    return max(1, math.ceil(len(response) / 4))


def parse_mcq_answer_letter(response: str, *, choices: Sequence[str] = MCQ_LETTERS) -> str | None:
    """Extract a final multiple-choice answer letter from model text."""

    allowed = {str(choice).strip().upper() for choice in choices if str(choice).strip()}
    if not allowed:
        raise ValueError("MCQ choices must not be empty")
    text = str(response or "").strip()
    if not text:
        return None
    patterns = [
        r"(?is)(?:final\s+answer|answer|ans|correct\s+option)\s*(?:is|:|-)?\s*[\(\[]?\s*([A-J])\s*[\)\].,;:]?",
        r"(?is)(?:option|choice)\s*[\(\[]?\s*([A-J])\s*[\)\].,;:]?",
        r"(?is)^\s*[\(\[]?\s*([A-J])\s*[\)\].,;:]?\s*$",
    ]
    for pattern in patterns:
        matches = [
            match.group(1).upper()
            for match in re.finditer(pattern, text)
            if match.group(1).upper() in allowed
        ]
        if matches:
            return matches[-1]
    # The bare-word fallback must never treat the English pronoun "I" as an
    # answer ("I do not know."); an explicit form ("Answer: I", "(I)") is still
    # matched by the keyword patterns above.
    standalone = [
        match.group(1).upper()
        for match in re.finditer(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", text)
        if match.group(1).upper() in allowed and match.group(1).upper() != "I"
    ]
    return standalone[-1] if standalone else None


def _mcq_gold_letter(row: Mapping[str, Any]) -> str:
    extra = _extra_info(row)
    for key in ("gold_letter", "answer_idx", "answer_letter"):
        value = extra.get(key)
        if value is not None and str(value).strip():
            letter = str(value).strip().upper()
            if letter not in MCQ_LETTERS:
                raise ValueError(f"extra_info.{key} is not a supported MCQ letter: {value!r}")
            return letter
    raise ValueError("MCQ row is missing extra_info.gold_letter")


def _mcq_choices(row: Mapping[str, Any]) -> list[str]:
    options = _extra_info(row).get("options")
    if isinstance(options, Mapping):
        letters = [str(key).strip().upper() for key in options]
        return [letter for letter in MCQ_LETTERS if letter in letters]
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes, bytearray)):
        return list(MCQ_LETTERS[: len(options)])
    return list(MCQ_LETTERS[:4])


def _length_adjusted_score(
    score: float,
    response: str,
    row: Mapping[str, Any],
) -> float | None:
    extra = _extra_info(row)
    center = extra.get("length_adjustment_center")
    penalty = extra.get("length_adjustment_penalty_per_500_chars")
    if center is None and penalty is None:
        return None
    if center is None or penalty is None:
        raise ValueError(
            "length_adjustment_center and length_adjustment_penalty_per_500_chars must be set together"
        )
    center_value = float(center)
    penalty_value = float(penalty)
    if center_value < 0 or penalty_value < 0:
        raise ValueError("length adjustment center and penalty must be non-negative")
    return float(score) - penalty_value * ((len(response) - center_value) / 500.0)


def _score_record(
    row_id: str,
    response: str,
    data_source: str | None,
    n_criteria: int,
    dataset_row: Mapping[str, Any],
    result: OfficialResult,
) -> dict[str, Any]:
    rubric = _rubric(dataset_row)
    tags = _example_tags(dataset_row)
    return {
        "id": row_id,
        "data_source": data_source,
        "eval_kind": "rubric",
        "response": response,
        "response_length": len(response),
        "response_chars": len(response),
        "response_tokens": _response_tokens(response),
        "n_criteria": n_criteria,
        "satisfied": result.satisfied,
        "weighted_score": result.weighted_score,
        "score": result.weighted_score,
        "accuracy": None,
        "example_tags": tags,
        "tag_scores": _rubric_tag_scores(rubric, result),
        "length_adjusted_score": _length_adjusted_score(
            result.weighted_score,
            response,
            dataset_row,
        ),
        "raw": result.raw,
        "usage": result.usage,
        "verdict_adapter": result.verdict_adapter,
        "error": None,
    }


def _mcq_score_record(
    row_id: str,
    response: str,
    data_source: str | None,
    dataset_row: Mapping[str, Any],
) -> dict[str, Any]:
    gold = _mcq_gold_letter(dataset_row)
    predicted = parse_mcq_answer_letter(response, choices=_mcq_choices(dataset_row))
    correct = predicted == gold
    return {
        "id": row_id,
        "data_source": data_source,
        "eval_kind": "mcq",
        "response": response,
        "response_length": len(response),
        "response_chars": len(response),
        "response_tokens": _response_tokens(response),
        "n_criteria": None,
        "predicted_letter": predicted,
        "gold_letter": gold,
        "accuracy": 1.0 if correct else 0.0,
        "weighted_score": 1.0 if correct else 0.0,
        "score": 1.0 if correct else 0.0,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _plain_extra(row: dict[str, Any]) -> dict[str, Any]:
    extra = row.get("extra_info")
    if isinstance(extra, str):
        row = dict(row)
        row["extra_info"] = json.loads(extra)
    return row


def load_questions(paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            row = _plain_extra(row)
            rid = str(_extra_info(row).get("id", row.get("id")))
            if rid in out:
                raise ValueError(f"duplicate question id {rid!r} ({path})")
            out[rid] = row
    return out


def _data_source(row: Mapping[str, Any] | None, rid: str) -> str:
    if row is not None and row.get("data_source") is not None:
        return str(row["data_source"])
    return rid.split(":")[0]


def score_one(judge, r: Mapping[str, Any], questions: Mapping[str, dict], args) -> dict[str, Any]:
    rid = str(r["id"])
    drow = questions.get(rid)
    src = _data_source(drow, rid)
    resp = r.get("response") or ""
    if drow is None:
        return {"id": rid, "data_source": src, "error": "OfflineLookupError: id not in parquet", "response": resp}
    if not resp:
        return {"id": rid, "data_source": src, "error": r.get("error") or "empty response", "response": resp}
    if resp.startswith(GENERATION_SKIPPED_PREFIX):
        return {"id": rid, "data_source": src, "error": "prompt exceeds max_model_len; generation skipped", "response": ""}
    rubric = _extra_info(drow).get("rubric")
    try:
        if isinstance(rubric, list) and rubric:
            res = score_with_retry(judge, drow.get("prompt", ""), resp, rubric,
                                   budget_429=args.budget_429, parse_attempts=args.parse_attempts)
            return _score_record(rid, resp, src, len(rubric), drow, res)
        return _mcq_score_record(rid, resp, src, drow)
    except Exception as exc:  # noqa: BLE001 -- recorded as an error row, retried on resume
        return {"id": rid, "data_source": src, "error": f"{type(exc).__name__}: {str(exc)[:200]}", "response": resp}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the module docstring / README.md for the input and output schema.")
    ap.add_argument("--questions", action="append", required=True, metavar="PARQUET",
                    help="suite parquet(s) holding every id in --responses (repeatable)")
    ap.add_argument("--responses", required=True, help="responses JSONL {id, response} from generate.py")
    ap.add_argument("--scores", required=True, help="output scores JSONL (appended; resumable)")
    ap.add_argument("--judge-role", default="LITE",
                    help="judge role for judge_from_env: LITE = second evaluator (paper: Doubao-lite, default); "
                         "'' = evaluation judge; ALT = third-family evaluator")
    ap.add_argument("--workers", type=int, default=32, help="concurrent questions (default 32)")
    ap.add_argument("--judge-max-tokens", type=int, default=4096, help="judge max_tokens (paper: 4096)")
    ap.add_argument("--judge-timeout", type=float, default=180.0, help="per-request timeout in seconds")
    ap.add_argument("--thinking", choices=["disabled", "provider-default"], default="disabled",
                    help="'disabled' sends thinking={type: disabled} as in the paper; 'provider-default' omits it")
    ap.add_argument("--budget-429", type=float, default=3600.0,
                    help="seconds a question may keep retrying HTTP 429 before it is written as an error")
    ap.add_argument("--parse-attempts", type=int, default=3, help="extra attempts after an unparseable judge reply")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N response rows")
    args = ap.parse_args(argv)

    with open(args.responses, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    done: set[str] = set()
    if os.path.exists(args.scores):
        with open(args.scores, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if not r.get("error"):
                        done.add(str(r["id"]))
    pending = [r for r in rows if str(r["id"]) not in done]
    questions = load_questions(args.questions)
    print(f"rows {len(rows)} done {len(done)} pending {len(pending)}", flush=True)

    judge = judge_from_env(
        args.judge_role or None,
        verdict_mode="hard",
        thinking="disabled" if args.thinking == "disabled" else "enabled",
        max_concurrency=args.workers,
        max_tokens=args.judge_max_tokens,
        timeout=args.judge_timeout,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.scores)), exist_ok=True)
    lock = threading.Lock()
    t0 = time.time()
    nok = nbad = 0
    with open(args.scores, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(score_one, judge, r, questions, args) for r in pending]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if rec.get("error"):
                    nbad += 1
                else:
                    nok += 1
            if i % 500 == 0 or i == len(pending):
                rc = retry_counts()
                print(f"  {i}/{len(pending)} ok {nok} bad {nbad} {time.time() - t0:.0f}s "
                      f"retry429 {rc['429']} retryparse {rc['parse']}", flush=True)
    rc = retry_counts()
    print(f"done: ok {nok} bad {nbad} -> {args.scores} retry429 {rc['429']} retryparse {rc['parse']} "
          f"gaveup429 {rc['gaveup_429']} gaveupparse {rc['gaveup_parse']}", flush=True)
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
