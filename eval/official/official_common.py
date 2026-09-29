#!/usr/bin/env python3
"""Shared plumbing for the official-benchmark scorers in this directory.

Library module (no CLI). Provides

* response loading: ``--responses NAME=PATH`` arguments pointing at JSONL files with one
  object per row, ``{"id": str, "response": str, ...}`` (extra fields ignored, empty
  responses skipped); the response text is passed to the judge verbatim;
* the judge: the evaluation judge of ``reward/judge_client.py`` (no role unless
  ``--judge-role`` is given), always with temperature 0 and thinking disabled; each scorer
  passes the max_tokens of the original run;
* ``retry_parsed_call``: up to 3 judge calls until the benchmark's parser accepts the output;
* append-only JSONL grades (``<work-dir>/grades.<NAME>.jsonl``; re-running skips finished items);
* ``paired_summary``: per-run means and paired deltas with a percentile bootstrap
  (5000 resamples, seed 67).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reward"))
from judge_client import JudgeUnavailableError, judge_from_env  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def split_runs(value: str) -> list[str]:
    runs = [item.strip() for item in value.split(",") if item.strip()]
    if not runs:
        raise ValueError("--runs must contain at least one run name")
    return runs


def parse_responses_arg(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"--responses must be NAME=PATH, got: {value!r}")
        runs[name.strip()] = Path(path.strip())
    return runs


def load_responses(path: Path) -> dict[str, str]:
    responses: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            response = row.get("response")
            if response:
                responses[str(row["id"])] = str(response)
    return responses


def grades_path(work_dir: Path, run_name: str) -> Path:
    return work_dir / f"grades.{safe_name(run_name)}.jsonl"


def load_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        _, _, text = text.partition("\n")
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("judge output is not a JSON object")
    return value


def configure_judge(role: str | None, qpm: int | None, workers: int):
    """Evaluation judge: temperature 0 (set per call), thinking disabled."""
    return judge_from_env(role, thinking="disabled", qpm=qpm, max_concurrency=workers)


def call_judge(judge, messages: list[dict], max_tokens: int = 3000) -> tuple[str | None, str | None]:
    try:
        return judge.complete(messages, max_tokens=max_tokens, temperature=0).raw, None
    except (JudgeUnavailableError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:300]}"


def retry_parsed_call(
    judge, messages: list[dict], parser: Callable[[str], object], *, max_tokens: int = 3000, attempts: int = 3
) -> tuple[object | None, str | None, str | None]:
    last_error = None
    last_raw = None
    for _ in range(attempts):
        raw, error = call_judge(judge, messages, max_tokens=max_tokens)
        last_raw = raw
        if raw is None:
            last_error = error
            continue
        try:
            return parser(raw), None, raw
        except Exception as exc:  # noqa: BLE001
            last_error = f"parse:{type(exc).__name__}:{exc}"
    return None, last_error, last_raw


def percentile_interval(values: list[float], *, samples: int = 5000, seed: int = 67) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def paired_summary(per_run: dict[str, dict[str, float]], *, scale: float = 1.0) -> dict:
    runs = list(per_run)
    common = set.intersection(*(set(per_run[run]) for run in runs)) if runs else set()
    result = {"common_ids": len(common), "models": {}, "paired_deltas": {}}
    for run in runs:
        values = list(per_run[run].values())
        result["models"][run] = {
            "n": len(values),
            "mean": round(scale * sum(values) / len(values), 4) if values else None,
            "common_mean": round(scale * sum(per_run[run][key] for key in common) / len(common), 4) if common else None,
        }
    for i, left in enumerate(runs):
        for right in runs[i + 1 :]:
            pair_common = set(per_run[left]) & set(per_run[right])
            deltas = [scale * (per_run[right][key] - per_run[left][key]) for key in sorted(pair_common)]
            result["paired_deltas"][f"{right} minus {left}"] = {
                "n": len(deltas),
                "delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
                "ci95": [round(x, 4) for x in percentile_interval(deltas)] if deltas else None,
            }
    return result


def add_run_args(cmd, *, default_workers: int = 12) -> None:
    cmd.add_argument("--responses", action="append", required=True, metavar="NAME=PATH",
                     help="responses JSONL {id, response} for one named run (repeatable)")
    cmd.add_argument("--work-dir", type=Path, required=True, help="directory for grades.<NAME>.jsonl")
    cmd.add_argument("--limit", type=int, default=0, help="grade only the first N matched ids per run")
    cmd.add_argument("--judge-role", default=None, help="judge_client role (default: none = evaluation judge)")
    cmd.add_argument("--qpm", type=int, default=None, help="requests/minute (default: RUBRIC_JUDGE_[ROLE_]QPM)")
    cmd.add_argument("--workers", type=int, default=default_workers)


def add_summarize_args(cmd) -> None:
    cmd.add_argument("--runs", required=True, help="comma-separated run NAMEs (as given to run --responses)")
    cmd.add_argument("--work-dir", type=Path, required=True)
