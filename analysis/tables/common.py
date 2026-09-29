# -*- coding: utf-8 -*-
"""Shared readers and small statistics for the table scripts in this directory.

Library module (no CLI). Everything here reuses ``eval/healthbench_judge.py`` for the
HealthBench question loader and the official per-question score, so the tables and the
grader agree on one formula:

    score(verdicts, rubric) = clip(sum(weight of met criteria) / sum(positive weights), 0, 1)

Input file kinds used across the scripts (all JSONL unless noted):

* questions: HealthBench-layout parquet (``prompt``, ``extra_info.rubric``; ids from
  ``extra_info.id``), read with ``healthbench_judge.load_questions``.
* responses: one row per question, ``{"id", "response", "score", "satisfied", "error",
  "data_source"}``. ``score``/``satisfied`` are the second evaluator's per-question
  HealthBench score and per-criterion verdicts (Doubao-lite in the paper; the same fields
  ``healthbench_judge.py summarize`` reads). A row with a truthy ``error`` is a failed
  evaluation of that question. ``data_source`` is optional; when present, rows whose
  ``data_source`` differs from the suite being read (``healthbench_consensus`` /
  ``healthbench_full``) are skipped, so one file may hold both suites.
* grades: verdict rows ``{"arm", "id", "k", "met"}`` as written by
  ``eval/healthbench_judge.py run`` or ``grade_hb.py`` (``met`` is null on a failed call).
  One grades file may hold many arms; a spec names the file and the arm key inside it.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PKG, "eval"))
sys.path.insert(0, os.path.join(PKG, "reward"))

import healthbench_judge as hbj  # noqa: E402

score = hbj.score  # official HealthBench per-question score (see module docstring)


def load_spec(path: str, section: str) -> dict:
    """Read one section of the arm-spec JSON; relative paths resolve against the spec's directory."""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    return _resolve(spec[section], base)


def _resolve(node, base):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in _PATH_KEYS and isinstance(v, str):
                out[k] = _path(v, base)
            elif k in _PATH_KEYS and isinstance(v, list) and all(isinstance(x, str) for x in v):
                out[k] = [_path(x, base) for x in v]
            else:
                out[k] = _resolve(v, base)
        return out
    if isinstance(node, list):
        return [_resolve(x, base) for x in node]
    return node


# keys whose string (or list-of-string) values are file paths
_PATH_KEYS = {"path", "responses", "hbfull_responses", "consensus", "hbfull", "ids", "grades",
              "medqa", "researchqa", "healthbench_full", "subset_ids", "consensus_ids",
              "writingbench", "creative_writing_v3", "arena_hard_v2", "gpqa_diamond"}


def _path(v, base):
    return v if os.path.isabs(v) else os.path.normpath(os.path.join(base, v))


def rubrics(parquet: str) -> dict[str, list[dict]]:
    return {rid: q["rubric"] for rid, q in hbj.load_questions(parquet).items()}


def read_lite(path: str, source: str, rub: dict[str, list[dict]]):
    """id -> (second-evaluator score, response chars) over clean rows of one suite.

    A later clean row for the same id wins. Also re-derives each score from ``satisfied`` +
    rubric and returns (mismatches, checked) as a consistency check."""
    out, mism, checked = {}, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("data_source") not in (None, source) or r.get("error") or r.get("score") is None:
                continue
            rid = str(r["id"])
            out[rid] = (float(r["score"]), len(r.get("response") or ""))
            sat = r.get("satisfied")
            if rid in rub and sat is not None and len(sat) == len(rub[rid]):
                checked += 1
                mism += abs(score([bool(x) for x in sat], rub[rid]) - float(r["score"])) > 1e-6
    return out, mism, checked


def read_responses(path: str, source: str) -> dict[str, dict]:
    """id -> row for every row of the suite without an ``error`` (the paired-set filter of the graders)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("data_source") not in (None, source) or r.get("error"):
                continue
            out[str(r["id"])] = r
    return out


class Grades:
    """Verdicts for a set of (grades file, arm key) pairs, read once per file.

    ``met[(path, arm)][(id, k)] -> bool`` holds successful verdicts; ``attempted[(path, arm)]``
    is every id that has any row (met or failed), i.e. the ids the grader was pointed at."""

    def __init__(self, refs):
        want = defaultdict(set)
        for ref in refs:
            want[ref["path"]].add(ref["arm"])
        self.met = defaultdict(dict)
        self.attempted = defaultdict(set)
        for path, arms in want.items():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["arm"] not in arms:
                        continue
                    key = (path, r["arm"])
                    self.attempted[key].add(str(r["id"]))
                    if r.get("met") is not None:
                        self.met[key][(str(r["id"]), int(r["k"]))] = bool(r["met"])

    def full(self, ref, rid, rubric) -> bool:
        g = self.met[(ref["path"], ref["arm"])]
        return all((rid, k) in g for k in range(len(rubric)))

    def item(self, ref, rid, rubric) -> float:
        g = self.met[(ref["path"], ref["arm"])]
        return score([g[(rid, k)] for k in range(len(rubric))], rubric)

    def per_id(self, ref, rub, pool=None) -> dict[str, float]:
        """id -> score for every id (of ``pool``, default all rubric ids) the arm is fully graded on."""
        ids = rub if pool is None else pool
        return {rid: self.item(ref, rid, rub[rid]) for rid in ids if rid in rub and self.full(ref, rid, rub[rid])}


def prompt_set(cfg: dict | None, rub: dict) -> set[str]:
    """The HealthBench-full prompt set.

    ``{"grades": PATH}``: ids on which EVERY arm present in that grades file has at least one
    successful verdict (the paper's 4,563-prompt set: the seed-42 grading run, whose own paired
    set was the ids every graded arm answered without an evaluation error).
    ``{"ids": PATH}``: a JSON list or one id per line. Absent: every question id."""
    if not cfg:
        return set(rub)
    if "ids" in cfg:
        return set(read_ids(cfg["ids"]))
    per_arm = defaultdict(set)
    with open(cfg["grades"], encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("met") is not None:
                per_arm[r["arm"]].add(str(r["id"]))
    return set.intersection(*per_arm.values())


def read_ids(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith(("[", "{")):
        obj = json.loads(text)
        return [str(x) for x in (obj["ids"] if isinstance(obj, dict) else obj)]
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def sd(v):
    """Sample standard deviation (ddof=1), as everywhere in the paper."""
    return statistics.stdev(v) if len(v) > 1 else float("nan")


def boot_ci(vals, seed=0, B=1000):
    """Percentile bootstrap of the mean: sorted[int(.025 B)], sorted[int(.975 B)]."""
    import random
    rng = random.Random(seed)
    m = len(vals)
    s = sorted(sum(vals[rng.randrange(m)] for _ in range(m)) / m for _ in range(B))
    return s[int(0.025 * B)], s[int(0.975 * B)]


# two-sided 95% Student-t quantiles t_{0.975, df}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_interval(diffs):
    """Mean, sd (ddof=1) and 95% t-interval (df = n-1) of paired per-seed differences."""
    n = len(diffs)
    m = statistics.mean(diffs)
    if n < 2:
        return m, float("nan"), (float("nan"), float("nan"))
    s = sd(diffs)
    h = T975[n - 1] * s / math.sqrt(n)
    return m, s, (m - h, m + h)


def dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
