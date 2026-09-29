#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-criterion HealthBench grading with the request and retry semantics behind the paper's tables.

This is ``eval/healthbench_judge.py run`` (same verbatim official GRADER_TEMPLATE, same
conversation rendering, same verdict parser, same grades format ``{"arm","id","k","met"}``), with
the four differences that decide WHICH items end up fully graded, and therefore the common item
sets of every table:

1. Three attempts per verdict. A call that fails in transport OR returns an unparsable /
   non-boolean ``criteria_met`` is re-asked up to three times in total before an error row
   (``met: null``) is written. (``healthbench_judge.py`` records a parse failure at once.)
2. Evaluation-error gating. Responses rows carrying a truthy ``error`` (the evaluation pass of
   that question failed, e.g. the second evaluator could not score it) are not graded, so the
   paired set is the ids every named arm answered cleanly.
3. A fixed prompt set. ``--ids FILE`` or ``--ids-from-grades FILE`` restricts grading to a given set;
   the latter reproduces the HealthBench-full prompt set: ids on which every arm in an earlier
   grades file has at least one verdict (common.prompt_set).
4. Passes. The pending (not yet successfully graded) criteria are re-submitted up to ``--passes``
   times (default 3); whatever still fails stays out of that arm.

Request settings: temperature 0, ``max_tokens`` 3000 (``--max-tokens``), and for the evaluation judge
thinking disabled (set ``RUBRIC_JUDGE_THINKING=disabled``; the third-family judge was called without
a thinking field, i.e. leave ``RUBRIC_JUDGE_ALT_THINKING`` unset).

  python3 grade_hb.py --questions healthbench_consensus.parquet \
      --responses base=base.jsonl --responses atomic_s42=atomic_s42.jsonl ... \
      --grades grades_consensus.jsonl [--judge-role ALT] [--ids ids.txt] [--passes 3] [--workers 16]

Judge endpoint: ``reward/judge_client.py`` ``judge_from_env(role)`` (RUBRIC_JUDGE[_<ROLE>]_*).
Roles: none = evaluation judge (DeepSeek-V4-Pro), LITE = second evaluator, ALT = third-family judge.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import common as K
from judge_client import JudgeUnavailableError, judge_from_env

hbj = K.hbj


def grade_one(judge, max_tokens, task):
    arm, rid, k, conversation, item = task
    prompt = hbj.grader_prompt(conversation, item)
    err = None
    for _ in range(3):
        try:
            raw = judge.complete([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0).raw
        except JudgeUnavailableError as exc:
            err = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            continue
        try:
            return {"arm": arm, "id": rid, "k": k, "met": hbj.parse_verdict(raw), "chars": len(raw)}
        except (ValueError, AttributeError) as exc:  # json.JSONDecodeError is a ValueError
            err = "parse:%s" % type(exc).__name__
    return {"arm": arm, "id": rid, "k": k, "met": None, "err": err}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", required=True, help="HealthBench-layout parquet")
    ap.add_argument("--responses", action="append", required=True, metavar="NAME=PATH",
                    help="arm name and responses JSONL (repeatable); NAME becomes the grades 'arm' key")
    ap.add_argument("--grades", required=True, help="append-only verdicts JSONL (resumable)")
    ap.add_argument("--source", default=None,
                    help="keep only responses rows whose data_source equals this (rows without the field are kept)")
    ap.add_argument("--ids", default=None, help="grade only these ids (JSON list or one per line)")
    ap.add_argument("--ids-from-grades", default=None,
                    help="grade only ids on which every arm of this earlier grades file has a verdict")
    ap.add_argument("--judge-role", default=None, help="judge_client role (default: evaluation judge; LITE; ALT)")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="grade only the first N paired ids")
    args = ap.parse_args()

    arms = hbj.parse_responses_arg(args.responses)
    qs = hbj.load_questions(args.questions)
    rows = {a: K.read_responses(p, args.source) if args.source else _clean(p) for a, p in arms.items()}
    ids = sorted(set(qs).intersection(*[set(v) for v in rows.values()]))
    if args.ids:
        want = set(K.read_ids(args.ids))
        ids = [i for i in ids if i in want]
    if args.ids_from_grades:
        want = K.prompt_set({"grades": args.ids_from_grades}, qs)
        ids = [i for i in ids if i in want]
    if args.limit:
        ids = ids[: args.limit]

    judge = judge_from_env(args.judge_role, max_concurrency=args.workers)
    grades_path = Path(args.grades)
    grades_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    for p in range(1, args.passes + 1):
        done = set(hbj.load_grades(args.grades))
        tasks = []
        for rid in ids:
            q = qs[rid]
            for arm in arms:
                conv = hbj.conversation_text(q["prompt"], rows[arm][rid]["response"])
                for k, item in enumerate(q["rubric"]):
                    if (arm, rid, k) not in done:
                        tasks.append((arm, rid, k, conv, item))
        print("pass %d: paired ids %d | pending %d | already graded %d" % (p, len(ids), len(tasks), len(done)), flush=True)
        if not tasks:
            break
        t0, n, bad = time.time(), 0, 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex, grades_path.open("a", encoding="utf-8") as out:
            for f in as_completed([ex.submit(grade_one, judge, args.max_tokens, t) for t in tasks]):
                r = f.result()
                with lock:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    out.flush()
                n += 1
                bad += r["met"] is None
                if n % 500 == 0:
                    print("  %d/%d (err %d) %.0f/min" % (n, len(tasks), bad, n / max(1e-9, (time.time() - t0) / 60)), flush=True)
        print("  pass %d done: %d graded, %d errors" % (p, n, bad), flush=True)


def _clean(path):
    """Rows with a response and no evaluation error (no data_source filter)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if not r.get("error") and r.get("response"):
                    out[str(r["id"])] = r
    return out


if __name__ == "__main__":
    main()
