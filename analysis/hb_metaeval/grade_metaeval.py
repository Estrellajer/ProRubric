#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HealthBench meta-evaluation: grade every (completion, criterion) row with a model grader.

Produces the grader labels behind the appendix paragraph "Agreement with physicians" and
Table ``tab:judge_agreement`` (the evaluation judge, DeepSeek-V4-Pro in the paper, on all
29,511 rows of HealthBench's public meta-evaluation file).

Prompt construction and output parsing are the official openai/simple-evals code (commit
652c89d0ca9df547706735883097e9537d40dc47; fetch it with ``fetch_simple_evals.sh``):
``HealthBenchMetaEval.__call__`` builds ``prompt_str = "\\n\\n".join("role: content")`` over the
row's ``prompt`` plus the ``completion`` as a final assistant turn and fills ``GRADER_TEMPLATE``
with it and the row's ``rubric``; ``parse_json_to_dict`` reads ``{"criteria_met", "explanation"}``.
Only the transport differs: the evaluation judge of ``reward/judge_client.py`` (no role by
default; ``--judge-role`` selects another), one user message, temperature 0, thinking disabled,
max_tokens 3000. The official loop retries unparsable output forever; here a row gets up to 6
attempts and is then written with ``criteria_met: null`` and an ``error`` (never dropped).

Input ``--data``: the public meta-evaluation file ``2025-05-07-06-14-12_oss_meta_eval.jsonl``
(simple-evals ``healthbench_meta_eval.py`` INPUT_PATH; one JSON object per row with ``prompt``,
``completion``, ``rubric``, ``category``, ``completion_id``, ``prompt_id``, ``binary_labels``,
``anonymized_physician_ids``).

Output ``--out``: append-only JSONL, one object per row, keyed by the row's 0-based line index
``i``: ``{"i", "completion_id", "prompt_id", "category", "criteria_met", "explanation", "judge",
"attempts"}`` (or ``criteria_met: null`` plus ``error``/``raw``). Resumable: rows already graded
with a label are skipped.

  python3 grade_metaeval.py --simple-evals-dir DEST --data DEST/2025-05-07-06-14-12_oss_meta_eval.jsonl \\
      --out grades_metaeval.jsonl [--limit 5] [--qpm 150] [--workers 32]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reward"))
from judge_client import JudgeUnavailableError, judge_from_env  # noqa: E402

MAX_TOKENS = 3000
ATTEMPTS = 6


def official_prompt(row, grader_template):
    # verbatim from HealthBenchMetaEval.__call__.fn
    convo_with_response = row["prompt"] + [dict(content=row["completion"], role="assistant")]
    prompt_str = "\n\n".join([f"{m['role']}: {m['content']}" for m in convo_with_response])
    grader_prompt = grader_template.replace("<<conversation>>", prompt_str)
    grader_prompt = grader_prompt.replace("<<rubric_item>>", row["rubric"])
    return grader_prompt


def grade(judge, i, row, grader_template, parse_json_to_dict):
    prompt, last, err = official_prompt(row, grader_template), None, None
    for attempt in range(ATTEMPTS):
        try:
            content = judge.complete([{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS, temperature=0).raw
        except (JudgeUnavailableError, ValueError) as exc:
            err = "%s: %s" % (type(exc).__name__, str(exc)[:300])
            continue
        last = content
        d = parse_json_to_dict(content)
        if d.get("criteria_met") is True or d.get("criteria_met") is False:
            return {"i": i, "completion_id": row["completion_id"], "prompt_id": row["prompt_id"], "category": row["category"],
                    "criteria_met": d["criteria_met"], "explanation": d.get("explanation", ""), "judge": judge.model,
                    "attempts": attempt + 1}
        err = "bad_json"
    return {"i": i, "completion_id": row["completion_id"], "prompt_id": row["prompt_id"], "category": row["category"],
            "criteria_met": None, "error": err, "raw": (last or "")[:500], "judge": judge.model}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="2025-05-07-06-14-12_oss_meta_eval.jsonl")
    ap.add_argument("--out", required=True, help="append-only grades JSONL (resumable)")
    ap.add_argument("--simple-evals-dir", default=None,
                    help="directory containing the simple_evals/ package (see fetch_simple_evals.sh); "
                         "omit if it is already importable via PYTHONPATH")
    ap.add_argument("--judge-role", default=None, help="judge_client role (default: none = evaluation judge)")
    ap.add_argument("--limit", type=int, default=None, help="grade at most N pending rows")
    ap.add_argument("--qpm", type=int, default=None, help="requests/minute (default: RUBRIC_JUDGE_[ROLE_]QPM)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.simple_evals_dir:
        sys.path.insert(0, os.path.abspath(a.simple_evals_dir))
    from simple_evals.healthbench_eval import GRADER_TEMPLATE, parse_json_to_dict

    judge = judge_from_env(a.judge_role, thinking="disabled", qpm=a.qpm, max_concurrency=a.workers)
    rows = [json.loads(l) for l in open(a.data)]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            if r.get("criteria_met") is not None:
                done.add(r["i"])
    todo = [(i, r) for i, r in enumerate(rows) if i not in done][: a.limit]
    print("rows %d | already graded %d | this run %d | judge %s | workers %d"
          % (len(rows), len(done), len(todo), judge.model, a.workers), flush=True)
    t0, n, bad = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        futures = [ex.submit(grade, judge, i, r, GRADER_TEMPLATE, parse_json_to_dict) for i, r in todo]
        for fu in as_completed(futures):
            g = fu.result()
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            bad += g.get("criteria_met") is None
            if n % 500 == 0 or n == len(todo):
                print("  %d/%d (errors %d) %.0f/min" % (n, len(todo), bad, n / max(1e-9, (time.time() - t0) / 60)), flush=True)


if __name__ == "__main__":
    main()
