#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResearchQA official rubric-coverage score with the evaluation judge.

Produces the ResearchQA column of Table ``tab:main_results``.

Protocol (realliyifei/ResearchQA ``compute_coverage.py``, commit
747a9a1330f097a0e20672240cb47e3cf02500ae): per answer, the rubric items are judged in batches
of 8 with the upstream prompt (``build_prompt``, reproduced verbatim below as HEAD +
"Response: ..." + "Questions: ..." + "Output:") on a 5-level scale (Not at all / Barely /
Moderately / Mostly / Completely -> 1..5); each level is normalized (x-1)/4 and averaged over
the rubric; coverage = mean over answers x 100. An answer counts only if every batch parsed to
exactly as many labels as it has rubric items (up to 3 attempts per batch). The official judge
(gpt-4.1-mini, T=0) is replaced by the evaluation judge (temperature 0, thinking disabled,
max_tokens 3000).

Inputs
  --questions   the ResearchQA valid split (703 queries), either the upstream ``valid.json``
                (Hugging Face dataset realliyifei/ResearchQA; list of {"id", "query", "rubric":
                [{"rubric_item", ...}]}) or a parquet with ``extra_info.id`` and
                ``extra_info.rubric`` = [{"criterion"}] in the same order. Ids are compared with
                an optional "researchqa_valid:" prefix removed.
  --responses   NAME=PATH, JSONL {"id", "response"}
Outputs
  <work-dir>/grades.<NAME>.jsonl  one row per (id, batch start): {"run","id","batch","scores":[1..5]|null,"err"}
  <work-dir>/summary.json         per-run coverage (x100) and paired deltas with bootstrap CIs

  python3 researchqa_official_coverage.py run --questions valid.json --work-dir rqa \\
      --responses rubric_rl=rubric_rl.jsonl --responses prorubric=prorubric.jsonl
  python3 researchqa_official_coverage.py summarize --questions valid.json --work-dir rqa --runs rubric_rl,prorubric
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import official_common as common  # noqa: E402

RATINGS = {"Not at all": 1, "Barely": 2, "Moderately": 3, "Mostly": 4, "Completely": 5}
BATCH = 8
MAX_TOKENS = 3000
ID_PREFIX = "researchqa_valid:"
_lock = threading.Lock()

HEAD = ("Please judge the following questions based on the response below.\n"
        "For each question, select one of the following ratings to indicate the extent to which the response addresses the question:\n"
        "Not at all, Barely, Moderately, Mostly, Completely\n\n"
        "Definitions:\n"
        "- Not at all: *totally uninferable*\n"
        "- Barely: *unmentioned but inferrable*\n"
        "- Moderately: *mentioned but misses important details*\n"
        "- Mostly: *mentioned but misses some details*\n"
        "- Completely: *mentioned with sufficient details*\n\n"
        "Only output one of the five phrases for each question, separated by newlines, and nothing else.\n\n")


def build_prompt(response, questions):
    return HEAD + f"Response: {response}\n" + "Questions:\n" + "\n".join(questions) + "\n\nOutput:"


def norm_id(value):
    value = str(value)
    return value[len(ID_PREFIX):] if value.startswith(ID_PREFIX) else value


def load_rubrics(path):
    out = {}
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq
        for r in pq.read_table(path).to_pylist():
            ei = r["extra_info"]
            out[norm_id(ei["id"])] = [str(x["criterion"]) for x in ei["rubric"]]
    else:
        for item in json.load(open(path, encoding="utf-8")):
            out[norm_id(item["id"])] = [str(x["rubric_item"]) for x in item["rubric"]]
    return out


def load_responses(path):
    return {norm_id(k): v for k, v in common.load_responses(Path(path)).items()}


def parse(content, n):
    lines = [l.strip().strip("-*•").strip() for l in content.strip().splitlines() if l.strip()]
    vals = []
    for l in lines:
        l = l.split(".", 1)[1].strip() if l[:2].rstrip(".").isdigit() and "." in l[:3] else l
        for k in RATINGS:
            if l.lower().startswith(k.lower()):
                vals.append(RATINGS[k]); break
    return vals if len(vals) == n else None


def grade(judge, task):
    run, rid, b, response, qs = task
    err = None
    for attempt in range(3):
        content, cerr = common.call_judge(judge, [{"role": "user", "content": build_prompt(response, qs)}], max_tokens=MAX_TOKENS)
        if content is None:
            err = cerr; continue
        vals = parse(content, len(qs))
        if vals is not None:
            return {"run": run, "id": rid, "batch": b, "scores": vals}
        err = "parse_mismatch:%d/%d" % (len(parse(content, 10**6) or []), len(qs))
    return {"run": run, "id": rid, "batch": b, "scores": None, "err": err}


def run_cmd(a):
    judge = common.configure_judge(a.judge_role, a.qpm, a.workers)
    rub = load_rubrics(a.questions)
    a.work_dir.mkdir(parents=True, exist_ok=True)
    for run, responses_path in common.parse_responses_arg(a.responses).items():
        resp = load_responses(responses_path); done = set()
        gp = common.grades_path(a.work_dir, run)
        if os.path.exists(gp):
            for line in open(gp):
                g = json.loads(line)
                if g.get("scores") is not None: done.add((g["id"], g["batch"]))
        ids = sorted(i for i in resp if i in rub)
        if a.limit: ids = ids[:a.limit]
        tasks = []
        for rid in ids:
            qs = rub[rid]
            for b in range(0, len(qs), BATCH):
                if (rid, b) not in done: tasks.append((run, rid, b, resp[rid], qs[b:b + BATCH]))
        print(run, "answers", len(ids), "| batches pending", len(tasks), "| done", len(done), flush=True)
        bad = 0
        with ThreadPoolExecutor(max_workers=a.workers) as ex, open(gp, "a") as out:
            for f in as_completed([ex.submit(grade, judge, t) for t in tasks]):
                r = f.result(); bad += r.get("scores") is None
                with _lock: out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
        print(run, "done; batch errors", bad, flush=True)


def summarize(a):
    rub = load_rubrics(a.questions); per = {}
    for run in common.split_runs(a.runs):
        sc = {}
        for line in open(common.grades_path(a.work_dir, run)):
            g = json.loads(line)
            if g.get("scores") is not None: sc.setdefault(norm_id(g["id"]), {})[g["batch"]] = g["scores"]
        cov = {}
        for rid, batches in sc.items():
            n = len(rub[rid]); vals = []
            for b in range(0, n, BATCH):
                if b not in batches: vals = None; break
                vals.extend(batches[b])
            if vals and len(vals) == n: cov[rid] = sum((v - 1) / 4 for v in vals) / n
        per[run] = cov
    runs = list(per); common_ids = set.intersection(*(set(per[r]) for r in runs))
    rng = random.Random(0)

    def boot(vals):
        m = len(vals); out = sorted(sum(vals[rng.randrange(m)] for _ in range(m)) / m for _ in range(2000)); return round(100 * out[50], 2), round(100 * out[1950], 2)

    res = {"judge": "evaluation judge (substitute for gpt-4.1-mini)", "protocol": "ResearchQA compute_coverage.py: 5-level Likert, (x-1)/4, batches of 8, T=0", "split": "valid (703)", "common_answers": len(common_ids), "coverage_pct": {}, "paired_deltas_pct": {}}
    for r in runs:
        allv = list(per[r].values()); res["coverage_pct"][r] = {"n": len(allv), "all": round(100 * sum(allv) / len(allv), 2), "common": round(100 * sum(per[r][i] for i in common_ids) / len(common_ids), 2)}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            # sorted: the bootstrap resamples by position, so fix the order (a bare set is hash-ordered)
            d = [per[runs[j]][k] - per[runs[i]][k] for k in sorted(common_ids)]
            res["paired_deltas_pct"]["%s minus %s" % (runs[j], runs[i])] = {"delta": round(100 * sum(d) / len(d), 2), "ci95": boot(d)}
    json.dump(res, open(os.path.join(a.work_dir, "summary.json"), "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="judge every rubric batch of every answer")
    r.add_argument("--questions", required=True, help="ResearchQA valid.json (or parquet with extra_info.id/rubric)")
    common.add_run_args(r)
    s = sub.add_parser("summarize", help="coverage per run and paired deltas")
    s.add_argument("--questions", required=True, help="ResearchQA valid.json (or parquet with extra_info.id/rubric)")
    common.add_summarize_args(s)
    a = ap.parse_args(); run_cmd(a) if a.cmd == "run" else summarize(a)
