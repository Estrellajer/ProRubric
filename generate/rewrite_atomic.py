#!/usr/bin/env python3
"""atomic-rw (science control): rewrite each atomic criterion ONE-TO-ONE in ProRubric's style.

The science control of Section 5.3 ("atomic-rw") separates "the criterion text was rewritten" from "criteria are
grouped": every atomic criterion is rewritten on its own into the achieves / fails style of ProRubric's dimensions,
while count, order, weight and aggregation stay those of Rubric-RL. Only the body after the RaR-Science
"<title>: <Category> Criteria:" prefix is rewritten; the prefix is kept. Generation uses the same generator and request
settings as ProRubric's dimensions (the evaluation judge role, temperature 0, 3,000 max tokens, thinking disabled) and
at most two regenerations; a question whose generation keeps failing keeps its original text (it is never dropped).

    python3 rewrite_atomic.py smoke --input rar-science/train.parquet --output rw_smoke.json --n 5
    python3 rewrite_atomic.py run   --input rar-science/train.parquet --output rw.jsonl [--rpm 150 --workers 24]

Output (run): one JSON line per question, {"id", "ok", "criteria": [rewritten criterion strings], "err"?}; resumable
(ids already written with ok=true are skipped). Package it with ../data/build_atomic_rw.py.
Input rows need extra_info.id, extra_info.problem and extra_info.rubric (list of {criterion, weight}).
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reward"))

import pyarrow.parquet as pq  # noqa: E402

from generate_dimensions import MAX_TOKENS, call_generator, parse_json_to_dict  # noqa: E402
from judge_client import RubricJudge  # noqa: E402

PREFIX = re.compile(r"^(.*?\b(?:Essential|Important|Optional|Pitfall)\s+Criteria:\s*)(.+)$", re.S)
MAX_REGEN = 2

PROMPT = """You are rewriting the items of an atomic grading checklist for an answer to a science question. Each item is currently phrased as a small "the response should mention X" check. Rewrite EACH item, separately and one-to-one, so that it:
1. States what a GOOD answer ACHIEVES on that specific point for THIS question, and what would make an answer FAIL that point;
2. Can be judged only by reading and understanding the answer -- NOT by scanning for a keyword or a single phrase;
3. Is NOT phrased as a list of terms to check off.

Hard constraints:
- Exactly one rewritten item per input item, same idx. Do not merge, split, drop, reorder or add items.
- Do not add requirements the original item does not contain, and do not remove any it does.
- Keep the polarity: an item that describes something the answer must NOT do stays a must-NOT item.
- 1-3 sentences per item, self-contained, no reference to other items.

# Question
{question}

# Atomic items (idx: text)
{items}

Return ONLY a JSON object, no markdown:
{{"items": [{{"idx": <int>, "criterion": "<rewritten text>"}}, ...]}}"""


def split(c):
    m = PREFIX.match(c)
    return (m.group(1), m.group(2).strip()) if m else ("", c.strip())


def rows(path):
    for r in pq.read_table(path, columns=["extra_info"]).to_pylist():
        e = r["extra_info"]
        yield e["id"], e["problem"], [x["criterion"] for x in e["rubric"]]


def one(judge, qid, problem, crits):
    parts = [split(c) for c in crits]
    items = "\n".join(f"{i}: {body}" for i, (_, body) in enumerate(parts))
    last = None
    for _ in range(MAX_REGEN + 1):
        content, err = call_generator(judge, PROMPT.format(question=problem, items=items))
        if content is None:
            last = err
            continue
        try:
            got = {int(x["idx"]): str(x["criterion"]).strip() for x in parse_json_to_dict(content)["items"]}
        except Exception as e:  # noqa: BLE001 -- any malformed output is a failed attempt
            last = f"parse:{type(e).__name__}"
            continue
        if sorted(got) != list(range(len(parts))) or not all(got.values()):
            last = f"bad_indices:{sorted(got)[:12]} want {len(parts)}"
            continue
        return {"id": qid, "ok": True, "criteria": [pre + got[i] for i, (pre, _) in enumerate(parts)]}
    return {"id": qid, "ok": False, "err": last, "criteria": crits}  # keep the original text; never drop a question


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["smoke", "run"])
    ap.add_argument("--input", required=True, help="atomic science training parquet (RaR-Science train split)")
    ap.add_argument("--output", required=True, help="run: resumable jsonl; smoke: side-by-side json")
    ap.add_argument("--n", type=int, default=5, help="smoke: number of questions")
    ap.add_argument("--rpm", type=int, default=150)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    judge = RubricJudge(qpm=a.rpm, max_concurrency=a.workers, max_tokens=MAX_TOKENS, thinking="disabled")
    data = list(rows(a.input))
    if a.mode == "smoke":
        out = [one(judge, *d) for d in data[: a.n]]
        json.dump(out, open(a.output, "w"), ensure_ascii=False, indent=1)
        for d, o in zip(data[: a.n], out):
            print("=" * 20, d[0], "ok" if o["ok"] else o["err"])
            for x, y in zip(d[2], o["criteria"]):
                print("  -", x, "\n  +", y)
        return
    done = set()
    if os.path.exists(a.output):
        for line in open(a.output):
            r = json.loads(line)
            if r["ok"]:
                done.add(r["id"])
    todo = [d for d in data if d[0] not in done]
    print(f"rows {len(data)} done {len(done)} todo {len(todo)}", flush=True)
    lock = threading.Lock()
    n = bad = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.output, "a") as fh:
        for f in as_completed([ex.submit(one, judge, *d) for d in todo]):
            r = f.result()
            n += 1
            bad += not r["ok"]
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            if n % 500 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} failed {bad} {n / max(1e-9, (time.time() - t0) / 60):.0f}/min", flush=True)
    print(f"done {n} failed {bad}", flush=True)


if __name__ == "__main__":
    main()
