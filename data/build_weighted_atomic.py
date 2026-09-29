#!/usr/bin/env python3
"""Weighted control: Rubric-RL's medical checklist with re-weighted criteria (Table 3, "reweighted (Weighted)").

Is the collapse of Rubric-RL a weighting problem rather than a compensation problem? This arm keeps every atomic
criterion, its text and its independent payout -- only the weights change:

  CRITICAL      (answering it wrong makes the answer wrong or unsafe)        weight x 3
  NORMAL        (everything else)                                            weight x 1
  FORMAT        (pure structure / length / formatting / summary demands)      weight x 0
  DUPLICATE     (near-duplicate of another criterion on the same list)        weight x 0
  OVERCLINICAL  (extra tests, imaging, referral or escalation a careful       weight x -1
                 clinician would not require here)

Negative weight is reserved for over-clinical demands; formatting and duplicate items are only zeroed. The rule is
deliberately coarse: a finely tuned weighting would amount to choosing a favourable configuration for one baseline.

The labels come from one call per question with all of that question's criteria, by a model that is neither the
training-reward judge nor the evaluation judge (paper: Doubao-lite; role LITE by default, see
../reward/judge_client.py). Request settings: temperature 0, 3,000 max tokens, thinking disabled, up to three tries,
then all-NORMAL as the fallback (counted).

    python3 build_weighted_atomic.py --input rubrichub-medical/train.parquet --work work/ --dry-run 20
    python3 build_weighted_atomic.py --input rubrichub-medical/train.parquet --work work/ --output release/medical-weighted

Resumable: labels are appended to <work>/labels.jsonl and re-runs skip ids already there.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reward"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from _common import tree_hash  # noqa: E402

MAX_TOKENS = 3000
MULT = {"CRITICAL": 3.0, "NORMAL": 1.0, "FORMAT": 0.0, "DUPLICATE": 0.0, "OVERCLINICAL": -1.0}

PROMPT = """You are auditing the scoring checklist of a medical question so that its items can be weighted.

QUESTION:
{q}

CHECKLIST ITEMS:
{items}

Label every item with exactly one of:
- CRITICAL: without it the answer to THIS question is wrong or unsafe. Facts that are merely useful are not critical.
- OVERCLINICAL: it requires extra tests, imaging, referral or escalation that a careful clinician would not require
  for this question. Only clinical actions count here, never wording or structure.
- FORMAT: it only demands structure, headings, length, a summary section or formatting, not content.
- DUPLICATE: its content requirement is already covered by an earlier item on this list.
- NORMAL: anything else. Most items are NORMAL; a typical list has 1-3 CRITICAL and few or no OVERCLINICAL.

Return ONLY a JSON object {{"labels": ["...", "..."]}} with exactly {n} labels, in the order the items are listed."""

_lock = threading.Lock()


def log(msg):
    print(time.strftime("[%H:%M:%S]") + " " + msg, flush=True)


def rubric_of(row):
    e = row["extra_info"]
    e = json.loads(e) if isinstance(e, str) else e
    return e, e["rubric"]


def question_of(row):
    p = row["prompt"]
    if isinstance(p, list):
        return "\n\n".join(str(m.get("content", "")) for m in p)
    return str(p)


def call_labeller(judge, prompt):
    from judge_client import JudgeUnavailableError
    try:
        return judge.complete([{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS, temperature=0).raw
    except JudgeUnavailableError:
        return None


def label_one(judge, row):
    e, rub = rubric_of(row)
    rid = str(e.get("id"))
    items = "\n".join("%d. %s" % (i, c["criterion"]) for i, c in enumerate(rub, 1))
    prompt = PROMPT.format(q=question_of(row)[:6000], items=items[:12000], n=len(rub))
    for _ in range(3):
        content = call_labeller(judge, prompt)
        if content is None:
            continue
        try:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            labels = json.loads(m.group(0))["labels"]
        except Exception:  # noqa: BLE001 -- malformed output is a failed try
            continue
        labels = [str(x).strip().upper() for x in labels]
        if len(labels) == len(rub) and all(x in MULT for x in labels):
            return {"id": rid, "labels": labels}
    return {"id": rid, "labels": ["NORMAL"] * len(rub), "fallback": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="atomic medical training parquet (Rubric-RL's training set)")
    ap.add_argument("--work", required=True, help="work directory for labels.jsonl (resumable)")
    ap.add_argument("--output", help="release directory to write (omit with --dry-run)")
    ap.add_argument("--dry-run", type=int, default=0, help="label N questions and print the weight changes")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rpm", type=int, default=600)
    ap.add_argument("--judge-role", default="LITE", help="judge_client role of the labeller (default LITE)")
    a = ap.parse_args()
    if not a.dry_run and not a.output:
        ap.error("--output is required unless --dry-run is given")
    from judge_client import judge_from_env

    judge = judge_from_env(a.judge_role, qpm=a.rpm, max_concurrency=a.workers, max_tokens=MAX_TOKENS,
                           thinking="disabled")
    os.makedirs(a.work, exist_ok=True)
    labels_path = os.path.join(a.work, "labels.jsonl")
    table = pq.read_table(a.input)
    rows = table.to_pylist()
    log("source: %d rows" % len(rows))
    done = {}
    if os.path.exists(labels_path):
        for line in open(labels_path):
            r = json.loads(line)
            done[r["id"]] = r["labels"]
        log("already labelled: %d" % len(done))
    todo = [r for r in rows if str(rubric_of(r)[0].get("id")) not in done]
    if a.dry_run:
        todo = todo[: a.dry_run]
    log("to label: %d" % len(todo))
    t0 = time.time()
    n = fb = 0
    out = open(labels_path, "a")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(label_one, judge, r) for r in todo]):
            r = f.result()
            with _lock:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
            done[r["id"]] = r["labels"]
            n += 1
            fb += bool(r.get("fallback"))
            if n % 200 == 0:
                log("labelled %d/%d (fallback %d) %.0f/min" % (n, len(todo), fb, n / max(1e-9, (time.time() - t0) / 60)))
    out.close()
    log("labelling done: %d new, %d fallback" % (n, fb))
    counts = {k: 0 for k in MULT}
    for labels in done.values():
        for x in labels:
            counts[x] += 1
    log("label mix: %s" % counts)
    if a.dry_run:
        for r in todo[:5]:
            e, rub = rubric_of(r)
            labels = done[str(e.get("id"))]
            print("--", str(e.get("id")))
            for c, lab in zip(rub, labels):
                print("   %-9s %6.1f -> %6.1f  %s" % (lab, c["weight"], c["weight"] * MULT[lab], c["criterion"][:90]))
        return

    # the release: same rows, only extra_info.rubric weights change
    new_rows = []
    changed = 0
    for row in rows:
        e, rub = rubric_of(row)
        labels = done.get(str(e.get("id")))
        if labels is None or len(labels) != len(rub):
            labels = ["NORMAL"] * len(rub)
        e2 = json.loads(json.dumps(e))
        for c, lab in zip(e2["rubric"], labels):
            w = float(c["weight"]) * MULT[lab]
            changed += w != float(c["weight"])
            c["weight"] = w
            c["weight_label"] = lab
        r2 = dict(row)
        r2["extra_info"] = json.dumps(e2, ensure_ascii=False) if isinstance(row["extra_info"], str) else e2
        new_rows.append(r2)
    os.makedirs(a.output, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(new_rows), os.path.join(a.output, "train.parquet"))
    man = {"arm": "Weighted (Rubric-RL, weighted criteria)", "rows_out": len(new_rows),
           "labeller_model": judge.model, "multipliers": MULT, "criteria_relabelled": changed, "label_mix": counts,
           "rule": "CRITICAL x3, NORMAL x1, FORMAT/DUPLICATE x0, OVERCLINICAL x-1; every atomic criterion kept, "
                   "payout still independent",
           "built_at": datetime.now(timezone.utc).isoformat()}
    json.dump(man, open(os.path.join(a.output, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    open(os.path.join(a.output, "README.md"), "w").write(
        "# Weighted medical checklist\n\nThe atomic medical checklist with re-weighted criteria: critical items x3,\n"
        "formatting and duplicate items zeroed, over-clinical demands negative, everything else unchanged. Criteria\n"
        "text, count and independent payout are identical to the Rubric-RL training set; only\n"
        "extra_info.rubric[*].weight differs (and a weight_label field records the label).\n")
    tree, nfiles, total = tree_hash(a.output)
    print(json.dumps({"rows": len(new_rows), "criteria_relabelled": changed, "label_mix": counts,
                      "tree": tree, "file_count": nfiles, "total_bytes": total}))


if __name__ == "__main__":
    main()
