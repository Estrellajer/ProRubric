# -*- coding: utf-8 -*-
"""Re-exposure fixing rate from training rollout dumps (supplementary to the appendix "Training dynamics").

A complement to the checkpoint-based fixing rates of score_checkpoints.py that needs no extra judge calls: it reads
the training judge's own verdicts from the rollout dump of a run trained with `trainer.rollout_data_dir` set.

Each dump file <step>.jsonl holds one record per rollout; every record carries its group's 8-rollout x K-dimension
training-judge verdict matrix (`rubric_judge_satisfied`, identical within a group), the group id
(`rubric_judge_group_id`), `judge_failed`, and `input` (the rendered prompt; the text before "\\nassistant" is the
question key). For a prompt drawn twice, a dimension unsatisfied at the first exposure (fewer than half of the 8
rollouts satisfy it) is "fixed" if at least half satisfy it at the second. Questions are binned by their
first-exposure state (zero / middle / one_short / all over majority-satisfied dimensions, as in
score_checkpoints.py), by the stage of the first exposure, and by the dimension's own first-exposure rate
(p1 = 0/8 vs 1-3/8: the regression-to-the-mean control).

--upto N uses steps 1..N only, for EVERY run, so a table from runs still training is compared with a reference on
the same window (second exposures are censored at N either way).

    python3 reexposure.py --run "ProRubric s42=DUMP_DIR" --run "Rubric-RL s42=DUMP_DIR" [--upto 300] --out reexposure.json
"""
import argparse, collections, json, os

STATES = ("zero", "middle", "one_short")


def state(maj):
    k, s = len(maj), sum(maj)
    return "zero" if s == 0 else ("all" if s == k else ("one_short" if s == k - 1 else "middle"))


def run(D, upto):
    first, fx, nprompts, npairs = {}, collections.defaultdict(lambda: [0, 0]), 0, 0
    last = max(int(f.split(".")[0]) for f in os.listdir(D) if f.endswith(".jsonl"))
    upto = min(upto, last if last >= 300 else last - 1)   # while training, the newest file may still be written
    for step in range(1, upto + 1):
        seen = set()
        for l in open("%s/%d.jsonl" % (D, step)):
            r = json.loads(l)
            g = r["rubric_judge_group_id"]
            if r.get("judge_failed") or g in seen: continue
            seen.add(g)
            M = r["rubric_judge_satisfied"]; M = json.loads(M) if isinstance(M, str) else M
            if not M or not M[0]: continue
            key = r["input"].split("\nassistant")[0]; k = len(M[0])
            p = [sum(row[d] for row in M) / len(M) for d in range(k)]
            if key not in first:
                first[key] = (step, p); nprompts += 1; continue
            t1, p1 = first[key]; npairs += 1
            s = state([x >= 0.5 for x in p1]); stage = "t1<=100" if t1 <= 100 else ("t1 101-200" if t1 <= 200 else "t1>200")
            for d in range(k):
                if p1[d] >= 0.5: continue
                b = "p1=0/8" if p1[d] == 0 else "p1=1-3/8"
                for cell in ((s, "all", "all"), (s, stage, "all"), (s, "all", b)):
                    f = fx[cell]; f[0] += 1; f[1] += p[d] >= 0.5
    return {"steps_used": upto, "prompts": nprompts, "pairs": npairs,
            "cells": {"|".join(c): {"n": v[0], "fixed": v[1], "rate": v[1] / v[0]} for c, v in fx.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DUMP_DIR",
                    help="a run's rollout dump directory (files <step>.jsonl); repeatable")
    ap.add_argument("--upto", type=int, default=300)
    ap.add_argument("--out", required=True, help="output JSON")
    a = ap.parse_args()
    runs = dict(s.split("=", 1) for s in a.run)
    res = {name: run(D, a.upto) for name, D in runs.items() if os.path.isdir(D)}
    json.dump(res, open(a.out, "w"), indent=1)
    cell = lambda r, c: ("%.1f%% (n=%d)" % (100 * r["cells"][c]["rate"], r["cells"][c]["n"])) if c in r["cells"] else "-"
    print("| run | steps | pairs | slice | zero | middle | one_short |\n|---|---|---|---|---|---|---|")
    for name, r in res.items():
        for sl, (a2, b2) in (("all", ("all", "all")), ("t1<=100", ("t1<=100", "all")), ("t1 101-200", ("t1 101-200", "all")),
                             ("p1=0/8", ("all", "p1=0/8")), ("p1=1-3/8", ("all", "p1=1-3/8"))):
            print("| %s | 1-%d | %d | %s | %s |" % (name, r["steps_used"], r["pairs"], sl,
                                                   " | ".join(cell(r, "%s|%s|%s" % (s, a2, b2)) for s in STATES)))
    print("written", a.out)


if __name__ == "__main__":
    main()
