#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HealthBench-consensus appropriateness for arm groups kept OUT of the ablation-table item set:
the SFT paragraph of the appendix (SFT vs. Base / Rubric-RL / ProRubric at 4B and 8B) and the
8B aggregation controls (raw-AND against Rubric-RL / ProRubric, three seeds).

Each named comparison has its own item set: the consensus ids that every listed cell answered
without an evaluation error AND on which every listed cell has an evaluation-judge verdict for
every criterion. That set is printed next to every number.

Per group: c = mean over cells (seeds) of the per-cell mean official score x100, with the sample
sd over seeds (ddof=1). Per pair: per-id seed means are differenced and averaged over the common
ids; the 95% interval is a percentile bootstrap over ids (seed 0, ``boots`` resamples), i.e. a
question-sampling interval; seed spread is the sd above.

  python3 paired_consensus.py --spec arms.json --which sft_4b [--out out/sft_4b.json]
  python3 paired_consensus.py --spec arms.json --list

Spec: the ``"paired_consensus"`` section of the arm-spec JSON, one entry per comparison:
  {"questions": PARQUET, "boots": 1000,
   "groups": {"Base": [CELL], "Rubric-RL": [CELL, CELL, CELL], ...},
   "pairs": [["SFT", "Base"], ...]}          # each pair is (target, reference): target - reference
  CELL = {"seed": 42, "responses": PATH, "grades": {"path": PATH, "arm": KEY}}
"""
from __future__ import annotations

import argparse
import json

import common as K


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True, help="arm-spec JSON (uses its 'paired_consensus' section)")
    ap.add_argument("--which", help="name of the comparison inside that section")
    ap.add_argument("--list", action="store_true", help="list the comparisons in the spec and exit")
    ap.add_argument("--out", default=None, help="optional output JSON")
    args = ap.parse_args()
    S = K.load_spec(args.spec, "paired_consensus")
    if args.list or not args.which:
        print("\n".join(k for k in S if not k.startswith("_")))
        return
    P = S[args.which]
    groups = P["groups"]
    cells = [(g, i, c) for g, cs in groups.items() for i, c in enumerate(cs)]
    qs = K.rubrics(P["questions"])
    resp = {(g, i): K.read_responses(c["responses"], "healthbench_consensus") for g, i, c in cells}
    G = K.Grades([c["grades"] for _, _, c in cells])

    cand = sorted(set(qs) & set.intersection(*[set(v) for v in resp.values()]))
    for g, i, c in cells:
        print("%-12s seed %s fully graded: %d / %d" % (g, c.get("seed"), sum(G.full(c["grades"], r, qs[r]) for r in cand), len(cand)))
    ids = [r for r in cand if all(G.full(c["grades"], r, qs[r]) for _, _, c in cells)]
    n = len(ids)
    print("\ncommon fully graded ids (%s): %d\n" % (", ".join("%s x%d" % (g, len(cs)) for g, cs in groups.items()), n))
    if not n:
        return
    ds = {(g, i): {r: G.item(c["grades"], r, qs[r]) for r in ids} for g, i, c in cells}

    out = {"which": args.which, "n": n, "groups": {}, "pairs": {}}
    print("| group | seeds | c mean +/- sd | per seed |")
    print("|---|---|---|---|")
    for g, cs in groups.items():
        v = [100 * sum(ds[(g, i)].values()) / n for i in range(len(cs))]
        m = sum(v) / len(v)
        out["groups"][g] = {"per_seed": v, "mean": m, "sd": K.sd(v)}
        print("| %s | %d | %.1f +/- %.1f | %s |" % (g, len(cs), m, K.sd(v), " / ".join("%.1f" % x for x in v)))

    mean_id = {g: {r: sum(ds[(g, i)][r] for i in range(len(cs))) / len(cs) for r in ids} for g, cs in groups.items()}
    B = int(P.get("boots", 1000))
    print("\n| pair | delta (seed mean, paired over %d ids) [95%% bootstrap] |" % n)
    print("|---|---|")
    for tgt, ref in P["pairs"]:
        d = [mean_id[tgt][r] - mean_id[ref][r] for r in ids]
        lo, hi = K.boot_ci(d, seed=0, B=B)
        point = 100 * sum(d) / n
        out["pairs"]["%s - %s" % (tgt, ref)] = {"delta": point, "ci95": [100 * lo, 100 * hi]}
        print("| %s - %s | %+.1f%s [%+.1f, %+.1f] |" % (tgt, ref, point, "*" if lo > 0 or hi < 0 else "", 100 * lo, 100 * hi))
    if args.out:
        K.dump(out, args.out)


if __name__ == "__main__":
    main()
