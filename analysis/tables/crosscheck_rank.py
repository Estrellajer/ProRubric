#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Third-family ranking check: Table tab:gpt_crosscheck and the Spearman correlation quoted with it
(appendix, notes on the experiments: rho = 0.84 over eleven arms).

Both judges grade the same HealthBench-consensus responses with the official template (one call
per criterion): the evaluation judge (DeepSeek-V4-Pro; grades written by ``grade_hb.py`` or
``eval/healthbench_judge.py run``) and a third-family judge (GPT-5.6-luna; the same grader run with
``--judge-role ALT``). Per judge and arm, a question counts when the arm has a verdict for every
criterion; the per-question official score is averaged x100.

Reported:
  table     third-family score of each arm on its OWN fully graded ids, with that n
            ("Paired items" column; 3,652--3,671 in the paper)
  rho       Spearman correlation between the two judges' arm scores, three ways:
              common    : every arm on the ids fully graded for ALL arms by BOTH judges  (paper)
              own sets  : each arm on its own fully graded ids per judge
              subset    : each arm on its own ids within an optional ``subset_ids`` list
  pairs     paired differences on the ids both arms share, percentile bootstrap (seed 0, 1000)

  python3 crosscheck_rank.py --spec arms.json [--out out/crosscheck.json] [--tex-dir out/tex]

Spec: the ``"crosscheck"`` section of the arm-spec JSON:
  {"questions": PARQUET, "subset_ids": PATH (optional),
   "arms": [{"name": "Rubric-RL", "section": "Baselines",
             "pro": {"path": GRADES, "arm": KEY}, "alt": {"path": GRADES, "arm": KEY}}, ...],
   "pairs": [["Rubric-RL", "ProRubric"], ...]}      # (x, y): reports y - x
"""
from __future__ import annotations

import argparse
import os
import random

import common as K


def per_arm(G, ref, qs):
    """id -> score over every id the arm has any successful verdict for and is fully graded on."""
    ids = {i for (i, _) in G.met[(ref["path"], ref["arm"])]}
    return {i: G.item(ref, i, qs[i]) for i in ids if i in qs and G.full(ref, i, qs[i])}


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for t in range(i, j + 1):
            r[o[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    c = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return c / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5


def mean_on(d, ids):
    ids = [i for i in ids if i in d]
    return 100 * sum(d[i] for i in ids) / len(ids), len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True, help="arm-spec JSON (uses its 'crosscheck' section)")
    ap.add_argument("--out", default=None, help="optional output JSON")
    ap.add_argument("--tex-dir", default=None, help="also write the tab:gpt_crosscheck body here")
    args = ap.parse_args()
    S = K.load_spec(args.spec, "crosscheck")
    qs = K.rubrics(S["questions"])
    arms = S["arms"]
    G = K.Grades([a[j] for a in arms for j in ("pro", "alt")])
    ds = {a["name"]: per_arm(G, a["pro"], qs) for a in arms}
    L = {a["name"]: per_arm(G, a["alt"], qs) for a in arms}
    names = [a["name"] for a in arms]
    sub = set(K.read_ids(S["subset_ids"])) if S.get("subset_ids") else None

    common = set.intersection(*[set(L[a]) & set(ds[a]) for a in names])
    print("common ids (all %d arms, both judges): %d" % (len(names), len(common)))
    rows = []
    for a in names:
        lf, lfn = mean_on(L[a], L[a])
        df, dfn = mean_on(ds[a], ds[a])
        lc, _ = mean_on(L[a], common)
        dc, _ = mean_on(ds[a], common)
        r = {"arm": a, "alt_own": lf, "alt_own_n": lfn, "pro_own": df, "pro_own_n": dfn, "alt_common": lc, "pro_common": dc}
        if sub is not None:
            r["alt_sub"], r["alt_sub_n"] = mean_on(L[a], [i for i in L[a] if i in sub])
            r["pro_sub"], r["pro_sub_n"] = mean_on(ds[a], [i for i in ds[a] if i in sub])
        rows.append(r)
        print("%-28s third-family own %.2f (%d) | evaluation judge own %.2f (%d) | common: %.2f vs %.2f"
              % (a, lf, lfn, df, dfn, lc, dc))
    rho = {"common": spearman([r["alt_common"] for r in rows], [r["pro_common"] for r in rows]),
           "own_sets": spearman([r["alt_own"] for r in rows], [r["pro_own"] for r in rows])}
    if sub is not None:
        rho["subset"] = spearman([r["alt_sub"] for r in rows], [r["pro_sub"] for r in rows])
    for k, v in rho.items():
        print("Spearman rho (%s): %.4f" % (k, v))

    pairs = {}
    for x, y in S.get("pairs", []):
        ids = sorted(set(L[x]) & set(L[y]))
        n = len(ids)
        dl = [L[y][i] - L[x][i] for i in ids]
        rng = random.Random(0)
        bs = sorted(sum(dl[rng.randrange(n)] for _ in range(n)) / n for _ in range(1000))
        pairs["%s - %s" % (y, x)] = {"n": n, "delta": 100 * sum(dl) / n, "ci95": [100 * bs[25], 100 * bs[975]],
                                     "unpaired_own_sets": mean_on(L[y], L[y])[0] - mean_on(L[x], L[x])[0]}
        print("%s - %s: paired n=%d %+.2f [%+.2f, %+.2f]; difference of own-set scores %+.2f" % (
            y, x, n, pairs["%s - %s" % (y, x)]["delta"], 100 * bs[25], 100 * bs[975], pairs["%s - %s" % (y, x)]["unpaired_own_sets"]))
    if args.out:
        K.dump({"n_common": len(common), "rows": rows, "rho": rho, "pairs": pairs}, args.out)
    if args.tex_dir:
        os.makedirs(args.tex_dir, exist_ok=True)
        lines = ["\\begin{tabular}{@{}l|cc@{}}", "\\toprule", "\\textbf{Arm} & \\textbf{Score} & \\textbf{Paired items} \\\\", "\\midrule"]
        sec = None
        for a, r in zip(arms, rows):
            if a.get("section") != sec:
                sec = a.get("section")
                lines.append("\\textit{%s} & & \\\\" % sec)
            lines.append("\\hspace{0.8em}%s & %.1f & %s \\\\" % (a.get("label", a["name"]), r["alt_own"], "{:,}".format(r["alt_own_n"])))
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(args.tex_dir, "gpt_crosscheck.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
