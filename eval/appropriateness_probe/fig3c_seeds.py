#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-seed probe differences from the untrained model (Figure ``fig:controls_compound`` c; Sections 5.4-5.5).

For each domain of the probe sample, restricted to the items on which every
listed arm has all four criteria graded (the "common probe sample"), this
reports, per training seed, the mean paired difference in probe score
between a trained arm and the untrained model (x100), then the mean and
sample sd (ddof=1) over seeds. Paper numbers read from it: Rubric-RL changes
appropriateness by -28.9 (medicine), -11.2 (science), -7.1 (dialogue) and
+16.0 (writing). ``--paired`` additionally reports the seed-matched
difference between two families (e.g. ProRubric minus Rubric-RL: in science
+7.5 on average, at least +5.0 on every seed).

  python3 fig3c_seeds.py --sample sample.json --grades grades.jsonl \\
      --family Rubric-RL=A,A43,A44 --family ProRubric=B,B43,B44 \\
      --paired ProRubric:Rubric-RL [--out fig3c_seeds.json]

Families list their arms in seed order (seed 42, 43, 44); the i-th arms of
two families are treated as the same seed. Grades and sample are those of
``probe.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe import domain_sets, load_grades, macro  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("--sample", required=True, help="probe sample.json")
    ap.add_argument("--grades", required=True, help="probe grades JSONL")
    ap.add_argument("--base", default="base", help="untrained arm key")
    ap.add_argument("--family", action="append", required=True, metavar="NAME=ARM,ARM,ARM",
                    help="a trained arm family, arms in seed order (repeatable)")
    ap.add_argument("--paired", action="append", default=[], metavar="FAMILY1:FAMILY2",
                    help="also report seed-matched FAMILY1 minus FAMILY2 (repeatable)")
    ap.add_argument("--domains", default=None, help="comma-separated domains to report (default: all in the sample)")
    ap.add_argument("--out", default=None, help="write the results as JSON")
    args = ap.parse_args()

    sample = json.load(open(args.sample, encoding="utf-8"))
    g, _ = load_grades(args.grades)
    families = {}
    for spec in args.family:
        name, _, arms = spec.partition("=")
        families[name] = arms.split(",")
    all_arms = [args.base] + [a for arms in families.values() for a in arms]
    wanted = set(args.domains.split(",")) if args.domains else None

    out = {}
    for dom, sets in domain_sets(sample).items():
        if wanted and dom not in wanted:
            continue
        ids = [(s, i) for s in sets for i in sample[s]["ids"] if all(len(g.get((s, i, a), {})) == 4 for a in all_arms)]
        row = {"n": len(ids), "of": sum(len(sample[s]["ids"]) for s in sets)}
        if not ids:
            out[dom] = row
            print(dom, row)
            continue
        per_seed = {}
        for fam, arms in families.items():
            v = [100 * st.mean(macro(g, s, i, a) - macro(g, s, i, args.base) for s, i in ids) for a in arms]
            per_seed[fam] = v
            row[fam] = [round(x, 1) for x in v]
            row[fam + "_mean"] = round(st.mean(v), 1)
            row[fam + "_sd"] = round(st.stdev(v), 1) if len(v) > 1 else 0.0
        for pair in args.paired:
            f1, _, f2 = pair.partition(":")
            v = [a - b for a, b in zip(per_seed[f1], per_seed[f2])]
            key = f"{f1} minus {f2}"
            row[key] = [round(x, 1) for x in v]
            row[key + "_mean"] = round(st.mean(v), 1)
            row[key + "_sd"] = round(st.stdev(v), 1) if len(v) > 1 else 0.0
        row["base_c"] = round(100 * st.mean(macro(g, s, i, args.base) for s, i in ids), 1)
        out[dom] = row
        print(dom, row)
    if args.out:
        json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
