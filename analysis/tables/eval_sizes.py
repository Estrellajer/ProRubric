#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table tab:eval_sizes: sizes of the common evaluation sets, read from the outputs of
ablations.py (HealthBench-consensus and HealthBench-full common sets) and main_results.py
(MedQA / GPQA-Diamond / ResearchQA intersections; for the writing and dialogue suites the
largest per-question count of a benchmark's score files, which every counted file reaches to
within 5%).

  python3 eval_sizes.py --ablations out/ablations.json --main out/main_results.json [--tex out/tex/eval_sizes.tex]
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ablations", required=True, help="JSON written by ablations.py")
    ap.add_argument("--main", required=True, help="JSON written by main_results.py")
    ap.add_argument("--tex", default=None, help="optional path for the LaTeX tabular")
    args = ap.parse_args()
    a = json.load(open(args.ablations, encoding="utf-8"))
    m = json.load(open(args.main, encoding="utf-8"))
    rows = [("Medicine", None),
            ("HealthBench-consensus (both judges)", a["n_consensus"]),
            ("HealthBench-full (Doubao-lite)", a["n_gfull_lite"]),
            ("HealthBench-full (DeepSeek-V4-Pro)", a["n_gfull_pro"]),
            ("MedQA", m["ids"]["medqa"]),
            ("Science", None),
            ("GPQA-Diamond", m["ids"]["gpqa_diamond"]),
            ("ResearchQA", m["ids"]["researchqa"]),
            ("Writing", None),
            ("WritingBench", m["suite_max"].get("WritingBench")),
            ("Creative Writing v3", m["suite_max"].get("Creative")),
            ("Dialogue", None),
            ("Arena-Hard v2", m["suite_max"].get("Arena"))]
    lines = ["\\begin{tabular}{@{}l|r@{}}", "\\toprule", "\\textbf{Benchmark} & \\textbf{Questions} \\\\", "\\midrule"]
    for name, n in rows:
        print("%-40s %s" % (name, "" if n is None else n))
        lines.append("\\textit{%s} & \\\\" % name if n is None else "\\hspace{0.8em}%s & %s \\\\" % (name, "{:,}".format(n)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    if args.tex:
        with open(args.tex, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
