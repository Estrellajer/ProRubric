#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-domain results: Table tab:main_results (tables/main_results.tex) and tab:per_seed_domains.

Step 1 computes one number per (size, arm, seed, benchmark) from per-question score files;
step 2 averages seeds, takes differences from Base, marks best / second best, forms the
seven-benchmark average, and the paired per-seed difference of that average between two arms
with a 95% Student-t interval over seeds (df = n_seeds - 1).

Per benchmark (every value x100 unless stated):

  WritingBench / Creative-v3 / Arena-Hard v2
      input: per-question score files, JSONL ``{"id", "score"}`` in the upstream unit: WritingBench
      = mean of the query's criterion scores (1-10; a question counts only if every criterion is
      scored), Creative-v3 = the isolated rubric piece score (0-20), Arena-Hard v2 = mean of the
      two-order outcomes vs. the category baseline (0-1). Scaled by 10 / 5 / 100.
      Several files per cell are repeated evaluations of one checkpoint and are averaged. A file
      counts only if its question count is >= 0.9 x the largest file of the cell AND >= 0.95 x the
      largest file of that benchmark over all cells (so a partially graded file never passes).
      Cells for the writing suites come from the writing-domain policy, Arena from the
      dialogue-domain policy (the spec decides which files a cell points at).
  MedQA
      input: the evaluation's scores JSONL (rows ``{"id", "accuracy", "error", "data_source"}``;
      ``data_source`` optional, must be ``medqa`` when present). Mean accuracy over the ids that
      every MedQA cell of the spec answered.
  GPQA-Diamond
      input: one or more scores JSONL per cell (repeats; ``data_source`` ``gpqa_diamond``). Common
      ids = intersection of the first file of every cell; value = mean over repeats.
  ResearchQA
      input: coverage grades JSONL per cell (repeats), rows ``{"id", "batch", "scores"}``: the
      rubric of question ``id`` is judged in batches of 8 criteria starting at index ``batch``, each
      on the 5-level scale (1..5). Per question: every batch present, coverage = mean((s-1)/4)
      over all criteria; rubric sizes come from the ResearchQA parquet (``extra_info.id``,
      ``extra_info.rubric``). Common ids = intersection of the first file of every cell.
  HealthBench (HB-full, evaluation judge)
      input: a grades file + arm key per cell. Each cell is scored on its OWN fully graded ids
      within the HB-full prompt set (spec ``healthbench_prompt_set``, see common.prompt_set);
      ``complete`` = graded on >= 0.99 of the set.

Per-seed values are rounded to two decimals before averaging, as in the paper's pipeline.

  python3 main_results.py --spec arms.json --out out/main_results.json [--tex-dir out/tex]

Spec: the ``"main_results"`` section of the arm-spec JSON (see README.md and arms.example.json).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import pyarrow.parquet as pq

import common as K

SUITES = {"writingbench": ("WritingBench", 10.0), "creative_writing_v3": ("Creative", 5.0),
          "arena_hard_v2": ("Arena", 100.0)}
COLUMNS = ["WritingBench", "Creative", "Arena", "HealthBench", "MedQA", "GPQA", "ResearchQA"]


def read_scores(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("score") is not None:
                    out[str(r["id"])] = float(r["score"])
    return out


def wc_scores(cells):
    """(cell index, column) -> (value, n files averaged, largest file n), with the two completeness floors."""
    raw = {}
    for ci, c in enumerate(cells):
        for suite, (col, k) in SUITES.items():
            for f in c.get(suite, []):
                s = read_scores(f)
                if s:
                    raw.setdefault((ci, col), []).append((len(s), k * sum(s.values()) / len(s)))
    suite_max = {}
    for (ci, col), runs in raw.items():
        suite_max[col] = max(suite_max.get(col, 0), max(n for n, _ in runs))
    res = {}
    for key, runs in raw.items():
        best = max(n for n, _ in runs)
        full = [v for n, v in runs if n >= 0.9 * best and n >= 0.95 * suite_max[key[1]]]
        if full:
            res[key] = (round(statistics.mean(full), 2), len(full), best)
    dropped = sorted((key, n, suite_max[key[1]]) for key, runs in raw.items() for n, _ in runs if n < 0.95 * suite_max[key[1]])
    return res, suite_max, dropped


def mcq(path, source):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("data_source") in (None, source) and not r.get("error") and r.get("accuracy") is not None:
                out[str(r["id"])] = float(r["accuracy"])
    return out or None


def coverage(path, rub):
    sc = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            if g.get("scores") is not None:
                sc.setdefault(str(g["id"]), {})[g["batch"]] = g["scores"]
    cov = {}
    for rid, b in sc.items():
        n = rub.get(rid)
        if not n:
            continue
        vals = []
        for k in range(0, n, 8):
            if k not in b:
                vals = None
                break
            vals.extend(b[k])
        if vals and len(vals) == n:
            cov[rid] = sum((v - 1) / 4 for v in vals) / n
    return cov or None


def researchqa_sizes(parquet):
    out = {}
    for r in pq.read_table(parquet).to_pylist():
        ei = r["extra_info"]
        if isinstance(ei, str):
            ei = json.loads(ei)
        out[str(ei["id"])] = len(ei["rubric"])
    return out


def compute(S):
    cells = S["cells"]
    wc, suite_max, dropped = wc_scores(cells)

    med = {ci: mcq(c["medqa"], "medqa") for ci, c in enumerate(cells) if c.get("medqa")}
    med = {ci: v for ci, v in med.items() if v}
    med_ids = set.intersection(*[set(v) for v in med.values()]) if med else set()
    sci = {}
    for ci, c in enumerate(cells):
        per = [p for p in (mcq(f, "gpqa_diamond") for f in c.get("gpqa_diamond", [])) if p]
        if per:
            sci[ci] = per
    sci_ids = set.intersection(*[set(p[0]) for p in sci.values()]) if sci else set()
    rq = {}
    if any(c.get("researchqa") for c in cells):
        rub = researchqa_sizes(S["questions"]["researchqa"])
        for ci, c in enumerate(cells):
            per = [p for p in (coverage(f, rub) for f in c.get("researchqa", [])) if p]
            if per:
                rq[ci] = per
    rq_ids = set.intersection(*[set(p[0]) for p in rq.values()]) if rq else set()

    hb = {}
    hb_refs = [c["healthbench"] for c in cells if c.get("healthbench")]
    if hb_refs:
        hq = K.rubrics(S["questions"]["healthbench_full"])
        pset = K.prompt_set(S.get("healthbench_prompt_set"), hq)
        G = K.Grades(hb_refs)
        for ci, c in enumerate(cells):
            if c.get("healthbench"):
                v = G.per_id(c["healthbench"], hq, pool=pset)
                if v:
                    hb[ci] = {"value": round(100 * sum(v.values()) / len(v), 2), "n": len(v),
                              "complete": len(v) >= 0.99 * len(pset)}

    rows = {}
    for ci, c in enumerate(cells):
        row = {}
        for col in ("WritingBench", "Creative", "Arena"):
            v = wc.get((ci, col))
            if v:
                row[col] = {"value": v[0], "reps": v[1]}
        if ci in med:
            row["MedQA"] = {"value": round(100 * sum(med[ci][i] for i in med_ids) / len(med_ids), 2)}
        if ci in sci:
            row["GPQA"] = {"value": round(statistics.mean(100 * sum(p[i] for i in sci_ids) / len(sci_ids) for p in sci[ci]), 2),
                           "reps": len(sci[ci])}
        if ci in rq:
            row["ResearchQA"] = {"value": round(statistics.mean(100 * sum(p[i] for i in rq_ids) / len(rq_ids) for p in rq[ci]), 2),
                                 "reps": len(rq[ci])}
        if ci in hb:
            row["HealthBench"] = hb[ci]
        if row:
            rows["%s/%s/s%s" % (c["size"], c["arm"], c["seed"])] = row
    return {"ids": {"medqa": len(med_ids), "gpqa_diamond": len(sci_ids), "researchqa": len(rq_ids),
                    "healthbench_prompt_set": len(pset) if hb_refs else 0},
            "suite_max": suite_max, "dropped_partial_files": [[cells[k[0]]["arm"], k[1], n, m] for k, n, m in dropped],
            "rows": rows}


def aggregate(out, S):
    """Seed means per (size, arm, column), 7-benchmark averages, paired per-seed interval."""
    rows = out["rows"]
    by = {}
    for key, row in rows.items():
        size, arm, seed = key.split("/")
        for col, v in row.items():
            by.setdefault((size, arm, col), {})[seed] = v["value"]
    agg = {}
    for (size, arm, col), per in by.items():
        vals = list(per.values())
        agg.setdefault(size, {}).setdefault(arm, {})[col] = {
            "per_seed": per, "mean": statistics.mean(vals), "sd": K.sd(vals), "k": len(vals)}
    for size, arms in agg.items():
        for arm, cols in arms.items():
            if all(c in cols for c in COLUMNS):
                cols["Average"] = {"mean": statistics.mean(cols[c]["mean"] for c in COLUMNS)}
                seeds = set.intersection(*[set(cols[c]["per_seed"]) for c in COLUMNS])
                cols["Average"]["per_seed"] = {s: statistics.mean(cols[c]["per_seed"][s] for c in COLUMNS) for s in sorted(seeds)}
    paired = {}
    for size, a, b in S.get("paired_average", []):
        pa, pb = agg[size][a]["Average"]["per_seed"], agg[size][b]["Average"]["per_seed"]
        seeds = sorted(set(pa) & set(pb))
        d = [pb[s] - pa[s] for s in seeds]
        m, s, ci = K.t_interval(d)
        paired["%s: %s - %s" % (size, b, a)] = {"seeds": seeds, "per_seed": d, "mean": m, "sd": s, "ci95_t": ci}
        for col in COLUMNS:
            ca, cb = agg[size][a][col]["per_seed"], agg[size][b][col]["per_seed"]
            dd = [cb[x] - ca[x] for x in sorted(set(ca) & set(cb))]
            paired["%s: %s - %s" % (size, b, a)].setdefault("columns", {})[col] = {
                "per_seed": dd, "mean": statistics.mean(dd), "sd": K.sd(dd)}
    return agg, paired


# ---------------------------------------------------------------- LaTeX bodies
def _delta(v):
    v = round(v, 1)
    return ("\\posdelta{%.1f}" if v >= 0 else "\\negdelta{%.1f}") % abs(v)


def write_main_tex(agg, S, d):
    L = S["layout"]
    head = [
        "\\begin{tabular}{l|ll l ll ll l}", "\\toprule",
        "\\multirow{2}{*}{\\textbf{Method}} & \\multicolumn{2}{c}{\\textbf{Writing}} & \\multicolumn{1}{c}{\\textbf{Dialogue}} & "
        "\\multicolumn{2}{c}{\\textbf{Clinical Medicine}} & \\multicolumn{2}{c}{\\textbf{Scientific Reasoning}} & "
        "\\multicolumn{1}{c}{\\textbf{Overall}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-9}",
        "& " + " & ".join("\\multicolumn{1}{c}{\\textbf{%s}}" % h for h in
                          ("WritingBench", "Creative-v3", "Arena-Hard", "HealthBench", "MedQA", "GPQA", "ResearchQA", "Average")) + " \\\\",
        "\\midrule"]
    rows = list(head)
    cols = COLUMNS[:3] + ["HealthBench", "MedQA", "GPQA", "ResearchQA", "Average"]
    for bi, (size, model, items) in enumerate(L["main_results"]):
        if bi:
            rows.append("\\midrule")
        rows.append("\\multicolumn{9}{@{}l}{\\textit{%s}} \\\\" % model)
        A = agg[size]
        base = items[0][0]
        rank = {}
        for col in cols:
            vals = sorted({round(A[arm][col]["mean"], 1) for arm, *_ in items if col in A.get(arm, {})}, reverse=True)
            rank[col] = vals[:2]
        for arm, label, highlight in items:
            cells = []
            for col in cols:
                x = A.get(arm, {}).get(col)
                if x is None:
                    cells.append("--")
                    continue
                v = round(x["mean"], 1)
                s = "%.1f" % x["mean"]
                if v == rank[col][0]:
                    s = "\\textbf{%s}" % s
                elif len(rank[col]) > 1 and v == rank[col][1]:
                    s = "\\underline{%s}" % s
                if arm != base and col in A[base]:
                    s += _delta(x["mean"] - A[base][col]["mean"])
                cells.append(s)
            name = ("\\rowcolor{ourhighlight} \\multicolumn{1}{l|}{\\hspace{0.8em}%s}" if highlight else "\\hspace{0.8em}%s") % label
            rows.append(name + " & " + " & ".join(cells) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(d, "main_results.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def write_per_seed_tex(agg, paired, S, d):
    size, a, b = S["layout"]["per_seed_domains"]
    A = agg[size]
    p = paired["%s: %s - %s" % (size, b, a)]
    groups = [("Writing", [("WritingBench", "WritingBench"), ("Creative", "Creative-v3")]),
              ("Dialogue", [("Arena", "Arena-Hard")]),
              ("Medicine", [("HealthBench", "HealthBench"), ("MedQA", "MedQA")]),
              ("Science", [("GPQA", "GPQA-Diamond"), ("ResearchQA", "ResearchQA")])]
    ps = lambda x: " / ".join("%.1f" % x["per_seed"][s] for s in sorted(x["per_seed"]))  # noqa: E731
    rows = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l|cc cc c@{}}", "\\toprule",
            "\\multirow{2}{*}{\\textbf{Benchmark}} & \\multicolumn{2}{c}{\\textbf{Rubric-RL}} & \\multicolumn{2}{c}{\\textbf{\\method{}}} & "
            "\\multirow{2}{*}{\\textbf{Paired $\\boldsymbol{\\Delta}$}} \\\\",
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
            "& \\textbf{per seed} & \\textbf{mean} & \\textbf{per seed} & \\textbf{mean} & \\\\", "\\midrule"]
    for g, items in groups:
        rows.append("\\textit{%s} & & & & & \\\\" % g)
        for col, lab in items:
            x, y, dd = A[a][col], A[b][col], p["columns"][col]
            rows.append("\\hspace{0.8em}%s & %s & %.1f\\std{%.1f} & %s & %.1f\\std{%.1f} & %+.2f\\std{%.2f} \\\\" % (
                lab, ps(x), x["mean"], x["sd"], ps(y), y["mean"], y["sd"], dd["mean"], dd["sd"]))
    xa, xb = A[a]["Average"], A[b]["Average"]
    va, vb = list(xa["per_seed"].values()), list(xb["per_seed"].values())
    rows += ["\\midrule", "Seven-benchmark average & %s & %.1f\\std{%.1f} & %s & %.1f\\std{%.1f} & \\textbf{%+.2f}\\std{%.2f} \\\\" % (
        ps(xa), statistics.mean(va), K.sd(va), ps(xb), statistics.mean(vb), K.sd(vb), p["mean"], p["sd"]),
             "\\bottomrule", "\\end{tabular*}"]
    with open(os.path.join(d, "per_seed_domains.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True, help="arm-spec JSON (uses its 'main_results' section)")
    ap.add_argument("--out", required=True, help="output JSON: per-seed rows, seed aggregates, paired average")
    ap.add_argument("--tex-dir", default=None, help="also write main_results.tex and per_seed_domains.tex here")
    args = ap.parse_args()
    S = K.load_spec(args.spec, "main_results")
    out = compute(S)
    agg, paired = aggregate(out, S)
    out["aggregate"], out["paired_average"] = agg, paired
    K.dump(out, args.out)
    print(json.dumps(out["ids"]), len(out["rows"]), "rows")
    for size, arms in agg.items():
        for arm, cols in arms.items():
            print("%s %-10s " % (size, arm) + "  ".join("%s %.1f" % (c, cols[c]["mean"]) for c in COLUMNS + ["Average"] if c in cols))
    for k, v in paired.items():
        print("paired average %s: %s -> %+.2f +/- %.2f, 95%% t-interval [%+.2f, %+.2f]" % (
            k, " / ".join("%+.2f" % x for x in v["per_seed"]), v["mean"], v["sd"], *v["ci95_t"]))
    if args.tex_dir:
        os.makedirs(args.tex_dir, exist_ok=True)
        write_main_tex(agg, S, args.tex_dir)
        if S["layout"].get("per_seed_domains"):
            write_per_seed_tex(agg, paired, S, args.tex_dir)
        print("wrote LaTeX bodies to", args.tex_dir)


if __name__ == "__main__":
    main()
