#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Medical ablations at 4B: Table tab:consensus_ablations (tables/ablations_main.tex), Table
tab:ablations_full (tables/consensus_analysis.tex), tab:per_seed_ablations, tab:seeds, and the
HealthBench rows of tab:eval_sizes.

Four scores per (arm, seed) cell, all x100:

  c_pro   HealthBench-consensus, evaluation judge (DeepSeek-V4-Pro) verdicts -> official score
  c_lite  HealthBench-consensus, second evaluator (Doubao-lite): the responses file's ``score``
  g_lite  HealthBench-full, second evaluator: the responses file's ``score``
  g_pro   HealthBench-full, evaluation judge verdicts -> official score
  len     mean response characters on the consensus items

Common item sets, one per axis, cut only by the arms with ``"role": "table"`` plus the Base row:

  consensus  ids that are clean for the second evaluator (no error, score present) AND have an
             evaluation-judge verdict for every criterion, in every cut cell; c_pro, c_lite and
             length are all computed on this one set.
  HB-full g_lite  ids clean for the second evaluator in every cut cell.
  HB-full g_pro   ids of the HB-full prompt set (spec ``hbfull_prompt_set``) with an
             evaluation-judge verdict for every criterion in every cut cell.

A cut cell must first pass a completeness floor on each axis (distinct ids): second evaluator
>= 0.95 x the parquet pool; evaluation judge on consensus >= 0.95 x the ids its grader attempted
(any row in the grades file); evaluation judge on HB-full >= 0.95 x the prompt set. A failing cell
is listed under ``excluded`` and does not cut that axis.

Arms with ``"role": "appendix"`` never cut the sets: each is scored on the common set restricted
to the ids it covers, and a value is reported only when it covers >= 0.95 of the set. Arms with
``"role": "side"`` (single-seed controls) are additionally paired with the reference cells (spec
``side_reference``, default Rubric-RL s42 / ProRubric s42 / Base) on the ids those cells share.

Scores: HealthBench official, sum(met signed weights) / sum(positive weights), clipped to [0, 1],
mean over items. sd is the sample sd (ddof=1) over seeds.

  python3 ablations.py --spec arms.json --out out/ablations.json [--tex-dir out/tex]

Spec: the ``"ablations"`` section of the arm-spec JSON (see README.md and arms.example.json).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import common as K


def cells_of(arms, roles):
    return [(a["name"], i, c) for a in arms if a["role"] in roles for i, c in enumerate(a["cells"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True, help="arm-spec JSON (uses its 'ablations' section)")
    ap.add_argument("--out", required=True, help="output JSON with every per-seed value, n and coverage")
    ap.add_argument("--tex-dir", default=None, help="also write the LaTeX table bodies here")
    args = ap.parse_args()
    S = K.load_spec(args.spec, "ablations")
    arms = S["arms"]

    cq = K.rubrics(S["questions"]["consensus"])
    hq = K.rubrics(S["questions"]["hbfull"])
    pool_c, pool_h = len(cq), len(hq)

    cells = cells_of(arms, {"table", "base", "appendix", "side"})
    t3cells = cells_of(arms, {"table", "base"})
    cid = lambda lab, i: "%s#%d" % (lab, i)  # noqa: E731

    lite_c, lite_h, check = {}, {}, {}
    for lab, i, c in cells:
        k = cid(lab, i)
        lite_c[k], m1, n1 = K.read_lite(c["responses"], "healthbench_consensus", cq)
        lite_h[k], m2, n2 = K.read_lite(c.get("hbfull_responses", c["responses"]), "healthbench_full", hq)
        check[k] = (m1, n1, m2, n2)

    G = K.Grades([c[x] for _, _, c in cells for x in ("consensus_grades", "hbfull_grades") if x in c])
    s42_set = K.prompt_set(S.get("hbfull_prompt_set"), hq)

    NONE = {"path": None, "arm": None}
    cov = {}
    for lab, i, c in cells:
        k = cid(lab, i)
        gc, gh = c.get("consensus_grades", NONE), c.get("hbfull_grades", NONE)
        cpro = {rid for rid in lite_c[k] if rid in cq and G.full(gc, rid, cq[rid])}
        cpro_any = {rid for rid in cq if G.full(gc, rid, cq[rid])}
        hpro = {rid for rid in s42_set if rid in hq and G.full(gh, rid, hq[rid])}
        cov[k] = {"c_lite": len(set(lite_c[k]) & set(cq)), "c_pro": len(cpro_any),
                  "c_pro_attempted": len(G.attempted[(gc["path"], gc["arm"])] & set(cq)), "c_both": len(cpro),
                  "g_lite": len(set(lite_h[k]) & set(hq)), "g_pro": len(hpro), "_cboth": cpro, "_hpro": hpro}

    def ok(k, axis):
        c = cov[k]
        if axis == "c":
            # evaluation judge: against the ids its grader was pointed at (the grader only ever grades a
            # paired set), not against the parquet pool
            return c["c_lite"] >= 0.95 * pool_c and c["c_pro"] >= 0.95 * c["c_pro_attempted"]
        if axis == "gl":
            return c["g_lite"] >= 0.95 * pool_h
        return c["g_pro"] >= 0.95 * len(s42_set)

    excluded = {ax: [(lab, i) for lab, i, _ in t3cells if not ok(cid(lab, i), ax)] for ax in ("c", "gl", "gp")}
    inc = {ax: [cid(lab, i) for lab, i, _ in t3cells if ok(cid(lab, i), ax)] for ax in ("c", "gl", "gp")}
    ids_c = sorted(set.intersection(*[cov[k]["_cboth"] for k in inc["c"]]))
    if S.get("consensus_ids"):
        # a frozen consensus set (the paper's 3,140): verdicts completed by a later retry pass can add
        # ids to the live intersection; the frozen set must still lie inside it
        frozen = K.read_ids(S["consensus_ids"])
        assert set(frozen) <= set(ids_c), "frozen consensus ids are not all in the live intersection"
        ids_c = sorted(frozen)
    ids_gl = sorted(set(hq).intersection(*[set(lite_h[k]) for k in inc["gl"]]))
    ids_gp = sorted(set.intersection(*[cov[k]["_hpro"] for k in inc["gp"]]))

    res = {}
    for lab, i, c in cells:
        k = cid(lab, i)
        gc, gh = c.get("consensus_grades", NONE), c.get("hbfull_grades", NONE)
        r = {"seed": c.get("seed"), "cov": {x: v for x, v in cov[k].items() if not x.startswith("_")},
             "lite_formula_mismatch": check[k]}
        sc = [x for x in ids_c if x in cov[k]["_cboth"]]
        sl = [x for x in ids_gl if x in lite_h[k]]
        sp = [x for x in ids_gp if x in cov[k]["_hpro"]]
        if sc and len(sc) >= 0.95 * len(ids_c):
            r["c_pro"] = 100 * sum(G.item(gc, x, cq[x]) for x in sc) / len(sc)
            r["c_lite"] = 100 * sum(lite_c[k][x][0] for x in sc) / len(sc)
            r["len"] = sum(lite_c[k][x][1] for x in sc) / len(sc)
            r["n_c"] = len(sc)
        if sl and len(sl) >= 0.95 * len(ids_gl):
            r["g_lite"] = 100 * sum(lite_h[k][x][0] for x in sl) / len(sl)
            r["n_gl"] = len(sl)
        if sp and len(sp) >= 0.95 * len(ids_gp):
            r["g_pro"] = 100 * sum(G.item(gh, x, hq[x]) for x in sp) / len(sp)
            r["n_gp"] = len(sp)
        res.setdefault(lab, {})[i] = r

    # side arms: paired with the reference cells on the ids they all share (all three axes)
    ref_names = S.get("side_reference", ["Rubric-RL", "ProRubric", "Base"])
    by_name = {a["name"]: a for a in arms}
    refs = [(n, 0) for n in ref_names]
    side = [a["name"] for a in arms if a["role"] == "side"]
    groups = [(lab, [(lab, 0)] + refs) for lab in side]
    if len(side) > 1:
        groups.append(("all side arms jointly", [(lab, 0) for lab in side] + refs))
    paired = {}
    for glab, grp in groups:
        keys = [cid(*g) for g in grp]
        pc = sorted(set.intersection(*[cov[k]["_cboth"] for k in keys]))
        pl = sorted(set(hq).intersection(*[set(lite_h[k]) for k in keys]))
        pp = sorted(set.intersection(*[cov[k]["_hpro"] for k in keys]))
        d = {"n_c": len(pc), "n_gl": len(pl), "n_gp": len(pp)}
        for (lab, i), k in zip(grp, keys):
            c = by_name[lab]["cells"][i]
            gc, gh = c.get("consensus_grades", NONE), c.get("hbfull_grades", NONE)
            d[lab] = {"c_pro": 100 * sum(G.item(gc, x, cq[x]) for x in pc) / len(pc) if pc else None,
                      "c_lite": 100 * sum(lite_c[k][x][0] for x in pc) / len(pc) if pc else None,
                      "len": sum(lite_c[k][x][1] for x in pc) / len(pc) if pc else None,
                      "g_lite": 100 * sum(lite_h[k][x][0] for x in pl) / len(pl) if pl else None,
                      "g_pro": 100 * sum(G.item(gh, x, hq[x]) for x in pp) / len(pp) if pp else None}
        paired[glab] = d

    agg = {}
    for a in arms:
        lab, n = a["name"], len(a["cells"])
        agg[lab] = {}
        for m in ("c_pro", "c_lite", "g_lite", "g_pro", "len"):
            v = [res[lab][i][m] for i in range(n) if m in res[lab][i]]
            agg[lab][m] = {"per_seed": [res[lab][i].get(m) for i in range(n)],
                           "mean": statistics.mean(v) if v else None, "sd": K.sd(v) if v else None, "k": len(v)}
    out = {"pool_consensus": pool_c, "pool_hbfull": pool_h, "hbfull_prompt_set": len(s42_set),
           "n_consensus": len(ids_c), "n_gfull_lite": len(ids_gl), "n_gfull_pro": len(ids_gp),
           "excluded": {ax: [(lab, i) for lab, i in v] for ax, v in excluded.items()},
           "cells": res, "agg": agg, "paired_side": paired}
    K.dump(out, args.out)
    report(out, arms)
    if args.tex_dir:
        write_tex(out, S, args.tex_dir)


def fmt(a, m, d=1, scale=1.0):
    x = a[m]
    if x["mean"] is None:
        return "-"
    ps = " / ".join("-" if v is None else "%.*f" % (d, v / scale) for v in x["per_seed"])
    if x["k"] > 1:
        return "%.*f +/- %.*f (%s)" % (d, x["mean"] / scale, d, x["sd"] / scale, ps)
    return "%.*f" % (d, x["mean"] / scale)


def report(out, arms):
    print("pools: consensus %d, HB-full %d (prompt set %d)" % (out["pool_consensus"], out["pool_hbfull"], out["hbfull_prompt_set"]))
    print("n: consensus %d | HB-full second evaluator %d | HB-full evaluation judge %d"
          % (out["n_consensus"], out["n_gfull_lite"], out["n_gfull_pro"]))
    print("excluded:", json.dumps(out["excluded"]))
    print("\n| arm | seed | c_lite ids | c_pro ids (full / attempted) | g_lite ids | g_pro ids | score re-derivation mismatches c/g |")
    print("|---|---|---|---|---|---|---|")
    for a in arms:
        for i, _ in enumerate(a["cells"]):
            r = out["cells"][a["name"]][i]
            c, m = r["cov"], r["lite_formula_mismatch"]
            print("| %s | %s | %d | %d / %d | %d | %d | %d/%d, %d/%d |" % (
                a["name"], r["seed"], c["c_lite"], c["c_pro"], c["c_pro_attempted"], c["g_lite"], c["g_pro"],
                m[0], m[1], m[2], m[3]))
    print("\n| arm | c_pro | c_lite | g_lite | g_pro | length (k chars) |")
    print("|---|---|---|---|---|---|")
    for a in arms:
        g = out["agg"][a["name"]]
        print("| %s | %s | %s | %s | %s | %s |" % (a["name"], fmt(g, "c_pro"), fmt(g, "c_lite"), fmt(g, "g_lite"),
                                                   fmt(g, "g_pro"), fmt(g, "len", 1, 1000.0)))
    print("\nside arms paired with the reference cells:")
    print(json.dumps(out["paired_side"], indent=1))
    print("\nper-cell n (consensus, HB-full second evaluator, HB-full evaluation judge) for non-cutting arms:")
    for a in arms:
        if a["role"] in ("appendix", "side"):
            print(a["name"], [(r.get("n_c"), r.get("n_gl"), r.get("n_gp")) for r in out["cells"][a["name"]].values()])


# ---------------------------------------------------------------- LaTeX bodies
def _v(agg, arm, m, std=True, scale=1.0, suffix=""):
    x = agg[arm][m]
    if x["mean"] is None:
        return "--"
    s = "%.1f%s" % (x["mean"] / scale, suffix)
    if std and x["k"] > 1:
        s += "\\std{%.1f}" % (x["sd"] / scale)
    return s


def write_tex(out, S, d):
    os.makedirs(d, exist_ok=True)
    agg, L = out["agg"], S["layout"]
    n_seeds = {a["name"]: len(a["cells"]) for a in S["arms"]}
    FLAG = {True: "\\facton", False: "\\factoff", None: "\\factna"}

    # tables/ablations_main.tex (tab:consensus_ablations): c = c_pro, g = g_pro
    rows = ["\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l|cc@{}}", "\\toprule",
            "\\textbf{Reward} & $\\boldsymbol{c}$ & $\\boldsymbol{g}$ \\\\"]
    for sec, items in L["ablations_main"]:
        rows += ["\\midrule", "\\textit{%s} & & \\\\" % sec]
        rows += ["\\hspace{0.6em}%s & %s & %s \\\\" % (lab, _v(agg, arm, "c_pro"), _v(agg, arm, "g_pro")) for arm, lab in items]
    rows += ["\\bottomrule", "\\end{tabular*}"]
    _write(d, "ablations_main.tex", rows)

    # tables/consensus_analysis.tex (tab:ablations_full)
    rows = ["\\begin{tabular}{l|cc cccc cc}", "\\toprule",
            "\\multirow{2}{*}{\\textbf{Configuration}} & \\multicolumn{2}{c}{\\textbf{Reward structure}} & "
            "\\multicolumn{4}{c}{\\textbf{Score}} & \\multirow{2}{*}{\\textbf{Len.}} & \\multirow{2}{*}{\\textbf{$n$}} \\\\",
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-7}",
            " & \\textbf{grouped} & \\textbf{appr.\\ crit.} & $\\boldsymbol{g_{\\mathrm{lite}}}$ & $\\boldsymbol{g_{\\mathrm{pro}}}$ & "
            "$\\boldsymbol{c_{\\mathrm{lite}}}$ & $\\boldsymbol{c_{\\mathrm{pro}}}$ & & \\\\", "\\midrule"]
    for si, (sec, items) in enumerate(L["ablations_full"]):
        if si:
            rows.append("")
        rows.append("\\addlinespace[0.5ex]\\textit{%s} & & & & & & & & \\\\" % sec)
        for arm, lab, grouped, appr in items:
            rows.append("\\hspace{0.8em}%s & %s & %s & %s & %s & %s & %s & %s & %d \\\\" % (
                lab, FLAG[grouped], FLAG[appr], _v(agg, arm, "g_lite"), _v(agg, arm, "g_pro", std=False),
                _v(agg, arm, "c_lite"), _v(agg, arm, "c_pro"), _v(agg, arm, "len", std=False, scale=1000.0, suffix="k"),
                n_seeds[arm]))
    rows += ["\\bottomrule", "\\end{tabular}"]
    _write(d, "consensus_analysis.tex", rows)

    # tab:per_seed_ablations
    def ps(arm, m):
        x = agg[arm][m]
        return " / ".join("--" if v is None else "%.1f" % v for v in x["per_seed"]) + (
            "\\std{%.1f}" % x["sd"] if x["k"] > 1 else "")
    rows = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l|ccc@{}}", "\\toprule",
            "\\textbf{Arm} & $\\boldsymbol{c_{\\mathrm{pro}}}$ \\textbf{per seed} & $\\boldsymbol{c_{\\mathrm{lite}}}$ "
            "\\textbf{per seed} & $\\boldsymbol{g_{\\mathrm{lite}}}$ \\textbf{per seed} \\\\", "\\midrule"]
    for sec, items in L["per_seed_ablations"]:
        rows.append("\\textit{%s} & & & \\\\" % sec)
        rows += ["\\hspace{0.8em}%s & %s & %s & %s \\\\" % (lab, ps(arm, "c_pro"), ps(arm, "c_lite"), ps(arm, "g_lite"))
                 for arm, lab in items]
    rows += ["\\bottomrule", "\\end{tabular*}"]
    _write(d, "per_seed_ablations.tex", rows)

    # tables/seeds.tex (tab:seeds): per-seed c for the two headline arms under both judges
    a0, a1 = L["seeds"]
    rows = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}c|rrrrrr@{}}", "\\toprule",
            "& \\multicolumn{3}{c}{\\textbf{DeepSeek-V4-Pro}} & \\multicolumn{3}{c}{\\textbf{Doubao-lite}} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            "\\textbf{Seed} & \\textbf{Rubric-RL} & \\textbf{\\method{}} & $\\boldsymbol{\\Delta}$ & \\textbf{Rubric-RL} & "
            "\\textbf{\\method{}} & $\\boldsymbol{\\Delta}$ \\\\", "\\midrule"]
    seeds = [c.get("seed") for c in next(a for a in S["arms"] if a["name"] == a0)["cells"]]
    for i, s in enumerate(seeds):
        v = [agg[a][m]["per_seed"][i] for m in ("c_pro", "c_lite") for a in (a0, a1)]
        rows.append("%s & %.1f & %.1f & %+.1f & %.1f & %.1f & %+.1f \\\\" % (s, v[0], v[1], v[1] - v[0], v[2], v[3], v[3] - v[2]))
    mv = [agg[a][m]["mean"] for m in ("c_pro", "c_lite") for a in (a0, a1)]
    rows += ["\\midrule", "Mean & %.1f & %.1f & %+.1f & %.1f & %.1f & %+.1f \\\\" % (
        mv[0], mv[1], mv[1] - mv[0], mv[2], mv[3], mv[3] - mv[2]), "\\bottomrule", "\\end{tabular*}"]
    _write(d, "seeds.tex", rows)
    print("wrote LaTeX bodies to", d)


def _write(d, name, rows):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
