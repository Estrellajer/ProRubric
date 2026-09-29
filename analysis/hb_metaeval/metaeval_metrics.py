#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HealthBench meta-evaluation agreement metrics with the official simple-evals function.

Produces Table ``tab:judge_agreement`` (appendix "Agreement with physicians"): the grader
column per theme and "All criteria", and the physician column.

The metric is simple-evals' ``compute_metrics_for_rater_by_class`` (commit 652c89d), called
exactly as ``HealthBenchMetaEval.__call__`` calls it for the model grader (self_pred = grader
label, other_preds = the row's physician labels, cluster = the row's consensus criterion) and,
in the verbatim per-physician loop, for each physician against the other physicians on the
same row. Reported, all from the official keys:
  * official pooled score: ``pairwise_model_f1_balanced`` (HealthBenchMetaEval's final score);
  * paper aggregation (HealthBench paper Sec. 8.1, Table 5): per criterion
    ``"<cluster>: pairwise_*_f1_balanced"``, then the unweighted mean over the 34 criteria;
    physicians = per criterion the n-weighted mean over physicians (this reproduces the
    HealthBench paper's physician average 0.647 on the public file: 0.6475). This is the
    "All criteria" row of the table;
  * per theme (cluster-name prefix; the theme rows of the table) and per criterion, with the
    physician percentile of the grader per criterion.
Rows the grader failed on are counted and excluded, never imputed.

Inputs: the grades JSONL of ``grade_metaeval.py`` (rows keyed by line index ``i``) and the
meta-evaluation file ``2025-05-07-06-14-12_oss_meta_eval.jsonl`` it was graded from.

  python3 metaeval_metrics.py grades_metaeval.jsonl --data DEST/2025-05-07-06-14-12_oss_meta_eval.jsonl \\
      --simple-evals-dir DEST [--out metaeval_metrics.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

K = "pairwise_%s_f1_balanced"
THEMES = ("emergency_referrals", "context_seeking", "global_health", "health_data_tasks", "communication", "hedging", "complex_responses")


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def physician_lists(examples, compute_metrics_for_rater_by_class):
    # verbatim from HealthBenchMetaEval.__call__
    physician_rating_lists = defaultdict(lambda: ([], [], []))
    for example in examples:
        for i in range(len(example["binary_labels"])):
            physician_id = example["anonymized_physician_ids"][i]
            self_pred = example["binary_labels"][i]
            other_preds = example["binary_labels"][:i] + example["binary_labels"][i + 1:]
            cluster = example["category"]
            physician_rating_lists[physician_id][0].append(self_pred)
            physician_rating_lists[physician_id][1].append(other_preds)
            physician_rating_lists[physician_id][2].append(cluster)
    out = defaultdict(dict)
    for pid, (a, b, c) in physician_rating_lists.items():
        for k, v in compute_metrics_for_rater_by_class(self_pred_list=a, other_preds_list=b, cluster_list=c, model_or_physician="physician").items():
            out[k][pid] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("grades", help="grades JSONL written by grade_metaeval.py")
    ap.add_argument("--data", required=True, help="2025-05-07-06-14-12_oss_meta_eval.jsonl")
    ap.add_argument("--simple-evals-dir", default=None,
                    help="directory containing the simple_evals/ package (see fetch_simple_evals.sh)")
    ap.add_argument("--out", help="optional path for the full JSON result (includes per-criterion values)")
    a = ap.parse_args()
    if a.simple_evals_dir:
        sys.path.insert(0, os.path.abspath(a.simple_evals_dir))
    from simple_evals.healthbench_meta_eval import compute_metrics_for_rater_by_class

    examples = [json.loads(l) for l in open(a.data)]
    g = {}
    for l in open(a.grades):
        r = json.loads(l)
        if r.get("criteria_met") is not None:
            g[r["i"]] = r["criteria_met"]
    keep = [i for i in range(len(examples)) if i in g]
    missing = len(examples) - len(keep)
    ex = [examples[i] for i in keep]
    model = compute_metrics_for_rater_by_class(self_pred_list=[g[i] for i in keep], other_preds_list=[e["binary_labels"] for e in ex],
                                               cluster_list=[e["category"] for e in ex], model_or_physician="model")
    phys = physician_lists(ex, compute_metrics_for_rater_by_class)
    cats = sorted({e["category"] for e in ex})
    per = {}
    for c in cats:
        m = model.get("%s: %s" % (c, K % "model"), {}).get("value")
        pv = [(v["n"], v["value"]) for v in phys.get("%s: %s" % (c, K % "physician"), {}).values() if v["value"] is not None]
        pw = sum(n * x for n, x in pv) / sum(n for n, _ in pv) if pv else None
        pct = (100 * sum(1 for _, x in pv if x < m) / len(pv)) if (pv and m is not None) else None
        per[c] = {"model": m, "physician_wavg": pw, "model_physician_percentile": pct, "rows": sum(1 for e in ex if e["category"] == c)}
    theme = defaultdict(list)
    for c, v in per.items():
        name = c.split(":", 1)[1] if c.startswith("cluster:") else c
        theme[next((t for t in THEMES if name.startswith(t)), name)].append(v)
    res = {"rows_total": len(examples), "rows_graded": len(keep), "rows_missing": missing,
           "official_pooled_model_f1_balanced": model.get(K % "model", {}).get("value"),   # absent when one class never occurs (tiny samples)
           "paper_agg_model_mean_over_criteria": mean_or_none([v["model"] for v in per.values()]),
           "paper_agg_physician_mean_over_criteria": mean_or_none([v["physician_wavg"] for v in per.values()]),
           "per_theme": {t: {"model": mean_or_none([x["model"] for x in v]), "physician_wavg": mean_or_none([x["physician_wavg"] for x in v]), "criteria": len(v)} for t, v in theme.items()},
           "per_criterion": per}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_criterion",)}, indent=1))


if __name__ == "__main__":
    main()
