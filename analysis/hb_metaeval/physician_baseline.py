#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physician-physician agreement baseline on HealthBench's meta-evaluation, official simple-evals code.

No judge calls. Computes the physician column of Table ``tab:judge_agreement`` independently of
any grader (``metaeval_metrics.py`` recomputes it on the graded rows; both agree when every row
is graded). The per-physician loop below is copied verbatim from simple-evals'
``HealthBenchMetaEval.__call__`` (commit 652c89d, where it is inline); the metric is the official
``compute_metrics_for_rater_by_class``. Aggregation across physicians: n-weighted mean (the
HealthBench paper reports a weighted average, Table 5) and the simple mean, overall and per
category (consensus criterion).

  python3 physician_baseline.py --data DEST/2025-05-07-06-14-12_oss_meta_eval.jsonl \\
      --simple-evals-dir DEST [--out physician_baseline.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--data", required=True, help="2025-05-07-06-14-12_oss_meta_eval.jsonl")
ap.add_argument("--simple-evals-dir", default=None,
                help="directory containing the simple_evals/ package (see fetch_simple_evals.sh)")
ap.add_argument("--out", default="physician_baseline.json")
args = ap.parse_args()
if args.simple_evals_dir:
    sys.path.insert(0, os.path.abspath(args.simple_evals_dir))
from simple_evals.healthbench_meta_eval import compute_metrics_for_rater_by_class  # noqa: E402

examples = [json.loads(l) for l in open(args.data)]
cats = defaultdict(int); labels = 0; phys = set()
for e in examples:
    cats[e["category"]] += 1; labels += len(e["binary_labels"]); phys.update(e["anonymized_physician_ids"])
print("rows (completion x criterion):", len(examples), "| physician labels:", labels, "| physicians:", len(phys), "| categories:", len(cats))
print("model meta-examples (grader label vs each physician label):", labels)
# --- verbatim from HealthBenchMetaEval.__call__ ---
physician_rating_lists = defaultdict(lambda: ([], [], []))
for example in examples:
    for i in range(len(example["binary_labels"])):
        physician_id = example["anonymized_physician_ids"][i]
        self_pred = example["binary_labels"][i]
        other_preds = (
            example["binary_labels"][:i] + example["binary_labels"][i + 1 :]
        )
        cluster = example["category"]
        physician_rating_lists[physician_id][0].append(self_pred)
        physician_rating_lists[physician_id][1].append(other_preds)
        physician_rating_lists[physician_id][2].append(cluster)
physician_agreement_metric_lists = defaultdict(dict)
for physician_id, (physician_rating_list, other_preds_list, cluster_list) in physician_rating_lists.items():
    physician_agreement_metrics = compute_metrics_for_rater_by_class(
        self_pred_list=physician_rating_list, other_preds_list=other_preds_list,
        cluster_list=cluster_list, model_or_physician="physician")
    for k, v in physician_agreement_metrics.items():
        physician_agreement_metric_lists[k][physician_id] = v
# --- end verbatim ---
def agg(key):
    vals = [(v["n"], v["value"]) for v in physician_agreement_metric_lists.get(key, {}).values() if v["value"] is not None]
    if not vals: return None
    w = sum(n * x for n, x in vals) / sum(n for n, _ in vals)
    return round(w, 4), round(statistics.mean(x for _, x in vals), 4), len(vals)
out = {"overall": agg("pairwise_physician_f1_balanced")}
print("physician pairwise_physician_f1_balanced overall: n-weighted, simple mean, #physicians =", out["overall"])
for c in sorted(cats):
    r = agg("%s: pairwise_physician_f1_balanced" % c); out[c] = r
    print("  %-70s %s (rows %d)" % (c, r, cats[c]))
json.dump({"counts": {"rows": len(examples), "physician_labels": labels, "physicians": len(phys), "categories": dict(cats)}, "physician_f1_balanced": out},
          open(args.out, "w"), indent=1)
