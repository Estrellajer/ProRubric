#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a RubricHub_v1 domain parquet (RuRL/rurbichub_v1_<Domain>.parquet) into the release contract
(prompt / data_source / extra_info{id, problem, rubric[{criterion, weight}], rubric_correct, opsd_context_kind, schema_version}),
sampling a train subset and a disjoint heldout. Used for writing (--train-n 12000) and dialogue ("Chat",
--train-n 9000); the medical sets come from build_medical_split.py.

  python3 build_domain_split.py --input rurbichub_v1_Writing.parquet --domain Writing \
      --data-source rubrichub_writing --train-n 12000 --heldout-n 300 \
      --output release/rubrichub-writing-v1 [--dry-run]
"""
import argparse
import hashlib
import json
import os
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq

from _common import sha256_file


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="source RubricHub_v1 domain parquet (RuRL/rurbichub_v1_<Domain>.parquet)")
    ap.add_argument("--output", required=True, help="output release directory")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--data-source", required=True)
    ap.add_argument("--train-n", type=int, default=12000)
    ap.add_argument("--heldout-n", type=int, default=300)
    ap.add_argument("--min-items", type=int, default=5)
    ap.add_argument("--max-items", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows = pq.read_table(a.input).to_pylist()
    seen = set()
    keep = []
    drop = {"dup": 0, "items": 0, "weights": 0, "empty": 0}
    for r in rows:
        prompt = [{"role": m.get("role"), "content": m.get("content")} for m in (r.get("prompt") or []) if m.get("content")]
        text = "\n".join(m["content"] for m in prompt).strip()
        if not text or not prompt:
            drop["empty"] += 1
            continue
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in seen:
            drop["dup"] += 1
            continue
        rub = [{"criterion": str(it.get("criterion", "")).strip(), "weight": float(it.get("points", 0))}
               for it in (r.get("Rubrics") or []) if str(it.get("criterion", "")).strip()]
        if not (a.min_items <= len(rub) <= a.max_items):
            drop["items"] += 1
            continue
        if sum(max(0.0, x["weight"]) for x in rub) <= 0:
            drop["weights"] += 1
            continue
        seen.add(key)
        keep.append({"prompt": prompt, "data_source": a.data_source, "extra_info": {
            "schema_version": "opd-rubric-repro/v1", "id": f"{a.data_source}:{key[:16]}", "problem": text,
            "rubric": rub, "rubric_correct": "\n".join(f"- [{x['weight']:g}] {x['criterion']}" for x in rub),
            "opsd_context_kind": "none", "source_index": int(r.get("__index_level_0__", -1)), "ability": r.get("ability"),
            "lang": "zh" if re.search(r"[一-鿿]", text) else "other"}})
    rng = random.Random(a.seed)
    rng.shuffle(keep)
    heldout = keep[: a.heldout_n]
    train = keep[a.heldout_n: a.heldout_n + a.train_n]
    stats = {"source_rows": len(rows), "kept": len(keep), "dropped": drop, "train": len(train), "heldout": len(heldout),
             "items_per_prompt_mean": round(sum(len(x["extra_info"]["rubric"]) for x in keep) / len(keep), 1),
             "zh_share": round(sum(x["extra_info"]["lang"] == "zh" for x in keep) / len(keep), 3)}
    print(json.dumps(stats, ensure_ascii=False))
    if a.dry_run:
        return
    os.makedirs(a.output, exist_ok=True)
    for name, part in (("train.parquet", train), ("heldout.parquet", heldout)):
        pq.write_table(pa.Table.from_pylist(part), os.path.join(a.output, name))
    artifact_id = os.path.basename(a.output.rstrip("/"))
    man = {"schema_version": "opd-rubric-repro-release/v1", "artifact_id": artifact_id, "created": "2026-09-01",
           "source": {"rubrichub": f"hf:sojuL/RubricHub_v1@3837d55971473a872e84879c88f708b8da3ec2ef RuRL/rurbichub_v1_{a.domain}.parquet ({len(rows)} rows)"},
           "members": {"train.parquet": f"{len(train)} {a.domain} prompts, seed {a.seed}, disjoint from heldout", "heldout.parquet": f"{len(heldout)} prompts"},
           "contract": "prompt(list[chat]) / data_source / extra_info{problem, rubric[list{criterion,weight}], rubric_correct, opsd_context_kind, id, lang}",
           "filters": {"dedupe": "sha256(prompt text)", "items": f"{a.min_items}<=n<={a.max_items}", "positive_weight_sum": True}, "stats": stats,
           "files": {n: {"bytes": os.path.getsize(os.path.join(a.output, n)), "sha256": sha256_file(os.path.join(a.output, n))}
                     for n in ("train.parquet", "heldout.parquet")}}
    with open(os.path.join(a.output, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    with open(os.path.join(a.output, "README.md"), "w") as f:
        f.write(f"# {artifact_id}\n\n"
                f"Atomic RubricHub {a.domain} base set in the OPD release contract: {len(train)} train and "
                f"{len(heldout)} heldout prompts (disjoint split, seed {a.seed}), each with a "
                f"{a.min_items}-{a.max_items}-item weighted checklist. Source: hf:sojuL/RubricHub_v1, "
                f"RuRL/rurbichub_v1_{a.domain}.parquet.\n")
    total = sum(os.path.getsize(os.path.join(a.output, n)) for n in os.listdir(a.output))
    print(json.dumps({"out_dir": a.output, "train_sha256": man["files"]["train.parquet"]["sha256"],
                      "file_count": len(os.listdir(a.output)), "total_bytes": total}))


if __name__ == "__main__":
    main()
