#!/usr/bin/env python3
"""Package the dimension generator's jsonl into a ProRubric training release (the keep rules).

This is the step between generate/generate_dimensions.py and the control transforms
in this directory: the generator emits one criterion set per question, and this
turns them into a train.parquet the trainer reads exactly like the atomic
release.

  python3 build_prorubric_release.py --base-parquet atomic/train.parquet \
      --gen-jsonl out/medical_dimensions.jsonl --out-dir release/medical-prorubric \
      --artifact-id medical-prorubric --mode protocol [--dry-run]

Keep a question iff its generation is valid-or-repaired, its criteria cover
every atomic index exactly once, the criteria count matches the mode
(2-5 for protocol, exactly 1 for k1, the K=1 control), and every criterion weight -- the exact
sum of |atomic weight| over its assigned indices -- is positive.

Row layout after packaging: extra_info.rubric becomes the dimensions
[{criterion, weight}], the original atomic list moves to
extra_info.rubric_atomic, and the generation metadata (names, the generator's
own weights, the index groups, repair flags) is kept as a JSON string in
extra_info.rir_generation. Everything else in the row is untouched.
"""
import argparse
import collections
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from _common import sha256_file


def load_generations(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["id"]] = rec  # regenerations append; last write wins
    return out


def check(gen, atomic, mode, min_words=0):
    """Return (rir_rubric, reason); reason None means keep."""
    if not (gen.get("valid") or gen.get("repaired")):
        return None, "invalid:%s" % (gen.get("reason") or "unknown")
    crit = gen.get("criteria") or []
    n = len(atomic)
    if mode == "k1" and len(crit) != 1:
        return None, "k1_needs_exactly_one_criterion:%d" % len(crit)
    if mode == "protocol" and not (2 <= len(crit) <= 5):
        return None, "n_criteria_out_of_range:%d" % len(crit)
    seen, rubric = [], []
    for c in crit:
        idx = [int(i) for i in (c.get("atomic_indices") or [])]
        if not idx:
            return None, "criterion_without_indices"
        if any(i < 1 or i > n for i in idx):
            return None, "index_out_of_range"
        seen.extend(idx)
        weight = sum(abs(float(atomic[i - 1]["weight"])) for i in idx)
        if weight <= 0:
            return None, "nonpositive_weight"
        desc = str(c.get("description") or "").strip()
        if not desc or len(desc.split()) < min_words:
            return None, "criterion_text_too_short"
        rubric.append({"criterion": desc, "weight": float(weight)})
    if sorted(seen) != list(range(1, n + 1)):
        return None, "coverage_not_exactly_once"
    return rubric, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-parquet", required=True, help="the atomic release this is derived from")
    ap.add_argument("--gen-jsonl", required=True, help="generator output (generate/generate_dimensions.py)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-id", required=True)
    ap.add_argument("--mode", choices=["protocol", "k1"], required=True)
    ap.add_argument("--exclude-ids", help="json list of ids, or {id: reason}, to drop")
    ap.add_argument("--upstream", default="", help="upstream artifact id, recorded in the manifest")
    ap.add_argument("--min-words", type=int, default=0,
                    help="drop a question if any criterion is shorter than this (v3 rules: 0, so only empty text is dropped)")
    ap.add_argument("--dry-run", action="store_true", help="print the keep/drop stats and write nothing")
    args = ap.parse_args()

    rows = pq.read_table(args.base_parquet).to_pylist()
    base_sha = sha256_file(args.base_parquet)
    generations = load_generations(args.gen_jsonl)

    manual = {}
    if args.exclude_ids:
        with open(args.exclude_ids) as f:
            ex = json.load(f)
        manual = ex if isinstance(ex, dict) else {i: "manual_exclude" for i in ex}

    kept, excluded, repaired_kept, ncrit, words = [], {}, 0, [], []
    generator_models = set()
    for row in rows:
        ei = dict(row["extra_info"])
        rid = ei["id"]
        gen = generations.get(rid)
        if gen is None:
            excluded[rid] = "no_generation"
            continue
        if rid in manual:
            excluded[rid] = "manual:%s" % manual[rid]
            continue
        atomic = [{"criterion": str(x["criterion"]), "weight": float(x["weight"])} for x in ei["rubric"]]
        if gen.get("n_atomic") not in (None, len(atomic)):
            excluded[rid] = "n_atomic_mismatch"
            continue
        rubric, reason = check(gen, atomic, args.mode, args.min_words)
        if reason:
            excluded[rid] = reason
            continue
        repaired_kept += bool(gen.get("repaired"))
        ncrit.append(len(rubric))
        words.extend(len(x["criterion"].split()) for x in rubric)
        generator_models.add(gen.get("model"))
        ei["rubric_atomic"] = atomic
        ei["rubric"] = rubric
        ei["rir_generation"] = json.dumps({
            "names": [c.get("name") for c in gen["criteria"]],
            "model_weights": [c.get("weight") for c in gen["criteria"]],
            "atomic_indices": [c.get("atomic_indices") for c in gen["criteria"]],
            "repaired": bool(gen.get("repaired")),
            "valid": bool(gen.get("valid")),
            "desc_word_counts": gen.get("desc_word_counts"),
            "model": gen.get("model"),
            "timestamp": gen.get("timestamp"),
            "mode": args.mode,
        }, ensure_ascii=False)
        kept.append({"prompt": row["prompt"], "data_source": row["data_source"], "extra_info": ei})

    stats = {
        "rows_in": len(rows),
        "rows_out": len(kept),
        "rows_excluded": len(excluded),
        "rows_repaired_kept": repaired_kept,
        "exclusion_reasons": dict(sorted(collections.Counter(v.split(":")[0] for v in excluded.values()).items())),
        "criteria_per_question": {
            "min": min(ncrit) if ncrit else None,
            "max": max(ncrit) if ncrit else None,
            "mean": round(sum(ncrit) / len(ncrit), 2) if ncrit else None,
        },
        "criterion_words": {
            "median": sorted(words)[len(words) // 2] if words else None,
            "mean": round(sum(words) / len(words), 1) if words else None,
        },
    }
    print(json.dumps(stats, ensure_ascii=False))
    if args.dry_run:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    out_parquet = os.path.join(args.out_dir, "train.parquet")
    pq.write_table(pa.Table.from_pylist(kept), out_parquet)
    manifest = {
        "schema_version": "prorubric-release/v3-rules",
        "artifact_id": args.artifact_id,
        "mode": args.mode,
        "source": {"artifact": args.upstream, "train_parquet": args.base_parquet,
                   "train_sha256": base_sha, "rows": len(rows)},
        "outputs": {"train.parquet": sha256_file(out_parquet)},
        "generator": {"models": sorted(m for m in generator_models if m), "jsonl": args.gen_jsonl},
        "weight_rule": "each dimension weight = exact sum of |atomic weight| over its assigned atomic indices; "
                       "the generator's own proposed weights are kept in extra_info.rir_generation.model_weights",
        "exclusion_rule": "keep iff generation valid-or-repaired, criteria cover every atomic index exactly once, "
                          "n_criteria %s, every weight > 0, non-empty criterion text (min_words=%d)%s"
                          % ("== 1" if args.mode == "k1" else "in [2,5]", args.min_words,
                             "; plus manual exclusions" if manual else ""),
        "schema_note": "same columns as the atomic release; extra_info.rubric = dimensions [{criterion,weight}]; "
                       "extra_info.rubric_atomic = the original atomic list; extra_info.rir_generation = json string; "
                       "other extra_info fields untouched",
        **stats,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "excluded_ids.json"), "w") as f:
        json.dump(excluded, f, indent=0, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# %s\n\nProRubric release (%s mode), built by build_prorubric_release.py from %s and %s. "
                "See manifest.json.\n" % (args.artifact_id, args.mode,
                                          os.path.basename(args.base_parquet), os.path.basename(args.gen_jsonl)))
    print(json.dumps({"out_dir": args.out_dir, "train_sha256": manifest["outputs"]["train.parquet"]}))


if __name__ == "__main__":
    main()
