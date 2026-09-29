#!/usr/bin/env python3
"""raw-AND control for a single non-medical domain (writing, dialogue, science).

Same grouping and weights as that domain's ProRubric set, but each dimension is
the verbatim conjunction of its atomic criteria. Unlike the medical raw-AND arm,
negative-weight atoms (the science domain has ~13%) are rendered as
'[must NOT hold]' sub-conditions, and rows without an atomic mapping are
skipped rather than asserted.
"""
import argparse
import json

from _common import parse_generation, read_rows, tree_hash, write_release

HEAD = ("ALL of the following must be satisfied for this criterion to count as met; if any one of them is not "
        "satisfied, the criterion is NOT met:")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="source domain ProRubric train.parquet")
    ap.add_argument("--output", required=True, help="output release directory")
    ap.add_argument("--artifact-id", required=True)
    ap.add_argument("--derived-from", required=True, help="name of the upstream domain ProRubric set (recorded in the manifest)")
    args = ap.parse_args()

    rows = read_rows(args.input)
    n_groups, n_atoms, wmis, nneg, skipped = [], [], 0, 0, 0
    out_rows = []
    for r in rows:
        ei = r["extra_info"]
        g = parse_generation(ei)
        atoms = ei.get("rubric_atomic") or []
        dims = ei.get("rubric") or []
        idx = (g or {}).get("atomic_indices")
        if not idx or len(idx) != len(dims) or not atoms:
            skipped += 1
            continue
        new = []
        for grp, rc in zip(idx, dims):
            parts = []
            for k, i in enumerate(grp, 1):
                a = atoms[i - 1]
                txt = str(a["criterion"]).strip()
                w = float(a["weight"])
                if w < 0:
                    nneg += 1
                    parts.append(f"({k}) [must NOT hold] {txt}")
                else:
                    parts.append(f"({k}) {txt}")
            wsum = sum(abs(float(atoms[i - 1]["weight"])) for i in grp)
            if abs(wsum - float(rc["weight"])) > 1e-6:
                wmis += 1
            new.append({"criterion": HEAD + " " + " ".join(parts), "weight": float(rc["weight"])})
            n_atoms.append(len(grp))
        n_groups.append(len(new))
        ei["rubric_rir_v1"] = dims
        ei["rubric"] = new
        g["and_only"] = {"criterion_rule": "verbatim atomic criteria of the group under an all-of-the-following header; negative-weight atoms as must-NOT-hold",
                         "weight_rule": "identical to the domain ProRubric set"}
        ei["rir_generation"] = json.dumps(g)
        out_rows.append(r)

    manifest = {
        "artifact_id": args.artifact_id,
        "derived_from": {"artifact": args.derived_from},
        "rows_out": len(out_rows),
        "rows_skipped_no_mapping": skipped,
        "criteria_per_question": {"min": min(n_groups), "max": max(n_groups), "mean": sum(n_groups) / len(n_groups)},
        "atoms_per_criterion": {"min": min(n_atoms), "max": max(n_atoms), "mean": sum(n_atoms) / len(n_atoms)},
        "negative_atoms_rendered": nneg,
        "weight_mismatch_criteria": wmis,
        "transform": ("each ProRubric dimension replaced by the verbatim conjunction of its own atomic criteria "
                      "(same atomic_indices grouping, same weight); no rewrite, no failure clause, no model call; "
                      "ProRubric text kept in extra_info.rubric_rir_v1"),
    }
    readme = (f"# {args.artifact_id}\n\nraw-AND control of {args.derived_from}: same grouping and weights, "
              "verbatim atomic conjunctions, no rewrite or failure clause. Val sets NOT included; bind val as the domain ProRubric arm does.\n")
    m = write_release(args.output, out_rows, manifest, readme)
    hh, file_count, total = tree_hash(args.output)
    print(json.dumps({"artifact": args.artifact_id, "rows": m["rows_out"], "skipped": skipped,
                      "cpq": m["criteria_per_question"]["mean"], "apc": m["atoms_per_criterion"]["mean"],
                      "neg": nneg, "wmis": wmis, "tree": hh, "file_count": file_count, "total_bytes": total}))


if __name__ == "__main__":
    main()
