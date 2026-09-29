#!/usr/bin/env python3
"""raw-AND control (medical layout): ProRubric's grouping with the criteria kept verbatim.

Same grouping as ProRubric (extra_info.rir_generation.atomic_indices), same
per-dimension weight (= sum of |atomic weight| over the group), but the text of
each dimension is the group's atomic criteria verbatim, joined into one
conjunctive criterion ("all of the following must be satisfied"). No rewriting,
no failure clauses, no model call.
"""
import argparse
import json

from _common import parse_generation, read_rows, sha256_file, write_release

HEAD = ("ALL of the following must be satisfied for this criterion to count as met; if any one of them is not "
        "satisfied, the criterion is NOT met:")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="source ProRubric train.parquet (build_prorubric_release.py output)")
    ap.add_argument("--output", required=True, help="output release directory")
    ap.add_argument("--artifact-id", default="medical-raw-and")
    ap.add_argument("--derived-from", default="medical-prorubric")
    args = ap.parse_args()

    rows = read_rows(args.input)
    n_groups, n_atoms_per_group, weight_mismatch = [], [], 0
    for r in rows:
        ei = r["extra_info"]
        g = parse_generation(ei)
        atoms = ei["rubric_atomic"]
        dims = ei["rubric"]
        assert len(g["atomic_indices"]) == len(dims), r.get("id")
        new = []
        for grp, rc in zip(g["atomic_indices"], dims):
            parts = [atoms[i - 1]["criterion"].strip() for i in grp]
            w = sum(abs(float(atoms[i - 1]["weight"])) for i in grp)
            if abs(w - float(rc["weight"])) > 1e-6:
                weight_mismatch += 1
            text = HEAD + " " + " ".join(f"({k}) {p}" for k, p in enumerate(parts, 1))
            new.append({"criterion": text, "weight": float(rc["weight"])})
            n_atoms_per_group.append(len(grp))
        n_groups.append(len(new))
        ei["rubric_rir_v3"] = dims
        ei["rubric"] = new
        g["and_only"] = {"criterion_rule": "verbatim atomic criteria of the group joined under an all-of-the-following header",
                         "weight_rule": "identical to ProRubric (sum of |atomic weight| over the group)"}
        ei["rir_generation"] = json.dumps(g)

    manifest = {
        "schema_version": "prorubric-release/raw-and-v1", "artifact_id": args.artifact_id,
        "derived_from": {"artifact": args.derived_from, "train_sha256": sha256_file(args.input)},
        "rows_out": len(rows),
        "transform": ("for every row, each ProRubric dimension is replaced by the conjunction of its own atomic criteria "
                      "(verbatim text, same atomic_indices grouping, same weight); prompt and every other column untouched; "
                      "the ProRubric text is kept in extra_info.rubric_rir_v3 for audit"),
        "criteria_per_question": {"min": min(n_groups), "max": max(n_groups), "mean": sum(n_groups) / len(n_groups)},
        "atoms_per_criterion": {"min": min(n_atoms_per_group), "max": max(n_atoms_per_group),
                                "mean": sum(n_atoms_per_group) / len(n_atoms_per_group)},
        "weight_mismatch_rows": weight_mismatch,
    }
    readme = (f"# {args.artifact_id}\n\nraw-AND control: {args.derived_from} grouping and weights, but each "
              "dimension is the verbatim conjunction of its atomic criteria (no rewrite, no failure clause). "
              "Val sets NOT included; bind val to the atomic arm.\n")
    m = write_release(args.output, rows, manifest, readme)
    print(json.dumps({k: m[k] for k in ("rows_out", "outputs", "criteria_per_question", "atoms_per_criterion", "weight_mismatch_rows")}, indent=1))
    e = rows[0]["extra_info"]
    print("SAMPLE:", e["rubric"][0]["criterion"][:700], "| w", e["rubric"][0]["weight"])


if __name__ == "__main__":
    main()
