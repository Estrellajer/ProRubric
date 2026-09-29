#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped-mean transform (an extra; the paper does not report this arm).

Same atom subset, grouping and weights as the raw-AND control, but every atom
is its own criterion scored independently (additive weighted mean). The group
weight is split back to its atoms in proportion to |atomic weight| (sign
preserved). No rewrite, no failure clause, no model call.
"""
import argparse
import json

from _common import parse_generation, read_rows, sha256_file, write_release


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="source ProRubric train.parquet")
    ap.add_argument("--output", required=True, help="output release directory")
    ap.add_argument("--artifact-id", default="medical-grouped-mean")
    ap.add_argument("--derived-from", default="medical-prorubric")
    args = ap.parse_args()

    rows = read_rows(args.input)
    n_crit, n_atoms_total, covered, neg = [], [], [], 0
    for r in rows:
        ei = r["extra_info"]
        g = parse_generation(ei)
        atoms = ei["rubric_atomic"]
        dims = ei["rubric"]
        assert len(g["atomic_indices"]) == len(dims), r.get("id")
        new = []
        seen = set()
        for grp, rc in zip(g["atomic_indices"], dims):
            wsum = sum(abs(float(atoms[i - 1]["weight"])) for i in grp)
            for i in grp:
                a = atoms[i - 1]
                w = float(a["weight"])
                if w < 0:
                    neg += 1
                new.append({"criterion": a["criterion"].strip(),
                            "weight": float(rc["weight"]) * abs(w) / wsum * (1 if w >= 0 else -1)})
                seen.add(i)
        n_crit.append(len(new))
        n_atoms_total.append(len(atoms))
        covered.append(len(seen) / len(atoms))
        ei["rubric_rir_v3"] = dims
        ei["rubric"] = new
        g["grouped_mean"] = {"criterion_rule": "verbatim atomic criteria of the ProRubric groups, each scored independently",
                             "weight_rule": "group weight split back to its atoms in proportion to |atomic weight|"}
        ei["rir_generation"] = json.dumps(g)

    manifest = {
        "schema_version": "prorubric-release/grouped-mean-v1", "artifact_id": args.artifact_id,
        "derived_from": {"artifact": args.derived_from, "train_sha256": sha256_file(args.input)},
        "rows_out": len(rows),
        "transform": ("each ProRubric dimension replaced by its own atomic criteria (verbatim, scored independently, "
                      "group weight split by |w|); prompt untouched"),
        "criteria_per_question": {"min": min(n_crit), "max": max(n_crit), "mean": sum(n_crit) / len(n_crit)},
        "atoms_available_per_question": {"mean": sum(n_atoms_total) / len(n_atoms_total)},
        "atom_coverage_fraction": {"mean": sum(covered) / len(covered), "min": min(covered)},
        "negative_weight_atoms": neg,
    }
    readme = (f"# {args.artifact_id}\n\nGrouped-mean transform: raw-AND subset/grouping/weights, atoms scored independently "
              "(additive). Val sets NOT included; bind val to the atomic arm.\n")
    m = write_release(args.output, rows, manifest, readme)
    print(json.dumps({k: m[k] for k in ("rows_out", "outputs", "criteria_per_question",
                                        "atoms_available_per_question", "atom_coverage_fraction",
                                        "negative_weight_atoms")}, indent=1))


if __name__ == "__main__":
    main()
