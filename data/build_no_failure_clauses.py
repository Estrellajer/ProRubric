#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ProRubric w/o failure clauses: ProRubric's dimensions with their failure-clause sentences
("... fails if ...") removed. Same grouping, weights, prompts; the only change
is the criterion text loses its failure clause(s). Mechanical regex on sentence
boundaries, no model call. Criteria whose remainder would be < 40 chars keep
their original text (counted in the manifest).
"""
import argparse
import json
import re

from _common import parse_generation, read_rows, sha256_file, write_release

PAT = re.compile(r"(?i)(?:^|(?<=[.!?]\s))([^.!?]*\bfails?(?: this dimension| the criterion)? if\b[^.!?]*[.!?])")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="source ProRubric train.parquet (build_prorubric_release.py output)")
    ap.add_argument("--output", required=True, help="output release directory")
    ap.add_argument("--artifact-id", default="medical-prorubric-no-failure-clauses")
    ap.add_argument("--derived-from", default="medical-prorubric")
    args = ap.parse_args()

    rows = read_rows(args.input)
    n_crit = n_stripped = n_kept_short = 0
    lb = la = 0
    for r in rows:
        ei = r["extra_info"]
        dims = ei["rubric"]
        new = []
        for c in dims:
            n_crit += 1
            text = c["criterion"]
            s = PAT.sub("", text).strip()
            if s != text.strip():
                if len(s) < 40:
                    n_kept_short += 1
                    s = text
                else:
                    n_stripped += 1
                    lb += len(text)
                    la += len(s)
            new.append({"criterion": s, "weight": float(c["weight"])})
        ei["rubric_rir_v3"] = dims
        ei["rubric"] = new
        g = parse_generation(ei)
        g["noveto"] = {"rule": "sentences containing 'fails if' removed (regex on sentence boundaries); weights and grouping unchanged"}
        ei["rir_generation"] = json.dumps(g)

    manifest = {
        "schema_version": "prorubric-release/no-failure-clauses-v1", "artifact_id": args.artifact_id,
        "derived_from": {"artifact": args.derived_from, "train_sha256": sha256_file(args.input)},
        "rows_out": len(rows),
        "criteria": n_crit, "criteria_stripped": n_stripped, "criteria_kept_short": n_kept_short,
        "mean_len_before_after": [lb / max(1, n_stripped), la / max(1, n_stripped)],
        "transform": ("ProRubric dimensions with their 'fails if' sentences removed; prompt, grouping, weights untouched; "
                      "original text kept in extra_info.rubric_rir_v3"),
    }
    readme = (f"# {args.artifact_id}\n\nProRubric dimensions minus their failure-clause sentences. Val sets NOT included; "
              "bind val to the atomic arm.\n")
    m = write_release(args.output, rows, manifest, readme)
    print(json.dumps({k: m[k] for k in ("rows_out", "outputs", "criteria", "criteria_stripped",
                                        "criteria_kept_short", "mean_len_before_after")}, indent=1))


if __name__ == "__main__":
    main()
