#!/usr/bin/env python3
"""Package the atomic-rw science control from the atomic science parquet and ../generate/rewrite_atomic.py output.

Every row is copied; only extra_info.rubric[i].criterion is replaced by its one-to-one rewrite. The build fails on
any difference other than the text: same id order, same criterion count, same weights, same
"<title>: <Category> Criteria:" prefix. Rows whose generation failed keep their original text and are counted in the
manifest. The reward and aggregation are Rubric-RL's (independent per-criterion weighted sum; configs/atomic.yaml).

    python3 build_atomic_rw.py --input rar-science/train.parquet --rewrites rw.jsonl --output release/science-atomic-rw
"""
import argparse
import hashlib
import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq

PREFIX = re.compile(r"^(.*?\b(?:Essential|Important|Optional|Pitfall)\s+Criteria:\s*)", re.S)
GENERATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generate", "rewrite_atomic.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="atomic science training parquet")
    ap.add_argument("--rewrites", required=True, help="rewrite_atomic.py run output (jsonl)")
    ap.add_argument("--output", required=True, help="release directory to write")
    a = ap.parse_args()

    rw = {}
    for line in open(a.rewrites):
        r = json.loads(line)
        if r["ok"] or r["id"] not in rw:
            rw[r["id"]] = r
    t = pq.read_table(a.input)
    rows = t.to_pylist()
    missing = [r["extra_info"]["id"] for r in rows if r["extra_info"]["id"] not in rw]
    assert not missing, f"{len(missing)} ids not generated yet, e.g. {missing[:3]}"
    kept_orig = changed = items = 0
    lens_o = lens_n = 0
    for r in rows:
        e = r["extra_info"]
        g = rw[e["id"]]
        new = g["criteria"]
        assert len(new) == len(e["rubric"]), e["id"]
        for it, c in zip(e["rubric"], new):
            po, pn = PREFIX.match(it["criterion"]), PREFIX.match(c)
            assert (po.group(1) if po else "") == (pn.group(1) if pn else ""), (e["id"], it["criterion"][:60], c[:60])
            lens_o += len(it["criterion"])
            lens_n += len(c)
            items += 1
            changed += it["criterion"] != c
            it["criterion"] = c
        kept_orig += not g["ok"]
    os.makedirs(a.output, exist_ok=True)
    dst = os.path.join(a.output, "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), dst)
    chk = pq.read_table(dst)
    assert chk.num_rows == t.num_rows and chk.schema.equals(t.schema)
    man = {
        "arm": "atomic-rw (science)",
        "transformation": "every atomic criterion rewritten one-to-one in ProRubric's achieves/fails style by the "
                          "dimension generator (temperature 0); count, order, weights, title and category prefix "
                          "unchanged; aggregation unchanged (Rubric-RL).",
        "members": {"train.parquet": chk.num_rows},
        "items": items,
        "items_changed": changed,
        "rows_kept_original_after_failed_generation": kept_orig,
        "mean_chars_per_criterion": {"original": round(lens_o / items, 1), "rewritten": round(lens_n / items, 1)},
        "generator_script_sha256": hashlib.sha256(open(GENERATOR, "rb").read()).hexdigest(),
    }
    json.dump(man, open(os.path.join(a.output, "manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
