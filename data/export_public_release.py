#!/usr/bin/env python3
"""Export a training release (or a held-out set) as a self-contained public dataset file.

The training parquets carry columns the trainer needs but a reader does not (the
OPSD teacher context, permuted/self-golden control rubrics, schema strings). This keeps
only what defines the rubrics:

  python3 export_public_release.py protocol --release release/medical-prorubric --domain medical \
      --generator "DeepSeek-V4-Pro" --out public/medical_prorubric_train.parquet
  python3 export_public_release.py atomic --parquet atomic/heldout.parquet --domain medical \
      --out public/medical_atomic_heldout.parquet

`protocol` reads <release>/train.parquet as written by build_prorubric_release.py and writes one row per
question: id, domain, prompt, the original atomic checklist, and the dimensions (name, text with its failure
clause, weight, the atomic indices grouped into it), plus whether the generation was repaired.
`atomic` writes id, domain, prompt and the atomic checklist (held-out and evaluation sets).
Both also write <out>.sha256.
"""
import argparse
import hashlib
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

CRITERION = pa.struct([("criterion", pa.string()), ("weight", pa.float64())])
DIMENSION = pa.struct([("name", pa.string()), ("criterion", pa.string()), ("weight", pa.float64()),
                       ("atomic_indices", pa.list_(pa.int32()))])


def _load(value):
    return json.loads(value) if isinstance(value, str) else value


def _checklist(items):
    return [{"criterion": str(i["criterion"]), "weight": float(i["weight"])} for i in _load(items)]


def _write(rows, schema, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out, compression="zstd")
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    open(out + ".sha256", "w").write("%s  %s\n" % (digest, os.path.basename(out)))
    print("wrote %s: %d rows, sha256 %s" % (out, len(rows), digest[:16]))


def export_protocol(release, domain, generator, out):
    rows = []
    for rec in pq.read_table(os.path.join(release, "train.parquet")).to_pylist():
        ei = rec["extra_info"]
        gen = _load(ei["rir_generation"])
        dims = _checklist(ei["rubric"])
        names, groups = gen["names"], gen["atomic_indices"]
        assert len(dims) == len(names) == len(groups), ei["id"]
        rows.append({
            "id": ei["id"],
            "domain": domain,
            "prompt": ei["problem"],
            "atomic_rubric": _checklist(ei["rubric_atomic"]),
            "dimensions": [{"name": n, "criterion": d["criterion"], "weight": d["weight"],
                            "atomic_indices": [int(k) for k in g]}
                           for n, d, g in zip(names, dims, groups)],
            "repaired": str(gen.get("repaired")).lower() == "true",
            "generator": generator,
        })
    schema = pa.schema([("id", pa.string()), ("domain", pa.string()), ("prompt", pa.string()),
                        ("atomic_rubric", pa.list_(CRITERION)), ("dimensions", pa.list_(DIMENSION)),
                        ("repaired", pa.bool_()), ("generator", pa.string())])
    _write(rows, schema, out)


def export_atomic(parquet, domain, out):
    rows = []
    for rec in pq.read_table(parquet).to_pylist():
        ei = rec["extra_info"]
        rows.append({"id": ei["id"], "domain": domain, "prompt": ei["problem"],
                     "atomic_rubric": _checklist(ei["rubric"])})
    schema = pa.schema([("id", pa.string()), ("domain", pa.string()), ("prompt", pa.string()),
                        ("atomic_rubric", pa.list_(CRITERION))])
    _write(rows, schema, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("protocol")
    p.add_argument("--release", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--generator", required=True, help="public name of the dimension generator model")
    p.add_argument("--out", required=True)
    a = sub.add_parser("atomic")
    a.add_argument("--parquet", required=True)
    a.add_argument("--domain", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "protocol":
        export_protocol(args.release, args.domain, args.generator, args.out)
    else:
        export_atomic(args.parquet, args.domain, args.out)


if __name__ == "__main__":
    main()
