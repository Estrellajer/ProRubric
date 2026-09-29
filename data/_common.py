"""Shared I/O for the ProRubric data transforms.

Only parquet I/O, rir_generation parsing, and release-artifact writing are
shared; the per-arm criterion transforms live in the build_*.py scripts.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq


def read_rows(path):
    return pq.read_table(path).to_pylist()


def parse_generation(ei):
    g = ei.get("rir_generation")
    return json.loads(g) if isinstance(g, str) else g


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_release(outdir, rows, manifest, readme):
    """Write train.parquet + manifest.json + README.md; manifest gets outputs/built_at filled in."""
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows), dst)
    manifest = dict(manifest)
    manifest["outputs"] = {"train.parquet": sha256_file(dst)}
    manifest["built_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write(readme)
    return manifest


def tree_hash(outdir):
    """Stable hash over every file in the release tree (mirrors the domain-arm manifest check)."""
    hh = hashlib.sha256()
    files = []
    total = 0
    for cur, dirs, names in os.walk(outdir):
        dirs[:] = sorted(dirs)
        for name in sorted(names):
            p = os.path.join(cur, name)
            files.append((os.path.relpath(p, outdir), sha256_file(p)))
            total += os.path.getsize(p)
    for rel, ck in files:
        hh.update(rel.encode())
        hh.update(ck.encode())
    return hh.hexdigest(), len(files), total
