# -*- coding: utf-8 -*-
"""Compare the numbers a figure script draws with the JSON written by analysis/tables/ablations.py."""
import json
import sys


def check(json_path, expected):
    """expected: iterable of (arm name in the spec, metric, drawn value, scale). Prints each mismatch
    at one decimal; exits non-zero if any value differs."""
    agg = json.load(open(json_path, encoding="utf-8"))["agg"]
    bad = 0
    for arm, metric, drawn, scale in expected:
        mean = agg[arm][metric]["mean"]
        table = None if mean is None else round(mean / scale, 1)
        if table != round(drawn, 1):
            bad += 1
            print("MISMATCH %s %s: drawn %.1f, table %s" % (arm, metric, drawn, table))
    print("checked %d drawn values against %s: %d mismatches" % (len(expected), json_path, bad))
    if bad:
        sys.exit(1)
