#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rubric-free blind pairwise comparisons repeated over training seeds.

One script for two paper results, both "x vs y on fixed items, per seed":

* Table 1(a) / Table ``tab:rubric_free_full``: untrained model (x = base, one
  set of responses shared by every seed) vs Rubric-RL or ProRubric (y, one
  set of responses per seed) in medicine, science, dialogue and writing,
  judged once by the evaluation judge (paper: DeepSeek-V4-Pro) and once with
  ``--judge-role ALT`` (paper: GPT-5.6-luna). Items: the first 100 ids of the
  domain's item list (medicine: the 600-id sample of ``pairwise_domain.py
  sample``; other domains: the probe sample), the same for every seed.
* Equal-budget comparison (Section 5.3, Figure ``fig:pairwise_winrates`` b):
  Rubric-RL (x = A) vs ProRubric (y = B), both regenerated under a
  2,048-token ceiling, per seed, in medicine (the 600-id sample) and science
  (the 200 probe ResearchQA ids); evaluation judge only.

Prompts, rendering and the double-order protocol are those of
``pairwise_core.py``. A stable win needs the same winner in both orders;
everything else is tie/unstable. Per seed the y preference is
wins_y / (wins_x + wins_y); the summary gives its mean and sample sd (ddof=1)
over seeds, and x's win rate (= 1 - y preference) with the W/T/L counts summed
over seeds, which is how ``tab:rubric_free_full`` reports the untrained model.

Row spec (``--spec``), JSON:

  {"rows": [
    {"domain": "medicine", "x": "base", "y": "A", "label": "Untrained vs Rubric-RL",
     "questions": ["healthbench_consensus.parquet"], "data_source": "healthbench_consensus",
     "ids": "medicine_ids.json", "ids_sets": null, "first_n": 100,
     "responses": {"base": {"all": ["base.jsonl"]},
                   "A": {"42": ["a_s42.jsonl"], "43": ["a_s43.jsonl"], "44": ["a_s44.jsonl"]}}},
    ...]}

``responses[arm]`` maps a seed label (or "all") to one or more responses
files (several files are merged, e.g. two writing suites). ``first_n`` 0 or
absent uses every id. Relative paths are resolved against the spec file's
directory. Verdict rows (append-only, resumable):
  {"domain","seed","id","x","y","order","left","winner","first_pos","judge"}
``--extra-verdicts`` reads earlier verdicts in the same schema (e.g. the
seed-42 runs of ``pairwise_domain.py``) and uses those whose
(domain, x, y, seed) matches a row and whose id is one of its items.

  python3 pairwise_seeds.py run --spec table1a.json --verdicts v_eval.jsonl
  python3 pairwise_seeds.py run --spec table1a.json --verdicts v_alt.jsonl --judge-role ALT
  python3 pairwise_seeds.py summarize --spec table1a.json --verdicts v_eval.jsonl [--extra-verdicts s42.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pairwise_core import (  # noqa: E402
    PROMPTS,
    build_judge,
    conversation_text,
    judge_pair,
    load_arm,
    load_prompts,
    read_ids,
    resolve_winner,
)


def load_spec(path: str) -> dict:
    """Read the row spec; relative paths inside it are taken relative to the spec file."""
    root = Path(path).resolve().parent
    fix = lambda p: str(p if os.path.isabs(p) else root / p)  # noqa: E731
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in spec["rows"]:
        row["questions"] = [fix(p) for p in row["questions"]]
        row["ids"] = fix(row["ids"])
        row["responses"] = {arm: {seed: [fix(p) for p in files] for seed, files in by_seed.items()}
                            for arm, by_seed in row["responses"].items()}
    return spec


def row_seeds(row: dict) -> list[str]:
    seeds = {s for arm in (row["x"], row["y"]) for s in row["responses"][arm] if s != "all"}
    return sorted(seeds, key=lambda s: (len(s), s))


def row_items(row: dict) -> list[str]:
    ids = read_ids(row["ids"], row.get("ids_sets"))
    n = row.get("first_n") or 0
    return ids[:n] if n else ids


def arm_files(row: dict, arm: str, seed: str) -> list[str]:
    spec = row["responses"][arm]
    return spec.get(seed) or spec.get("all") or []


def row_key(row: dict) -> tuple[str, str, str]:
    return row["domain"], row["x"], row["y"]


def run(args) -> None:
    spec = load_spec(args.spec)
    out_path = Path(args.verdicts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("winner") is not None:
                    done.add((r["domain"], r["x"], r["y"], str(r["seed"]), r["id"], r["order"]))
    tasks, missing = [], defaultdict(int)
    material = {}
    for row in spec["rows"]:
        prompts = load_prompts(row["questions"])
        items = row_items(row)
        for seed in row_seeds(row):
            resp = {
                arm: load_arm(arm_files(row, arm, seed), data_source=row.get("data_source"))
                for arm in (row["x"], row["y"])
            }
            key = (*row_key(row), seed)
            material[key] = (prompts, resp)
            for rid in items:
                if rid not in prompts or rid not in resp[row["x"]] or rid not in resp[row["y"]]:
                    missing[key] += 1
                    continue
                for order in (0, 1):
                    if (*key, rid, order) not in done:
                        tasks.append((key, rid, order))
    print("pending", len(tasks), "done", len(done), "missing", {"/".join(k): v for k, v in missing.items()}, flush=True)
    judge = build_judge(args.judge_role, qpm=args.qpm, workers=args.workers)

    def one(task):
        key, rid, order = task
        domain, x, y, seed = key
        prompts, resp = material[key]
        left, right = (x, y) if order == 0 else (y, x)
        text = PROMPTS[domain].format(
            conv=conversation_text(prompts[rid]), a=resp[left][rid]["response"], b=resp[right][rid]["response"]
        )
        letter, err = judge_pair(judge, text)
        base = {"domain": domain, "seed": seed, "id": rid, "x": x, "y": y, "order": order, "judge": judge.model}
        if letter is None:
            return {**base, "winner": None, "err": err}
        return {**base, "left": left, "winner": resolve_winner(letter, left, right), "first_pos": letter == "A"}

    lock = threading.Lock()
    n = bad = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a", encoding="utf-8") as fh:
        for fut in as_completed([pool.submit(one, t) for t in tasks]):
            r = fut.result()
            n += 1
            bad += r["winner"] is None
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            if n % 200 == 0:
                print(f"{n}/{len(tasks)} err {bad} {n / ((time.time() - t0) / 60):.1f}/min", flush=True)
    print("PAIRWISE-SEEDS-DONE", n, "errors", bad, flush=True)


def tally(pairs: dict, x: str, y: str) -> dict:
    """pairs: id -> {order: winner}. Stable = same winner in both orders, else tie/unstable."""
    xs = ys = ties = 0
    for v in pairs.values():
        if len(v) < 2:
            continue
        if v[0] == v[1] and v[0] in (x, y):
            xs += v[0] == x
            ys += v[0] == y
        else:
            ties += 1
    return {"n": xs + ys + ties, "x": xs, "y": ys, "tie": ties, "y_pref": ys / (xs + ys) if xs + ys else float("nan")}


def summarize(args) -> None:
    spec = load_spec(args.spec)
    items = {row_key(r): set(row_items(r)) for r in spec["rows"]}
    V = defaultdict(lambda: defaultdict(dict))  # (domain, x, y, seed) -> id -> order -> winner

    def ingest(path: str, default_seed: str | None) -> None:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("winner") is None:
                continue
            seed = str(r.get("seed", default_seed))
            key = (r.get("domain"), r["x"], r["y"])
            if key in items and r["id"] in items[key]:
                V[(*key, seed)][r["id"]][r["order"]] = r["winner"]

    for path in args.extra_verdicts or []:
        ingest(path, args.extra_seed)
    ingest(args.verdicts, None)

    lines = [f"# Rubric-free pairwise over seeds ({Path(args.verdicts).name})", "",
             "Cell: y stable wins / x stable wins / tie or unstable (n) · y preference = y / (x + y).", "",
             "| row | " + " | ".join(f"seed {s}" for s in sorted({s for r in spec["rows"] for s in row_seeds(r)},
                                                                  key=lambda s: (len(s), s)))
             + " | y preference, mean ± sd | x W / T / L (sum) | x win rate, mean ± sd |"]
    all_seeds = sorted({s for r in spec["rows"] for s in row_seeds(r)}, key=lambda s: (len(s), s))
    lines.append("|---" * (len(all_seeds) + 4) + "|")
    results = []
    for row in spec["rows"]:
        x, y = row["x"], row["y"]
        cells, prefs = [], []
        wx = wy = wt = 0
        for seed in all_seeds:
            t = tally(V[(*row_key(row), seed)], x, y)
            if not t["n"]:
                cells.append("—")
                continue
            cells.append(f"{t['y']}/{t['x']}/{t['tie']} (n={t['n']}) · {100 * t['y_pref']:.1f}%")
            prefs.append(t["y_pref"])
            wx, wy, wt = wx + t["x"], wy + t["y"], wt + t["tie"]
        sd = 100 * st.stdev(prefs) if len(prefs) > 1 else 0.0
        mean = 100 * st.mean(prefs) if prefs else float("nan")
        label = row.get("label") or f"{x} vs {y}"
        lines.append(f"| {row['domain']}: {label} | " + " | ".join(cells)
                     + f" | {mean:.1f} ± {sd:.1f} (n={len(prefs)}) | {wx} / {wt} / {wy} | {100 - mean:.1f} ± {sd:.1f} |")
        results.append({"domain": row["domain"], "x": x, "y": y, "label": label, "seeds": len(prefs),
                        "y_pref_mean": mean, "sd": sd, "x_wins": wx, "ties": wt, "y_wins": wy})
    lines += ["", "± is the sample standard deviation over seeds (ddof=1)."]
    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run", help="judge every (row, seed, item) in both orders (resumable)")
    rp.add_argument("--spec", required=True, help="row spec JSON (see module docstring)")
    rp.add_argument("--verdicts", required=True, help="append-only verdicts JSONL (one file per judge)")
    rp.add_argument("--judge-role", default=None, help="judge_client role: none = evaluation judge, ALT = third family")
    rp.add_argument("--workers", type=int, default=16)
    rp.add_argument("--qpm", type=int, default=80)
    sm = sub.add_parser("summarize", help="per-seed and three-seed table")
    sm.add_argument("--spec", required=True)
    sm.add_argument("--verdicts", required=True)
    sm.add_argument("--extra-verdicts", action="append", default=None,
                    help="earlier verdicts in the same schema (repeatable), e.g. seed-42 pairwise_domain.py runs")
    sm.add_argument("--extra-seed", default="42", help="seed label for extra verdict rows that carry none")
    sm.add_argument("--out", default=None, help="also write the markdown table here")
    sm.add_argument("--json-out", default=None, help="also write the row summaries as JSON")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
