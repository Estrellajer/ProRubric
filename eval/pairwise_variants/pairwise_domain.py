#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rubric-free blind pairwise comparisons within one domain (seed-42 arms).

Covers the single-seed comparisons of Appendix "Rubric-Free Comparisons and
Length Controls": all pairs among a set of arms in medicine, science,
dialogue or writing, judged by the evaluation judge (paper: DeepSeek-V4-Pro)
or, with ``--judge-role ALT`` and ``--limit-per-pair 100``, by the
third-family judge (paper: GPT-5.6-luna). The same script, pointed at
responses regenerated under a 2,048-token ceiling, gives the seed-42 cells
of the equal-budget comparison (Figure ``fig:pairwise_winrates`` b).

Items
  medicine   600 ids: ``sample`` over the HealthBench-consensus questions,
             sha1(id) order over ids answered by every arm passed to it
             (the paper's sample used the untrained, Rubric-RL and ProRubric
             arms at 4B and at 8B), first 600.
  science    the 200 ResearchQA ids of the appropriateness-probe sample
             (``--ids probe_sample.json --ids-sets researchqa``).
  dialogue   the 200 Arena-Hard v2 ids of the probe sample (``--ids-sets arena``).
  writing    the 96 Creative-v3 + 104 WritingBench ids of the probe sample
             (``--ids-sets creative,writingbench``; two questions parquets,
             two responses files per arm).
  In science/dialogue/writing an id missing from any arm is dropped for every
  pair; in medicine it is skipped only for the pairs that need it.

  python3 pairwise_domain.py sample --questions hb_consensus.parquet \\
      --responses base=base.jsonl --responses A=atomic.jsonl ... --n 600 --out med_ids.json

  python3 pairwise_domain.py run --domain science \\
      --questions researchqa_valid.parquet --data-source researchqa_valid \\
      --ids probe_sample.json --ids-sets researchqa \\
      --responses base=base.jsonl --responses A=atomic.jsonl --responses B=prorubric.jsonl \\
      --pairs base:A,A:B,base:B --verdicts sci_verdicts.jsonl [--judge-role ALT --limit-per-pair 100]

  python3 pairwise_domain.py summarize [same arguments] [--out table.md]

Verdict rows (append-only JSONL, resumable, keyed by (id, x, y, order)):
  {"domain","seed","id","x","y","order","left","winner","first_pos","judge"}
``winner`` is an arm name or "tie"; ``order`` 0 shows x first. The same
schema is read by ``pairwise_seeds.py --extra-verdicts``.
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
    DOMAINS,
    PROMPTS,
    bootstrap_interval,
    build_judge,
    conversation_text,
    hash_key,
    judge_pair,
    load_arm,
    load_prompts,
    parse_named_paths,
    read_ids,
    resolve_winner,
)


def parse_pairs(value: str | None, arms: list[str]) -> list[tuple[str, str]]:
    if not value:
        return [(x, y) for i, x in enumerate(arms) for y in arms[i + 1:]]
    pairs = []
    for chunk in value.split(","):
        x, sep, y = chunk.partition(":")
        if not sep:
            raise ValueError(f"--pairs entries must be x:y, got {chunk!r}")
        pairs.append((x.strip(), y.strip()))
    return pairs


def load_inputs(args):
    named = parse_named_paths(args.responses)
    prompts = load_prompts(args.questions)
    arms = {
        name: load_arm(paths, data_source=args.data_source, skip_error_rows=args.domain == "medicine")
        for name, paths in named.items()
    }
    pairs = parse_pairs(args.pairs, list(named))
    for x, y in pairs:
        if x not in arms or y not in arms:
            raise ValueError(f"pair {x}:{y} names an arm without --responses")
    ids = read_ids(args.ids, args.ids_sets)
    if args.domain != "medicine":
        ids = [i for i in ids if i in prompts and all(i in arms[a] for a in arms)]
    return prompts, arms, pairs, ids


def pair_items(ids, arms, prompts, x, y, limit):
    """Ids used for pair (x, y): those both arms answered, optionally the first ``limit``."""
    sel = [i for i in ids if i in prompts and i in arms[x] and i in arms[y]]
    return sel[:limit] if limit else sel


def sample(args) -> None:
    named = parse_named_paths(args.responses)
    prompts = load_prompts(args.questions)
    arms = {name: load_arm(paths, data_source=args.data_source, skip_error_rows=True) for name, paths in named.items()}
    eligible = sorted(i for i in prompts if all(i in arms[a] for a in arms))
    chosen = sorted(eligible, key=hash_key)[: args.n]
    Path(args.out).write_text(json.dumps(chosen), encoding="utf-8")
    print(f"eligible {len(eligible)} -> sampled {len(chosen)} -> {args.out}")


def run(args) -> None:
    prompts, arms, pairs, ids = load_inputs(args)
    prompt_template = PROMPTS[args.domain]
    out_path = Path(args.verdicts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if row.get("winner") is not None:
                    done.add((row["id"], row["x"], row["y"], row["order"]))
    tasks = [
        (rid, x, y, order)
        for x, y in pairs
        for rid in pair_items(ids, arms, prompts, x, y, args.limit_per_pair)
        for order in (0, 1)
        if (rid, x, y, order) not in done
    ]
    print(f"items {len(ids)} | pending {len(tasks)} | done {len(done)}", flush=True)
    judge = build_judge(args.judge_role, qpm=args.qpm, workers=args.workers)

    def one(task):
        rid, x, y, order = task
        left, right = (x, y) if order == 0 else (y, x)
        text = prompt_template.format(
            conv=conversation_text(prompts[rid]), a=arms[left][rid]["response"], b=arms[right][rid]["response"]
        )
        letter, err = judge_pair(judge, text)
        row = {"domain": args.domain, "seed": args.seed, "id": rid, "x": x, "y": y, "order": order, "judge": judge.model}
        if letter is None:
            return {**row, "winner": None, "err": err}
        return {**row, "left": left, "winner": resolve_winner(letter, left, right), "first_pos": letter == "A"}

    lock = threading.Lock()
    n = bad = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a", encoding="utf-8") as fh:
        for fut in as_completed([pool.submit(one, t) for t in tasks]):
            row = fut.result()
            n += 1
            bad += row["winner"] is None
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if n % 300 == 0:
                print(f"{n}/{len(tasks)} err {bad} {n / ((time.time() - t0) / 60):.0f}/min", flush=True)
    print("PAIRWISE-DONE", n, "errors", bad, flush=True)


def summarize(args) -> None:
    prompts, arms, pairs, ids = load_inputs(args)
    V = defaultdict(dict)
    for line in open(args.verdicts, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("winner") is not None:
                V[(row["id"], row["x"], row["y"])][row["order"]] = row
    lines = [
        f"# Rubric-free blind pairwise, {args.domain}",
        "",
        "| pair (x vs y) | n | x stable wins | y stable wins | tie / unstable | x stable win rate [95%] | "
        "first-position rate | sign agreement with training-judge score (n) | chars x / y |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for x, y in pairs:
        keep = set(pair_items(ids, arms, prompts, x, y, args.limit_per_pair))
        rows = [(rid, v) for (rid, xx, yy), v in V.items() if xx == x and yy == y and len(v) == 2 and rid in keep]
        xs = ys = ties = 0
        first, agree = [], []
        for rid, v in rows:
            w0, w1 = v[0]["winner"], v[1]["winner"]
            first += [v[0]["first_pos"], v[1]["first_pos"]]
            if w0 == w1 and w0 in (x, y):
                xs += w0 == x
                ys += w0 == y
                sign = 1 if w0 == x else -1
                sx, sy = arms[x][rid]["score"], arms[y][rid]["score"]
                if sx is not None and sy is not None and sx != sy:
                    agree.append(int((sx > sy) == (sign > 0)))
            else:
                ties += 1
        dec = [1] * xs + [0] * ys
        lo, hi = bootstrap_interval(dec) if len(dec) >= 10 else (float("nan"), float("nan"))
        cx = st.mean(len(arms[x][rid]["response"]) for rid, _ in rows) if rows else float("nan")
        cy = st.mean(len(arms[y][rid]["response"]) for rid, _ in rows) if rows else float("nan")
        lines.append(
            f"| {x} vs {y} | {len(rows)} | {xs} | {ys} | {ties} ({100 * ties / max(1, len(rows)):.0f}%) | "
            f"{100 * xs / max(1, len(dec)):.1f} [{100 * lo:.1f}, {100 * hi:.1f}] | "
            f"{100 * st.mean(first) if first else float('nan'):.1f}% | "
            f"{100 * st.mean(agree) if agree else float('nan'):.1f} ({len(agree)}) | {cx:.0f} / {cy:.0f} |"
        )
    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample", help="sha1-hash sample of ids answered by every given arm (medicine items)")
    sp.add_argument("--questions", action="append", required=True, help="questions parquet (repeatable)")
    sp.add_argument("--responses", action="append", required=True, metavar="NAME=PATH")
    sp.add_argument("--data-source", default=None, help="keep only response rows with this data_source (or none)")
    sp.add_argument("--n", type=int, default=600)
    sp.add_argument("--out", required=True, help="output JSON id list")

    def common(p):
        p.add_argument("--domain", choices=DOMAINS, required=True, help="selects the judge prompt")
        p.add_argument("--questions", action="append", required=True, help="questions parquet (repeatable)")
        p.add_argument("--responses", action="append", required=True, metavar="NAME=PATH",
                       help="arm responses JSONL; repeat a NAME to merge several files (e.g. two writing suites)")
        p.add_argument("--data-source", default=None, help="keep only response rows with this data_source (or none)")
        p.add_argument("--ids", required=True, help="JSON id list, or a probe sample.json (then --ids-sets)")
        p.add_argument("--ids-sets", default=None, help="comma-separated probe sets, e.g. creative,writingbench")
        p.add_argument("--pairs", default=None, help="comma-separated x:y (default: all pairs in --responses order)")
        p.add_argument("--limit-per-pair", type=int, default=0,
                       help="judge only the first N eligible items per pair (the ALT cross-check used 100)")
        p.add_argument("--verdicts", required=True, help="append-only verdicts JSONL")

    rp = sub.add_parser("run", help="judge every pair in both orders (resumable)")
    common(rp)
    rp.add_argument("--judge-role", default=None, help="judge_client role: none = evaluation judge, ALT = third family")
    rp.add_argument("--seed", default="42", help="seed label written into verdict rows")
    rp.add_argument("--workers", type=int, default=24)
    rp.add_argument("--qpm", type=int, default=150)

    sm = sub.add_parser("summarize", help="stable-win table")
    common(sm)
    sm.add_argument("--out", default=None, help="also write the markdown table here")

    args = ap.parse_args()
    {"sample": sample, "run": run, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
