#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-domain appropriateness probe (Appendix "Appropriateness Probe across Domains", Table ``tab:probe``).

Every response is scored on the same four domain-independent criteria
(``CRITERIA``), one judge call per criterion, with the official HealthBench
grader template (``eval/healthbench_judge.py``; each criterion rendered as
``[1.0] <criterion>``). A response's probe score is the mean of its four
verdicts (x100). Differences between arms are paired over prompts, with a
95% bootstrap interval over prompts (5,000 resamples, seed 67).

Sets and items (paper): 200 prompts per domain -- HealthBench-consensus
(medicine), Arena-Hard v2 (dialogue), ResearchQA (science), and 96
Creative-v3 plus 104 WritingBench prompts (writing). Items are drawn once,
blind to grades, by ``sample``: among the ids answered by the untrained,
Rubric-RL and ProRubric seed-42 arms, the first n in sha1(id) order;
ResearchQA is first allocated proportionally over ``extra_info.general_domain``
(n * stratum / total, rounded), hash order within each stratum, trimmed to n
by hash order if rounding overshoots and topped up in global hash order if it
falls short. Other arms and seeds are graded on the same items later.

Subcommands (run once per set for ``sample`` and ``grade``):

  python3 probe.py sample --set researchqa --domain science --n 200 \\
      --questions researchqa_valid.parquet --data-source researchqa_valid \\
      --stratify-by general_domain \\
      --responses base=base.jsonl --responses A=atomic.jsonl --responses B=prorubric.jsonl \\
      --sample sample.json

  python3 probe.py grade --set researchqa --sample sample.json \\
      --questions researchqa_valid.parquet --data-source researchqa_valid \\
      --responses base=base.jsonl --responses A=atomic.jsonl --responses B=prorubric.jsonl \\
      --responses C=raw_and.jsonl --responses A43=atomic_s43.jsonl ... \\
      --grades grades.jsonl [--judge-role ALT] [--workers 16 --qpm 100]

  python3 probe.py summarize --sample sample.json --grades grades.jsonl [--out PROBE_RESULTS.md]

Arm keys (free-form; the summary's defaults follow the paper's naming):
  base = untrained model, A = Rubric-RL, B = ProRubric, C = raw-AND,
  G = Graded, AV = Rubric-RL + appr. criterion, RUSCA = RuscaRL, RGSD = OPSD,
  NV = ProRubric w/o failure clauses, AR = atomic-rw;
  a 43/44 suffix marks training seed 43/44 (no suffix = seed 42).

Inputs
  questions  parquet rows with ``prompt`` and ``extra_info.id``
             (``extra_info.general_domain`` for ResearchQA stratification).
  responses  JSONL rows ``{"id", "response", ["data_source"]}``; rows whose
             ``data_source`` is set and differs from ``--data-source`` are dropped.
Outputs
  sample.json  {set: {"domain", "eligible", "ids": [...]}}
  grades.jsonl {"set","id","arm","k","met","chars","judge"} (append-only, resumable;
               ``k`` indexes CRITERIA, ``chars`` = response length)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "reward"))

import pyarrow.parquet as pq  # noqa: E402

from healthbench_judge import conversation_text, grader_prompt  # noqa: E402
from judge_client import JudgeUnavailableError, judge_from_env  # noqa: E402
from pairwise import bootstrap_interval, strip_fence  # noqa: E402

CRITERIA = [
    "The response should address the user's primary request before adding supplementary material, unless safety or essential missing context requires clarification first.",
    "The response should adapt its content, assumptions, and level of detail to the user's stated context and constraints.",
    "The response should include only information that materially helps satisfy the request and should avoid unnecessary repetition or digression.",
    "The response should distinguish supported claims from uncertainty and, when missing information would materially change the answer, state the assumption or ask for clarification.",
]
CRITERION_NAMES = ["c1 primary request", "c2 context fit", "c3 no redundancy", "c4 uncertainty"]
JUDGE_MAX_TOKENS = 3000
JUDGE_ATTEMPTS = 3

CORE_ARMS = ["base", "A", "B"]  # sampling eligibility and the core comparisons
EXTRA_ARMS = ["C", "G", "AV", "RUSCA", "RGSD", "NV", "AR", "AR43", "A43", "A44", "B43", "B44"]
LABELS = {
    "base": "Untrained model", "A": "Rubric-RL", "B": "ProRubric", "C": "raw-AND", "G": "Graded",
    "AV": "Rubric-RL + appr. criterion", "RUSCA": "RuscaRL", "RGSD": "OPSD", "NV": "ProRubric w/o failure clauses",
    "AR": "atomic-rw, s42", "AR43": "atomic-rw, s43", "A43": "Rubric-RL, s43", "A44": "Rubric-RL, s44",
    "B43": "ProRubric, s43", "B44": "ProRubric, s44",
}


def hash_key(identifier: str) -> str:
    return hashlib.sha1(identifier.encode()).hexdigest()


def extra_info(row: dict) -> dict:
    extra = row.get("extra_info")
    return json.loads(extra) if isinstance(extra, str) else (extra or {})


def load_questions(paths: list[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            rows[str(extra_info(row).get("id"))] = row
    return rows


def load_responses(path: str, data_source: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if data_source and row.get("data_source") not in (None, data_source):
                continue
            if row.get("response"):
                out[str(row["id"])] = row["response"]
    return out


def parse_responses(values: list[str], data_source: str | None) -> dict[str, dict[str, str]]:
    arms: dict[str, dict[str, str]] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep:
            raise ValueError(f"--responses must be NAME=PATH, got {value!r}")
        arms[name.strip()] = load_responses(path.strip(), data_source)
    return arms


def draw(ids: list[str], n: int, strata_of: dict[str, str] | None) -> list[str]:
    if strata_of is None:
        return sorted(ids, key=hash_key)[:n]
    strata: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        strata[strata_of[i]].append(i)
    alloc = {s: round(n * len(v) / len(ids)) for s, v in strata.items()}
    chosen = [i for s, v in strata.items() for i in sorted(v, key=hash_key)[: alloc[s]]]
    chosen = sorted(chosen, key=hash_key)[:n] if len(chosen) > n else chosen
    while len(chosen) < n:
        for i in sorted(ids, key=hash_key):
            if i not in chosen:
                chosen.append(i)
                break
    return chosen


def sample(args) -> None:
    questions = load_questions(args.questions)
    arms = parse_responses(args.responses, args.data_source)
    missing = [a for a in args.core_arms.split(",") if a not in arms]
    if missing:
        raise ValueError(f"sampling needs --responses for the core arms {missing}")
    ids = sorted(i for i in questions if all(i in arms[a] for a in args.core_arms.split(",")))
    strata_of = (
        {i: str(extra_info(questions[i]).get(args.stratify_by)) for i in ids} if args.stratify_by else None
    )
    chosen = draw(ids, args.n, strata_of)
    path = Path(args.sample)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if args.set in data and not args.overwrite:
        raise SystemExit(f"set {args.set!r} already in {path}; pass --overwrite to redraw it")
    data[args.set] = {"domain": args.domain, "eligible": len(ids), "ids": chosen}
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{args.set}: eligible {len(ids)} -> sampled {len(chosen)}")


def build_judge(role: str | None, qpm: int, workers: int):
    # Evaluation judge: thinking disabled; ALT judge: no thinking field in the request.
    thinking = "enabled" if role and role.upper() == "ALT" else "disabled"
    return judge_from_env(role or None, qpm=qpm, max_concurrency=workers, thinking=thinking)


def grade(args) -> None:
    ids = json.loads(Path(args.sample).read_text(encoding="utf-8"))[args.set]["ids"]
    questions = load_questions(args.questions)
    arms = parse_responses(args.responses, args.data_source)
    out_path = Path(args.grades)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("met") is not None:
                    done.add((r["set"], r["id"], r["arm"], r["k"]))
    tasks = [
        (args.set, i, arm, k, conversation_text(questions[i]["prompt"], resp[i]), len(resp[i]))
        for i in ids
        for arm, resp in arms.items()
        if i in resp
        for k in range(len(CRITERIA))
        if (args.set, i, arm, k) not in done
    ]
    print("pending", len(tasks), "done", len(done), flush=True)
    judge = build_judge(args.judge_role, args.qpm, args.workers)

    def one(task):
        s, i, arm, k, conv, chars = task
        row = {"set": s, "id": i, "arm": arm, "k": k, "chars": chars, "judge": judge.model}
        text = grader_prompt(conv, {"criterion": CRITERIA[k], "weight": 1.0})
        err = None
        for _ in range(JUDGE_ATTEMPTS):
            try:
                raw = judge.complete([{"role": "user", "content": text}], max_tokens=JUDGE_MAX_TOKENS, temperature=0).raw
            except (JudgeUnavailableError, ValueError) as exc:
                err = f"{type(exc).__name__}: {str(exc)[:160]}"
                continue
            try:
                met = json.loads(strip_fence(raw)).get("criteria_met")
            except Exception as exc:  # noqa: BLE001  (unparseable answer = one failed attempt)
                err = f"parse:{type(exc).__name__}"
                continue
            if met is True or met is False:
                return {**row, "met": bool(met)}
            err = f"non_boolean:{met!r}"
        return {**row, "met": None, "err": err}

    lock = threading.Lock()
    n = bad = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a", encoding="utf-8") as fh:
        for fut in as_completed([pool.submit(one, t) for t in tasks]):
            r = fut.result()
            n += 1
            bad += r["met"] is None
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            if n % 200 == 0:
                print(f"{n}/{len(tasks)} err {bad} {n / ((time.time() - t0) / 60):.0f}/min", flush=True)
    print("GRADE-DONE", n, "errors", bad, flush=True)


def load_grades(path: str):
    g: dict[tuple, dict[int, bool]] = defaultdict(dict)
    chars: dict[tuple, int] = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("met") is not None:
            g[(r["set"], r["id"], r["arm"])][r["k"]] = r["met"]
            if r.get("chars") is not None:
                chars[(r["set"], r["id"], r["arm"])] = r["chars"]
    return g, chars


def domain_sets(sample_data: dict) -> dict[str, list[str]]:
    doms: dict[str, list[str]] = {}
    for name, entry in sample_data.items():
        doms.setdefault(entry["domain"], []).append(name)
    return doms


def domain_items(g, sample_data, sets, core, extras):
    """Paired items and arms for one domain: the core arms must be fully graded on an item;
    an extra arm joins (and the items shrink to where it is graded) if it covers >= 90% of them."""
    ids = [(s, i) for s in sets for i in sample_data[s]["ids"] if all(len(g.get((s, i, a), {})) == 4 for a in core)]
    joined = []
    for xa in extras:
        with_x = [(s, i) for s, i in ids if len(g.get((s, i, xa), {})) == 4]
        if ids and len(with_x) >= 0.9 * len(ids):
            ids = with_x
            joined.append(xa)
    return ids, core + joined, joined


def macro(g, s, i, arm) -> float:
    return st.mean(g[(s, i, arm)][k] for k in range(4))


def delta(g, ids, x, y, key="macro"):
    if key == "macro":
        d = [macro(g, s, i, x) - macro(g, s, i, y) for s, i in ids]
    else:
        d = [float(all(g[(s, i, x)][k] for k in range(4))) - float(all(g[(s, i, y)][k] for k in range(4))) for s, i in ids]
    lo, hi = bootstrap_interval(d)
    return 100 * st.mean(d), 100 * lo, 100 * hi


def summarize(args) -> None:
    sample_data = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    g, chars = load_grades(args.grades)
    core = args.core_arms.split(",")
    extras = [a for a in args.extra_arms.split(",") if a]
    untrained, ref, method = core
    lines = ["# Appropriateness probe", "", "Sample: " + ", ".join(f"{k} {len(v['ids'])}" for k, v in sample_data.items()), ""]
    reading, probe_rows = [], []
    for dom, sets in domain_sets(sample_data).items():
        ids, arms_here, joined = domain_items(g, sample_data, sets, core, extras)
        if not ids:
            continue
        lines += [f"## {dom} (n={len(ids)})", "",
                  "| arm | " + " | ".join(CRITERION_NAMES) + " | score (mean of 4) | all four met | chars |",
                  "|---|---|---|---|---|---|---|---|"]
        score = {}
        for arm in arms_here:
            per_k = [st.mean(g[(s, i, arm)][k] for s, i in ids) for k in range(4)]
            score[arm] = st.mean(macro(g, s, i, arm) for s, i in ids)
            all4 = st.mean(all(g[(s, i, arm)][k] for k in range(4)) for s, i in ids)
            ch = [chars[(s, i, arm)] for s, i in ids if (s, i, arm) in chars]
            lines.append(f"| {arm} | " + " | ".join(f"{100 * v:.1f}" for v in per_k)
                         + f" | {100 * score[arm]:.1f} | {100 * all4:.1f} | {st.mean(ch) if ch else float('nan'):.0f} |")
        pairs = [(ref, untrained), (method, ref), (method, untrained)] + [(xa, y) for xa in joined for y in core]
        lines += ["", "| comparison | score Δ [95%] | all-four Δ [95%] |", "|---|---|---|"]
        deltas = {}
        for x, y in pairs:
            m, lo, hi = delta(g, ids, x, y)
            m2, lo2, hi2 = delta(g, ids, x, y, "all4")
            deltas[(x, y)] = (m, lo, hi)
            star = "*" if (lo > 0 or hi < 0) else ""
            lines.append(f"| {x}−{y} | {m:+.1f}{star} [{lo:+.1f}, {hi:+.1f}] | {m2:+.1f} [{lo2:+.1f}, {hi2:+.1f}] |")
        lines.append("")
        # Pre-registered reading: collapse if Rubric-RL is >= 3 points below the untrained model with the
        # interval excluding zero; recovery if ProRubric beats Rubric-RL and is within 3 points of untrained.
        mA, loA, hiA = deltas[(ref, untrained)]
        _, loB, _ = deltas[(method, ref)]
        _, loBb, _ = deltas[(method, untrained)]
        verdict = "collapse" if (mA <= -3 and hiA < 0) else ("no collapse" if loA > -3 else "mixed")
        recovery = "recovers" if (loB > 0 and loBb > -3) else "does not recover"
        reading.append(f"- {dom}: {ref}−{untrained} {mA:+.1f} [{loA:+.1f}, {hiA:+.1f}] -> {verdict}; "
                       f"{method} {recovery} ({method}−{ref} lower bound {loB:+.1f}, {method}−{untrained} lower bound {loBb:+.1f})")
        # tab:probe layout: score, and difference from Rubric-RL (the untrained row is the negated Rubric-RL−untrained).
        probe_rows.append(f"| *{dom}* | | |")
        for arm in arms_here:
            if args.table_arms and arm not in args.table_arms.split(","):
                continue
            label = LABELS.get(arm, arm)
            if arm == ref:
                probe_rows.append(f"| {label} | {100 * score[arm]:.1f} | --- |")
                continue
            if arm == untrained:
                m, lo, hi = -mA, -hiA, -loA
            else:
                m, lo, hi = deltas[(arm, ref)]
            star = "*" if (lo > 0 or hi < 0) else ""
            probe_rows.append(f"| {label} | {100 * score[arm]:.1f} | {m:+.1f}{star} [{lo:+.1f}, {hi:+.1f}] |")
    lines += ["## Pre-registered reading", ""] + reading + ["",
              f"## Table tab:probe layout (difference from {LABELS.get(ref, ref)})", "",
              "| arm | score | Δ vs reference [95% CI] |", "|---|---|---|"] + probe_rows
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def set_inputs(p):
        p.add_argument("--set", required=True, help="set name, e.g. medical, arena, researchqa, creative, writingbench")
        p.add_argument("--questions", action="append", required=True, help="questions parquet (repeatable)")
        p.add_argument("--data-source", default=None, help="keep only response rows with this data_source (or none)")
        p.add_argument("--responses", action="append", required=True, metavar="ARM=PATH", help="responses JSONL per arm")
        p.add_argument("--sample", required=True, help="sample.json (created/extended by `sample`)")

    sp = sub.add_parser("sample", help="draw the blind item sample for one set")
    set_inputs(sp)
    sp.add_argument("--domain", required=True, help="domain label the set is summarized under")
    sp.add_argument("--n", type=int, required=True)
    sp.add_argument("--stratify-by", default=None, help="extra_info field for proportional strata (ResearchQA: general_domain)")
    sp.add_argument("--core-arms", default=",".join(CORE_ARMS), help="arms every sampled id must have a response from")
    sp.add_argument("--overwrite", action="store_true")

    gp = sub.add_parser("grade", help="grade every arm on the set's items, four criteria each (resumable)")
    set_inputs(gp)
    gp.add_argument("--grades", required=True, help="append-only grades JSONL")
    gp.add_argument("--judge-role", default=None, help="judge_client role: none = evaluation judge, ALT = third family")
    gp.add_argument("--workers", type=int, default=16)
    gp.add_argument("--qpm", type=int, default=100)

    su = sub.add_parser("summarize", help="per-criterion scores, paired deltas, tab:probe layout")
    su.add_argument("--sample", required=True)
    su.add_argument("--grades", required=True)
    su.add_argument("--core-arms", default=",".join(CORE_ARMS), help="untrained,Rubric-RL,ProRubric arm keys")
    su.add_argument("--extra-arms", default=",".join(EXTRA_ARMS),
                    help="further arms, in the order they are tried for the paired subset")
    su.add_argument("--table-arms", default=None,
                    help="arms to list in the tab:probe layout (default: every arm of the domain); "
                         "seed-43/44 arms there are compared with the seed-42 Rubric-RL arm")
    su.add_argument("--out", default=None)

    args = ap.parse_args()
    {"sample": sample, "grade": grade, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
