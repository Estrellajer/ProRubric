#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rubric-free blind pairwise usefulness judge.

Two responses to the same prompt are shown without any rubric; the judge
picks the more useful one for the user who wrote the last message. Every
comparison is judged twice with the response order swapped, so a verdict is
only counted when both orderings agree.

  python3 pairwise.py run \
      --questions healthbench_consensus.parquet \
      --responses base=base_scores.jsonl --responses atomic=atomic_scores.jsonl \
      --responses prorubric=prorubric_scores.jsonl \
      --pairs base:atomic,atomic:prorubric,base:prorubric \
      --verdicts verdicts.jsonl [--n 600] [--workers 24] [--qpm 150]

  python3 pairwise.py summarize \
      --questions ... --responses ... [same as run] \
      --pairs ... --verdicts verdicts.jsonl \
      [--grades grades.jsonl] [--out pairwise_results.md]

Inputs are the same parquet / NAME=responses.jsonl contract as
``healthbench_judge.py``. The responses files' ``score`` field (the
training-time judge score) is used for the sign-agreement column;
``--grades`` additionally enables agreement against the strict HealthBench
judge per-question scores.

Judging is resumable: verdicts are appended to ``--verdicts`` one JSON object
per line, keyed by ``(id, x, y, order)``; re-running skips finished verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reward"))
from judge_client import JudgeUnavailableError, RubricJudge  # noqa: E402

from healthbench_judge import load_grades, load_responses, score as healthbench_score  # noqa: E402

PROMPT = """You are reviewing two candidate responses to the same user in a health conversation. You do not have a checklist; judge usefulness to THIS user.

Which response would be more useful to the person who wrote the last user message? Weigh, in order: whether it answers what was actually asked; whether it fits this user's situation, role and level of detail; whether it is medically accurate and safe; and whether it avoids unnecessary length, repetition or digression. Longer is not better by itself.

Conversation:
<<<
{conv}
>>>

Response A:
<<<
{a}
>>>

Response B:
<<<
{b}
>>>

Answer with a JSON object only: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""


def strip_fence(raw: str) -> str:
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        _, _, rest = value.partition("\n")
        value = rest[:-3].strip()
    return value


def extra_info(row: dict) -> dict:
    extra = row.get("extra_info")
    if isinstance(extra, str):
        try:
            return json.loads(extra)
        except json.JSONDecodeError:
            return {}
    return extra or {}


def load_prompts(path: str) -> dict[str, object]:
    prompts: dict[str, object] = {}
    for row in pq.read_table(path).to_pylist():
        rid = str(extra_info(row).get("id") or row.get("id"))
        prompts[rid] = row["prompt"]
    return prompts


def conversation_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n\n".join(
        f"[{message.get('role', 'user')}] {message.get('content', '')}" for message in prompt
    )


def hash_key(identifier: str) -> str:
    return hashlib.sha1(identifier.encode()).hexdigest()


def parse_pairs(value: str | None, arms: list[str]) -> list[tuple[str, str]]:
    if not value:
        return [(left, right) for i, left in enumerate(arms) for right in arms[i + 1 :]]
    pairs: list[tuple[str, str]] = []
    for chunk in value.split(","):
        left, sep, right = chunk.partition(":")
        if not sep or not left.strip() or not right.strip():
            raise ValueError(f"--pairs entries must be x:y, got: {chunk!r}")
        pairs.append((left.strip(), right.strip()))
    return pairs


def sample_ids(prompts: dict[str, object], responses: dict[str, dict], n: int) -> list[str]:
    eligible = sorted(
        identifier
        for identifier in prompts
        if all(identifier in arm_responses for arm_responses in responses.values())
    )
    chosen = sorted(eligible, key=hash_key)
    return chosen[:n] if n else chosen


JUDGE_MAX_TOKENS = 3000
JUDGE_ATTEMPTS = 3


def judge_one(judge: RubricJudge, prompt_text: str) -> tuple[str | None, str | None]:
    """Return ("A" | "B" | "tie", None) or (None, error) after up to three calls
    (temperature 0, 3,000 max tokens, thinking disabled, as in the paper's runs)."""
    error = None
    for _ in range(JUDGE_ATTEMPTS):
        try:
            completion = judge.complete(
                [{"role": "user", "content": prompt_text}], max_tokens=JUDGE_MAX_TOKENS, temperature=0
            )
        except (JudgeUnavailableError, ValueError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:160]}"
            continue
        try:
            winner = json.loads(strip_fence(completion.raw)).get("winner")
        except Exception as exc:  # noqa: BLE001 -- an unparseable answer is one failed attempt
            error = f"parse:{type(exc).__name__}"
            continue
        if winner in ("A", "B", "tie"):
            return winner, None
        error = f"bad:{winner!r}"
    return None, error


def run(args: argparse.Namespace) -> None:
    arms_paths = {}
    # argparse appends; reuse the same NAME=PATH contract.
    for value in args.responses:
        name, sep, path = value.partition("=")
        if not sep:
            raise ValueError(f"--responses must be NAME=PATH, got: {value!r}")
        arms_paths[name.strip()] = path.strip()
    prompts = load_prompts(args.questions)
    responses = {name: load_responses(path) for name, path in arms_paths.items()}
    pairs = parse_pairs(args.pairs, list(arms_paths))
    for left, right in pairs:
        if left not in arms_paths or right not in arms_paths:
            raise ValueError(f"pair {left}:{right} names an unknown arm")
    ids = sample_ids(prompts, responses, args.n)

    verdicts_path = Path(args.verdicts)
    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if verdicts_path.exists():
        with verdicts_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("winner") is not None:
                    done.add((row["id"], row["x"], row["y"], row["order"]))

    tasks: list[tuple[str, str, str, int]] = []
    for rid in ids:
        for left, right in pairs:
            if rid not in responses[left] or rid not in responses[right]:
                continue
            for order in (0, 1):
                if (rid, left, right, order) not in done:
                    tasks.append((rid, left, right, order))
    print(f"sampled {len(ids)} | pending {len(tasks)} | done {len(done)}", flush=True)

    judge = RubricJudge(qpm=args.qpm, max_concurrency=args.workers, max_tokens=JUDGE_MAX_TOKENS,
                        thinking="disabled")
    lock = threading.Lock()
    started = time.time()
    finished = failed = 0

    def grade(task: tuple[str, str, str, int]) -> dict:
        rid, x, y, order = task
        left, right = (x, y) if order == 0 else (y, x)
        prompt_text = PROMPT.format(
            conv=conversation_text(prompts[rid]),
            a=responses[left][rid]["response"],
            b=responses[right][rid]["response"],
        )
        winner, error = judge_one(judge, prompt_text)
        if winner is None:
            return {"id": rid, "x": x, "y": y, "order": order, "winner": None, "err": error}
        resolved = "tie" if winner == "tie" else (left if winner == "A" else right)
        return {
            "id": rid,
            "x": x,
            "y": y,
            "order": order,
            "left": left,
            "winner": resolved,
            "first_pos": winner == "A",
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor, verdicts_path.open(
        "a", encoding="utf-8"
    ) as out:
        futures = [executor.submit(grade, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            finished += 1
            failed += row["winner"] is None
            if finished % 300 == 0:
                rate = finished / max(1e-9, (time.time() - started) / 60)
                print(f"{finished}/{len(tasks)} err {failed} {rate:.0f}/min", flush=True)
    print(f"PAIRWISE-DONE {finished} errors {failed}", flush=True)


def bootstrap_interval(values: list[float], *, samples: int = 5000, seed: int = 67) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


def summarize(args: argparse.Namespace) -> None:
    arms_paths: dict[str, str] = {}
    for value in args.responses:
        name, sep, path = value.partition("=")
        if not sep:
            raise ValueError(f"--responses must be NAME=PATH, got: {value!r}")
        arms_paths[name.strip()] = path.strip()
    prompts = load_prompts(args.questions)
    responses = {name: load_responses(path) for name, path in arms_paths.items()}
    pairs = parse_pairs(args.pairs, list(arms_paths))
    ids = set(sample_ids(prompts, responses, args.n))

    verdicts: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    with open(args.verdicts, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("winner") is not None and row["id"] in ids:
                verdicts[(row["id"], row["x"], row["y"])][row["order"]] = row

    strict_scores: dict[str, dict[str, float]] = {}
    if args.grades:
        from healthbench_judge import load_questions

        questions = load_questions(args.questions)
        grades = load_grades(args.grades)
        for name in arms_paths:
            strict_scores[name] = {}
            for rid in ids:
                rubric = questions.get(rid, {}).get("rubric")
                if not rubric:
                    continue
                vals = [grades.get((name, rid, k)) for k in range(len(rubric))]
                if vals and all(value is not None for value in vals):
                    strict_scores[name][rid] = healthbench_score(vals, rubric)

    lines = [
        "| pair (x vs y) | n | x stable wins | y stable wins | tie / inconsistent | "
        "x stable win rate [95%] | first-position preference | "
        "sign agreement vs strict judge [95%] | sign agreement vs training judge |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for x, y in pairs:
        rows = [
            (rid, both)
            for (rid, xx, yy), both in verdicts.items()
            if xx == x and yy == y and len(both) == 2
        ]
        x_wins = y_wins = ties = 0
        first_positions: list[float] = []
        agree_strict: list[float] = []
        agree_wide: list[float] = []
        for rid, both in rows:
            w0, w1 = both[0]["winner"], both[1]["winner"]
            first_positions.extend([float(both[0]["first_pos"]), float(both[1]["first_pos"])])
            if w0 == w1 and w0 in (x, y):
                if w0 == x:
                    x_wins += 1
                else:
                    y_wins += 1
                sign = 1 if w0 == x else -1
                if rid in strict_scores.get(x, {}) and rid in strict_scores.get(y, {}):
                    delta = strict_scores[x][rid] - strict_scores[y][rid]
                    if delta != 0:
                        agree_strict.append(float((delta > 0) == (sign > 0)))
                wide_x = responses[x].get(rid, {}).get("score")
                wide_y = responses[y].get(rid, {}).get("score")
                if wide_x is not None and wide_y is not None:
                    delta = float(wide_x) - float(wide_y)
                    if delta != 0:
                        agree_wide.append(float((delta > 0) == (sign > 0)))
            else:
                ties += 1
        decisions = [1.0] * x_wins + [0.0] * y_wins
        win_lo, win_hi = (
            bootstrap_interval(decisions) if len(decisions) >= 10 else (float("nan"), float("nan"))
        )
        strict_lo, strict_hi = (
            bootstrap_interval(agree_strict)
            if len(agree_strict) >= 10
            else (float("nan"), float("nan"))
        )

        def pct_ci(lo: float, hi: float) -> str:
            return f"[{100*lo:.1f}, {100*hi:.1f}]"

        lines.append(
            f"| {x} vs {y} | {len(rows)} | {x_wins} | {y_wins} | {ties} "
            f"({100*ties/max(1, len(rows)):.0f}%) | "
            f"{100*x_wins/max(1, len(decisions)):.1f} {pct_ci(win_lo, win_hi)} | "
            f"{100*statistics.mean(first_positions) if first_positions else float('nan'):.1f}% | "
            f"{100*statistics.mean(agree_strict) if agree_strict else float('nan'):.1f} "
            f"{pct_ci(strict_lo, strict_hi)} (n={len(agree_strict)}) | "
            f"{100*statistics.mean(agree_wide) if agree_wide else float('nan'):.1f} "
            f"(n={len(agree_wide)}) |"
        )

    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--questions", required=True, help="parquet with prompts")
        target.add_argument(
            "--responses",
            action="append",
            required=True,
            metavar="NAME=PATH",
            help="named run responses JSONL (repeatable)",
        )
        target.add_argument("--pairs", default=None, help="comma-separated x:y pairs (default: all pairs)")
        target.add_argument("--n", type=int, default=600, help="number of hash-sampled questions")

    run_parser = sub.add_parser("run", help="run blind pairwise judging (resumable)")
    add_common(run_parser)
    run_parser.add_argument("--verdicts", required=True, help="append-only verdicts JSONL")
    run_parser.add_argument("--workers", type=int, default=24)
    run_parser.add_argument("--qpm", type=int, default=150)

    sum_parser = sub.add_parser("summarize", help="tally stable-win rates from verdicts")
    add_common(sum_parser)
    sum_parser.add_argument("--verdicts", required=True)
    sum_parser.add_argument(
        "--grades", default=None, help="optional healthbench_judge grades for strict-judge agreement"
    )
    sum_parser.add_argument("--out", default=None, help="optional path to also write the markdown table")

    args = parser.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
