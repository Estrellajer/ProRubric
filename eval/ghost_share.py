#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ghost-credit share on HealthBench (Sec. 3.1 and App. C, Judges; 27.5% before training, 46.2 / 44.3 / 44.1%
for seeds 42 / 43 / 44 after): positive-weight criteria that the
training-time (wide) judge marks satisfied but the strict HealthBench judge
(``healthbench_judge.py`` grades) marks not met.

Zero judge calls: it joins the append-only grades JSONL produced by
``healthbench_judge.py run`` with the same responses JSONL files (whose
``satisfied`` field holds the training-time judge verdicts).

  python3 ghost_share.py \
      --questions healthbench_consensus.parquet \
      --responses atomic=atomic_scores.jsonl --responses prorubric=prorubric_scores.jsonl \
      --grades grades.jsonl [--out ghost_share.md]

Definitions (positive-weight criteria only):

  ghost_share = #(wide met & strict not met) / #(wide met)
  ghost_all   = #(wide met & strict not met) / #(all positive-weight criteria)
  strict_only = #(strict met & wide not met) / #(strict met)

The bootstrap 95% interval resamples questions, with the per-question ghost
share as the unit.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from healthbench_judge import load_grades, load_questions, load_responses, parse_responses_arg  # noqa: E402


def bootstrap_interval(values: list[float], *, samples: int = 2000, seed: int = 67) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


def analyze(
    questions: dict[str, dict],
    responses: dict[str, dict],
    grades: dict[tuple[str, str, int], bool],
) -> dict:
    per_question_rates: list[float] = []
    wide_met = strict_met = ghost = total = strict_only = 0
    questions_used = 0
    for rid, row in responses.items():
        question = questions.get(rid)
        if question is None:
            continue
        rubric = question["rubric"]
        wide_satisfied = row["satisfied"]
        # The caller passes this arm's verdicts keyed as (id, criterion_index).
        met_flags = [grades.get((rid, k)) for k in range(len(rubric))]
        if any(met is None for met in met_flags) or len(wide_satisfied) < len(rubric):
            continue
        questions_used += 1
        question_wide_met = 0
        question_ghost = 0
        for k, item in enumerate(rubric):
            if item["weight"] <= 0:
                continue
            total += 1
            wide = bool(wide_satisfied[k])
            strict = bool(met_flags[k])
            wide_met += int(wide)
            strict_met += int(strict)
            question_wide_met += int(wide)
            if wide and not strict:
                ghost += 1
                question_ghost += 1
            if strict and not wide:
                strict_only += 1
        if question_wide_met:
            per_question_rates.append(question_ghost / question_wide_met)
    return {
        "questions": questions_used,
        "positive_criteria": total,
        "wide_pass_rate": wide_met / total if total else None,
        "strict_pass_rate": strict_met / total if total else None,
        "ghost_share": ghost / wide_met if wide_met else None,
        "ghost_share_ci95": (
            bootstrap_interval(per_question_rates) if len(per_question_rates) >= 30 else (None, None)
        ),
        "ghost_all": ghost / total if total else None,
        "strict_only_share": strict_only / strict_met if strict_met else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", required=True, help="parquet with prompts and rubrics")
    parser.add_argument(
        "--responses",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="named run responses JSONL (repeatable)",
    )
    parser.add_argument("--grades", required=True, help="verdicts JSONL from healthbench_judge.py run")
    parser.add_argument("--out", default=None, help="optional path to also write the markdown table")
    args = parser.parse_args()

    arms = parse_responses_arg(args.responses)
    questions = load_questions(args.questions)
    all_grades = load_grades(args.grades)

    lines = [
        "| arm | questions | positive criteria | wide pass % | strict pass % | "
        "ghost share (wide met, strict not) [95%] | ghost / all criteria | strict-only / strict met |",
        "|---|---|---|---|---|---|---|---|",
    ]
    results = {}
    for name, path in arms.items():
        responses = load_responses(path)
        # Restrict the shared grade table to this arm's verdicts.
        arm_grades = {
            (rid, k): met
            for (arm, rid, k), met in all_grades.items()
            if arm == name
        }
        stats = analyze(questions, responses, arm_grades)
        results[name] = stats
        lo, hi = stats["ghost_share_ci95"]
        ci = f"[{100*lo:.1f}, {100*hi:.1f}]" if lo is not None else "n/a"
        def pct(value):
            return "n/a" if value is None else f"{100*value:.1f}"
        lines.append(
            f"| {name} | {stats['questions']} | {stats['positive_criteria']} | "
            f"{pct(stats['wide_pass_rate'])} | {pct(stats['strict_pass_rate'])} | "
            f"**{pct(stats['ghost_share'])}** {ci} | {pct(stats['ghost_all'])} | "
            f"{pct(stats['strict_only_share'])} |"
        )

    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
