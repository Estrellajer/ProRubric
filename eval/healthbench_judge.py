#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HealthBench strict cross-judge: regrade model responses with the official
HealthBench GRADER_TEMPLATE, one judge call per rubric criterion.

The judge is any OpenAI-compatible endpoint exposed through the shared
``reward/judge_client.py`` client (same client as the training-time reward);
configure it with the RUBRIC_JUDGE_* environment variables documented there.

  python3 healthbench_judge.py run \
      --questions healthbench_consensus.parquet \
      --responses atomic=atomic_scores.jsonl --responses prorubric=prorubric_scores.jsonl \
      --grades grades.jsonl [--limit N] [--workers 12] [--qpm 120]

  python3 healthbench_judge.py summarize \
      --questions healthbench_consensus.parquet \
      --responses atomic=atomic_scores.jsonl --responses prorubric=prorubric_scores.jsonl \
      --grades grades.jsonl [--out summary.json]

Input contracts:

- ``--questions`` is a parquet file whose rows carry ``prompt`` (a string or a
  list of ``{"role", "content"}`` messages) and ``extra_info.rubric``, a list
  of ``{"criterion": str, "weight": float}`` items. Row ids are taken from
  ``extra_info.id`` / ``extra_info.sample_id`` / ``extra_info.prompt_id``
  (then the top-level equivalents).
- ``--responses NAME=PATH`` (repeatable) names a run and points at a JSONL
  file with one object per row: ``id``, ``response``, and optionally the
  training-time judge's own ``satisfied`` booleans and ``score`` (used only
  for the agreement columns in ``summarize``).

Grading is resumable: verdicts are appended to ``--grades`` one JSON object
per line, keyed by ``(arm, id, criterion_index)``; re-running skips every
verdict already present.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reward"))

import pyarrow.parquet as pq  # noqa: E402

from judge_client import JudgeUnavailableError, RubricJudge  # noqa: E402

# GRADER_TEMPLATE below is the verbatim HealthBench grading prompt from
# openai/simple-evals (MIT license), healthbench_eval.py, GRADER_TEMPLATE.
# Upstream: https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
# Do not edit; the acceptance check diffs it against the upstream string.
GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def row_id(row: dict, fallback_index: int) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = {}
    extra = extra or {}
    for key in ("id", "sample_id", "prompt_id"):
        value = extra.get(key) if isinstance(extra, dict) else None
        if value is not None:
            return str(value)
    for key in ("id", "sample_id", "prompt_id"):
        if row.get(key) is not None:
            return str(row[key])
    return str(fallback_index)


def load_questions(path: str) -> dict[str, dict]:
    questions: dict[str, dict] = {}
    for index, row in enumerate(pq.read_table(path).to_pylist()):
        rid = row_id(row, index)
        extra = row.get("extra_info")
        if isinstance(extra, str):
            extra = json.loads(extra)
        rubric = [
            {"criterion": str(item["criterion"]), "weight": float(item["weight"])}
            for item in (extra or {}).get("rubric", [])
        ]
        questions[rid] = {"prompt": row["prompt"], "rubric": rubric}
    return questions


def load_responses(path: str) -> dict[str, dict]:
    responses: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("response"):
                continue
            responses[str(row["id"])] = {
                "response": row["response"],
                "satisfied": list(row.get("satisfied") or []),
                "score": row.get("score"),
            }
    return responses


def parse_responses_arg(values: list[str]) -> dict[str, str]:
    arms: dict[str, str] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"--responses must be NAME=PATH, got: {value!r}")
        arms[name.strip()] = path.strip()
    return arms


def conversation_text(prompt, response: str) -> str:
    # Matches openai/simple-evals healthbench_eval.py grade_sample: the model
    # response is appended as the final assistant turn, then every turn is
    # rendered as "role: content" joined by blank lines.
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
            for message in prompt
        ]
    messages = messages + [{"role": "assistant", "content": response}]
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)


def grader_prompt(conversation: str, item: dict) -> str:
    # Upstream renders a rubric item as "[points] criterion" (RubricItem.__str__).
    rubric_item = f"[{item['weight']}] {item['criterion']}"
    return GRADER_TEMPLATE.replace("<<conversation>>", conversation).replace(
        "<<rubric_item>>", rubric_item
    )


def strip_fence(raw: str) -> str:
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        first_line, _, rest = value.partition("\n")
        if first_line.strip().lower() in {"```", "```json"}:
            value = rest[:-3].strip()
    return value


def parse_verdict(raw: str) -> bool:
    payload = json.loads(strip_fence(raw))
    met = payload.get("criteria_met")
    if met is True or met is False:
        return bool(met)
    raise ValueError(f"criteria_met is not boolean: {met!r}")


def score(verdicts: list[bool], rubric: list[dict]) -> float:
    """HealthBench score: achieved points over possible positive points.

    Negative-weight rubric items contribute their (negative) points when met,
    but only positive points enter the denominator. This is the official
    HealthBench definition (openai/simple-evals calculate_score), clipped to [0, 1]
    as in the training reward. It is NOT a per-criterion average: averaging
    verdicts weights every criterion equally and inflates the base score
    (e.g. 25.7 -> 37.6) because HealthBench criteria carry unequal weights.
    """
    positive_total = sum(max(0.0, item["weight"]) for item in rubric)
    if positive_total <= 0:
        raise ValueError("rubric needs a positive total weight")
    achieved = sum(item["weight"] * int(bool(met)) for item, met in zip(rubric, verdicts))
    return max(0.0, min(1.0, achieved / positive_total))


def load_grades(path: str) -> dict[tuple[str, str, int], bool]:
    grades: dict[tuple[str, str, int], bool] = {}
    grades_path = Path(path)
    if not grades_path.exists():
        return grades
    with grades_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("met") is not None:
                grades[(row["arm"], str(row["id"]), int(row["k"]))] = bool(row["met"])
    return grades


def grade_one(judge: RubricJudge, task: tuple[str, str, int, str, dict]) -> dict:
    arm, rid, k, conversation, item = task
    try:
        completion = judge.complete(
            [{"role": "user", "content": grader_prompt(conversation, item)}]
        )
        met = parse_verdict(completion.raw)
        return {"arm": arm, "id": rid, "k": k, "met": met}
    except (JudgeUnavailableError, ValueError, json.JSONDecodeError) as exc:
        return {"arm": arm, "id": rid, "k": k, "met": None, "err": f"{type(exc).__name__}: {str(exc)[:200]}"}


def run(args: argparse.Namespace) -> None:
    arms = parse_responses_arg(args.responses)
    questions = load_questions(args.questions)
    responses = {name: load_responses(path) for name, path in arms.items()}
    grades_path = Path(args.grades)
    grades_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_grades(args.grades)

    common_ids = sorted(set(questions).intersection(*(set(rows) for rows in responses.values())))
    if args.limit:
        common_ids = common_ids[: args.limit]

    tasks: list[tuple[str, str, int, str, dict]] = []
    for rid in common_ids:
        question = questions[rid]
        for arm in arms:
            conversation = conversation_text(question["prompt"], responses[arm][rid]["response"])
            for k, item in enumerate(question["rubric"]):
                if (arm, rid, k) not in done:
                    tasks.append((arm, rid, k, conversation, item))
    print(
        f"paired ids {len(common_ids)} | tasks pending {len(tasks)} | already graded {len(done)}",
        flush=True,
    )

    judge = RubricJudge(qpm=args.qpm, max_concurrency=args.workers)
    lock = threading.Lock()
    started = time.time()
    finished = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor, grades_path.open(
        "a", encoding="utf-8"
    ) as out:
        futures = [executor.submit(grade_one, judge, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            finished += 1
            failed += row["met"] is None
            if finished % 200 == 0:
                rate = finished / max(1e-9, (time.time() - started) / 60)
                print(f"graded {finished}/{len(tasks)} (err {failed}) {rate:.0f}/min", flush=True)
    print(f"done {finished} errors {failed}", flush=True)


def bootstrap_delta(
    left: list[float], right: list[float], *, samples: int = 1000, seed: int = 0
) -> tuple[float, list[float]]:
    rng = random.Random(seed)
    n = len(left)

    def draw() -> float:
        return sum(right[rng.randrange(n)] - left[rng.randrange(n)] for _ in range(n)) / n

    deltas = sorted(draw() for _ in range(samples))
    point = sum(r - l for l, r in zip(left, right)) / n
    return point, [deltas[int(0.025 * samples)], deltas[min(samples - 1, int(0.975 * samples))]]


def summarize(args: argparse.Namespace) -> None:
    arms = parse_responses_arg(args.responses)
    questions = load_questions(args.questions)
    responses = {name: load_responses(path) for name, path in arms.items()}
    grades = load_grades(args.grades)

    per_question: dict[str, dict[str, float]] = {name: {} for name in arms}
    wide_score: dict[str, dict[str, float]] = {name: {} for name in arms}
    agreement: dict[str, list[int]] = {name: [0, 0] for name in arms}
    for rid, question in questions.items():
        rubric = question["rubric"]
        for name in arms:
            row = responses[name].get(rid)
            if row is None:
                continue
            verdicts = [grades.get((name, rid, k)) for k in range(len(rubric))]
            if any(met is None for met in verdicts):
                continue
            per_question[name][rid] = score(verdicts, rubric)
            if row["score"] is not None:
                wide_score[name][rid] = float(row["score"])
            wide_satisfied = row["satisfied"]
            for k, met in enumerate(verdicts):
                if k < len(wide_satisfied):
                    agreement[name][0] += int(bool(wide_satisfied[k]) == bool(met))
                    agreement[name][1] += 1

    common_ids = sorted(set.intersection(*(set(rows) for rows in per_question.values())))
    if not common_ids:
        print("no fully graded rows present for every named arm")
        return

    names = list(arms)
    result: dict = {
        "paired_rows": len(common_ids),
        "graded_calls": len(grades),
        "strict_judge": {},
        "paired_deltas": [],
        "training_judge_same_rows": {},
        "criterion_agreement_strict_vs_training": {},
    }
    for name in names:
        values = [per_question[name][rid] for rid in common_ids]
        entry = {"n": len(values), "mean": sum(values) / len(values)}
        result["strict_judge"][name] = entry
        wide_values = [wide_score[name][rid] for rid in common_ids if rid in wide_score[name]]
        if wide_values:
            result["training_judge_same_rows"][name] = {
                "n": len(wide_values),
                "mean": sum(wide_values) / len(wide_values),
            }
        if agreement[name][1]:
            result["criterion_agreement_strict_vs_training"][name] = (
                agreement[name][0] / agreement[name][1]
            )
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            l_values = [per_question[left][rid] for rid in common_ids]
            r_values = [per_question[right][rid] for rid in common_ids]
            delta, ci95 = bootstrap_delta(l_values, r_values)
            entry = {"pair": f"{right} minus {left}", "delta": delta, "ci95": ci95}
            wide_common = [
                rid
                for rid in common_ids
                if rid in wide_score[left] and rid in wide_score[right]
            ]
            if wide_common:
                wide_delta, wide_ci95 = bootstrap_delta(
                    [wide_score[left][rid] for rid in wide_common],
                    [wide_score[right][rid] for rid in wide_common],
                )
                entry["training_judge_delta"] = wide_delta
                entry["training_judge_ci95"] = wide_ci95
            result["paired_deltas"].append(entry)

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run", help="grade responses with the official HealthBench prompt")
    run_parser.add_argument("--questions", required=True, help="parquet with prompts and rubrics")
    run_parser.add_argument(
        "--responses",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="named run responses JSONL (repeatable)",
    )
    run_parser.add_argument("--grades", required=True, help="append-only verdicts JSONL")
    run_parser.add_argument("--limit", type=int, default=0, help="grade only the first N paired ids")
    run_parser.add_argument("--workers", type=int, default=12)
    run_parser.add_argument("--qpm", type=int, default=120)

    sum_parser = sub.add_parser("summarize", help="score graded verdicts (same score() as run)")
    sum_parser.add_argument("--questions", required=True)
    sum_parser.add_argument("--responses", action="append", required=True, metavar="NAME=PATH")
    sum_parser.add_argument("--grades", required=True)
    sum_parser.add_argument("--out", default=None, help="optional path to also write the summary JSON")

    args = parser.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
