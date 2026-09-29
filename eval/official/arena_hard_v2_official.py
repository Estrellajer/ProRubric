#!/usr/bin/env python3
"""Arena-Hard v2 double-order judging against the category-specific official baselines.

Produces the Arena-Hard v2 column of Table ``tab:main_results``.

Protocol (lmarena/arena-hard-auto, commit 196f6b826783b3da7310e361a805fa36f0be83f3):
questions ``data/arena-hard-v2.0/question.jsonl``; per question the baseline answer of the
category's baseline model from ``utils/judge_utils.py`` ``JUDGE_SETTINGS`` (o3-mini-2025-01-31
or gemini-2.0-flash-001, answers from ``data/arena-hard-v2.0/model_answer/``) and the
category's ``system_prompt`` (verbatim, loaded from the upstream checkout). Every question is
judged in both orders (``baseline_A``: baseline as Assistant A; ``candidate_A``: candidate as
Assistant A) with the upstream user-prompt layout (PROMPT_TEMPLATE below); the last
``[[A>>B]]``-style label in the judgment is the verdict. Per question, each game gives the
candidate 1 / 0.5 / 0 for win / tie / loss (strength ignored); question score = mean of the
two games; benchmark score = mean over questions x 100 (a paired win rate against the
baselines, not the upstream Bradley-Terry / style-controlled fit). The upstream
strength-weighted preference (">>" counted three times) is kept as a diagnostic.
The upstream judge is replaced by the evaluation judge (temperature 0, thinking disabled,
max_tokens 16000, up to 3 attempts per game).

The response text is sent to the judge verbatim (no reasoning-trace stripping); the paper's
Arena-Hard v2 responses were generated with thinking disabled.

Inputs
  --upstream-dir   arena-hard-auto checkout at the pinned commit (see fetch_upstream.sh)
  --responses      NAME=PATH, JSONL {"id": "<question uid>" (an "arena_hard_v2:" prefix is stripped), "response"}
Outputs
  <work-dir>/grades.<NAME>.jsonl  one row per (id, order): {"run","id","category","order",
                                  "baseline_model","verdict","error","judge","raw"}
  <work-dir>/summary.json         per-run win rate (x100), paired deltas, weighted preference, per-category win rate

  python3 arena_hard_v2_official.py run --upstream-dir up/arena-hard-auto --work-dir arena \\
      --responses rubric_rl=rubric_rl.jsonl --responses prorubric=prorubric.jsonl
  python3 arena_hard_v2_official.py summarize --work-dir arena --runs rubric_rl,prorubric
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import official_common as common  # noqa: E402

BASELINE_MODELS = ("o3-mini-2025-01-31", "gemini-2.0-flash-001")
PROMPT_TEMPLATE = (
    "<|User Prompt|>\n{QUESTION}\n\n<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n"
    "<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n"
    "<|The End of Assistant B's Answer|>"
)
VERDICT_RE = re.compile(r"\[\[([AB<>=]+)\]\]|\[([AB<>=]+)\]", re.IGNORECASE)


def load_judge_settings(upstream: Path) -> dict:
    spec = importlib.util.spec_from_file_location("arena_judge_utils", upstream / "utils/judge_utils.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.JUDGE_SETTINGS


def load_questions(upstream: Path) -> dict[str, dict]:
    return {row["uid"]: row for row in common.load_jsonl(upstream / "data/arena-hard-v2.0/question.jsonl")}


def load_baselines(upstream: Path) -> dict[str, dict[str, str]]:
    result = {}
    for model in BASELINE_MODELS:
        result[model] = {}
        for row in common.load_jsonl(upstream / f"data/arena-hard-v2.0/model_answer/{model}.jsonl"):
            content = row["messages"][-1]["content"]
            answer = content["answer"] if isinstance(content, dict) else content
            result[model][str(row["uid"])] = str(answer)
    return result


def parse_verdict(raw: str) -> str:
    matches = [left or right for left, right in VERDICT_RE.findall(raw.upper())]
    valid = [value for value in matches if value in {"A>>B", "A>B", "A=B", "B=A", "B>A", "B>>A", "A<<B", "A<B", "B<<A", "B<A"}]
    if not valid:
        raise ValueError("no valid Arena verdict")
    return valid[-1]


def candidate_outcome(label: str, candidate_position: str) -> float:
    a_strength = {"A>>B": 3, "B<<A": 3, "A>B": 1, "B<A": 1, "A=B": 0, "B=A": 0, "B>A": -1, "A<B": -1, "B>>A": -3, "A<<B": -3}[label]
    candidate_strength = a_strength if candidate_position == "A" else -a_strength
    if candidate_strength > 0:
        return 1.0
    if candidate_strength < 0:
        return 0.0
    return 0.5


def official_weighted_observations(label: str, candidate_position: str) -> list[float]:
    strength = 3 if "<<" in label or ">>" in label else 1
    return [candidate_outcome(label, candidate_position)] * strength


def scores_by_id(path: Path) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    by_id = {}
    for row in common.load_jsonl(path):
        if row.get("verdict") is not None:
            by_id.setdefault(str(row["id"]), {})[str(row["order"])] = row
    win_rate, weighted_rate, categories = {}, {}, {}
    for row_id, games in by_id.items():
        if set(games) != {"baseline_A", "candidate_A"}:
            continue
        observations = [
            candidate_outcome(games["baseline_A"]["verdict"], "B"),
            candidate_outcome(games["candidate_A"]["verdict"], "A"),
        ]
        weighted = (
            official_weighted_observations(games["baseline_A"]["verdict"], "B")
            + official_weighted_observations(games["candidate_A"]["verdict"], "A")
        )
        win_rate[row_id] = sum(observations) / 2
        weighted_rate[row_id] = sum(weighted) / len(weighted)
        categories[row_id] = games["baseline_A"]["category"]
    return win_rate, weighted_rate, categories


def run(args) -> None:
    questions = load_questions(args.upstream_dir)
    settings = load_judge_settings(args.upstream_dir)
    baselines = load_baselines(args.upstream_dir)
    judge = common.configure_judge(args.judge_role, args.qpm, args.workers)
    for run_name, responses_path in common.parse_responses_arg(args.responses).items():
        raw_responses = common.load_responses(responses_path)
        responses = {key.removeprefix("arena_hard_v2:"): value for key, value in raw_responses.items()}
        ids = sorted(set(responses) & set(questions))
        if args.limit:
            ids = ids[: args.limit]
        grade_path = common.grades_path(args.work_dir, run_name)
        done = {(str(row["id"]), str(row["order"])) for row in common.load_jsonl(grade_path) if row.get("verdict")}
        tasks = [(uid, order) for uid in ids for order in ("baseline_A", "candidate_A") if (uid, order) not in done]
        print(f"{run_name}: matched_answers={len(ids)} pending_games={len(tasks)} done={len(done)}", flush=True)

        def grade(task):
            uid, order = task
            question = questions[uid]
            category = question["category"]
            baseline_model = settings[category]["baseline"]
            baseline = baselines[baseline_model][uid]
            candidate = responses[uid]
            answer_a, answer_b = (baseline, candidate) if order == "baseline_A" else (candidate, baseline)
            prompt = PROMPT_TEMPLATE.format(QUESTION=question["prompt"], ANSWER_A=answer_a, ANSWER_B=answer_b)
            value, error, raw = common.retry_parsed_call(
                judge,
                [{"role": "system", "content": settings[category]["system_prompt"]}, {"role": "user", "content": prompt}],
                parse_verdict, max_tokens=16000,
            )
            return {
                "run": run_name, "id": uid, "category": category, "order": order,
                "baseline_model": baseline_model, "verdict": value, "error": error,
                "judge": judge.model, "raw": raw,
            }

        errors = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for future in as_completed([executor.submit(grade, task) for task in tasks]):
                row = future.result()
                errors += row["verdict"] is None
                common.append_jsonl(grade_path, row)
        print(f"{run_name}: completed={len(tasks) - errors} errors={errors}", flush=True)


def summarize(args) -> None:
    per_run = {}
    weighted = {}
    category_map = {}
    for run_name in common.split_runs(args.runs):
        path = common.grades_path(args.work_dir, run_name)
        per_run[run_name], weighted[run_name], category_map[run_name] = scores_by_id(path)
    result = {
        "benchmark": "Arena-Hard-v2",
        "protocol": "official category-specific baseline and prompts; A/B plus B/A; simplified paired win-rate",
        "judge": "evaluation judge (substitute for the official judge)",
        "official_difference": "reports win-rate, not fitted Bradley-Terry/style-controlled score; official strength-weighted preference included as diagnostic",
        **common.paired_summary(per_run, scale=100.0),
        "weighted_preference_pct": {},
        "category_win_rate_pct": {},
    }
    for run_name in per_run:
        vals = list(weighted[run_name].values())
        result["weighted_preference_pct"][run_name] = round(100 * sum(vals) / len(vals), 4) if vals else None
        result["category_win_rate_pct"][run_name] = {}
        for category in sorted(set(category_map[run_name].values())):
            category_values = [score for uid, score in per_run[run_name].items() if category_map[run_name][uid] == category]
            result["category_win_rate_pct"][run_name][category] = {"n": len(category_values), "mean": round(100 * sum(category_values) / len(category_values), 4)}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run", help="judge every question in both orders")
    run_cmd.add_argument("--upstream-dir", type=Path, required=True, help="arena-hard-auto checkout")
    common.add_run_args(run_cmd)
    summ = sub.add_parser("summarize", help="score graded runs")
    common.add_summarize_args(summ)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
