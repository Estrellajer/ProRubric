#!/usr/bin/env python3
"""Creative Writing Benchmark v3, isolated rubric score (Elo/Glicko component excluded).

Produces the Creative-v3 column of Table ``tab:main_results``.

Protocol (EQ-bench/creative-writing-bench, commit c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b):
32 prompts x 3 seed modifiers (the first three ``seed_modifiers`` of each prompt) = 96 tasks.
Each piece is judged once with the upstream ``data/creative_writing_judging_prompt.txt``
(verbatim, filled with the base writing prompt, the response, the 22 criteria of
``creative_writing_criteria.txt`` as a bulleted list and the ``negative_criteria.txt`` names);
the ``[Scores]`` section is parsed for 0-20 scores of the official criteria. Piece score =
mean over the scored criteria, with the nine negative criteria inverted (20 - x); benchmark
score = mean over pieces x 5. The judge prompt tells the judge to omit criteria that do not
apply, so a piece need not carry all 22. The recommended upstream judge is replaced by the
evaluation judge (temperature 0, thinking disabled, max_tokens 4096, up to 3 attempts).

Inputs
  --upstream-dir   creative-writing-bench checkout at the pinned commit (see fetch_upstream.sh)
  --responses      NAME=PATH, JSONL {"id": "creative_writing_v3:<prompt_id>:iter<1..3>", "response"}
Outputs
  <work-dir>/grades.<NAME>.jsonl  one row per piece: {"run","id","scores":{criterion: 0-20},"error","judge","raw"}
  <work-dir>/summary.json         per-run means (x5) and paired deltas (bootstrap unit: the 96 tasks)

  python3 creative_writing_v3_official.py build-data --upstream-dir up/creative-writing-bench --output-dir gen_inputs/cw3
  python3 creative_writing_v3_official.py run --upstream-dir up/creative-writing-bench --work-dir cw3 \\
      --responses rubric_rl=rubric_rl.jsonl --responses prorubric=prorubric.jsonl
  python3 creative_writing_v3_official.py summarize --upstream-dir up/creative-writing-bench --work-dir cw3 --runs rubric_rl,prorubric
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import official_common as common  # noqa: E402

SOURCE_COMMIT = "c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b"


def upstream_files(upstream: Path) -> dict[str, Path]:
    return {
        "prompts": upstream / "data/creative_writing_prompts_v3.json",
        "criteria": upstream / "data/creative_writing_criteria.txt",
        "negative": upstream / "data/negative_criteria.txt",
        "judge_prompt": upstream / "data/creative_writing_judging_prompt.txt",
    }


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_items(upstream: Path) -> dict[str, dict]:
    prompts = json.loads(upstream_files(upstream)["prompts"].read_text())
    items = {}
    for prompt_id, source in prompts.items():
        modifiers = source.get("seed_modifiers") or []
        if len(modifiers) < 3:
            raise ValueError(f"prompt {prompt_id} has only {len(modifiers)} seed modifiers")
        for iteration in range(1, 4):
            modifier = modifiers[iteration - 1]
            row_id = f"creative_writing_v3:{prompt_id}:iter{iteration}"
            items[row_id] = {
                "prompt_id": str(prompt_id),
                "iteration": iteration,
                "base_prompt": source["writing_prompt"],
                "seed_modifier": modifier,
                "problem": source["writing_prompt"].replace("<SEED>", modifier),
                "category": source.get("category"),
                "title": source.get("title"),
            }
    return items


def parse_scores(raw: str) -> dict[str, float]:
    if "[Scores]" not in raw:
        raise ValueError("missing [Scores] section")
    score_section = raw.rsplit("[Scores]", 1)[1]
    scores = {}
    for pattern in (
        r"(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)",
        r"(.*?):\s*\[(?:Score\s+)?(-?\d+(?:\.\d+)?)\]",
    ):
        for name, value in re.findall(pattern, score_section):
            score = float(value)
            if not 0 <= score <= 20:
                raise ValueError(f"score outside 0-20 for {name.strip()}: {score}")
            scores[name.strip()] = score
    if not scores:
        raise ValueError("no rubric scores parsed")
    return scores


def parse_official_scores(raw: str, criteria: list[str]) -> dict[str, float]:
    parsed = parse_scores(raw)
    selected = {name: parsed[name] for name in criteria if name in parsed}
    if not selected:
        raise ValueError("no official rubric scores parsed")
    return selected


def piece_score(scores: dict[str, float], negative: set[str]) -> float | None:
    values = []
    for metric, score in scores.items():
        if not 0 <= score <= 20:
            raise ValueError(f"score outside 0-20 for {metric}: {score}")
        value = 20.0 - score if metric in negative else score
        values.append(value)
    return sum(values) / len(values) if values else None


def scores_by_id(path: Path, criteria: list[str], negative: set[str]) -> dict[str, float]:
    result = {}
    for row in common.load_jsonl(path):
        if row.get("scores"):
            official = {name: float(row["scores"][name]) for name in criteria if name in row["scores"]}
            score = piece_score(official, negative)
            if score is not None:
                result[str(row["id"])] = score
    return result


def build_data(args) -> None:
    """Generation input: 96 rows, prompt = writing_prompt with <SEED> replaced by the seed modifier."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = upstream_files(args.upstream_dir)
    criteria = load_lines(files["criteria"])
    negative = set(load_lines(files["negative"]))
    rows = []
    for row_id, item in load_items(args.upstream_dir).items():
        rubric = [
            {
                "criterion": (f"Lower incidence is better: {name}" if name in negative else name),
                "weight": 1.0,
            }
            for name in criteria
        ]
        rows.append(
            {
                "prompt": [{"role": "user", "content": item["problem"]}],
                "data_source": "creative_writing_v3",
                "extra_info": {
                    "id": row_id,
                    "problem": item["problem"],
                    "base_problem": item["base_prompt"],
                    "seed_modifier": item["seed_modifier"],
                    "iteration": item["iteration"],
                    "rubric": rubric,
                },
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "prompts.parquet"
    pq.write_table(pa.Table.from_pylist(rows), target)
    manifest = {
        "schema_version": "creative-writing-v3",
        "rows": len(rows),
        "prompts": len(rows) // 3,
        "iterations": 3,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": common.sha256(files["prompts"]),
        "output_sha256": common.sha256(target),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def run(args) -> None:
    files = upstream_files(args.upstream_dir)
    items = load_items(args.upstream_dir)
    criteria = load_lines(files["criteria"])
    negative = load_lines(files["negative"])
    template = files["judge_prompt"].read_text()
    judge = common.configure_judge(args.judge_role, args.qpm, args.workers)
    for run_name, responses_path in common.parse_responses_arg(args.responses).items():
        responses = common.load_responses(responses_path)
        ids = sorted(set(responses) & set(items))
        if args.limit:
            ids = ids[: args.limit]
        grade_path = common.grades_path(args.work_dir, run_name)
        done = {str(row["id"]) for row in common.load_jsonl(grade_path) if row.get("scores")}
        pending = [row_id for row_id in ids if row_id not in done]
        print(f"{run_name}: matched_answers={len(ids)} pending={len(pending)} done={len(done)}", flush=True)

        def grade(row_id):
            item = items[row_id]
            prompt = template.format(
                writing_prompt=item["base_prompt"],
                test_model_response=responses[row_id],
                creative_writing_criteria="\n".join("- " + name for name in criteria),
                lower_is_better_criteria=", ".join(negative),
            )

            def parse_complete(raw_text):
                # The official prompt says to omit criteria that do not apply to
                # the piece, so a valid response need not contain all 22 names.
                return parse_official_scores(raw_text, criteria)

            value, error, raw = common.retry_parsed_call(
                judge, [{"role": "user", "content": prompt}], parse_complete, max_tokens=4096
            )
            return {
                "run": run_name, "id": row_id, "scores": value, "error": error,
                "judge": judge.model, "raw": raw,
            }

        errors = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for future in as_completed([executor.submit(grade, row_id) for row_id in pending]):
                row = future.result()
                errors += not bool(row["scores"])
                common.append_jsonl(grade_path, row)
        print(f"{run_name}: completed={len(pending) - errors} errors={errors}", flush=True)


def summarize(args) -> None:
    files = upstream_files(args.upstream_dir)
    criteria = load_lines(files["criteria"])
    negative = set(load_lines(files["negative"]))
    per_run = {
        run_name: scores_by_id(common.grades_path(args.work_dir, run_name), criteria, negative)
        for run_name in common.split_runs(args.runs)
    }
    result = {
        "benchmark": "CreativeWriting-V3 isolated rubric",
        "protocol": "official 22 dimensions on 0-20; nine negative dimensions inverted; mean multiplied by 5",
        "judge": "evaluation judge (substitute for the recommended upstream judge)",
        "elo": "not computed: official leaderboard Elo/Glicko requires canonical historical run pool and adaptive pairwise matchups",
        "bootstrap_unit": "iteration task (32 prompts x 3 seed iterations = 96 tasks)",
        **common.paired_summary(per_run, scale=5.0),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-data", help="write the generation-input parquet")
    build.add_argument("--upstream-dir", type=Path, required=True, help="creative-writing-bench checkout")
    build.add_argument("--output-dir", type=Path, required=True)
    run_cmd = sub.add_parser("run", help="judge every piece once")
    run_cmd.add_argument("--upstream-dir", type=Path, required=True, help="creative-writing-bench checkout")
    common.add_run_args(run_cmd)
    summ = sub.add_parser("summarize", help="score graded runs")
    summ.add_argument("--upstream-dir", type=Path, required=True, help="creative-writing-bench checkout")
    common.add_summarize_args(summ)
    args = parser.parse_args()
    if args.command == "build-data":
        build_data(args)
    elif args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
