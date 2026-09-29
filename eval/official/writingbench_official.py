#!/usr/bin/env python3
"""WritingBench official query-dependent-criteria score with the evaluation judge.

Produces the WritingBench column of Table ``tab:main_results``.

Protocol (X-PLUG/WritingBench, commit ae2d5176449b7b769815482641d35926f26793eb): every query
in ``benchmark_query/benchmark_all.jsonl`` carries its own checklist of five criteria; each
criterion is judged separately with the upstream ``prompt.py`` (``evaluate_system`` as system
message, ``evaluate_prompt.format(query, response, criteria)`` as user message; both used
verbatim, loaded from the upstream checkout) and scored 1-10 as ``{"score", "reason"}``.
Question score = mean over its criteria; the benchmark score = mean over questions x 10.
The upstream critic model is replaced by the evaluation judge (temperature 0, thinking
disabled, max_tokens 3000, up to 3 attempts per criterion).

Inputs
  --upstream-dir   WritingBench checkout at the pinned commit (see fetch_upstream.sh)
  --responses      NAME=PATH, JSONL {"id": <benchmark_all.jsonl "index" as str>, "response"}
Outputs
  <work-dir>/grades.<NAME>.jsonl  one row per (id, criterion_index): {"run","id",
                                  "criterion_index","criterion_name","score","reason","error","judge","raw"}
  <work-dir>/summary.json         per-run means (x10) and paired deltas

  python3 writingbench_official.py build-data --upstream-dir up/WritingBench --output-dir gen_inputs/writingbench
  python3 writingbench_official.py run --upstream-dir up/WritingBench --work-dir wb \\
      --responses rubric_rl=rubric_rl.jsonl --responses prorubric=prorubric.jsonl
  python3 writingbench_official.py summarize --upstream-dir up/WritingBench --work-dir wb --runs rubric_rl,prorubric
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import official_common as common  # noqa: E402

SOURCE_COMMIT = "ae2d5176449b7b769815482641d35926f26793eb"


def benchmark_file(upstream: Path) -> Path:
    return upstream / "benchmark_query" / "benchmark_all.jsonl"


def load_prompt_module(upstream: Path):
    spec = importlib.util.spec_from_file_location("writingbench_prompt", upstream / "prompt.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_benchmark(upstream: Path) -> dict[str, dict]:
    rows = {}
    with benchmark_file(upstream).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["index"])] = row
    return rows


def criterion_text(criterion: dict) -> str:
    name = str(criterion.get("name", "")).strip()
    details = [f"{key}: {value}" for key, value in criterion.items() if key != "name"]
    return name + (("\n" + "\n".join(details)) if details else "")


def parse_grade(raw: str) -> dict:
    value = common.parse_json_object(raw)
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or int(score) != score or not 1 <= int(score) <= 10:
        raise ValueError(f"invalid WritingBench score: {score!r}")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("missing score reason")
    return {"score": int(score), "reason": reason}


def question_scores(path: Path, benchmark: dict[str, dict]) -> dict[str, float]:
    by_id: dict[str, dict[int, float]] = {}
    for row in common.load_jsonl(path):
        if row.get("score") is not None:
            by_id.setdefault(str(row["id"]), {})[int(row["criterion_index"])] = float(row["score"])
    result = {}
    for row_id, scores in by_id.items():
        expected = len(benchmark[row_id]["checklist"])
        if set(scores) == set(range(expected)):
            result[row_id] = sum(scores.values()) / expected
    return result


def build_data(args) -> None:
    """Generation input: one row per query, prompt = the query as a single user turn."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for row_id, source in load_benchmark(args.upstream_dir).items():
        rubric = [{"criterion": criterion_text(item), "weight": 1.0} for item in source["checklist"]]
        rows.append(
            {
                "prompt": [{"role": "user", "content": source["query"]}],
                "data_source": "writingbench",
                "extra_info": {"id": row_id, "problem": source["query"], "rubric": rubric},
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "writingbench.parquet"
    pq.write_table(pa.Table.from_pylist(rows), target)
    manifest = {
        "schema_version": "writingbench-v1",
        "rows": len(rows),
        "criteria_per_row": sorted({len(row["extra_info"]["rubric"]) for row in rows}),
        "source_commit": SOURCE_COMMIT,
        "source_sha256": common.sha256(benchmark_file(args.upstream_dir)),
        "output_sha256": common.sha256(target),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def run(args) -> None:
    prompt_module = load_prompt_module(args.upstream_dir)
    benchmark = load_benchmark(args.upstream_dir)
    judge = common.configure_judge(args.judge_role, args.qpm, args.workers)
    for run_name, responses_path in common.parse_responses_arg(args.responses).items():
        responses = common.load_responses(responses_path)
        ids = sorted(set(responses) & set(benchmark))
        if args.limit:
            ids = ids[: args.limit]
        grade_path = common.grades_path(args.work_dir, run_name)
        done = {
            (str(row["id"]), int(row["criterion_index"]))
            for row in common.load_jsonl(grade_path)
            if row.get("score") is not None
        }
        tasks = []
        for row_id in ids:
            source = benchmark[row_id]
            for index, criterion in enumerate(source["checklist"]):
                if (row_id, index) not in done:
                    tasks.append((row_id, index, source, criterion))
        print(f"{run_name}: matched_answers={len(ids)} pending_criteria={len(tasks)} done={len(done)}", flush=True)

        def grade(task):
            row_id, index, source, criterion = task
            user_prompt = prompt_module.evaluate_prompt.format(
                query=source["query"], response=responses[row_id], criteria=criterion
            )
            value, error, raw = common.retry_parsed_call(
                judge,
                [
                    {"role": "system", "content": prompt_module.evaluate_system},
                    {"role": "user", "content": user_prompt},
                ],
                parse_grade,
            )
            return {
                "run": run_name,
                "id": row_id,
                "criterion_index": index,
                "criterion_name": criterion.get("name"),
                "score": value["score"] if value else None,
                "reason": value["reason"] if value else None,
                "error": error,
                "judge": judge.model,
                "raw": raw,
            }

        errors = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for future in as_completed([executor.submit(grade, task) for task in tasks]):
                row = future.result()
                errors += row["score"] is None
                common.append_jsonl(grade_path, row)
        print(f"{run_name}: completed={len(tasks) - errors} errors={errors}", flush=True)


def summarize(args) -> None:
    benchmark = load_benchmark(args.upstream_dir)
    per_run = {}
    for run_name in common.split_runs(args.runs):
        per_run[run_name] = question_scores(common.grades_path(args.work_dir, run_name), benchmark)
    result = {
        "benchmark": "WritingBench",
        "protocol": "official prompt.py; each of five dynamic criteria scored 1-10; question mean multiplied by 10",
        "judge": "evaluation judge (substitute for the configurable official LLM evaluator)",
        **common.paired_summary(per_run, scale=10.0),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-data", help="write the generation-input parquet")
    build.add_argument("--upstream-dir", type=Path, required=True, help="WritingBench checkout")
    build.add_argument("--output-dir", type=Path, required=True)
    run_cmd = sub.add_parser("run", help="judge every (answer, criterion)")
    run_cmd.add_argument("--upstream-dir", type=Path, required=True, help="WritingBench checkout")
    common.add_run_args(run_cmd)
    summ = sub.add_parser("summarize", help="score graded runs")
    summ.add_argument("--upstream-dir", type=Path, required=True, help="WritingBench checkout")
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
