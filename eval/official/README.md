# Official benchmark scorers (WritingBench, Creative-v3, Arena-Hard v2, ResearchQA)

These four scripts produce the WritingBench, Creative-v3, Arena-Hard v2 and ResearchQA
columns of Table `tab:main_results` (protocols described in the appendix "Benchmark Scoring
and Judge Configurations"). Each follows its upstream benchmark's scoring protocol unchanged;
the only substitutions are (i) the judge, which is the package's evaluation judge
(DeepSeek-V4-Pro in the paper) in place of each benchmark's official judge, and (ii) the
response source, which is a generic JSONL file in place of the upstream sample generators.
Prompts, criteria, questions and baseline answers are read from the upstream repositories at
pinned commits; nothing from them is redistributed here.

| Script | Benchmark (upstream, pinned) | Unit judged | Score reported (x100 scale) |
|---|---|---|---|
| `writingbench_official.py` | [X-PLUG/WritingBench](https://github.com/X-PLUG/WritingBench) @ `ae2d5176449b7b769815482641d35926f26793eb` | each of a query's 5 query-dependent criteria, 1-10, with upstream `prompt.py` (`evaluate_system` + `evaluate_prompt`) | mean over criteria, then over the 1,000 queries, x10 |
| `creative_writing_v3_official.py` | [EQ-bench/creative-writing-bench](https://github.com/EQ-bench/creative-writing-bench) @ `c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b` | each piece once with `creative_writing_judging_prompt.txt`; 22 criteria scored 0-20 | isolated rubric score: mean over scored criteria (9 negative criteria inverted as 20-x), then over the 96 pieces (32 prompts x 3 seed modifiers), x5; the Elo/Glicko leaderboard component is not computed |
| `arena_hard_v2_official.py` | [lmarena/arena-hard-auto](https://github.com/lmarena/arena-hard-auto) @ `196f6b826783b3da7310e361a805fa36f0be83f3` | each question in both orders against the category's baseline (`JUDGE_SETTINGS`: o3-mini-2025-01-31 or gemini-2.0-flash-001) with the category's system prompt | per game 1/0.5/0 for candidate win/tie/loss; mean of the two orders, then over questions, x100 (paired win rate; the strength-weighted preference is reported as a diagnostic, no Bradley-Terry fit) |
| `researchqa_official_coverage.py` | [realliyifei/ResearchQA](https://github.com/realliyifei/ResearchQA) `compute_coverage.py` @ `747a9a1330f097a0e20672240cb47e3cf02500ae`; data: HF dataset `realliyifei/ResearchQA` @ `bf8a4cfef073ecfc0275c57acf8ca960e4dc79d6`, `valid.json` | rubric items in batches of 8, 5-level scale (Not at all ... Completely -> 1..5), upstream prompt verbatim | (x-1)/4 averaged over the rubric, then over the 703 validation queries, x100 |

`official_common.py` is the shared library (response loading, judge setup, retry, JSONL
grades, paired bootstrap).

## Judge configuration

All four scripts call `../../reward/judge_client.py` with the evaluation judge
(`RUBRIC_JUDGE_BASE_URL`, `RUBRIC_JUDGE_API_KEY`, `RUBRIC_JUDGE_MODEL`, no role;
`--judge-role ROLE` reads `RUBRIC_JUDGE_<ROLE>_*` instead). Every request is a
chat-completions call with temperature 0 and thinking disabled; max_tokens is 3000
(WritingBench, ResearchQA), 4096 (Creative-v3) and 16000 (Arena-Hard v2). A judgment whose
output the benchmark's parser rejects is re-requested up to 3 times in total and otherwise
recorded with `error` set (it then does not count toward the score). `--qpm` and `--workers`
only control throughput.

## Setup

```
bash fetch_upstream.sh up      # downloads the pinned upstream files into up/ and checks their MD5
export RUBRIC_JUDGE_BASE_URL=... RUBRIC_JUDGE_API_KEY=... RUBRIC_JUDGE_MODEL=...
```

Layout produced: `up/WritingBench/`, `up/creative-writing-bench/`, `up/arena-hard-auto/`,
`up/ResearchQA/valid.json`.

## Inputs: responses

Each evaluated model (one checkpoint, one evaluation) is one named run:
`--responses NAME=PATH`, repeatable, where PATH is a JSONL file with one object per
benchmark item, `{"id": ..., "response": ...}` (other fields ignored; empty responses
skipped). The response string is passed to the judge verbatim. Ids:

| Benchmark | `id` |
|---|---|
| WritingBench | the `index` field of `benchmark_query/benchmark_all.jsonl`, as a string |
| Creative-v3 | `creative_writing_v3:<prompt_id>:iter<k>`, k = 1..3 (the k-th seed modifier) |
| Arena-Hard v2 | the question `uid` (an `arena_hard_v2:` prefix is accepted and stripped) |
| ResearchQA | the `id` of `valid.json` (a `researchqa_valid:` prefix is accepted and stripped) |

Only ids present in both the responses file and the benchmark are judged; `summarize`
reports each run over its fully judged items (`n`) and over the items judged for every run
listed (`common_ids`).

Generation inputs. `writingbench_official.py build-data` and
`creative_writing_v3_official.py build-data` write the generation-input parquet
(`prompt` = one user turn with the query / the seeded writing prompt, `extra_info.id` as
above). Arena-Hard v2 prompts are the `prompt` field of `question.jsonl` as one user turn; the
paper's generation input holds 748 of the 750 questions (two very long `hard_prompt`
questions, uids `6c69551e80664df5` and `e54eb46a3f6247c4`, are not included), so Arena-Hard
v2 scores are over 748 questions.
ResearchQA prompts are the `query` field as the user turn, preceded by the system message
"Answer the following research question thoroughly and accurately. Ground your answer in
the relevant scientific literature where appropriate." All policies in the paper generate
in non-thinking mode; in particular the Arena-Hard v2 numbers in Table `tab:main_results`
come from responses generated with thinking disabled, and no reasoning trace is stripped
before judging.

## Commands

Run names below are examples (`rubric_rl` = Rubric-RL, `prorubric` = ProRubric; any names work).

```
# WritingBench
python3 writingbench_official.py run --upstream-dir up/WritingBench --work-dir out/wb \
    --responses rubric_rl=rubric_rl.wb.jsonl --responses prorubric=prorubric.wb.jsonl
python3 writingbench_official.py summarize --upstream-dir up/WritingBench --work-dir out/wb --runs rubric_rl,prorubric

# Creative Writing v3
python3 creative_writing_v3_official.py run --upstream-dir up/creative-writing-bench --work-dir out/cw3 \
    --responses rubric_rl=rubric_rl.cw3.jsonl --responses prorubric=prorubric.cw3.jsonl
python3 creative_writing_v3_official.py summarize --upstream-dir up/creative-writing-bench --work-dir out/cw3 --runs rubric_rl,prorubric

# Arena-Hard v2
python3 arena_hard_v2_official.py run --upstream-dir up/arena-hard-auto --work-dir out/arena \
    --responses rubric_rl=rubric_rl.arena.jsonl --responses prorubric=prorubric.arena.jsonl
python3 arena_hard_v2_official.py summarize --work-dir out/arena --runs rubric_rl,prorubric

# ResearchQA coverage
python3 researchqa_official_coverage.py run --questions up/ResearchQA/valid.json --work-dir out/rqa \
    --responses rubric_rl=rubric_rl.rqa.jsonl --responses prorubric=prorubric.rqa.jsonl
python3 researchqa_official_coverage.py summarize --questions up/ResearchQA/valid.json --work-dir out/rqa --runs rubric_rl,prorubric

# generation inputs
python3 writingbench_official.py build-data --upstream-dir up/WritingBench --output-dir gen/writingbench
python3 creative_writing_v3_official.py build-data --upstream-dir up/creative-writing-bench --output-dir gen/creative_writing_v3
```

`run` is resumable: judgments are appended to `<work-dir>/grades.<NAME>.jsonl` and
re-running skips every unit that already has a parsed result. `--limit N` judges only the
first N matched ids per run.

## Outputs

`<work-dir>/grades.<NAME>.jsonl`, one line per judged unit:

- WritingBench: `{"run", "id", "criterion_index", "criterion_name", "score" (1-10), "reason", "error", "judge", "raw"}`
- Creative-v3: `{"run", "id", "scores": {criterion: 0-20}, "error", "judge", "raw"}`
- Arena-Hard v2: `{"run", "id", "category", "order" ("baseline_A"|"candidate_A"), "baseline_model", "verdict" (e.g. "A>B"), "error", "judge", "raw"}`
- ResearchQA: `{"run", "id", "batch" (index of the batch's first rubric item), "scores": [1..5] | null, "err"}`

`<work-dir>/summary.json` (also printed): per-run means on the x100 scale
(`models[NAME].mean` for WritingBench, Creative-v3 and Arena-Hard v2;
`coverage_pct[NAME].all` for ResearchQA), paired deltas between every pair of runs with
95% percentile-bootstrap intervals over items (5,000 resamples, seed 67; ResearchQA: 2,000
resamples, seed 0), and for Arena-Hard v2 the strength-weighted preference and per-category
win rates. A cell of Table `tab:main_results` averages this per-run score over repeated
evaluations of the same checkpoint (where present) and then over training seeds (mean and
sample sd, ddof=1); that aggregation is not part of these scripts.
