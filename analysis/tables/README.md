# Paper tables from per-run scores

These scripts turn per-question scores and per-criterion verdicts into the numbers of the
paper's tables. None of them calls a model except `grade_hb.py`. Every script takes one
**arm-spec JSON** (`--spec`) that maps the paper's arms and seeds to input files. A worked
example with placeholder paths is in `arms.example.json`.

| Script | Produces |
|---|---|
| `ablations.py` | Table `tab:consensus_ablations` (`tables/ablations_main.tex`), `tab:ablations_full` (`tables/consensus_analysis.tex`), `tab:per_seed_ablations`, `tab:seeds` (`tables/seeds.tex`), and the HealthBench sizes of `tab:eval_sizes`. It also covers the step-300 check in the training-dynamics appendix (retrained runs as non-cutting arms). |
| `main_results.py` | Table `tab:main_results` (`tables/main_results.tex`): seven benchmarks, 4B and 8B, seed means, differences from Base, the seven-benchmark average, and the paired 95% interval of the average. Also `tab:per_seed_domains`. |
| `paired_consensus.py` | The SFT paragraph of the appendix notes (SFT against Base / Rubric-RL / ProRubric on consensus, 4B and 8B), and the 8B aggregation controls (raw-AND against Rubric-RL / ProRubric). |
| `crosscheck_rank.py` | Table `tab:gpt_crosscheck` (third-family judge, GPT-5.6-luna) and its Spearman correlation with the evaluation judge (rho = 0.84 over eleven arms). |
| `eval_sizes.py` | Table `tab:eval_sizes`, built from the two JSON outputs above. |
| `grade_hb.py` | Per-criterion HealthBench grading that writes the verdicts the scripts above consume (see "Grading" below). |
| `common.py` | Shared readers: spec loading, grades, second-evaluator scores, prompt set, sd, bootstrap, t-interval. |

The figure scripts in `../../figures/` draw these numbers. Run them with
`--ablations-json` to check the drawn values against `ablations.py`'s output.

## Conventions shared by all scripts

- **HealthBench score.** Per question, the official score is the sum of the weights of the met
  criteria divided by the sum of the positive weights, clipped to [0, 1]. It is averaged over
  questions and multiplied by 100. The code is `eval/healthbench_judge.py`'s `score()`, imported,
  not copied.
- **Seed spread.** The sample sd (ddof = 1). Base is one checkpoint.
- **Paired item sets.** Every comparison uses the intersection of the questions that all
  participating runs have been scored on without an evaluation failure, and each script prints
  that n. Arms that are not part of a table's definition never shrink that table's set: the roles
  `appendix` and `side` in `ablations.py`, and the separate comparisons in `paired_consensus.py`.
- **Relative paths** inside the spec resolve against the spec file's directory.

## Input files

| Kind | Format |
|---|---|
| questions | HealthBench-layout parquet: `prompt`, `extra_info.rubric` = `[{"criterion", "weight"}]`, ids from `extra_info.id` |
| responses | JSONL, one row per question: `{"id", "response", "score", "satisfied", "error", "data_source"}`. `score` and `satisfied` are the **second evaluator's** per-question HealthBench score and per-criterion verdicts (Doubao-lite in the paper). This is the same field convention `eval/healthbench_judge.py summarize` uses. A truthy `error` marks a failed evaluation of that question. `data_source` is optional (`healthbench_consensus` / `healthbench_full`); when it is present, one file can hold both suites. |
| grades | JSONL verdicts `{"arm", "id", "k", "met"}` (`met` is null on a failed call), as written by `grade_hb.py` or `eval/healthbench_judge.py run`. One file can hold many arms, so the spec names both the file and the arm key: `{"path": ..., "arm": ...}`. |
| WritingBench / Creative-v3 / Arena-Hard v2 | Per-question score JSONL `{"id", "score"}` in the upstream unit, one file per evaluation (see below) |
| MedQA / GPQA-Diamond | The evaluation's score JSONL: `{"id", "accuracy", "error", "data_source"}` |
| ResearchQA | Coverage grades JSONL `{"id", "batch", "scores"}`: the criteria of question `id` are judged in batches of 8, starting at index `batch`, on a 1–5 scale. The ResearchQA parquet supplies rubric sizes (`extra_info.id`, `extra_info.rubric`). |

The per-question writing and dialogue scores come from the upstream scorers (see
`eval/README.md`):

- **WritingBench.** The mean of the query's criterion scores (1–10). A question counts only if
  every criterion was scored.
- **Creative-v3.** The isolated rubric piece score (0–20).
- **Arena-Hard v2.** The mean of the two-order outcomes against the category baseline (0–1).

`main_results.py` scales these three by 10, 5 and 100.

## Arm names

The spec uses the paper's arm names. The file keys in `arms.example.json` are short descriptive
tokens:

| Paper name | Example key | What differs |
|---|---|---|
| Base | `base` | untrained model |
| Rubric-RL | `atomic` | atomic checklist, weighted sum |
| ProRubric | `prorubric` | dimensions with failure clauses |
| raw-AND | `raw_and` | ProRubric's grouping of the verbatim criteria |
| Graded | `graded` | four-level credit per dimension |
| ProRubric w/o failure clauses | `prorubric_nofail` | failure clauses removed |
| K=1 | `k1` | all dimensions merged into one |
| Rubric-RL + appr. criterion / ProRubric + appr. criterion | `atomic_appr` / `prorubric_appr` | fixed appropriateness criterion appended |
| Length-matched | `lenmatch` | 4,096-token response limit |
| Weighted | `weighted` | Rubric-RL with weighted criteria |
| Rubric-RL / ProRubric, beta = 0.005, 0.01, 0.04 | `atomic_kl005` ... `prorubric_kl04` | KL coefficient |
| ProRubric, dimensions from another generator | `prorubric_altgen` | dimensions regenerated by a second generator |
| RuscaRL, OPSD, SFT | `ruscarl`, `opsd`, `sft` | baselines |
| Rubric-RL / ProRubric, retrained | `*_retrained` | the retrained runs of the training-dynamics appendix at step 300 |

## Spec schema

Each script reads one top-level section of the spec. Every section may also carry a `_comment`
key, which the scripts ignore.

### `ablations` (ablations.py)

```json
{
 "questions": {"consensus": "healthbench_consensus.parquet", "hbfull": "healthbench_full.parquet"},
 "hbfull_prompt_set": {"grades": "hbfull_pro_seed42_arms.jsonl"},
 "consensus_ids": "consensus_common_ids.txt",
 "side_reference": ["Rubric-RL", "ProRubric", "Base"],
 "arms": [
  {"name": "Rubric-RL", "role": "table",
   "cells": [{"seed": 42, "responses": "responses/atomic_s42.jsonl",
              "consensus_grades": {"path": "grades/consensus_pro.jsonl", "arm": "atomic_s42"},
              "hbfull_grades": {"path": "grades/hbfull_pro.jsonl", "arm": "atomic_s42"}}, ...]},
  ...],
 "layout": {"ablations_main": [...], "ablations_full": [...], "per_seed_ablations": [...], "seeds": ["Rubric-RL", "ProRubric"]}
}
```

A cell's `responses` file supplies both suites through `data_source`. If the HealthBench-full
rows live in a separate file, add `"hbfull_responses": PATH` to the cell.

**`role`** controls how an arm affects the common item sets:

| Role | Effect |
|---|---|
| `table` | Its cells cut the common item sets. |
| `base` | The untrained row; it also cuts the sets. |
| `appendix` | Scored on the common sets restricted to the ids it covers. It never cuts them, and a value is reported only when the arm covers at least 95% of the set. |
| `side` | A single-seed control. Handled like `appendix`, and additionally paired with the `side_reference` cells (seed index 0) on the ids they share. |

The paper uses these roles as follows:

- `table`: the ten arms of `tab:consensus_ablations`.
- `appendix`:
  - RuscaRL, OPSD, the KL sweep and Weighted;
  - the three retrained runs of the training-dynamics appendix, so their step-300 scores are
    computed on the ablation table's own item set without moving it.
- `side`: the "another generator" control.

**Other fields.**

- **`hbfull_prompt_set`** is the HealthBench-full prompt set: the ids on which every arm in the
  given grades file has at least one verdict. In the paper this is the seed-42 grading run
  (4,563 prompts). Use `{"ids": FILE}` to give the list directly; omit the key to use every
  question.
- **`consensus_ids`** is optional. It freezes the consensus set to a given list, and the script
  asserts that the list is still inside the live intersection. The paper's set has 3,140 ids. A
  later retry pass completed one more verdict, so the live intersection is 3,141, and without the
  frozen list some cells move by 0.1.
- **`layout`** gives the rows and LaTeX labels of the four rendered tables, as
  `[section title, [[arm name, label, ...], ...]]`. `ablations_full` rows also carry the "grouped"
  and "appr. crit." flags: `true` / `false` / `null` render as `\facton` / `\factoff` / `\factna`.
  The rendered bodies use the paper's macros (`\method`, `\std`, `\facton`, ...). Bold emphasis
  in `tables/consensus_analysis.tex` was added by hand.

### `main_results` (main_results.py)

```json
{
 "questions": {"researchqa": "researchqa_valid.parquet", "healthbench_full": "healthbench_full.parquet"},
 "healthbench_prompt_set": {"grades": "hbfull_pro_seed42_arms.jsonl"},
 "cells": [{"size": "4B", "arm": "atomic", "seed": 42,
            "writingbench": [rep1, rep2, rep3], "creative_writing_v3": [...], "arena_hard_v2": [...],
            "medqa": "medqa_scores.jsonl", "gpqa_diamond": [...], "researchqa": [...],
            "healthbench": {"path": "grades/hbfull_pro.jsonl", "arm": "4B_atomic_s42"}}, ...],
 "paired_average": [["4B", "atomic", "prorubric"], ...],
 "layout": {"main_results": [["4B", "Qwen3-4B", [["base", "Base", false], ..., ["prorubric", "\\textbf{+ \\method{} (Ours)}", true]]], ...],
            "per_seed_domains": ["4B", "atomic", "prorubric"]}
}
```

**Cells.**

- A cell is one (size, arm, seed).
- A list holds repeated evaluations of the same checkpoint, which are averaged.
- The writing suites point at the writing-domain policy, Arena-Hard at the dialogue-domain
  policy, MedQA and HealthBench at the medical policy, and GPQA and ResearchQA at the science
  policy. SFT is one model for all four domains.
- The Arena-Hard files are the non-thinking regeneration used in the paper.

**Rules per benchmark.**

- **Writing and dialogue files.** A file counts only if it has at least 0.9 times the question
  count of the largest file in its cell and at least 0.95 times that of the largest file of its
  benchmark.
- **MedQA.** Scored on the ids every MedQA cell answered.
- **GPQA and ResearchQA.** Scored on the ids answered by the first file of every cell.
- **HealthBench.** Each cell is scored on its own fully graded ids within the prompt set, and is
  marked `complete` when it covers at least 99% of the set.
- **Rounding.** Per-seed values are rounded to two decimals, then averaged.

**Summary rows.** The seven-benchmark average is the mean of the seven column means. The paired
interval uses the per-seed averages, `pb - pa`, with mean, sd and a Student-t interval on n − 1
degrees of freedom. For the paper's 4B Rubric-RL vs. ProRubric comparison this is +1.30 ± 0.46,
[+0.17, +2.44].

**Rendered deltas.** The script renders each difference from Base as the difference of the
unrounded seed means.

### `paired_consensus` (paired_consensus.py)

One entry per comparison (`--which NAME`, `--list`):

```json
"sft_4b": {"questions": "healthbench_consensus.parquet", "boots": 1000,
           "groups": {"Base": [CELL], "Rubric-RL": [CELL, CELL, CELL], "ProRubric": [...], "SFT": [...]},
           "pairs": [["SFT", "Base"], ["SFT", "Rubric-RL"], ["SFT", "ProRubric"]]}
CELL = {"seed": 42, "responses": PATH, "grades": {"path": PATH, "arm": KEY}}
```

**Item set.** The consensus ids that every cell answered without an evaluation error and on which
every cell is fully graded by the evaluation judge.

**Reported values.**

- **Groups.** The mean ± sd over the group's cells.
- **Pairs.** The difference of per-id seed means, with a percentile bootstrap over ids
  (seed 0, `boots` resamples: 1000 for the SFT comparisons, 2000 for the 8B controls).

### `crosscheck` (crosscheck_rank.py)

```json
{"questions": "healthbench_consensus.parquet", "subset_ids": "optional id list",
 "arms": [{"name": "Rubric-RL", "label": "Rubric-RL", "section": "Baselines",
           "pro": {"path": "grades/consensus_pro.jsonl", "arm": "atomic_s42"},
           "alt": {"path": "grades/consensus_alt.jsonl", "arm": "atomic_s42"}}, ...],
 "pairs": [["Rubric-RL", "ProRubric"], ...]}
```

**Table scores.** Each arm's third-family score on its own fully graded ids. The paper reports
3,652–3,671 items per arm.

**Correlation.**

- The paper's rho is the `common` variant: every arm is scored on the ids fully graded for all
  arms by both judges. That gives 0.836.
- The script also prints rho on each arm's own sets (0.864) and within `subset_ids`.

**Pairs.** Paired third-family differences, with the difference of own-set scores alongside. The
paper's "+7.9" (raw-AND over Rubric-RL) and "+4.6" (appropriateness criterion appended to
ProRubric vs. to Rubric-RL) are differences of own-set scores.

## Grading (`grade_hb.py`) and how it relates to `eval/healthbench_judge.py`

`grade_hb.py` reuses `healthbench_judge.py`'s pieces verbatim: the official `GRADER_TEMPLATE`, the
conversation rendering, the rubric-item rendering `[weight] criterion`, `parse_verdict`, and the
grades format. The per-criterion prompt the judge sees is therefore byte-identical.

It exists because the paper's grading differed from `healthbench_judge.py run` in four ways. All
four change which items end up fully graded, and so change the common sets and the numbers:

1. **Three attempts per verdict.** A transport failure, or an unparsable or non-boolean
   `criteria_met`, is re-asked up to three times in total before `met: null` is written.
   `healthbench_judge.py` records a parse failure at once.
2. **Evaluation-error gating.** Responses rows with a truthy `error` are not graded, so the paired
   set is the ids every named arm answered cleanly. `healthbench_judge.py` grades any row that
   has a response.
3. **Fixed prompt set.** `--ids FILE` restricts grading to a list. `--ids-from-grades FILE`
   restricts it to the HealthBench-full prompt set: the ids on which every arm of an earlier
   grades file has a verdict. The later seeds were graded only on the seed-42 prompt set.
4. **Passes.** The pending criteria are re-submitted up to `--passes 3` times; whatever still
   fails stays out of that arm.

**Request settings.** Temperature 0 and `max_tokens` 3000. For the evaluation judge, thinking is
disabled with `RUBRIC_JUDGE_THINKING=disabled`. The third-family judge was called without a
thinking field, so leave `RUBRIC_JUDGE_ALT_THINKING` unset and run with `--judge-role ALT`.

Transport details (connection pools, rate limiting, prompt-cache routing hints) are not part of
the protocol. They are left to `reward/judge_client.py`.

## Commands

```bash
cd analysis/tables
# 1. grade (evaluation judge; then the third-family judge on the same responses)
export RUBRIC_JUDGE_BASE_URL=... RUBRIC_JUDGE_API_KEY=... RUBRIC_JUDGE_MODEL=... RUBRIC_JUDGE_THINKING=disabled
python3 grade_hb.py --questions healthbench_consensus.parquet --source healthbench_consensus \
    --responses atomic_s42=responses/atomic_s42.jsonl --responses prorubric_s42=responses/prorubric_s42.jsonl \
    --grades grades/consensus_pro.jsonl
python3 grade_hb.py --questions healthbench_full.parquet --source healthbench_full \
    --responses atomic_s43=responses/atomic_s43.jsonl --ids-from-grades grades/hbfull_pro_seed42_arms.jsonl \
    --grades grades/hbfull_pro.jsonl
python3 grade_hb.py --judge-role ALT --questions healthbench_consensus.parquet --source healthbench_consensus \
    --responses atomic_s42=responses/atomic_s42.jsonl --grades grades/consensus_alt.jsonl

# 2. tables
python3 ablations.py        --spec arms.json --out out/ablations.json    --tex-dir out/tex
python3 main_results.py     --spec arms.json --out out/main_results.json --tex-dir out/tex
python3 eval_sizes.py       --ablations out/ablations.json --main out/main_results.json --tex out/tex/eval_sizes.tex
python3 paired_consensus.py --spec arms.json --which sft_4b
python3 paired_consensus.py --spec arms.json --which sft_8b
python3 paired_consensus.py --spec arms.json --which controls_8b
python3 crosscheck_rank.py  --spec arms.json --out out/crosscheck.json --tex-dir out/tex

# 3. figures
python3 ../../figures/plot_diagnosis_compound.py --out-dir out/fig --ablations-json out/ablations.json
python3 ../../figures/plot_controls_compound.py  --out-dir out/fig --ablations-json out/ablations.json
python3 ../../figures/plot_length_utility.py     --out-dir out/fig --ablations-json out/ablations.json
```

Dependencies: Python 3.9+, `pyarrow`. `grade_hb.py` also needs the judge client's
dependencies.
