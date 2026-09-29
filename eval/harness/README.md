# Evaluation harness: response generation and per-question scoring

Every evaluation in the paper starts from one responses file per checkpoint,
generated over the benchmark suites, and one per-question scores file derived
from it. These two scripts produce both.

| Script | What it computes | Paper |
|---|---|---|
| `generate.py` | vLLM responses of one checkpoint over one or more suite parquets | the responses behind every table and figure that scores a trained policy |
| `score_records.py` | per-question scores: multiple-choice accuracy (answer-letter parser) and the official HealthBench score from one judge call per criterion | MedQA and GPQA-Diamond columns of `tab:main_results` and `tab:per_seed_domains`; with `--judge-role LITE` (Doubao-lite), the `c_lite` / `g_lite` columns of `tab:ablations_full`, `tab:per_seed_ablations`, and the Doubao-lite half of `tab:seeds`, the HealthBench-full (Doubao-lite) size in `tab:eval_sizes`, the open markers of `fig:pairwise_winrates`(a), and the Doubao-lite numbers in the "Judges" paragraph of `app:sec5_notes` |

The DeepSeek-V4-Pro columns (`c_pro`, `g_pro`) are graded separately from the
same responses by `../../analysis/tables/grade_hb.py`. WritingBench, Creative-v3,
Arena-Hard v2 and ResearchQA are scored by `../official/`. The scores file
written here is the file that `../../analysis/tables/` calls the "responses"
input (for HealthBench) and the "MedQA / GPQA-Diamond score JSONL".

## Generation settings

Qwen3 in non-thinking mode (chat template rendered with `enable_thinking=False`),
up to 8,192 new tokens, temperature 0.7, top-p 0.8, vLLM per-request seed 42
(43 and 44 for the seed replicates). The equal-budget comparison regenerates with
`--max-tokens 2048`. Rows of all input parquets are concatenated in the order
given and split into contiguous shards, one vLLM engine per GPU
(`max_model_len` 32,768, `gpu_memory_utilization` 0.85). A prompt longer than
`max_model_len - max_tokens - 16` tokens after the chat template is not
generated; its response is `[[GENERATION_SKIPPED: prompt exceeds max_model_len]]`
and scoring records it as a failed question.

```
python3 generate.py --model /path/to/checkpoint \
    --input healthbench_full.parquet --input healthbench_consensus.parquet \
    --input medqa_usmle_4opt.parquet --output medicine/responses.jsonl

python3 generate.py --model /path/to/checkpoint \
    --input gpqa_diamond.parquet --input researchqa_valid.parquet \
    --output science/responses.jsonl

# equal-budget comparison
python3 generate.py --model /path/to/checkpoint --input healthbench_consensus.parquet \
    --max-tokens 2048 --output budget2048/responses.jsonl
```

vLLM and GPUs are needed only to generate; `--help` works without them.

## Scoring

Rubric suites (rows whose `extra_info.rubric` is non-empty) are graded one call
per criterion with the official HealthBench `GRADER_TEMPLATE` from
`../healthbench_judge.py` (the rubric item is rendered as `[weight] criterion`,
the conversation as `role: content` turns with the response appended as the last
assistant turn), temperature 0, thinking disabled, `max_tokens` 4096. The
per-question score is the official HealthBench score (`healthbench_judge.score`:
met weights over positive weights, clipped to [0, 1]). All other rows are
multiple-choice: the evaluation harness's parser extracts the last answer letter
(keyword forms first, then a bare standalone letter, never the pronoun "I") and
compares it with `extra_info.gold_letter`.

A rate-limited request (HTTP 429) is retried with jittered exponential backoff
(at most 600 s per wait) for up to `--budget-429` seconds (default 3600); an
unparseable judge reply is retried up to `--parse-attempts` times (default 3). A
retry regrades the whole question. Other failures are written as error rows.
The output is appended and the run is resumable: ids with a clean row are
skipped, and errored ids are retried on the next run.

```
export RUBRIC_JUDGE_BASE_URL=https://your-endpoint/v1
export RUBRIC_JUDGE_API_KEY=...
export RUBRIC_JUDGE_LITE_MODEL=<second-evaluator model id>   # falls back to RUBRIC_JUDGE_MODEL

python3 score_records.py \
    --questions healthbench_full.parquet --questions healthbench_consensus.parquet \
    --questions medqa_usmle_4opt.parquet \
    --responses medicine/responses.jsonl --scores medicine/scores.jsonl --judge-role LITE

python3 score_records.py --questions gpqa_diamond.parquet \
    --responses science/responses.jsonl --scores science/scores.jsonl
```

`--judge-role ''` uses the evaluation judge's variables (`RUBRIC_JUDGE_*`),
`--judge-role ALT` the third-family evaluator's. Multiple-choice rows need no
judge, but a judge model must still be configured.

## Inputs and outputs

**Suite parquet** (`--input` / `--questions`), one row per question:

| Field | Content |
|---|---|
| `prompt` | string, or `[{"role", "content"}]` messages |
| `data_source` | suite name: `healthbench_full`, `healthbench_consensus`, `medqa`, `gpqa_diamond`, ... |
| `extra_info.id` | `<suite>:<n>`, unique across all suites of one run |
| `extra_info.rubric` | rubric suites: `[{"criterion", "weight", "tags"?}]` |
| `extra_info.gold_letter`, `extra_info.options` | multiple-choice suites: gold letter A–J; options as a letter-keyed mapping or a list (default choices A–D) |
| `extra_info.example_tags` | optional; copied into the score row |

`extra_info` may be a struct or a JSON string.

**Responses** (`generate.py` output): JSONL `{"id", "response"}`, one row per
question.

**Scores** (`score_records.py` output): JSONL, one row per scored question.

| Row kind | Fields |
|---|---|
| rubric | `id`, `data_source`, `eval_kind: "rubric"`, `response`, `satisfied` (one bool per criterion, rubric order), `score` = `weighted_score` (official HealthBench score, 0–1), `n_criteria`, `tag_scores`, `example_tags`, `raw` (judge replies), `usage`, `response_chars`, `error: null` |
| multiple choice | `id`, `data_source`, `eval_kind: "mcq"`, `response`, `predicted_letter`, `gold_letter`, `accuracy` (0/1), `score`, `error: null` |
| failed | `id`, `data_source`, `error` (message), `response` |
