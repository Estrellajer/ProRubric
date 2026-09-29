# HealthBench meta-evaluation: agreement with physicians

Scripts behind the appendix paragraph "Agreement with physicians" and Table
`tab:judge_agreement`: the evaluation judge (DeepSeek-V4-Pro in the paper) grades
HealthBench's public meta-evaluation set (29,511 response-criterion pairs over the 34
consensus criteria, 60,896 physician labels) with the official grader prompt, and its
labels are scored against the physicians' with the official simple-evals metric.

| Script | Judge calls | Computes |
|---|---|---|
| `fetch_simple_evals.sh DEST` | none | Downloads the pinned simple-evals files and the meta-evaluation data file (see below). |
| `grade_metaeval.py` | one per row (29,511) | Grader label `criteria_met` for every row. |
| `metaeval_metrics.py` | none | Balanced F1 of the grader and of physicians: per criterion, per theme, over all criteria (the table), plus the official pooled score. |
| `physician_baseline.py` | none | Physician-vs-other-physicians balanced F1 from the labels alone (a grader-independent check of the physician column). |

## Upstream code and data

All prompt construction, output parsing and metric code is the official
[openai/simple-evals](https://github.com/openai/simple-evals) code at commit
`652c89d0ca9df547706735883097e9537d40dc47` (MIT license, Copyright (c) 2024 OpenAI).
It is not redistributed here; `fetch_simple_evals.sh DEST` downloads
`common.py`, `types.py`, `healthbench_eval.py`, `healthbench_meta_eval.py`,
`sampler/chat_completion_sampler.py` and `LICENSE` into `DEST/simple_evals/` (adding empty
`__init__.py` files so the directory is importable as a package) and checks their MD5.
The scripts import from it via `--simple-evals-dir DEST` (or `DEST` on `PYTHONPATH`):

- `simple_evals.healthbench_eval.GRADER_TEMPLATE`, `parse_json_to_dict` (grading);
- `simple_evals.healthbench_meta_eval.compute_metrics_for_rater_by_class` (metrics).

Importing these modules needs Python >= 3.10 and `blobfile numpy pandas jinja2 requests tqdm openai`.

The data file is `2025-05-07-06-14-12_oss_meta_eval.jsonl` (about 136 MB), the `INPUT_PATH`
of simple-evals' `healthbench_meta_eval.py`:
`https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl`.
`fetch_simple_evals.sh` downloads it to `DEST/`. Each line has `prompt` (messages),
`completion`, `rubric` (the criterion text), `category` (`cluster:<theme>_<criterion>`),
`binary_labels`, `anonymized_physician_ids`, `completion_id`, `prompt_id`.

## Grader configuration

`grade_metaeval.py` rebuilds the official grader prompt exactly as
`HealthBenchMetaEval.__call__` does (conversation plus the completion as the last assistant
turn, rendered as `role: content` blocks, filled into `GRADER_TEMPLATE` with the rubric item)
and parses the reply with the official `parse_json_to_dict`. Only the transport differs:
the request goes through `../../reward/judge_client.py` with the evaluation judge
(`RUBRIC_JUDGE_*`, no role; `--judge-role` selects another), a single user message,
temperature 0, thinking disabled, max_tokens 3000. The official loop retries unparsable output
indefinitely; here each row gets up to 6 attempts and is then written with
`criteria_met: null` (never dropped). In the paper's run every row received a label.

## Commands

```
bash fetch_simple_evals.sh se                         # -> se/simple_evals/, se/2025-05-07-06-14-12_oss_meta_eval.jsonl
export RUBRIC_JUDGE_BASE_URL=... RUBRIC_JUDGE_API_KEY=... RUBRIC_JUDGE_MODEL=...
python3 grade_metaeval.py --simple-evals-dir se --data se/2025-05-07-06-14-12_oss_meta_eval.jsonl \
    --out grades_metaeval.jsonl [--qpm 150] [--workers 32] [--limit 5]
python3 metaeval_metrics.py grades_metaeval.jsonl --simple-evals-dir se \
    --data se/2025-05-07-06-14-12_oss_meta_eval.jsonl --out metaeval_metrics.json
python3 physician_baseline.py --simple-evals-dir se \
    --data se/2025-05-07-06-14-12_oss_meta_eval.jsonl --out physician_baseline.json
```

`grade_metaeval.py` output (append-only, resumable; keyed by the row's 0-based line index):
`{"i", "completion_id", "prompt_id", "category", "criteria_met", "explanation", "judge", "attempts"}`.

## Reading the table from `metaeval_metrics.json`

- **All criteria** row: `paper_agg_model_mean_over_criteria` (grader) and
  `paper_agg_physician_mean_over_criteria` (physicians). This is the HealthBench paper's
  aggregation: balanced F1 against each physician label within a criterion
  (`"<cluster>: pairwise_model_f1_balanced"`), then the unweighted mean over the 34 criteria;
  for physicians, the per-criterion value is the n-weighted mean over physicians. On the
  public file the physician value is 0.6475 (HealthBench paper: 0.647).
- **Theme** rows: `per_theme[<theme>]` (mean over the theme's criteria; `criteria` is the
  count column). Themes are the cluster-name prefixes `emergency_referrals`,
  `global_health`, `communication`, `context_seeking`, `hedging`, `health_data_tasks`,
  `complex_responses`.
- `official_pooled_model_f1_balanced` is simple-evals' own single pooled score
  (`HealthBenchMetaEval` final score; 0.660 for the paper's grader), reported for reference.
- `per_criterion` adds each criterion's grader value, physician value, and the grader's
  percentile among physicians.

With the paper's grader labels, `metaeval_metrics.py` gives 0.621 (grader) and 0.648
(physicians) over all criteria, printed as 0.62 and 0.65 in the table.
