# Rubric-free blind pairwise comparisons (all domains, seeds, equal budget)

Code for Table 1(a), Appendix Table `tab:rubric_free_full`, the equal-budget
comparison of Section 5.3 / Figure `fig:pairwise_winrates`(b), and the
single-seed pairwise comparisons of Appendix "Rubric-Free Comparisons and
Length Controls".

The judge receives the user prompt and two anonymous responses, no rubric,
and returns `{"winner": "A" | "B" | "tie"}`. Each pair is judged in both
orders; a stable win or loss needs the same winner in both, everything else
counts as tie/unstable and is excluded from the conditional win rate.

Judge prompts (the text the judge sees is exactly the paper's):

| domain | prompt |
|---|---|
| medicine | `../pairwise.py` `PROMPT` (reused) |
| science | `pairwise_core.SCIENCE_PROMPT` |
| dialogue, writing | `pairwise_core.REQUEST_PROMPT` |

Request settings: temperature 0, `max_tokens` 3000, up to three judge calls
per verdict. Evaluation judge (no role; paper: DeepSeek-V4-Pro) with thinking
disabled; `--judge-role ALT` (paper: GPT-5.6-luna) without a thinking field.
Configure with `RUBRIC_JUDGE_*` / `RUBRIC_JUDGE_ALT_*` (`../../reward/judge_client.py`).

## Files

| file | what it computes | paper |
|---|---|---|
| `pairwise_core.py` | library: prompts, loaders, judge call, winner mapping | |
| `pairwise_domain.py sample` | medicine item sample: first 600 ids in sha1(id) order among those answered by every given arm | items |
| `pairwise_domain.py run / summarize` | all pairs among seed-42 arms in one domain; stable W/T/L, x win rate with bootstrap CI (5,000 resamples, seed 67), first-position rate, sign agreement with the training-judge `score`, mean lengths | appendix pairwise comparisons; seed-42 cells of Table 1(a) and of the equal-budget comparison |
| `pairwise_seeds.py run / summarize` | one x-vs-y comparison per row over training seeds, fixed items; per-seed W/T/L, y preference mean and sd (ddof=1) over seeds, x W/T/L summed over seeds and x win rate | Table 1(a), `tab:rubric_free_full` (both judges); equal-budget 83% / 79% |
| `table1a_spec.example.json` | row spec for Table 1(a) / `tab:rubric_free_full` (replace the response paths) | |
| `equal_budget_spec.example.json` | row spec for the 2,048-token comparison | |
| `ids/medicine_600.json` | the 600 HealthBench-consensus ids of the medicine comparisons | |

Items in the paper: medicine uses `ids/medicine_600.json` (sampled over the
untrained, Rubric-RL and ProRubric arms at 4B and 8B); science, dialogue and
writing use the appropriateness-probe sample
(`../appropriateness_probe/sample.json`: sets `researchqa`, `arena`,
`creative,writingbench`). Table 1(a) uses the first 100 ids of each list for
every seed and both judges; the equal-budget comparison uses all 600 / 200.
In science, dialogue and writing an id without a response from every arm is
dropped for all pairs; in medicine it is skipped only for the pairs that need it.

`tab:rubric_free_full` reports the untrained model's side: the `x W / T / L
(sum)` and `x win rate` columns of `pairwise_seeds.py summarize` for rows with
`x = base`. Table 1(a) and the equal-budget figure report the trained arm's
side (`y preference`).

## Inputs and outputs

- questions: parquet rows with `prompt` and `extra_info.id`.
- responses: JSONL per arm (and seed), `{"id", "response", ["score"],
  ["data_source"], ["error"]}`; `score` is the training-time judge score,
  used only for the sign-agreement column. Several files for one arm are
  merged (writing: Creative-v3 and WritingBench).
- verdicts (both scripts, append-only, resumable):
  `{"domain","seed","id","x","y","order","left","winner","first_pos","judge"}`;
  `winner` is an arm key or `"tie"`, `order` 0 shows x as response A.

Arm keys: `base` = untrained model, `A` = Rubric-RL, `B` = ProRubric,
`C` = raw-AND, `G` = Graded (any keys work; they only have to match between
`--responses`/spec and `--pairs`).

## Commands

```
# medicine items (provided as ids/medicine_600.json)
python3 pairwise_domain.py sample --questions healthbench_consensus.parquet \
    --data-source healthbench_consensus \
    --responses base=base_4b.jsonl --responses A=rubric_rl_4b.jsonl --responses B=prorubric_4b.jsonl \
    --responses base8b=base_8b.jsonl --responses A8b=rubric_rl_8b.jsonl --responses B8b=prorubric_8b.jsonl \
    --n 600 --out ids/medicine_600.json

# seed-42 comparisons in one domain (science shown)
python3 pairwise_domain.py run --domain science \
    --questions researchqa_valid.parquet --data-source researchqa_valid \
    --ids ../appropriateness_probe/sample.json --ids-sets researchqa \
    --responses base=sci/base.jsonl --responses A=sci/rubric_rl_s42.jsonl \
    --responses B=sci/prorubric_s42.jsonl --responses C=sci/raw_and_s42.jsonl \
    --pairs base:A,A:B,base:B,A:C,C:B,base:C --verdicts sci_s42.jsonl
python3 pairwise_domain.py summarize [same arguments] --out sci_s42.md
#   third-family judge on the first 100 items per pair:
#   ... run [same] --pairs base:A --judge-role ALT --limit-per-pair 100 --verdicts sci_s42_alt.jsonl

# Table 1(a) / tab:rubric_free_full, both judges
python3 pairwise_seeds.py run --spec table1a_spec.json --verdicts t1a_eval.jsonl
python3 pairwise_seeds.py run --spec table1a_spec.json --verdicts t1a_alt.jsonl --judge-role ALT
python3 pairwise_seeds.py summarize --spec table1a_spec.json --verdicts t1a_eval.jsonl
python3 pairwise_seeds.py summarize --spec table1a_spec.json --verdicts t1a_alt.jsonl

# equal budget (responses regenerated with a 2,048-token ceiling)
python3 pairwise_seeds.py run --spec equal_budget_spec.json --verdicts eqb.jsonl --qpm 150 --workers 24
python3 pairwise_seeds.py summarize --spec equal_budget_spec.json --verdicts eqb.jsonl
```

Seed-42 verdicts already produced by `pairwise_domain.py` can be reused
instead of re-judged: `pairwise_seeds.py summarize --extra-verdicts
sci_s42.jsonl ...` keeps those whose (domain, x, y, seed) matches a spec row
and whose id is one of its items.
