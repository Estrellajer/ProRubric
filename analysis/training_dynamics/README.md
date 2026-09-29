# Training dynamics

Code behind the first paragraph of Sec. 5.4 ("Under additive aggregation, appropriateness collapses early"),
Figure `fig:training_dynamics` and the appendix paragraph "Training dynamics", plus the K=1 identical-reward share
(Sec. 5.3 and the appendix paragraph "Aggregation controls at 8B").

| File | Computes | Paper |
|---|---|---|
| `score_checkpoints.py` | scores every saved checkpoint of the dynamics replicates: appropriateness c on 300 HealthBench-consensus prompts; g_train and ProRubric dimension satisfaction on 300 medical training prompts (training judge); fixing rates grouped by the untrained answer's state; mean ± sd over seeds; ordering and separation checks | Sec. 5.4 (55.8 → 35.9 within 50 steps, 50.3 / 49.3 at step 50, 23.8 / 35.3 / 38.0 at step 300, "every Rubric-RL seed below every seed of the other two"); appendix "Training dynamics" (fixing-rate ordering "under all three rewards and on every seed") |
| `plot_training_dynamics.py` | the figure, from `summary.json` or from its embedded per-seed values (the paper's numbers) | Figure `fig:training_dynamics` |
| `identical_reward_groups.py` | share of sampled groups whose 8 rollouts get identical rewards, from training logs | K=1: "about two thirds", 63–66% (63.9 / 65.5 / 63.4 on seeds 42 / 43 / 44) |
| `reexposure.py` | fixing rate of dimensions between a prompt's first and second draw during training, from rollout dumps | supplementary check of the same ordering; not quoted in the paper |
| `healthbench_consensus_300_ids.json` | the 300 HealthBench-consensus ids scored at every checkpoint | — |

## The runs

The dynamics replicates repeat the medical 4B recipes of Table `tab:consensus_ablations` for Rubric-RL
(`configs/atomic.yaml`) and ProRubric (`configs/prorubric.yaml`) at seeds
42 / 43 / 44, with the same data and seeds. The change is config-only: `trainer.max_actor_ckpt_to_keep` raised so that
every 50th checkpoint (steps 50…300) survives, and `trainer.rollout_data_dir` set so the rollout dumps
`reexposure.py` reads are written. They are new runs, not the table's checkpoints, and all nine are reported.

Each saved checkpoint answers 600 prompts, decoded like every other evaluation (8192 tokens, T 0.7, top_p 0.8,
seed 42, thinking off):

- the 300 medical training prompts of the incentive audit (`analysis/incentive_audit`, `<workdir>/medicine`);
- the 300 HealthBench-consensus prompts in `healthbench_consensus_300_ids.json`, a fixed draw (ordered by a salted
  sha256 of the id) from the common set of Table `tab:consensus_ablations`.

## Judges and request settings

| Quantity | Judge (role) | Call |
|---|---|---|
| g_train (checklist, rubric key `A`) and dimension satisfaction (ProRubric dimensions, key `B`) | training-reward judge (`TRAIN`; paper: Doubao-mini) | `RubricJudge.score`, hard verdicts, one whole-rubric call per prompt, `max_tokens` 1024, thinking disabled: the incentive audit's scoring call |
| appropriateness c | evaluation judge (no role by default; paper: a second deployment of DeepSeek-V4-Pro, the same one for every point and for step 0) | official HealthBench `GRADER_TEMPLATE`, one call per criterion, temperature 0, thinking disabled, `max_tokens` 3000, up to 3 attempts when a call fails or its verdict does not parse; score = `eval/healthbench_judge.py` `score()` |

The consensus grader reuses `eval/healthbench_judge.py`'s `grader_prompt`, `conversation_text`, `parse_verdict`,
`score` and grades layout. Its own `grade_one` is kept here because the paper's runs used `max_tokens` 3000 and
three attempts per criterion, while `eval/healthbench_judge.py`'s `grade_one` uses the client default and one attempt.

Step 0 is the untrained model for every arm and seed. Its audit verdicts are the incentive audit's
`scores/A.jsonl` / `scores/B.jsonl` rows with `variant == "orig"` (same judge, same call); its consensus answers
are graded with `--base` under the key `base` by the same judge as the checkpoints.

## Inputs

- `--point ARM:SEED:STEP=PATH` (repeatable) or `--points-json FILE` with `{"ARM": {"SEED": {"STEP": PATH}}}`.
  `PATH` is a responses JSONL, `{"id", "response"}` per row, holding the checkpoint's answers to both prompt sets.
  Rows are joined by id. Arm keys used below: `rubric-rl` (Rubric-RL), `prorubric` (ProRubric); any key works, and the plot maps keys to labels with `--arm KEY=LABEL`.
- `--audit-dir`: the incentive audit's medicine directory: `prompts.jsonl`
  (`{"id", "problem", "rubrics": {"A": [...], "B": [...]}}`) and `scores/<KEY>.jsonl` for step 0.
- `--questions`: HealthBench-consensus parquet (layout of `eval/healthbench_judge.py`); `--ids`: the id list.
- `identical_reward_groups.py` and the optional stability section read the trainer's `tracking/events.jsonl`
  (`{"step", "data": {metric: value}}` per line). `reexposure.py` reads the rollout dump directory
  (`<step>.jsonl`, records with `rubric_judge_satisfied`, `rubric_judge_group_id`, `judge_failed`, `input`).

## Outputs

`--out-dir` receives append-only, resumable verdict files and the summary:

- `audit_verdicts.jsonl`: `{"point", "rubric", "id", "satisfied": [[bool]], "score", "chars"}`;
- `consensus_grades.jsonl`: `{"arm": point, "id", "k", "met"}` (point key `ARM/sSEED/stNNN`, step 0 = `base`);
- `summary.json` / `summary.md`:
  - `base`: step 0;
  - `seeds[SEED].points["ARM/STEP"]`: `g_train`, `dimensions`, `consensus_c` (x100, with n);
  - `seeds[SEED].dimensions_by_base_state["ARM/STEP"]`: per state (`zero` = no dimension met by the untrained answer,
    `middle`, `one_short` = all but one met, `all`) the dimension satisfaction and the fixing rate (the share of
    dimensions unmet at step 0 that are met at step N), plus `overall` (satisfaction before and after, fixed and lost
    rates);
  - `mean_over_seeds`: mean ± sd (ddof=1), filled only when every seed has the point;
  - `dimensions_by_base_state_mean`;
  - `fixing_rate_ordering_at_check_step`: whether zero < middle < one-short holds per arm and seed;
  - `consensus_c_all_seeds_below`;
  - optionally `reference_check` and `stability`.

## Commands

```bash
# role TRAIN = training-reward judge; no role = evaluation judge (see reward/judge_client.py)
export RUBRIC_JUDGE_TRAIN_MODEL=... RUBRIC_JUDGE_TRAIN_BASE_URL=... RUBRIC_JUDGE_TRAIN_API_KEY=...
export RUBRIC_JUDGE_MODEL=... RUBRIC_JUDGE_BASE_URL=... RUBRIC_JUDGE_API_KEY=...

# points.json: {"rubric-rl": {"42": {"50": "rrl_s42_st050.jsonl", ...}, "43": {...}, "44": {...}},
#               "prorubric": {...}}
python3 score_checkpoints.py audit --audit-dir W/medicine --points-json points.json --out-dir OUT
python3 score_checkpoints.py consensus --questions healthbench_consensus.parquet \
    --ids healthbench_consensus_300_ids.json --base untrained_answers.jsonl --points-json points.json --out-dir OUT
python3 score_checkpoints.py summary --audit-dir W/medicine --questions healthbench_consensus.parquet \
    --ids healthbench_consensus_300_ids.json --out-dir OUT --seeds 42,43,44
python3 plot_training_dynamics.py --summary OUT/summary.json --out-dir figures/

# K=1 identical-reward share
python3 identical_reward_groups.py --run "K=1 s42=k1_s42/events.jsonl" --run "K=1 s43=k1_s43/events.jsonl" \
    --run "K=1 s44=k1_s44/events.jsonl"

# optional: rollout-dump re-exposure table
python3 reexposure.py --run "ProRubric s42=prorubric_s42/rollout_dump" --run "Rubric-RL s42=rubric_rl_s42/rollout_dump" --out reexposure.json
```

Optional summary sections:

- `--reference-grades G --reference prorubric=K42,K43,K44`: compares each seed's step-300 c with the range of other
  runs graded on the same 300 ids, for example the table's step-300 checkpoints graded by
  `eval/healthbench_judge.py run --grades G` under the names `K42..K44`.
- `--events ARM:SEED=PATH --original-events ARM:SEED=PATH`: gradient-norm spikes (above 3x the run's median) and
  entropy drift (mean over steps 251–300 minus steps 151–200) per replicate. A replicate is flagged when its drift
  exceeds that arm's largest drift among the table seeds by more than 0.05.

The ProRubric fixing rates of the table's own step-300 checkpoints come from the same summary: add them as another
arm (e.g. `--point prorubric-table:42:300=...`) and read `dimensions_by_base_state` / `overall`.

## Not here

The step-300 check against Table `tab:consensus_ablations` in the appendix paragraph (24.3 / 35.7 / 37.6 on the
full common set with the table's judge deployment) scores each replicate's step-300 checkpoint exactly like a table
arm. It comes from the Table `tab:consensus_ablations` aggregation, with the nine replicates added as extra arms,
and not from this directory.
