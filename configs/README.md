# Training configurations

All RL training methods launch through the same entrypoint (`python -m verl.trainer.main_ppo`)
with the same reward manager (`RubricJudgeRewardManager` in
`../reward/rubric_judge.py`). They differ in the training data (which rubric the
parquet carries) and in the handful of override lines listed below. Nothing about
the optimizer is specific to ProRubric.

Each `*.yaml` is a Hydra override list: a flat YAML list of `key=value` strings,
passed after the entrypoint as command-line overrides. They compose on top of
verl's default `ppo_trainer.yaml` (the SFT list on `sft_trainer_engine`). Hydra
interpolations like `${oc.env:PRORUBRIC_TRAIN_PARQUET}` resolve at startup.

## Files

| File | Method / Baseline | Training data |
|---|---|---|
| `atomic.yaml` | Rubric-RL (explicit aggregation) | the atomic checklist (`../data/build_domain_split.py`; science: RaR-Science train split) |
| `prorubric.yaml` | **ProRubric** | the dimensions (`../generate/generate_dimensions.py` → `../data/build_prorubric_release.py`) |
| `graded.yaml` | Graded (4-level credit) | ProRubric's parquet, unchanged; `graded` verdict mode, two draws averaged |
| `ruscarl.yaml` | RuscaRL | scaffold-expanded atomic parquet (`../data/build_ruscarl.py`) |
| `opsd.yaml` | OPSD (following RGSD) | atomic parquet with `extra_info.opsd_context_kind` set (`../distill/README.md`) |
| `sft.yaml` | SFT | RubricHub's SFT corpus (`../data/build_sft_rubrichub.py`); entrypoint `verl.trainer.sft_trainer_ray` |

## Variants that change only the training parquet

These use `prorubric.yaml` (or `atomic.yaml` where the base is the checklist) with
`PRORUBRIC_TRAIN_PARQUET` pointing at the variant's parquet, plus the four
engineering knobs of "later variants" listed at the end of this file.

| Method / Baseline | Base config | Parquet built by |
|---|---|---|
| raw-AND | `prorubric.yaml` | `../data/build_raw_and.py` (medicine), `../data/build_raw_and_domain.py` (other domains) |
| ProRubric w/o failure clauses | `prorubric.yaml` | `../data/build_no_failure_clauses.py` |
| K=1 | `prorubric.yaml` | `generate_dimensions.py --variant k1` → `build_prorubric_release.py --mode k1` |
| ProRubric + appr. criterion | `prorubric.yaml` | `../data/build_appr_criterion.py --mode prorubric` |
| Rubric-RL + appr. criterion | `atomic.yaml` | `../data/build_appr_criterion.py --mode atomic` |
| Rubric-RL, weighted criteria | `atomic.yaml` | `../data/build_weighted_atomic.py` |
| ProRubric, dimensions from another generator | `prorubric.yaml` | `generate_dimensions.py --variant anchored` with `RUBRIC_JUDGE_MODEL` set to the other generator (paper: Doubao-lite) |
| atomic-rw (science) | `atomic.yaml` | `../generate/rewrite_atomic.py` → `../data/build_atomic_rw.py` |

## Configurations that change only overrides

Append these to the base list (later entries win in Hydra):

| Variant / Model | Base | Extra overrides |
|---|---|---|
| KL sweep, β ∈ {0.005, 0.01, 0.04} | `atomic.yaml` / `prorubric.yaml` | `actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=<β>` (`kl_loss_type=low_var_kl` is already set) |
| Length-matched | `atomic.yaml` | `data.max_response_length=4096 actor_rollout_ref.rollout.max_model_len=8192 +reward.rubric_judge.overlong_buffer_len=2048` |
| Qwen3-8B (any method) | the 4B list | `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384`, `PRORUBRIC_MODEL_PATH` → Qwen3-8B; SFT 8B: `data.micro_batch_size_per_gpu=1` |
| Training seeds 42 / 43 / 44 | any | `data.seed=<seed>` (SFT: `trainer.seed=<seed>`) |
| Writing, dialogue, science domains | any | `data.val_files=['${oc.env:PRORUBRIC_VAL_HELDOUT_PARQUET}']` (no HealthBench validation set) |
| OPSD outside medicine | `opsd.yaml` | `trainer.total_training_steps=469` (writing) / `352` (dialogue) / `716` (science): five epochs at batch 128, as 485 is in medicine |
| RuscaRL seeds 43 / 44 | `ruscarl.yaml` | `data.seed=<seed>` and a scaffold parquet built with `build_ruscarl.py --seed <seed>` (the schedule is pre-expanded per seed) |
| Training-dynamics replicates (App. "Training dynamics") | `atomic.yaml` / `prorubric.yaml` | `trainer.save_freq=50 trainer.test_freq=50 trainer.max_actor_ckpt_to_keep=6 trainer.rollout_data_dir=<dir>` (keep every 50-step checkpoint, dump rollouts with the per-dimension verdicts) |

In the length-matched variant the overlong buffer shrinks with the limit, so shaping
still starts at half the response limit (2,048 of 4,096 tokens).

## What differs between the listed files

Everything not listed is identical across the RL lists (GRPO, optimizer, vLLM,
judge wiring). `atomic` and `prorubric` differ in no training override at all,
only in the training parquet and the experiment name.

| Override | atomic / prorubric | graded | ruscarl | opsd |
|---|---|---|---|---|
| training judge verdict mode | hard | graded ×2 draws at T=0.7 | hard | — (judge scores validation only) |
| reward manager | `RubricJudgeRewardManager` | `RubricJudgeRepeatRewardManager` | same | same |
| `algorithm.adv_estimator` | `grpo` | `grpo` | `grpo` | `reinforce_plus_plus` |
| `data.train_batch_size` | 64 | 64 | 64 | 512 rows (= 64 groups × 8) | 128 |
| `actor_rollout_ref.rollout.n` | 8 | 8 | 8 | 1 | 1 |
| `actor.ppo_mini_batch_size` | 32 | 32 | 32 | 256 | 32 |
| `data.shuffle` | default | default | default | `False` (scaffold groups stay contiguous) | default |
| `data.max_prompt_length` | 4096 | 4096 | 4096 | 6144 | 4096 |
| `data.max_response_length` | 8192 | 8192 | 8192 | 8192 | 2048 |
| `rollout.max_model_len` | 12288 | 12288 | 12288 | 16384 | 10240 |
| `actor.optim.lr` | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 4.2e-6 |
| `trainer.total_training_steps` | 300 | 300 | 300 | 300 | 485 |
| overlong shaping (buffer 4096, factor 0.5) | yes | yes | yes | yes | no (2048 response cap) |
| `distillation.*` | — | — | — | — | OPSD teacher + clipped JSD loss |

The RuscaRL numbers (`n=1`, batch 512, mini-batch 256, `shuffle=False`, prompt
6144) and the OPSD numbers (485 steps, response 2048, lr 4.2e-6, n=1) follow
those methods' own settings (Table `tab:training` of the paper).

RuscaRL needs one engine change: verl mints a fresh `uid` per training row, which
would break the pre-expanded scaffold groups. `../engine_patches/0001-honor-dataset-uid.patch`
(7 lines, applies to verl v0.8.0) keeps a dataset-provided `uid`. OPSD needs the
engine extensions described in `../distill/README.md`.

## Judges and environment

The training reward and the evaluation use different judges (paper, Sec. 5.1 and
App. B.1): training rewards come from a Doubao-family model (Doubao-mini), which no
policy is evaluated with; in-training validation uses Doubao-lite; all reported
scores use DeepSeek-V4-Pro through `../eval/`. Any OpenAI-compatible endpoints work.
There is no default model: an unset model raises with the name of the variable.

| Env var | Meaning |
|---|---|
| `PRORUBRIC_MODEL_PATH` | base model (Qwen3-4B or Qwen3-8B) |
| `PRORUBRIC_TRAIN_PARQUET` | the target training parquet |
| `PRORUBRIC_VAL_HELDOUT_PARQUET` | held-out validation parquet (the atomic baseline's, for every method) |
| `PRORUBRIC_VAL_BENCH_PARQUET` | benchmark validation parquet (medicine: HealthBench held-out) |
| `PRORUBRIC_OUTPUT_DIR` | output root; checkpoints go to `$PRORUBRIC_OUTPUT_DIR/checkpoints` |
| `PRORUBRIC_CONTEXT_PROFILE` | (OPSD) teacher-context profile; defaults to `../distill/rgsd_rubric_v1.yaml` |
| `RUBRIC_JUDGE_TRAIN_BASE_URL` / `_API_KEY` / `_MODEL` | training-reward judge (paper: Doubao-mini) |
| `RUBRIC_JUDGE_VAL_BASE_URL` / `_API_KEY` / `_MODEL` | in-training validation judge (paper: Doubao-lite) |
| `RUBRIC_JUDGE_BASE_URL` / `_API_KEY` / `_MODEL` | shared fallback for any role setting left unset |
| `RUBRIC_JUDGE_TRAIN_MAX_TOKENS` | training judge output budget (runs: 1024) |
| `RUBRIC_JUDGE_TRAIN_THINKING` / `RUBRIC_JUDGE_VAL_THINKING` | thinking toggles (runs: both `disabled`) |
| `RUBRIC_JUDGE_QPM` / `_TPM` / `_MAX_CONCURRENCY` (and `_TRAIN_` / `_VAL_` variants) | rate limits |
| `RUBRIC_JUDGE_ENV_FILE` | optional `KEY=VALUE` file holding any of the above |

The reward manager loads via `reward.reward_manager.module.path=pkg://rubric_judge`,
so `reward/` must be on `PYTHONPATH`.

## These are the settings that ran

Each file is the override list of the runs behind the corresponding rows, with
paths, endpoints and credentials replaced by environment variables. One difference
is worth knowing if you compare configurations line by line: the first 4B Rubric-RL /
ProRubric pair (seed 42) ran before two engineering knobs were adopted, and every
later run (other seeds, controls, domains, 8B; `graded.yaml` and
`ruscarl.yaml` here) sets them:

| key | first 4B pair | every later run |
|---|---|---|
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | unset (verl default) | 32768 (4B) / 16384 (8B) |
| `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu` | unset (verl default) | 32768 (4B) / 16384 (8B) |
| `trainer.save_freq` | 100 | 50 |
| `trainer.test_freq` | 100 | 50 |

These bound the token budget per micro-batch under `use_dynamic_bsz` and set the
checkpoint and validation cadence; no optimization hyperparameter differs (the
paper's appendix on training configurations notes the cadence difference).

## Which engine

`atomic.yaml`, `prorubric.yaml`, `graded.yaml` and
`ruscarl.yaml` compose against upstream verl v0.8.0
(`7aed6b230776f963fa09509c10d9c3a767d1102c`); RuscaRL additionally needs the uid
patch above to train correctly. `opsd.yaml` sets `distillation.opsd.*` and the
generalized-JSD loss options, which upstream has no schema for; see
`../distill/README.md`. `../smoke.py --engine` composes every list against a verl
checkout and reports which compose.

`++data.prompt_template_mode=chat` uses `++` on purpose: upstream has no such key
(chat is its only behaviour) and the OPSD engine does; `++` sets-or-appends, so one
list works on both.

Example (8 GPUs, one node, as the configs assume):

```bash
export PYTHONPATH=$PWD/reward:$PYTHONPATH
export PRORUBRIC_MODEL_PATH=/path/to/Qwen3-4B
export PRORUBRIC_TRAIN_PARQUET=/path/to/train.parquet
export PRORUBRIC_VAL_HELDOUT_PARQUET=/path/to/heldout.parquet
export PRORUBRIC_VAL_BENCH_PARQUET=/path/to/healthbench_heldout.parquet
export PRORUBRIC_OUTPUT_DIR=/path/to/output
export RUBRIC_JUDGE_TRAIN_BASE_URL=... RUBRIC_JUDGE_TRAIN_API_KEY=... RUBRIC_JUDGE_TRAIN_MODEL=...
export RUBRIC_JUDGE_VAL_BASE_URL=...   RUBRIC_JUDGE_VAL_API_KEY=...   RUBRIC_JUDGE_VAL_MODEL=...

python -m verl.trainer.main_ppo \
  $(python3 -c "import yaml; [print(repr(x)) for x in yaml.safe_load(open('configs/prorubric.yaml'))]")
```
