# OPSD baseline: on-policy self-distillation with the rubric as the teacher's context

The paper's OPSD baseline follows RGSD: a frozen copy of the policy, shown the
question together with its rubric, acts as the teacher, and the student is trained
on-policy toward it with a clipped generalized Jensen-Shannon divergence. It is not a
separate training entrypoint: it runs through `verl.trainer.main_ppo`, the same
entrypoint as the other methods, on verl's own on-policy distillation teacher loop.
What it adds on top of upstream verl is the privileged teacher context (the rubric
shown to the teacher) and the generalized-JSD loss settings.

## Files

| File | Role |
|---|---|
| `opsd_context.py` | Pure OPSD teacher-prompt renderer (77 lines, standard library only). Reference copy of the rendering contract; the training path renders through the equivalent functions of the engine's teacher manager (see "Engine extensions"). |
| `rgsd_rubric_v1.yaml` | The teacher-context profile the OPSD runs used, verbatim. Declares the data fields the teacher prompt is built from and the `context_kind` the dataset must carry. |
| `distillation_overlay.yaml` | Hydra fragment with only the OPSD-relevant `distillation.*` fields (loss mode, JSD beta/clip, colocated teacher, privileged-context block); apply on top of verl's `verl/trainer/config/distillation/distillation.yaml`. `../configs/opsd.yaml` is the complete override list. |
| `upstream/` | A self-contained patch that adds the generalized-JSD top-k loss to upstream verl, and the scope of the remaining two pieces. |

## Upstream base

- `https://github.com/verl-project/verl.git`, tag **v0.8.0**, commit
  `7aed6b230776f963fa09509c10d9c3a767d1102c`.
- The teacher loop is upstream code: On-Policy Distillation (PR #5041), with
  `main_ppo_sync` support (#5997) and multi-teacher support (#6051); teacher
  colocate mode (#5723, #5745). OPSD runs colocated: `enable_resource_pool=false`,
  `n_gpus_per_node` matching the trainer topology.

## Upstream status

The teacher loop has been upstream since v0.8.0; the OPSD layer is not upstream at
v0.8.0, v0.9.0 (`483b8a00`) or `main` at `cf1649b2`: `DistillationLossConfig` offers
only `loss_mode` k1/k2/k3, with no generalized JSD, no student/teacher vocabulary
union, no per-token clip and no teacher-context profile. Composing `../configs/opsd.yaml`
against any of these fails at `distillation.enable_resource_pool`. The other methods'
lists compose on all three.

## Engine extensions this baseline needs

1. **Privileged teacher context.** A `distillation.opsd` config block
   (`enable`, `teacher_thinking`, `context_template`, `context_profile_path`,
   `require_same_model`, `model_identity`); a context-profile loader and validator and
   the teacher-prompt rendering in `verl/experimental/teacher_loop/teacher_manager.py`
   (profile load/validate, required-field resolution, chat-template application),
   consumed from the agent loop's OPSD branch and from `ray_trainer.py`; and a
   7-line wrapper in `ray_trainer.py` that presents the dataset's `extra_info`
   column as row-shaped records so profile field paths such as
   `extra_info.rubric_correct` resolve. `opsd_context.py` here is the
   dependency-free renderer of that prompt (`render_opsd_teacher_user_message`,
   `render_opsd_teacher_prompt_ids`, with `canonical_h0` / `legacy_user_template`
   modes; training uses the legacy user-message template).

2. **The context profile** `rgsd_rubric_v1.yaml`: the frozen teacher sees the problem
   plus the atomic rubric text, rendered as "Hidden evaluation criteria that a good
   response should satisfy", and is asked to answer without referencing them.
   `teacher_thinking: false`, `require_same_model: true`: the teacher must be the
   same checkpoint as the student (self-distillation), and the trainer fails closed
   on an identity mismatch.

3. **Loss settings** (`distillation_overlay.yaml`): `loss_mode=opsd`, generalized
   JSD with `opsd_jsd_beta=0.5`, per-token clip `opsd_jsd_token_clip=0.05`,
   `opsd_vocab_support=student_teacher_union`, `topk=128`, `kd_temperature=1.0`,
   `use_task_rewards=false` (no judge reward during training; the judge scores
   validation only).

## Data-side contract

The OPSD training parquet is Rubric-RL's atomic parquet with one extra field per row:

- **`extra_info.opsd_context_kind` must equal `"rubric_repro_v1"`**, the string
  declared by `context_kind` in `rgsd_rubric_v1.yaml`. The trainer compares the
  dataset value against the profile and raises
  `OPSD context kind mismatch: dataset=<x>, profile='rubric_repro_v1'` on any
  mismatch.
- The profile also requires two non-empty string fields per row, resolved by dotted
  path with no fallback: `extra_info.problem` (`problem_source`, the question) and
  `extra_info.rubric_correct` (`solution_source`, the rubric text shown to the
  teacher). Missing or empty fields fail hard.

The medicine and science builders (`../data/build_medical_split.py`,
`../data/build_science_split.py`) already write this value. For writing and dialogue
(`../data/build_domain_split.py` writes `"none"`), set it on every row; no other
change and no model call (`rubric_correct` is already present):

```python
import pyarrow as pa, pyarrow.parquet as pq
rows = pq.read_table("atomic/train.parquet").to_pylist()
for r in rows:
    r["extra_info"]["opsd_context_kind"] = "rubric_repro_v1"
pq.write_table(pa.Table.from_pylist(rows), "opsd/train.parquet")
```

## Running

Use `../configs/opsd.yaml` (it contains the overlay above). The teacher is an in-job
vLLM replica of the student checkpoint (`teacher_models.teacher_model.model_path`
identical to `actor_rollout_ref.model.path`, plus a matching `model_identity`).

## Notes

- Profile fields `render_mode` and `formal_run_eligible` in `rgsd_rubric_v1.yaml` are
  not read by the profile loader (which validates `schema_version`, `profile`,
  `context_kind`, `template`, `problem_source`, `solution_source`,
  `teacher_thinking`, `require_same_model`); they are metadata.
- The evaluated OPSD checkpoints are step 485 in medicine, 400 in writing, 350–352 in
  dialogue and 700 in science (paper appendix on training configurations).
