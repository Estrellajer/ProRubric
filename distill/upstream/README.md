# Bringing OPSD to upstream verl

The OPSD baseline is the one method that needs engine extensions beyond upstream verl.
They split into three pieces, smallest and most self-contained first. Piece 1 is
provided as a patch; pieces 2 and 3 are described.

## 1. The loss: `0001-generalized-jsd-topk.patch`

Adds `generalized_jsd_topk` alongside `forward_kl_topk`: the generalized
Jensen-Shannon divergence (Agarwal et al., arXiv:2306.13649) on the teacher's
top-k support, with `jsd_beta`, `jsd_token_clip` and `kd_temperature`.

Against upstream `main` at `cf1649b2`, 9 files, +344/-6:

| file | what |
|---|---|
| `verl/workers/config/distillation.py` | three fields and their validation |
| `verl/trainer/distillation/losses.py` | the 40-line maths core, the registered aggregation, and a kernel lookup by `loss_mode` |
| `verl/trainer/distillation/fsdp/losses.py` | the FSDP top-k kernel |
| `verl/trainer/config/distillation/*.yaml` | the keys, plus the four regenerated configs |
| `tests/workers/test_distillation_generalized_jsd_on_cpu.py` | 12 CPU tests |

The one API change: `compute_topk_loss` hardcoded `compute_forward_kl_topk` for
every mode, so the backend kernel is now looked up through
`TOPK_KERNEL_BY_LOSS_MODE`, falling back to forward KL; every existing mode keeps
the kernel it had.

The 12 new tests pass, upstream's existing distillation CPU tests still pass,
`ruff check` / `ruff format --check` are clean, and
`scripts/generate_trainer_config.sh` passes.

```bash
git clone https://github.com/verl-project/verl && cd verl
git checkout cf1649b2244a96c0ec7abea50cda9df291ee411b
git am < /path/to/0001-generalized-jsd-topk.patch
```

The patch's new test file carries verl's standard license header, as every file in
verl does.

## 2. The union support (described)

`opsd_vocab_support=student_teacher_union` scores the divergence on the union of the
teacher's and the student's top-k ids instead of the teacher's alone, so
student-confident ids are not silently dropped. Unlike piece 1 this needs the engine,
because the teacher has to supply log-probabilities for ids only the student proposed:

- `verl/workers/engine_workers.py`: build a fixed-width, duplicate-free union padded
  to `2*K` with `-inf`, and expose `compute_student_topk_ids` on the actor path
  (~120 lines, gated on the support mode);
- `verl/workers/engine/fsdp/transformer_impl.py`: assert the fused-kernel path emits
  the per-token fields the loss needs (~8 lines);
- `verl/trainer/distillation/fsdp/losses.py`: the union branch of the kernel
  (~45 lines), which the maths core in piece 1 already supports through its
  `support_mask` argument.

## 3. The teacher context (described)

The rubric-conditioned teacher prompt: `opsd.context_profile_path`,
`teacher_thinking`, `require_same_model`, `model_identity`. The upstream teacher
manager has no hook for constructing the teacher prompt, so this piece is a new
extension point rather than a new option:

- `verl/experimental/teacher_loop/opsd_context.py`: 77 lines, new file (`../opsd_context.py`);
- `verl/experimental/teacher_loop/teacher_manager.py`: profile loading and the
  dataset field contract (~120 lines);
- `verl/trainer/config/distillation/distillation.yaml`: the `opsd:` block.

## What reproduces the paper's OPSD numbers

The paper's OPSD runs used `opsd_vocab_support=student_teacher_union` with
`jsd_beta=0.5` and `jsd_token_clip=0.05`. Piece 1 alone lands the objective, not the
support, so all three pieces are needed; re-expressing the baseline on an objective
upstream already has would be a different baseline.
