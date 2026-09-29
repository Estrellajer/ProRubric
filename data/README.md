# Training data

Every script here is a deterministic post-process of a parquet, except
`build_weighted_atomic.py` (one labelling call per question) and
`build_sft_rubrichub.py` (needs the base model's tokenizer). They depend on
`pyarrow` and the standard library; run them from any directory.

```
RubricHub medical + HealthBench ──build_medical_split.py──┐
RubricHub writing / dialogue ─────build_domain_split.py───┼──▶ atomic train.parquet (Rubric-RL)
RaR-Science ──────────────────────build_science_split.py──┘
                                                       │
            ../generate/generate_dimensions.py ────────┤
                                                       ▼
                         build_prorubric_release.py ──▶ ProRubric train.parquet
                                                       │
       build_raw_and{,_domain}.py · build_no_failure_clauses.py · build_appr_criterion.py
```

## Sources

| Set | Source | Used for |
|---|---|---|
| RubricHub v1 | `sojuL/RubricHub_v1` on HuggingFace, revision `3837d55971473a872e84879c88f708b8da3ec2ef`, `RuRL/rurbichub_v1_{Medical,Writing,Chat}.parquet` | medicine, writing and dialogue training sets and their held-out sets |
| RubricHub v1 SFT | same repository, `sft_RuFT/rurbichub_v1_best_of_6samples_26k_sft_data.parquet` | the SFT baseline |
| RaR-Science | `ScaleAI/RaR-Science` on HuggingFace; the full train split (18,333 prompts) for training, held-out prompts from its validation split | science |

## Licensing & Attribution Notice

- **RubricHub Derivative Data (Apache License 2.0)**:
  The medical, writing, and dialogue training/held-out sets and SFT corpora are derived from [RubricHub](https://github.com/teqkilla/RubricHub) ([sojuL/RubricHub_v1](https://huggingface.co/datasets/sojuL/RubricHub_v1), Li et al., ACL 2026 / arXiv:2601.08430), released under the Apache License 2.0. In accordance with Section 4 of the Apache License, the original copyright notices are retained, and downstream restructuring (grouping into dimensions and failure clauses) is tracked in each generated manifest. See `../NOTICE` and `../LICENSE`.
- **RaR-Science Data**:
  The science training and held-out sets are derived from [ScaleAI/RaR-Science](https://huggingface.co/datasets/ScaleAI/RaR-Science) (Gunjal et al., ICLR 2026 / arXiv:2507.17746), hosted publicly on Hugging Face by Scale AI for academic research. Used in accordance with Hugging Face Terms of Service and academic fair use.

## Packaging the method

`build_prorubric_release.py --base-parquet <atomic train.parquet> --gen-jsonl <dimensions.jsonl>
--out-dir <dir> --artifact-id <name> --mode {protocol,k1}`

Keeps a question iff its generation is valid-or-repaired, its dimensions cover every
atomic index exactly once, the dimension count matches the mode (2–5 for
`protocol`; exactly 1 for `k1`, the K=1 control), and every dimension weight (the
exact sum of `|atomic weight|` over its assigned indices) is positive. Dropped ids
and their reasons go to `excluded_ids.json`. `--dry-run` prints the statistics and
writes nothing.

The paper's releases keep repaired rows in medicine (62 of 12,519) and exclude them
in writing, dialogue and science, where no semantic audit of the repairs was done
(164, 78 and 7 rows). To reproduce that, pass the repaired ids as exclusions:

```bash
python3 -c "import json,sys; print(json.dumps({r['id']: 'repaired_row_excluded' for r in map(json.loads, open(sys.argv[1])) if r.get('repaired')}))" \
    out/writing_dimensions.jsonl > writing_exclude.json
python3 build_prorubric_release.py ... --exclude-ids writing_exclude.json
```

After packaging, `extra_info.rubric` holds the dimensions, the original checklist
moves to `extra_info.rubric_atomic`, and the generation metadata lands in
`extra_info.rir_generation`, which is the contract the transforms below read.

## Controls and baselines

| Script | Paper row | What it does |
|---|---|---|
| `build_medical_split.py` | medicine atomic sets | RubricHub medical → 12,519 training and 300 held-out prompts (seed 42); HealthBench → 300 prompts at seed 42 (the in-training validation set) and a disjoint 300 at seed 43. |
| `build_domain_split.py` | writing and dialogue atomic sets | RubricHub domain parquet → release: dedupe by prompt hash, keep checklists of 5–60 items with positive total weight, then a seeded disjoint train/held-out split (writing 12,000, dialogue 9,000 training prompts; 300 held-out). |
| `build_science_split.py` | science atomic sets | RaR-Science → the full train split (18,333) and 300 held-out prompts from the validation split (seed 42); criteria `"<title>: <Category> Criteria: ..."` with RaR's categorical weights (Essential 1.0, Important 0.7, Optional 0.3, Pitfall −0.9). |
| `build_raw_and.py` | raw-AND (medicine) | Each dimension becomes the verbatim conjunction of its atomic criteria under an "ALL of the following must be satisfied" header; same grouping, same weights, no rewrite, no failure clause. |
| `build_raw_and_domain.py` | raw-AND (writing, dialogue, science) | Same, tolerant of rows without an atomic mapping (counted); negative-weight atoms become `[must NOT hold]` sub-conditions. |
| `build_no_failure_clauses.py` | ProRubric w/o failure clauses | Removes sentences matching `... fails if ...` from each dimension by regex; grouping and weights unchanged. A description shortened below 40 characters keeps its original text. Unmatched failure expressions remain. |
| `build_appr_criterion.py --mode prorubric` | ProRubric + appr. criterion | Appends one fixed appropriateness criterion; weight = mean dimension weight. Everything else byte-identical. |
| `build_appr_criterion.py --mode atomic` | Rubric-RL + appr. criterion | Appends the same criterion to the checklist; weight = `sum(|atomic w|)/K`, the same `1/(K+1)` share. Needs `--prorubric-input` for each question's K. |
| `build_weighted_atomic.py` | Rubric-RL, weighted criteria | Labels every criterion CRITICAL / NORMAL / FORMAT / DUPLICATE / OVERCLINICAL with one model call per question and multiplies weights by 3 / 1 / 0 / 0 / −1. |
| `build_atomic_rw.py` | atomic-rw (science) | Replaces each science criterion by its one-to-one rewrite from `../generate/rewrite_atomic.py`; count, order, weights and prefixes are asserted unchanged. |
| `build_ruscarl.py` | RuscaRL | Pre-expands the RuscaRL rollout groups (arXiv 2508.16949): T steps × 64 groups × G=8 rows sharing a `uid`; row i gets a scaffold of `round(λ·N)` atomic criteria appended to the last user message, `λ(t) = 1/(1+exp(α(t/T−t0)))`, t0=0.2, α=125, intra-group strength `(G−i)/(G−1)`. Step-major order, so `shuffle=False` with batch 512 reproduces the schedule. The reward rubric is untouched. One parquet per seed (`--seed`). |
| `build_sft_rubrichub.py` | SFT | RubricHub's SFT corpus → `messages` parquet; drops (never truncates) rows over 20,000 tokens. |
| `build_grouped_mean.py` | (extra, not in the paper) | Same atoms and grouping as raw-AND, each atom scored on its own with the group weight split by `|w|`. |

The OPSD training set is the atomic parquet with `extra_info.opsd_context_kind`
set; the medicine and science builders already set it, and the few lines that set it
for writing and dialogue are in `../distill/README.md`.

## Usage

```bash
python3 build_medical_split.py --rubrichub rurbichub_v1_Medical.parquet \
    --healthbench 2025-05-07-06-14-12_oss_eval.jsonl --output release/medical-atomic
python3 build_domain_split.py --input rurbichub_v1_Writing.parquet --domain Writing \
    --data-source rubrichub_writing --train-n 12000 --output release/writing-atomic
python3 build_domain_split.py --input rurbichub_v1_Chat.parquet --domain Chat \
    --data-source rubrichub_chat --train-n 9000 --output release/dialogue-atomic
python3 build_science_split.py --output release/science-atomic      # downloads ScaleAI/RaR-Science
python3 build_prorubric_release.py --base-parquet release/medical-atomic/train.parquet \
    --gen-jsonl out/medical_dimensions.jsonl --out-dir release/medical-prorubric \
    --artifact-id medical-prorubric --mode protocol
python3 build_raw_and.py            --input release/medical-prorubric/train.parquet --output release/medical-raw-and
python3 build_no_failure_clauses.py --input release/medical-prorubric/train.parquet --output release/medical-no-fc
python3 build_appr_criterion.py --mode prorubric --input release/medical-prorubric/train.parquet \
    --output release/medical-prorubric-appr
python3 build_appr_criterion.py --mode atomic --input release/medical-atomic/train.parquet \
    --prorubric-input release/medical-prorubric/train.parquet --output release/medical-atomic-appr
python3 build_weighted_atomic.py --input release/medical-atomic/train.parquet --work work/weighted \
    --output release/medical-weighted            # labeller: RUBRIC_JUDGE_LITE_* (falls back to RUBRIC_JUDGE_*)
python3 build_ruscarl.py --input release/medical-atomic/train.parquet --output release/medical-ruscarl-s42 \
    --artifact-id medical-ruscarl-s42 --derived-from medical-atomic --seed 42
python3 build_raw_and_domain.py --input release/science-prorubric/train.parquet \
    --output release/science-raw-and --artifact-id science-raw-and --derived-from science-prorubric
python3 build_atomic_rw.py --input release/science-atomic/train.parquet --rewrites out/science_rw.jsonl \
    --output release/science-atomic-rw
python3 build_sft_rubrichub.py --src rurbichub_v1_best_of_6samples_26k_sft_data.parquet \
    --model /path/to/Qwen3-4B --output release/rubrichub-sft
```

Every script writes a release directory with `train.parquet`, `manifest.json`
(transform, counts, consistency checks, sha256) and, for most, a `README.md`.
Validation sets are not produced by the transforms: every method validates on the
atomic baseline's held-out sets, so held-out numbers stay comparable across all methods.

## Field names

Parquet field names predate the paper's terminology and are kept because every
script reads them:

| Field | Meaning |
|---|---|
| `extra_info.rubric` | the rubric the reward judge scores (atomic criteria, or dimensions) |
| `extra_info.rubric_atomic` | the original atomic checklist of a ProRubric row |
| `extra_info.rir_generation` | JSON string: generator metadata, including `atomic_indices` (one group of 1-based indices into `rubric_atomic` per dimension) |
| `extra_info.rubric_rir_v3` / `rubric_rir_v1` | the ProRubric text a control replaced (kept for audit; medicine / other domains) |

## Input parquet contract of the transforms

Each row carries `extra_info` with `rir_generation` (JSON string or dict with
`atomic_indices`), `rubric_atomic` (list of `{"criterion", "weight"}`, 20–30 per
question) and `rubric` (the 2–5 dimensions). The transforms keep the original
dimensions under `extra_info.rubric_rir_v3` (or `rubric_rir_v1` for the domain
script), replace `extra_info.rubric`, and annotate `rir_generation` with the rule
they applied.
