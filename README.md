<div align="center">

# ProRubric: Protocol-Level Rubrics for Reinforcement Learning

<p align="center">
  <a href="https://arxiv.org/abs/2609.38847"><img src="https://img.shields.io/badge/arXiv-2609.38847-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/release/python-3112/"><img src="https://img.shields.io/badge/Python-3.11.2-brightgreen.svg" alt="Python"></a>
  <a href="https://github.com/verl-project/verl"><img src="https://img.shields.io/badge/RL_Engine-verl_v0.8.0-orange.svg" alt="verl"></a>
</p>

Official codebase for the paper:  
**Scoring Higher, Answering Worse: Mitigating Reward Hacking in Rubric-Based RL via Protocol-Level Rubrics**
</div>

---

## Overview

Rubric-based RL evaluates policy responses against an atomic checklist of 20–30 criteria and awards a weighted sum (explicit aggregation). Under standard additive aggregation, criteria compensate for one another: a policy that misses the essential clinical or logical decision can easily "buy" the lost points back with irrelevant fluff and excessive verbosity that nobody asked for.

**ProRubric** (Protocol-level Rubrics) preserves what the criteria ask for while reshaping how they are aggregated:
- **Offline Grouping**: An atomic checklist is grouped once, offline, into 2–5 protocol-level **dimensions**.
- **Conjunctive Logic & Failure Clauses**: A dimension scores only when *all* of its constituent criteria hold **and** its explicit **failure clause** ("fails if …") does not fire.
- **Summed Weight**: Each dimension carries the summed weight of its criteria.
- **Zero Overhead**: Purely a data-side change. Same RL trainer, same reward manager, same hyperparameters—what differs is solely the rubric structure carried in the training dataset.

---

## Methods and Baselines

| Method / Baseline | Training Rubric / Reward | Config |
|---|---|---|
| Rubric-RL | Atomic checklist with explicit weighted-sum aggregation | `configs/atomic.yaml` |
| **ProRubric** | Conjunctive dimensions with explicit failure clauses | `configs/prorubric.yaml` |
| RuscaRL | Atomic reward + rubric scaffolding in rollout prompts | `configs/ruscarl.yaml` |
| OPSD (following RGSD) | On-policy self-distillation with rubric as teacher context | `configs/opsd.yaml` |
| SFT | Supervised fine-tuning on RubricHub released responses | `configs/sft.yaml` |
| Graded | ProRubric dimensions evaluated with 0–3 graded credit | `configs/graded.yaml` |
| Controls & Ablations | raw-AND, w/o failure clauses, K=1, + appr. criterion, Weighted, Length-matched, KL sweep, alternative generator, atomic-rw, 8B | `configs/README.md` |

> See [`COVERAGE.md`](COVERAGE.md) for a complete 1-to-1 mapping from every paper table and figure to its reproduction scripts.

---

## Quick Start

### 1. Environment Setup

Experiments were executed with Python 3.11.2 and [verl](https://github.com/verl-project/verl) v0.8.0.

```bash
# Clone the repository
git clone https://github.com/Estrellajer/ProRubric.git
cd ProRubric

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation (Smoke Test)

Run the end-to-end smoke test suite (asserts data pipelines, generation transforms, verdict modes, and CLI interfaces using a local mock judge; requires no GPU or network):

```bash
python3 smoke.py
```

To also verify verl engine configuration composition:
```bash
VERL_CONFIG_DIR=/path/to/verl/verl/trainer/config python3 smoke.py --engine
```

---

## End-to-End Workflow (Medicine Example)

> **Tip**: To directly reproduce RL training without rebuilding datasets from raw sources, download the pre-packaged splits from [Released Rubric Datasets](#released-rubric-datasets) to your local `release/` directory and skip directly to **Step 3**.

### Step 1. Prepare Atomic Splits
```bash
python3 data/build_medical_split.py \
    --rubrichub rurbichub_v1_Medical.parquet \
    --healthbench 2025-05-07-06-14-12_oss_eval.jsonl \
    --output release/medical-atomic
```

### Step 2. Generate Protocol Dimensions
```bash
export RUBRIC_JUDGE_BASE_URL=https://...
export RUBRIC_JUDGE_API_KEY=your_key
export RUBRIC_JUDGE_MODEL=deepseek-v4-pro

# Group checklist into dimensions
python3 generate/generate_dimensions.py --variant anchored \
    --input release/medical-atomic/train.parquet \
    --output out/medical_dimensions.jsonl \
    --workers 24 --rpm 180

# Package the ProRubric training parquet
python3 data/build_prorubric_release.py \
    --base-parquet release/medical-atomic/train.parquet \
    --gen-jsonl out/medical_dimensions.jsonl \
    --out-dir release/medical-prorubric \
    --artifact-id medical-prorubric --mode protocol
```

### Step 3. Launch PPO Training
All RL training methods run through `verl.trainer.main_ppo` with Hydra configuration overrides:
```bash
export PYTHONPATH=$PWD/reward:$PYTHONPATH
export PRORUBRIC_MODEL_PATH=/path/to/Qwen3-4B
export PRORUBRIC_TRAIN_PARQUET=release/medical-prorubric/train.parquet
export PRORUBRIC_VAL_HELDOUT_PARQUET=release/medical-atomic/heldout.parquet
export PRORUBRIC_VAL_BENCH_PARQUET=release/medical-atomic/healthbench_heldout.parquet
export PRORUBRIC_OUTPUT_DIR=runs/medical_prorubric_4b

python3 -m verl.trainer.main_ppo \
  $(python3 -c "import yaml; [print(repr(x)) for x in yaml.safe_load(open('configs/prorubric.yaml'))]")
```

### Step 4. Evaluation & Grading
Grade coverage ($g$) and appropriateness ($c$) using HealthBench's official grader:
```bash
# Run grading
python3 eval/healthbench_judge.py run \
    --questions healthbench_consensus.parquet \
    --responses rubric_rl=atomic.jsonl \
    --responses prorubric=prorubric.jsonl \
    --grades grades.jsonl

# Summarize results
python3 eval/healthbench_judge.py summarize \
    --questions healthbench_consensus.parquet \
    --responses rubric_rl=atomic.jsonl \
    --responses prorubric=prorubric.jsonl \
    --grades grades.jsonl
```

---

## Judges Configuration

Judge models are configured via environment variables and accessed through standard OpenAI-compatible endpoints (`reward/judge_client.py`). **No judge model is hard-coded**.

| Role | Model in Paper | Environment Variables |
|---|---|---|
| Training Reward | Doubao-mini | `RUBRIC_JUDGE_TRAIN_{BASE_URL,API_KEY,MODEL}` |
| In-training Validation | Doubao-lite | `RUBRIC_JUDGE_VAL_{BASE_URL,API_KEY,MODEL}` |
| Main Evaluation & Dimension Generator | DeepSeek-V4-Pro | `RUBRIC_JUDGE_{BASE_URL,API_KEY,MODEL}` |
| Second Evaluator (Same Family) | Doubao-lite | `RUBRIC_JUDGE_LITE_*` |
| Third-Family Evaluator | GPT-5.6-luna | `RUBRIC_JUDGE_ALT_*` |
| Incentive Audit Perturbation Writer | Doubao-lite | `RUBRIC_JUDGE_EDIT_*` |

Variables fall back to the shared `RUBRIC_JUDGE_*` prefix if role-specific variables are omitted. You may also specify a `KEY=VALUE` file via `RUBRIC_JUDGE_ENV_FILE`.

---

## Repository Structure

```
├── configs/            # Hydra override configs per method (atomic, prorubric, graded, ruscarl, opsd, sft)
├── data/               # Dataset split builders, transforms, and public release export tools
├── generate/           # Atomic checklist -> protocol-level dimensions generation scripts
├── reward/             # Verl reward-loop plugin (rubric_judge.py) and judge client (judge_client.py)
├── distill/            # OPSD teacher-context engine extensions and generalized-JSD loss patch
├── engine_patches/     # Verl patch for RuscaRL rollout group UID preservation
├── eval/               # Official benchmark harnesses, pairwise arena, and HealthBench grader
├── analysis/           # Incentive audits, training dynamics probes, meta-evaluation, and table generation
├── figures/            # Publication figure plotting scripts
├── requirements.txt    # Pinned dependency specifications
└── smoke.py            # Local zero-dependency integration and smoke test suite
```

---

## Released Rubric Datasets

The pre-built ProRubric datasets (including protocol dimensions with failure clauses, atomic checklists, and held-out validation sets) are hosted for direct download:

- **Google Drive**: [ProRubric Public Release Data](https://drive.google.com/drive/folders/1X2IzyCaH9gkXANP6kSgLByxdW1gbdkJg?usp=drive_link)
- **Hugging Face**: *(Mirror repository coming soon)*

The release contains 16 core distribution files (8 `.parquet` splits and their corresponding `.sha256` checksums), alongside `MANIFEST.json` and dropped query logs under `excluded_ids/`.

### File Manifest & Statistics

| File | Rows | Description |
|---|---|---|
| `medical_prorubric_train.parquet` | 12,500 | RubricHub medical prompts with atomic checklists and protocol dimensions |
| `writing_prorubric_train.parquet` | 11,836 | RubricHub writing prompts with protocol dimensions |
| `science_prorubric_train.parquet` | 18,326 | RaR-Science training split with protocol dimensions |
| `dialogue_prorubric_train.parquet` | 8,922 | RubricHub dialogue prompts with protocol dimensions |
| `medical_atomic_heldout.parquet` | 300 | Held-out medical prompts with atomic checklists (seed 42) |
| `medical_atomic_healthbench_heldout.parquet` | 300 | HealthBench held-out validation set (seed 42) |
| `medical_atomic_healthbench_disjoint.parquet` | 300 | HealthBench disjoint validation set (seed 43) |
| `writing_atomic_heldout.parquet` | 300 | Held-out writing prompts with atomic checklists |
| `dialogue_atomic_heldout.parquet` | 300 | Held-out dialogue prompts with atomic checklists |
| `science_atomic_heldout.parquet` | 2,292 | Held-out science validation split |
| `science_atomic_heldout300.parquet` | 300 | Held-out science evaluation subset (300 prompts) |
| `excluded_ids/*.json` | — | Drop records and exclusion reasons for questions failing quality rules |
| `MANIFEST.json` | — | Manifest containing exact file byte sizes and SHA256 integrity digests |

### Parquet Schema (`*_prorubric_train.parquet`)

Each record in the training parquet files contains:
- `id` (`string`): Unique prompt identifier.
- `domain` (`string`): Domain label (`medical`, `writing`, `dialogue`, `science`).
- `prompt` (`string`): User instruction / query text.
- `atomic_rubric` (`list<{criterion: string, weight: float}>`): Original atomic checklist (20–30 criteria).
- `dimensions` (`list<{name: string, criterion: string, weight: float, atomic_indices: list<int>}>`): Protocol-level dimensions. The `criterion` field specifies conjunctive requirements terminating in an explicit `"fails if ..."` clause; `atomic_indices` are 1-based index pointers into `atomic_rubric`.
- `repaired` (`bool`): Flag indicating if automatic repair was applied to fix generator grouping discrepancies.
- `generator` (`string`): Model that produced the dimensions (`deepseek-v4-pro`).

### Structural Invariants & Filtering Rules
- **Disjoint Partition**: Every atomic criterion belongs to exactly one dimension.
- **Weight Conservation**: Each dimension's weight is the sum of absolute weights of its assigned criteria ($\sum |w_i|$).
- **Dimension Count**: 2–5 dimensions per prompt (mean: 4.6 in Medicine, 4.6 in Writing, 4.4 in Dialogue, 3.9 in Science).
- **Kept / Excluded Counts**: Medicine 12,500 / 19 (62 kept rows repaired); Writing 11,836 / 164; Dialogue 8,922 / 78; Science 18,326 / 7. Repaired rows were excluded in writing, dialogue, and science where manual repair audit was not performed.
- **Lineage & Licensing**: Derived from RubricHub v1 (`sojuL/RubricHub_v1`, Apache-2.0) and RaR-Science (`ScaleAI/RaR-Science`, research use, copyright Scale AI). See [`NOTICE`](NOTICE) and `public/README.md`.

---

## Dataset Resources

| Domain | Source Dataset | Purpose |
|---|---|---|
| Medicine | `sojuL/RubricHub_v1` (`RuRL/rurbichub_v1_Medical.parquet`) | Training & held-out evaluation |
| Writing | `sojuL/RubricHub_v1` (`RuRL/rurbichub_v1_Writing.parquet`) | Domain transfer training & held-out |
| Dialogue | `sojuL/RubricHub_v1` (`RuRL/rurbichub_v1_Chat.parquet`) | Domain transfer training & held-out |
| Science | `ScaleAI/RaR-Science` (Train split & validation held-out) | Domain transfer training & held-out |
| HealthBench | [openai/simple-evals](https://github.com/openai/simple-evals) | HealthBench-full (5k), consensus set, physician meta-eval |
| Benchmarks | MedQA, GPQA-Diamond, WritingBench, Creative-v3, Arena-Hard v2, ResearchQA | Multi-choice accuracy and official benchmark scoring |

---

## Nomenclature Reference

For consistency with ongoing code and data schemas, internal identifiers map to the paper as follows:

| Code Identifier | Paper Terminology |
|---|---|
| `protocol`, `B`, `rir` (e.g., `extra_info.rir_generation`, `rubric_rir_v3`) | ProRubric |
| `atomic`, `A` | Rubric-RL |
| `and`, `C` | raw-AND |
| `NOVETO` | ProRubric w/o failure clauses |
| `V` / `AV` | ProRubric + appr. criterion / Rubric-RL + appr. criterion |
| `G`, `graded` | Graded |
| `K1`, `k1` | K=1 |
| `rgsd` (profile filename, `rubric_repro_v1` context kind) | OPSD |
| `criterion` (inside generation prompts) | dimension |

---

## Citation

If you find this work, code, or datasets helpful, please cite our paper:

```bibtex
@article{liu2026scoring,
  title={Scoring Higher, Answering Worse: Mitigating Reward Hacking in Rubric-Based RL via Protocol-Level Rubrics},
  author={Liu, Maoqi and He, Junwei and Zhang, Bowen and Li, Feiran and Ma, Wentao and Lin, Rongyi and Zhong, Shuhan and Fang, Quan},
  journal={arXiv preprint arXiv:2609.38847},
  year={2026}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).  
See [NOTICE](NOTICE) for third-party compliance records and attribution notices.

