# Incentive audit (perturbation analysis)

What each training reward pays for, measured before any training: take the untrained model's answer to a
training prompt, apply one controlled edit, score the original and the edit with each arm's own training reward,
and read the mean reward change. A second score of the unedited original gives the judge-noise floor.

| Script | Produces |
|---|---|
| `incentive_audit.py` | Table 1(b) and Table `tab:incentive_audit_full` (Sec. 3.3, appendix "Sensitivity Analysis"); the reward side of Sec. 5.5 (filler vs substance in four domains, E1/E6 ratios, science's unchanged rewards); the reward side of the appendix "Where Added Material Sits" (E6L/E7L matched pairs and the pairing selection report) |
| `redundancy_probe.py` | The redundancy sub-score charges: Sec. 5.4 ("17 to 38 points of the redundancy sub-score", "25 to 38 for added substance"), Sec. 5.5 ("25 to 38 points against 17 to 29"), all prompts, and the probe side of "Where Added Material Sits" (the unequal-budget E7 pass and the common-budget E6L/E7L pass) |
| `appr_criterion_audit.py` | Appendix "A Fixed Appropriateness Criterion": how often the fixed criterion is judged unmet on the untrained answers (54.0%) and on edits (E1 81.7%, E4 57.0%, E3 80.0%, E5 56.9%), and the adjudicated unwarranted share (67.3% detail-only, 52.0% untrained answers) |

## Edits

| Key | Paper row | Edit |
|---|---|---|
| E1 | Name an item | append a closing paragraph naming 4–6 weakly covered checklist topics without adding information |
| E1b / E1b_clean | Name the topic | same with generic topic labels only; `E1b_clean` keeps only paddings the leak check found free of entities and facts (the paper row) |
| E2 | Satisfy a criterion in part | append one sentence meeting only the first requirement of an unmet compound criterion |
| E3 | Add a needless test | insert one mild, plausible inappropriate recommendation |
| E4 | Drop the key advice | delete the single most important correct sentence |
| E5 | Paraphrase | same content and length |
| E6 | added substance (Sec. 5.5) | append 2–3 checklist items actually satisfied with correct, specific content |
| E7 | integrated, unequal budget | rewrite that distributes new material through the answer, sized to E6's addition |
| E6L / E7L | appended vs integrated, common budget | both sized to 50% of the answer; paired afterwards where the two additions are within 25% of each other |

Medicine gets all ten edits; writing, dialogue and science get E1, E1b, E6, E7, E6L, E7L (E2–E5 are clinical).
The edit prompts are in `incentive_audit.py` verbatim.

## Arms

Arm keys are free-form; the report prints the paper name for these keys:

| Key | Paper name | Training parquet | Verdict mode | Draws × T |
|---|---|---|---|---|
| A | Rubric-RL | atomic checklist | hard | 1 × default |
| B | ProRubric | `data/build_prorubric_release.py` | hard | 1 × default |
| C | raw-AND | `data/build_raw_and.py` | hard | 1 × default |
| G | Graded | ProRubric dimensions | graded | 2 × 0.7 |
| V | ProRubric + appr. criterion | `data/build_appr_criterion.py --mode prorubric` | hard | 1 × default |
| AV | Rubric-RL + appr. criterion | `data/build_appr_criterion.py --mode atomic` | hard | 1 × default |
| NOVETO | ProRubric w/o failure clauses | `data/build_no_failure_clauses.py` | hard | 1 × default |
| K1 | K=1 | one dimension per question | hard | 1 × default |

"Default" temperature is the judge client's: greedy, sampling only when retrying a malformed verdict.
In writing, dialogue and science the paper audits A (Rubric-RL) and B (ProRubric). Use one work directory for all
four domains; each domain keeps its own arm spec.

Each training parquet row must carry `extra_info` with `id`, `problem` (prompt text) and `rubric`
(list of `{"criterion", "weight"}`). `sample` takes the prompt ids common to all arms of the domain, shuffles them
with `random.Random(seed)` and keeps the first `--n` (paper: 300, seed 0).

## Judges

All calls go through `reward/judge_client.py` (`RUBRIC_JUDGE_<ROLE>_*` variables, falling back to `RUBRIC_JUDGE_*`).

| Role | Used for | Paper model | Settings |
|---|---|---|---|
| `TRAIN` | scoring every variant with each arm's reward | Doubao-mini | max_tokens 1024, thinking disabled, arm's verdict mode |
| `EDIT` | writing edits; medical/non-medical labels; E1b leak check | Doubao-lite | temperature 0 first, 0.7/0.9 on retries; thinking disabled |
| (none) | redundancy probe; adjudicating the appropriateness criterion | DeepSeek-V4-Pro | temperature 0, max_tokens 3000, thinking disabled |

`--thinking env` leaves the thinking toggle to the environment (for endpoints that do not accept it).

## Commands

```bash
W=audit   # work directory
# medicine
python3 incentive_audit.py sample --workdir $W --domain medicine --n 300 --seed 0 --checklist-arm A \
  --arm A=rubric_rl/train.parquet --arm B=prorubric/train.parquet --arm C=raw_and/train.parquet \
  --arm G=prorubric/train.parquet:graded:2:0.7 --arm V=prorubric_appr/train.parquet \
  --arm AV=rubric_rl_appr/train.parquet --arm NOVETO=no_failure_clauses/train.parquet \
  --arm K1=k1/train.parquet
# generate the untrained model's answers to $W/medicine/gen_prompts.parquet
# (paper: Qwen3-4B, max 8192 tokens, T 0.7, top_p 0.8, seed 42, thinking off) -> base_medicine.jsonl {id, response}
python3 incentive_audit.py originals --workdir $W --domain medicine --responses base_medicine.jsonl
python3 incentive_audit.py classify  --workdir $W                       # medical-only subset of Sec. 5.5
python3 incentive_audit.py edit      --workdir $W --domain medicine
python3 incentive_audit.py edit      --workdir $W --domain medicine     # second pass: E7 is sized to E6
python3 incentive_audit.py leakcheck --workdir $W --domain medicine
python3 incentive_audit.py score     --workdir $W --domain medicine
# writing / dialogue / science: same steps (no classify) with two arms
python3 incentive_audit.py sample --workdir $W --domain science --n 300 --seed 0 \
  --arm A=science_rubric_rl/train.parquet --arm B=science_prorubric/train.parquet
# ...
python3 incentive_audit.py report --workdir $W          # prints every table, writes $W/report.json

# redundancy sub-score (evaluation judge)
python3 redundancy_probe.py --workdir $W --domains science,medicine,dialogue,writing --n 300 \
  --variants orig,E1,E6 --out probe/c3_e1_e6.jsonl
python3 redundancy_probe.py --workdir $W --domains science,medicine,dialogue,writing --n 300 \
  --variants orig,E1,E6,E7 --out probe/c3_e7.jsonl
python3 redundancy_probe.py --workdir $W --domains science,medicine,dialogue,writing --n 300 \
  --variants orig,E6L,E7L --out probe/c3_e7l.jsonl

# fixed appropriateness criterion (reads the V scores; adjudicates with the evaluation judge)
python3 appr_criterion_audit.py --audit-dir $W/medicine --out-dir appr_fp --n 150
```

## Reading the output

`report` prints one table per part, Δreward × 100 with a 95% bootstrap interval over prompts (2,000 resamples,
seed 0), the share of prompts whose reward rose / fell, n, and the count of prompts whose reward did not move:

- `medicine, all prompts` → Table 1(b) and `tab:incentive_audit_full` (also printed in that layout); the paraphrase
  row and `orig_rescore` column are the noise floor quoted in Sec. 3.3.
- `medicine, medical-only subset` → the medicine figures of Sec. 5.5 (E6 = added substance, E1/E6 ratio lines).
- `writing`, `dialogue`, `science` → the other domains of Sec. 5.5; the science E1 zero count is the "reward did
  not move" figure; the writing Rubric-RL E1 / E1b_clean cells are the writing column of `tab:incentive_audit_full`.
- `<domain>, E6L/E7L matched within 25%` and the pairing table → appendix "Where Added Material Sits" (matched n,
  realised expansion, within-pair length difference, lengths of matched vs unmatched originals).
- `<domain>, prompts with a usable E7` → the earlier unequal-budget comparison on its own prompt set.

`redundancy_probe.py` writes `<out>.report.json`: pass rate of the redundancy criterion per variant and, on the
prompts that have every requested variant, the paired pass rates and each edit's change from the base answer.

`appr_criterion_audit.py` writes `report.json` with the unmet rate per variant (`orig`, `E1`, `E3`, `E4`, `E5`, ...)
and the adjudicated unwarranted share for detail-only variants and for untrained answers.

The DAPO overlong penalty is not applied in the audit; `report` counts edits long enough to reach it.
