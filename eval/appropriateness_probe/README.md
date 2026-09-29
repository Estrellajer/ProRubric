# Cross-domain appropriateness probe

Code for Appendix "Appropriateness Probe across Domains" (Table `tab:probe`),
Figure `fig:controls_compound`(c), and the probe numbers quoted in
Sections 5.4 and 5.5.

The probe scores responses from every domain on the same four
domain-independent criteria, each judged separately with the official
HealthBench grader template (`../healthbench_judge.py`, criterion rendered as
`[1.0] <criterion>`):

1. addresses the user's primary request before supplementary material
   (unless safety or essential missing context requires clarification first);
2. adapts content, assumptions and level of detail to the user's stated context;
3. includes only material that helps and avoids repetition or digression
   (the "no redundancy" criterion, `c3`);
4. separates supported claims from uncertainty and states assumptions or asks
   when missing information would change the answer.

A response's score is the mean of its four verdicts (x100). Arm differences
are paired over prompts with a 95% bootstrap interval over prompts (5,000
resamples, seed 67). The judge is the evaluation judge (paper:
DeepSeek-V4-Pro; thinking disabled, temperature 0, `max_tokens` 3000, up to
three calls per verdict); `--judge-role ALT` grades with the third-family
judge instead.

## Files

| file | what it computes | paper |
|---|---|---|
| `probe.py sample` | draws the blind item sample of one set (sha1(id) order over ids answered by the untrained, Rubric-RL and ProRubric arms; ResearchQA stratified proportionally by `general_domain`) | item sets of `tab:probe` |
| `probe.py grade` | four criterion verdicts per (item, arm), resumable | |
| `probe.py summarize` | per-domain per-criterion pass rates, probe score, all-four rate, mean length; paired deltas with CIs; the pre-registered reading; a `tab:probe`-layout table | `tab:probe` |
| `fig3c_seeds.py` | per-seed and three-seed (mean, sd ddof=1) difference from the untrained model on the common probe sample; optional seed-matched family difference | `fig:controls_compound`(c); Sec. 5.5 "-28.9 / -11.2 / -7.1 / +16.0"; Sec. 5.4 science "+7.5, at least +5.0 per seed" |
| `sample.json` | the item sample used in the paper (200 HealthBench-consensus, 200 Arena-Hard v2, 200 ResearchQA, 96 Creative-v3 + 104 WritingBench ids) | |

## Inputs and outputs

- questions: parquet rows with `prompt` (string or `[{"role","content"}]`) and
  `extra_info.id` (`extra_info.general_domain` for ResearchQA).
- responses: JSONL per arm, `{"id", "response", ["data_source"]}`.
- `sample.json`: `{set: {"domain", "eligible", "ids": [...]}}`. Sets sharing a
  `domain` are summarized together (writing = creative + writingbench).
- grades: JSONL `{"set","id","arm","k","met","chars","judge"}`.

Arm keys used by the summary defaults (any key works; the defaults fix the
order in which extra arms join the paired subset, which matters because an
extra arm joins only if it is graded on at least 90% of the current items):

| key | paper name |
|---|---|
| `base` | Untrained model |
| `A` | Rubric-RL |
| `B` | ProRubric |
| `C` | raw-AND |
| `G` | Graded |
| `AV` | Rubric-RL + appr. criterion |
| `RUSCA` | RuscaRL |
| `RGSD` | OPSD |
| `NV` | ProRubric w/o failure clauses |
| `AR` | atomic-rw |
| suffix `43` / `44` | training seed 43 / 44 (no suffix: seed 42) |

In the `tab:probe` layout every arm is compared with the seed-42 Rubric-RL
arm; the untrained row is the negated Rubric-RL minus untrained difference.

## Commands

```
# 1) items (already provided as sample.json; shown for reproduction), one call per set
python3 probe.py sample --set medical --domain medicine --n 200 \
    --questions healthbench_consensus.parquet --data-source healthbench_consensus \
    --responses base=med/base.jsonl --responses A=med/rubric_rl_s42.jsonl --responses B=med/prorubric_s42.jsonl \
    --sample sample.json
python3 probe.py sample --set researchqa --domain science --n 200 --stratify-by general_domain \
    --questions researchqa_valid.parquet --data-source researchqa_valid \
    --responses base=sci/base.jsonl --responses A=sci/rubric_rl_s42.jsonl --responses B=sci/prorubric_s42.jsonl \
    --sample sample.json
#    likewise: --set arena --domain dialogue --n 200; --set creative --domain writing --n 96;
#              --set writingbench --domain writing --n 104

# 2) grading, one call per set, every arm to be reported
python3 probe.py grade --set researchqa --sample sample.json \
    --questions researchqa_valid.parquet --data-source researchqa_valid \
    --responses base=sci/base.jsonl --responses A=sci/rubric_rl_s42.jsonl --responses B=sci/prorubric_s42.jsonl \
    --responses C=sci/raw_and_s42.jsonl --responses AR=sci/atomic_rw_s42.jsonl \
    --responses A43=sci/rubric_rl_s43.jsonl --responses A44=sci/rubric_rl_s44.jsonl \
    --responses B43=sci/prorubric_s43.jsonl --responses B44=sci/prorubric_s44.jsonl ... \
    --grades grades.jsonl

# 3) tables
python3 probe.py summarize --sample sample.json --grades grades.jsonl \
    --table-arms base,A,B,C,IMPL,IMPL43,IMPL44,AR,AR43 --out PROBE_RESULTS.md
python3 fig3c_seeds.py --sample sample.json --grades grades.jsonl \
    --family Rubric-RL=A,A43,A44 --family ProRubric=B,B43,B44 --paired ProRubric:Rubric-RL --out fig3c_seeds.json
```

Judge configuration: `RUBRIC_JUDGE_BASE_URL`, `RUBRIC_JUDGE_API_KEY`,
`RUBRIC_JUDGE_MODEL` (and `RUBRIC_JUDGE_ALT_*` for `--judge-role ALT`); see
`../../reward/judge_client.py`.

The per-criterion columns of `summarize` give the probe's "no redundancy"
(`c3`) pass rates per arm. The Section 5.5 figures for what added material
costs on that criterion ("17 to 38 points of the redundancy sub-score") come
from applying criterion 3 alone to the incentive-audit perturbations, not
from this probe sample.
