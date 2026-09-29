# Generating ProRubric's dimensions

`generate_dimensions.py` is the method's only model step. For each question it
sends the atomic checklist (20–30 small "mentions X" items) to a generator model,
which partitions the items by protocol phase and rewrites each group as one
dimension: a short holistic description of what a good answer achieves on that
phase, ending in a failure clause ("fails if ..."), plus the set of atomic indices
it covers. The training algorithm, reward manager and data pipeline downstream are
unchanged. The prompts call a dimension a "criterion"; they are kept byte-for-byte
as run, because a reworded prompt produces a different rubric.

| `--variant` | Used for | Prompt semantics |
|---|---|---|
| `anchored` | RubricHub medicine (paper Box A.1) | An expert-reviewed anchor (`rubric_correct`, optionally `self_golden`) resolves contradictions between atomic items. |
| `writing` | RubricHub writing | `anchored` with the task described as a writing task and one added rule: write the dimensions in the question's language. |
| `dialogue` | RubricHub dialogue ("Chat") | the same, for a chat assistant response. |
| `science` | RaR-Science | Anchored as above, but atomic pitfalls (negative weights) are labelled inline as `[PITFALL, weight w -- an answer that does this FAILS]` and must be folded into the failure clause, never rewarded. |
| `k1` | the K=1 control | Exactly one dimension covering every atomic item, as one 5–10 sentence paragraph with its failure clause. Validation requires exactly one. |
| `healthbench` | not used for training in the paper | No anchor; signed atomic weights; failure modes folded into the failure clause. |

Packaging (`../data/build_prorubric_release.py`) sets each dimension's weight to
the exact sum of `|atomic weight|` over its assigned items; the generator's own
proposed weights are kept as metadata.

`rewrite_atomic.py` is the science control atomic-rw: it rewrites each atomic
criterion one-to-one in the same style (no grouping), with the same generator and
request settings; `../data/build_atomic_rw.py` packages it.

## Input

`--input` accepts a `.parquet` file of questions (the training-set layout), a
`.jsonl` file with one question per line, or a `.json` file (a dict keyed by
question id, or an array). Each question needs:

- **question text**: `messages` (list of `{role, content}`), `question`, `problem`
  (also `extra_info.problem`), or `prompt`; non-`messages` forms become one `user` turn;
- **atomic rubric**: a list of `{"criterion", "weight"}` from `rubric_atomic` or
  `rubric` (top level) or `extra_info.rubric_atomic` / `extra_info.rubric`; a
  question with none is a hard error;
- **anchor** (optional; `anchored` / `science` / `k1`): `rubric_correct` and/or
  `self_golden` (top level or `extra_info.*`); without one, the prompt says so;
- **id**: `id`, `extra_info.id`, `extra_info.sample_id`, `extra_info.tid`, then the
  row index. Resume matching and packaging join on this id.

## Output

`--output` is a JSONL file, one record per question, appended to and reused on
rerun (ids already present are skipped):

```json
{"id": "<question id>", "n_atomic": 24,
 "criteria": [{"name": "<=6 words", "description": "2-5 sentences ... fails if ...",
               "weight": 12.0, "atomic_indices": [1, 4, 7, 12]}],
 "n_criteria": 4, "repaired": false, "valid": true, "reason": null,
 "attempts": [{"regen": 0, "valid": true, "reason": "ok"}],
 "desc_word_counts": [38, 41, 29, 33], "model": "<model id>", "timestamp": "<UTC ISO-8601>"}
```

### Validation and repair (paper App. A.3)

1. The response must parse as JSON and contain 2–5 dimensions (exactly 1 for `k1`),
   each with `name` / `description` / `weight` / `atomic_indices`.
2. `atomic_indices` across all dimensions must cover every atomic index 1…n exactly
   once.
3. On failure the same prompt is retried up to 2 times (3 tries in total).
4. If every try fails, the last parseable output is repaired structurally: invalid
   dimensions are dropped, duplicate indices keep their first occurrence, orphan
   indices go to the numerically nearest dimension, and weights are recomputed.
   Descriptions are not regenerated. The record is written with `"repaired": true`;
   no question is dropped silently. Packaging then applies the keep rules.

## Model client

Calls go through `../reward/judge_client.py` (bearer auth against
`<base_url>/chat/completions`, rolling-window QPM/TPM throttling, adaptive
concurrency, 429/5xx backoff). Configure it with `RUBRIC_JUDGE_BASE_URL`,
`RUBRIC_JUDGE_API_KEY` and `RUBRIC_JUDGE_MODEL` (required; the paper's generator is
DeepSeek-V4-Pro), or point `RUBRIC_JUDGE_ENV_FILE` at a `KEY=VALUE` file. `--rpm`
and `--workers` set the rate limit and concurrency. Requests use temperature 0,
`max_tokens=3000` and disable provider-side thinking.

The generator control of Table 3 ("dimensions from another generator") reruns
`--variant anchored` on the same questions with `RUBRIC_JUDGE_MODEL` set to a
Doubao-lite model; nothing else changes.

## Usage

```bash
# medicine: anchored (writing: --variant writing, dialogue: --variant dialogue)
python3 generate_dimensions.py --variant anchored \
    --input release/medical-atomic/train.parquet --output out/medical_dimensions.jsonl \
    --workers 24 --rpm 180

# science
python3 generate_dimensions.py --variant science \
    --input release/science-atomic/train.parquet --output out/science_dimensions.jsonl \
    --workers 24 --rpm 180

# K=1 control
python3 generate_dimensions.py --variant k1 \
    --input release/medical-atomic/train.parquet --output out/medical_k1.jsonl

# atomic-rw (science control)
python3 rewrite_atomic.py run --input release/science-atomic/train.parquet --output out/science_rw.jsonl

# try the first 10 pending questions
python3 generate_dimensions.py --variant anchored --input q.jsonl --output out.jsonl --limit 10
```

Re-running resumes. At the end of every run a fidelity summary is printed
(valid / repaired counts, dimensions per question, description word counts);
`--fidelity path.json` also saves it.
