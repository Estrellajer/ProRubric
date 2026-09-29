#!/usr/bin/env python3
"""Generate ProRubric's dimensions from atomic rubrics (the method's only model step).

Each question's atomic checklist (20-30 small "mentions X" items) is sent to an
LLM, which returns 2-5 protocol-level dimensions, each
{name, description, weight, atomic_indices}, where atomic_indices partitions
the 1-based atomic items. Each description ends in a failure clause
("fails if ..."). The prompts call a dimension a "criterion"; they are kept verbatim.

Six prompt variants, each carrying the exact prompt used for the corresponding
rubric set of the paper:
  --variant anchored      RubricHub medicine: an
                          expert-reviewed anchor (rubric_correct / self_golden)
                          resolves contradictions between atomic items; the prompt
                          asks for weight = sum of assigned atomic weights
                          (packaging recomputes it as the sum of |atomic weight|).
  --variant writing       RubricHub writing: the anchored prompt with the task
                          described as a writing task and a same-language rule.
  --variant dialogue      RubricHub dialogue ("Chat"): the same, for a chat
                          assistant response.
  --variant healthbench   HealthBench-style questions (not used for training in
                          the paper): no anchor exists; atomic items carry
                          signed weights (negative = failure mode); criterion
                          weight = sum of |atomic weight| and failure modes are
                          folded into the criterion's FAIL conditions.
  --variant science       Science (RaR-Science): anchored, but atomic pitfalls (negative
                          weights) are labelled inline as PITFALL and criterion
                          weight = sum of |atomic weight|, so a criterion never
                          rewards a pitfall.
  --variant k1            The K=1 control: exactly ONE dimension covering all
                          atomic items, with its failure clause.

Per question: validate full coverage (every atomic index assigned exactly
once); regenerate up to 2 times on failure; if still invalid, auto-repair
(assign orphan indices to the nearest criterion, recompute weights) and flag
the row as repaired rather than dropping the question.

Resumable: re-running skips ids already present in the output jsonl.

Model calls go through the shared OpenAI-compatible client in
reward/judge_client.py (rolling-window rate limit, 429/5xx backoff).
Configure it via environment variables (or RUBRIC_JUDGE_ENV_FILE):
RUBRIC_JUDGE_BASE_URL, RUBRIC_JUDGE_API_KEY, RUBRIC_JUDGE_MODEL (the paper
used DeepSeek-V4-Pro; the generator control of Table 3 sets RUBRIC_JUDGE_MODEL
to a Doubao-lite model and changes nothing else).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reward"))

from judge_client import JudgeUnavailableError, RubricJudge  # noqa: E402

MAX_TOKENS = 3000
MAX_REGEN = 2  # semantic regeneration attempts on validation failure (0,1,2 => 3 tries)


GEN_PROMPT_ANCHORED = """You are designing a compact, protocol-level grading rubric for a medical Q&A answer, to replace an atomic checklist of many small "mentions X" items with 2-5 coarse-grained criteria that each describe what a GOOD answer ACHIEVES on one dimension.

# Question / conversation
{conversation}

# Existing atomic rubric items (for reference only -- do not just copy these into your descriptions as a checklist)
{atomic_items}

# Anchor for resolving contradictions (an expert-reviewed reference rubric for this same question -- some atomic items above may be redundant, mutually exclusive, or simply wrong; use this anchor to decide what a correct answer should actually contain)
{anchor}

# Task
Design between 2 and 5 protocol-level criteria. Each criterion is a paragraph, not a checklist item. Each criterion must:
1. state what a good answer ACHIEVES on that dimension for THIS SPECIFIC question, and what would make an answer FAIL that dimension;
2. be judgeable only by reading and understanding the whole answer holistically -- NOT by scanning for a keyword or a single sentence;
3. NOT be phrased as a list of "mentions A, B, C" terms to check off;
4. if two or more atomic items below contradict each other (e.g. reward mutually exclusive answers), resolve the contradiction explicitly using the anchor -- the description should state what the CORRECT position is, not just note that people disagree;
5. together with the other criteria, have every atomic item index from 1 to {n} assigned to exactly one criterion (pick whichever criterion is closest in topic, even after you rewrite its language into holistic phrasing);
6. carry a weight equal to the sum of the weights of the atomic items assigned to it.

# Output format
Return ONLY a JSON object, no markdown fences, no other text:
{{"criteria": [{{"name": "<=6 words", "description": "2-5 sentences, self-contained, states what ACHIEVES / FAILS this dimension", "weight": <number>, "atomic_indices": [<1-based ints into the atomic list above>]}}, ...]}}
The atomic_indices arrays across all criteria must union to exactly {{1, ..., {n}}} with no repeats and no omissions."""


GEN_PROMPT_HEALTHBENCH = """You are designing a compact, protocol-level grading rubric for a medical Q&A answer (HealthBench), to replace an atomic checklist of many small "mentions X" items with 2-5 coarse-grained criteria that each describe what a GOOD answer ACHIEVES on one dimension.

# Question / conversation
{conversation}

# Existing atomic rubric items (for reference only -- do not just copy these into your descriptions as a checklist)
Each item below carries a signed weight. A POSITIVE weight means the item describes something a good answer SHOULD contain or do -- reward it. A NEGATIVE weight means the item describes a FAILURE MODE, something a good answer should AVOID -- do not reward an answer for exhibiting it; instead fold it into your criterion's description as part of what would make the answer FAIL that dimension.
{atomic_items}

# Anchor for resolving contradictions
No independently expert-reviewed reference rubric is available for this question. If two or more atomic items below contradict each other (e.g. reward mutually exclusive answers, or a positive-weight item conflicts with a negative-weight item), resolve the contradiction yourself using your own best clinical judgment of what a correct, safe answer to THIS question should actually contain -- state what the CORRECT position is, do not just note that the items disagree.

# Task
Design between 2 and 5 protocol-level criteria. Each criterion is a paragraph, not a checklist item. Each criterion must:
1. state what a good answer ACHIEVES on that dimension for THIS SPECIFIC question, and what would make an answer FAIL that dimension (including any negative-weight failure modes assigned to it);
2. be judgeable only by reading and understanding the whole answer holistically -- NOT by scanning for a keyword or a single sentence;
3. NOT be phrased as a list of "mentions A, B, C" terms to check off;
4. if two or more atomic items below contradict each other, resolve the contradiction explicitly as instructed above -- the description should state what the CORRECT position is, not just note that people disagree;
5. together with the other criteria, have every atomic item index from 1 to {n} assigned to exactly one criterion (pick whichever criterion is closest in topic, even after you rewrite its language into holistic phrasing) -- this includes both positive- and negative-weight items;
6. carry a weight equal to the sum of the ABSOLUTE VALUES of the weights of the atomic items assigned to it, so the criterion weight is always positive even when it absorbs negative-weight failure-mode items.

# Output format
Return ONLY a JSON object, no markdown fences, no other text:
{{"criteria": [{{"name": "<=6 words", "description": "2-5 sentences, self-contained, states what ACHIEVES / FAILS this dimension", "weight": <number>, "atomic_indices": [<1-based ints into the atomic list above>]}}, ...]}}
The atomic_indices arrays across all criteria must union to exactly {{1, ..., {n}}} with no repeats and no omissions."""


GEN_PROMPT_SCIENCE = """You are designing a compact, protocol-level grading rubric for a medical Q&A answer, to replace an atomic checklist of many small "mentions X" items with 2-5 coarse-grained criteria that each describe what a GOOD answer ACHIEVES on one dimension.

# Question / conversation
{conversation}

# Existing atomic rubric items (for reference only -- do not just copy these into your descriptions as a checklist)
{atomic_items}

# Anchor for resolving contradictions (an expert-reviewed reference rubric for this same question -- some atomic items above may be redundant, mutually exclusive, or simply wrong; use this anchor to decide what a correct answer should actually contain)
{anchor}

# Task
Design between 2 and 5 protocol-level criteria. Each criterion is a paragraph, not a checklist item. Each criterion must:
1. state what a good answer ACHIEVES on that dimension for THIS SPECIFIC question, and what would make an answer FAIL that dimension;
2. be judgeable only by reading and understanding the whole answer holistically -- NOT by scanning for a keyword or a single sentence;
3. NOT be phrased as a list of "mentions A, B, C" terms to check off;
4. if two or more atomic items below contradict each other (e.g. reward mutually exclusive answers), resolve the contradiction explicitly using the anchor -- the description should state what the CORRECT position is, not just note that people disagree;
5. together with the other criteria, have every atomic item index from 1 to {n} assigned to exactly one criterion (pick whichever criterion is closest in topic, even after you rewrite its language into holistic phrasing);
6. carry a weight equal to the sum of the ABSOLUTE weights of the atomic items assigned to it (pitfall items count by magnitude); every criterion weight must be positive. Pitfall items must be folded into the criterion's FAIL conditions (what a good answer avoids), never rewarded.

# Output format
Return ONLY a JSON object, no markdown fences, no other text:
{{"criteria": [{{"name": "<=6 words", "description": "2-5 sentences, self-contained, states what ACHIEVES / FAILS this dimension", "weight": <number>, "atomic_indices": [<1-based ints into the atomic list above>]}}, ...]}}
The atomic_indices arrays across all criteria must union to exactly {{1, ..., {n}}} with no repeats and no omissions."""


GEN_PROMPT_K1 = """You are rewriting an atomic grading checklist for a medical Q&A answer into ONE holistic, protocol-level criterion: a single paragraph describing what a GOOD answer to this specific question achieves overall, and what would make it fail.

# Question / conversation
{conversation}

# Existing atomic rubric items (for reference only -- do not copy them as a checklist)
{atomic_items}

# Anchor for resolving contradictions (use it to decide what a correct answer should actually contain when atomic items conflict)
{anchor}

# Task
Write EXACTLY ONE criterion. Its description is one self-contained paragraph of 5-10 sentences that:
1. states what a good answer ACHIEVES for THIS SPECIFIC question (correct core answer, necessary coverage, safety/restraint, actionability, tone) and what would make an answer FAIL;
2. is judgeable only by reading the whole answer holistically -- NOT by scanning for keywords;
3. is NOT phrased as a list of "mentions A, B, C" items to tick off;
4. resolves any contradictions among the atomic items using the anchor, stating the CORRECT position;
5. covers every atomic item 1..{n} in holistic phrasing (all indices belong to this single criterion);
6. carries weight = the sum of the weights of ALL atomic items.

# Output format
Return ONLY a JSON object, no markdown fences, no other text, with exactly one element in "criteria":
{{"criteria": [{{"name": "<=6 words", "description": "5-10 sentences, self-contained, states what the answer ACHIEVES / FAILS overall", "weight": <number>, "atomic_indices": [1, 2, ..., {n}]}}]}}"""


# RubricHub writing and dialogue ("Chat"): the anchored prompt with the task description and a same-language rule.
GEN_PROMPT_WRITING = """You are designing a compact, protocol-level grading rubric for a written response to a writing task (essay, article, story, marketing copy, technical or creative writing), to replace an atomic checklist of many small "mentions X" items with 2-5 coarse-grained criteria that each describe what a GOOD answer ACHIEVES on one dimension.

# Question / conversation
{conversation}

# Existing atomic rubric items (for reference only -- do not just copy these into your descriptions as a checklist)
{atomic_items}

# Anchor for resolving contradictions (an expert-reviewed reference rubric for this same question -- some atomic items above may be redundant, mutually exclusive, or simply wrong; use this anchor to decide what a correct answer should actually contain)
{anchor}

# Task
Design between 2 and 5 protocol-level criteria. Each criterion is a paragraph, not a checklist item. Each criterion must:
1. state what a good answer ACHIEVES on that dimension for THIS SPECIFIC question, and what would make an answer FAIL that dimension;
2. be judgeable only by reading and understanding the whole answer holistically -- NOT by scanning for a keyword or a single sentence;
3. NOT be phrased as a list of "mentions A, B, C" terms to check off;
4. if two or more atomic items below contradict each other (e.g. reward mutually exclusive answers), resolve the contradiction explicitly using the anchor -- the description should state what the CORRECT position is, not just note that people disagree;
5. together with the other criteria, have every atomic item index from 1 to {n} assigned to exactly one criterion (pick whichever criterion is closest in topic, even after you rewrite its language into holistic phrasing);
6. carry a weight equal to the sum of the weights of the atomic items assigned to it;
7. be written in the SAME LANGUAGE as the question (Chinese question -> Chinese criteria; English -> English).

# Output format
Return ONLY a JSON object, no markdown fences, no other text:
{{"criteria": [{{"name": "<=6 words", "description": "2-5 sentences, self-contained, states what ACHIEVES / FAILS this dimension", "weight": <number>, "atomic_indices": [<1-based ints into the atomic list above>]}}, ...]}}
The atomic_indices arrays across all criteria must union to exactly {{1, ..., {n}}} with no repeats and no omissions."""


GEN_PROMPT_DIALOGUE = """You are designing a compact, protocol-level grading rubric for a chat assistant response to a user request, to replace an atomic checklist of many small "mentions X" items with 2-5 coarse-grained criteria that each describe what a GOOD answer ACHIEVES on one dimension.

# Question / conversation
{conversation}

# Existing atomic rubric items (for reference only -- do not just copy these into your descriptions as a checklist)
{atomic_items}

# Anchor for resolving contradictions (an expert-reviewed reference rubric for this same question -- some atomic items above may be redundant, mutually exclusive, or simply wrong; use this anchor to decide what a correct answer should actually contain)
{anchor}

# Task
Design between 2 and 5 protocol-level criteria. Each criterion is a paragraph, not a checklist item. Each criterion must:
1. state what a good answer ACHIEVES on that dimension for THIS SPECIFIC question, and what would make an answer FAIL that dimension;
2. be judgeable only by reading and understanding the whole answer holistically -- NOT by scanning for a keyword or a single sentence;
3. NOT be phrased as a list of "mentions A, B, C" terms to check off;
4. if two or more atomic items below contradict each other (e.g. reward mutually exclusive answers), resolve the contradiction explicitly using the anchor -- the description should state what the CORRECT position is, not just note that people disagree;
5. together with the other criteria, have every atomic item index from 1 to {n} assigned to exactly one criterion (pick whichever criterion is closest in topic, even after you rewrite its language into holistic phrasing);
6. carry a weight equal to the sum of the weights of the atomic items assigned to it;
7. be written in the SAME LANGUAGE as the question (Chinese question -> Chinese criteria; English -> English).

# Output format
Return ONLY a JSON object, no markdown fences, no other text:
{{"criteria": [{{"name": "<=6 words", "description": "2-5 sentences, self-contained, states what ACHIEVES / FAILS this dimension", "weight": <number>, "atomic_indices": [<1-based ints into the atomic list above>]}}, ...]}}
The atomic_indices arrays across all criteria must union to exactly {{1, ..., {n}}} with no repeats and no omissions."""


VARIANTS = {
    "anchored": {"prompt": GEN_PROMPT_ANCHORED, "abs_weights": False, "has_anchor": True,
                 "k_range": (2, 5), "pitfall_tags": False},
    "healthbench": {"prompt": GEN_PROMPT_HEALTHBENCH, "abs_weights": True, "has_anchor": False,
                    "k_range": (2, 5), "pitfall_tags": False},
    "science": {"prompt": GEN_PROMPT_SCIENCE, "abs_weights": True, "has_anchor": True,
                "k_range": (2, 5), "pitfall_tags": True},
    "k1": {"prompt": GEN_PROMPT_K1, "abs_weights": False, "has_anchor": True,
           "k_range": (1, 1), "pitfall_tags": False},
    "writing": {"prompt": GEN_PROMPT_WRITING, "abs_weights": False, "has_anchor": True,
                "k_range": (2, 5), "pitfall_tags": False},
    "dialogue": {"prompt": GEN_PROMPT_DIALOGUE, "abs_weights": False, "has_anchor": True,
                 "k_range": (2, 5), "pitfall_tags": False},
}


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_to_dict(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        first_line, _, rest = text.partition("\n")
        if rest.rstrip().endswith("```"):
            rest = rest.rstrip()[:-3]
        text = rest.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if m:
            return json.loads(m.group(0))
        raise


def build_conversation(messages: list[dict]) -> str:
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def build_atomic_items(rubric: list[dict], *, pitfall_tags: bool = False) -> str:
    lines = []
    for item in rubric:
        weight = float(item["weight"])
        if pitfall_tags and weight < 0:
            tag = f"[PITFALL, weight {abs(weight):g} -- an answer that does this FAILS]"
        elif pitfall_tags:
            tag = f"[weight {weight:g}]"
        else:
            tag = f"[weight {item['weight']}]"
        lines.append(f"{item['idx'] + 1}. {tag} {item['criterion']}")
    return "\n".join(lines)


def build_anchor(q: dict) -> str:
    parts = []
    if q.get("rubric_correct"):
        parts.append("Reference rubric:\n" + q["rubric_correct"])
    if q.get("self_golden"):
        parts.append("Reference (gold) answer notes:\n" + q["self_golden"])
    if not parts:
        parts.append("(no separate anchor provided -- use the atomic items above as-is, resolving any internal contradictions by majority weight)")
    return "\n\n".join(parts)


def call_generator(judge: RubricJudge, prompt: str) -> tuple[str | None, str | None]:
    try:
        completion = judge.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0,
        )
        return completion.raw, None
    except JudgeUnavailableError as exc:
        return None, f"{type(exc).__name__}:{exc}"


def validate(parsed: dict, n: int, *, k_range: tuple[int, int] = (2, 5)) -> tuple[bool, str]:
    lo, hi = k_range
    crit = parsed.get("criteria")
    if not isinstance(crit, list) or not (lo <= len(crit) <= hi):
        return False, f"criteria count {len(crit) if isinstance(crit, list) else 'n/a'} not in [{lo},{hi}]"
    seen = set()
    for c in crit:
        if not all(k in c for k in ("name", "description", "weight", "atomic_indices")):
            return False, "missing field"
        idxs = c["atomic_indices"]
        if not isinstance(idxs, list) or not idxs:
            return False, "empty atomic_indices"
        for i in idxs:
            if not isinstance(i, int) or not (1 <= i <= n):
                return False, f"bad index {i}"
            if i in seen:
                return False, f"duplicate index {i}"
            seen.add(i)
    if seen != set(range(1, n + 1)):
        missing = set(range(1, n + 1)) - seen
        return False, f"coverage gap, missing {sorted(missing)[:10]}"
    return True, "ok"


def auto_repair(parsed: dict, rubric: list[dict], n: int, *, abs_weights: bool, min_criteria: int = 2) -> dict:
    """Best-effort repair: drop invalid criteria shape, assign orphan/duplicate
    indices to the criterion whose existing indices are numerically closest,
    then recompute each criterion's weight as the sum of its assigned atomic
    weights. Guarantees full 1..n coverage with no duplicates."""
    crit = [c for c in (parsed.get("criteria") or []) if isinstance(c, dict) and c.get("name") and c.get("description")]
    if len(crit) < min_criteria:
        crit = [{"name": "Overall quality", "description": "Overall holistic quality of the answer for this question.", "weight": 0, "atomic_indices": []}]
    # de-duplicate indices across criteria, keeping first occurrence
    claimed: set[int] = set()
    for c in crit:
        idxs = c.get("atomic_indices") or []
        clean = []
        for i in idxs:
            if isinstance(i, int) and 1 <= i <= n and i not in claimed:
                clean.append(i)
                claimed.add(i)
        c["atomic_indices"] = clean
    orphans = [i for i in range(1, n + 1) if i not in claimed]
    for i in orphans:
        # nearest criterion by mean of its current indices; fall back to first criterion
        best_c, best_d = crit[0], float("inf")
        for c in crit:
            if not c["atomic_indices"]:
                continue
            mean_idx = sum(c["atomic_indices"]) / len(c["atomic_indices"])
            d = abs(mean_idx - i)
            if d < best_d:
                best_c, best_d = c, d
        best_c["atomic_indices"].append(i)
        claimed.add(i)
    wby = {item["idx"] + 1: item["weight"] for item in rubric}
    for c in crit:
        weights = (abs(wby[i]) for i in c["atomic_indices"]) if abs_weights else (wby[i] for i in c["atomic_indices"])
        c["weight"] = round(sum(weights), 2)
    crit = [c for c in crit if c["atomic_indices"]]
    return {"criteria": crit}


def load_questions(path: str) -> list[tuple[str, dict]]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        raw_rows = pq.read_table(path).to_pylist()
    elif path.endswith(".jsonl"):
        with open(path) as f:
            raw_rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            raw_rows = [dict(value, id=key) if "id" not in value else value for key, value in payload.items()]
        else:
            raw_rows = payload

    questions: list[tuple[str, dict]] = []
    for index, row in enumerate(raw_rows):
        ei = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        qid = str(row.get("id") or ei.get("id") or ei.get("sample_id") or ei.get("tid") or index)

        messages = row.get("messages")
        if not messages:
            question = row.get("question") or row.get("problem") or ei.get("problem") or row.get("prompt")
            if isinstance(question, list):
                messages = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in question]
            else:
                messages = [{"role": "user", "content": str(question)}]

        atomic = (row.get("rubric_atomic") or row.get("rubric")
                  or ei.get("rubric_atomic") or ei.get("rubric"))
        if not atomic:
            raise ValueError(
                f"question {qid!r} has no atomic rubric: looked for rubric_atomic / rubric at the "
                "row top level and under extra_info"
            )
        rubric = [
            {"idx": i, "criterion": str(item["criterion"]), "weight": item["weight"]}
            for i, item in enumerate(atomic)
        ]

        questions.append(
            (
                qid,
                {
                    "messages": messages,
                    "rubric": rubric,
                    "rubric_correct": row.get("rubric_correct") or ei.get("rubric_correct"),
                    "self_golden": row.get("self_golden") or ei.get("self_golden"),
                },
            )
        )
    return questions


def load_done(out_path: str) -> set[str]:
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add(str(d["id"]))
    return done


def process_one(judge: RubricJudge, variant: dict, qid: str, q: dict) -> dict:
    rubric = q["rubric"]
    n = len(rubric)
    prompt_kwargs = {
        "conversation": build_conversation(q["messages"]),
        "atomic_items": build_atomic_items(rubric, pitfall_tags=variant["pitfall_tags"]),
        "n": n,
    }
    if variant["has_anchor"]:
        prompt_kwargs["anchor"] = build_anchor(q)
    prompt = variant["prompt"].format(**prompt_kwargs)

    attempts = []
    parsed = None
    for regen in range(MAX_REGEN + 1):
        content, err = call_generator(judge, prompt)
        if err is not None:
            attempts.append({"regen": regen, "error": err})
            continue
        try:
            p = parse_json_to_dict(content)
        except Exception as exc:  # noqa: BLE001
            attempts.append({"regen": regen, "error": f"json_parse:{exc}", "raw": content[:500]})
            continue
        ok, reason = validate(p, n, k_range=variant["k_range"])
        attempts.append({"regen": regen, "valid": ok, "reason": reason})
        if ok:
            parsed = p
            break
        parsed = p  # keep last parseable attempt for possible repair
    repaired = False
    if parsed is None:
        parsed = {"criteria": []}
    ok, reason = validate(parsed, n, k_range=variant["k_range"])
    if not ok:
        parsed = auto_repair(parsed, rubric, n, abs_weights=variant["abs_weights"],
                             min_criteria=variant["k_range"][0])
        repaired = True
        ok2, reason2 = validate(parsed, n, k_range=variant["k_range"])
        if not ok2:
            reason = f"repair_failed:{reason2}"
    words = [len(c["description"].split()) for c in parsed["criteria"]]
    return {
        "id": qid,
        "n_atomic": n,
        "criteria": parsed["criteria"],
        "n_criteria": len(parsed["criteria"]),
        "repaired": repaired,
        "valid": ok,
        "reason": reason if not ok else None,
        "attempts": attempts,
        "desc_word_counts": words,
        "model": judge.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def summarize(out_path: str) -> dict:
    with open(out_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    n_crit = [r["n_criteria"] for r in rows]
    words = [w for r in rows for w in r["desc_word_counts"]]
    dist: dict[str, int] = {}
    for k in n_crit:
        dist[str(k)] = dist.get(str(k), 0) + 1
    return {
        "n_questions": len(rows),
        "valid_on_final": sum(1 for r in rows if r["valid"]),
        "repaired": sum(1 for r in rows if r["repaired"]),
        "n_criteria_distribution": dist,
        "n_criteria_mean": sum(n_crit) / len(n_crit) if n_crit else None,
        "desc_word_count_mean": sum(words) / len(words) if words else None,
        "desc_word_count_min": min(words) if words else None,
        "desc_word_count_max": max(words) if words else None,
        "coverage_rate": sum(1 for r in rows if r["valid"] or r["repaired"]) / len(rows) if rows else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS),
                    help="prompt variant: anchored (RubricHub medicine, with anchor), writing, dialogue, healthbench "
                         "(signed weights, no anchor), science (anchored + PITFALL labels, "
                         "absolute weights), k1 (the K=1 control, exactly one dimension)")
    ap.add_argument("--input", required=True,
                    help="input questions: .parquet or .jsonl (or a .json dict/array); rows need question/messages, "
                         "an atomic rubric list of {criterion, weight}, and optional rubric_correct/self_golden anchors")
    ap.add_argument("--output", required=True, help="output jsonl (one dimension set per question); appended to, resumable")
    ap.add_argument("--workers", type=int, default=8, help="concurrent generator requests (default 8)")
    ap.add_argument("--rpm", type=int, default=180, help="requests-per-minute cap shared across workers (default 180)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N pending questions")
    ap.add_argument("--fidelity", default=None, help="optional path for the fidelity summary JSON (also printed to stdout)")
    args = ap.parse_args()

    variant = VARIANTS[args.variant]
    questions = load_questions(args.input)
    done = load_done(args.output)
    todo = [(qid, q) for qid, q in questions if qid not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"total questions={len(questions)} already_done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print(json.dumps(summarize(args.output), ensure_ascii=False, indent=2))
        return

    judge = RubricJudge(
        qpm=args.rpm,
        max_concurrency=args.workers,
        max_tokens=MAX_TOKENS,
        thinking="disabled",
    )

    out_f = open(args.output, "a")
    out_lock = threading.Lock()
    counts = {"ok": 0, "fail": 0, "repaired": 0}
    t0 = time.monotonic()

    def work(item):
        qid, q = item
        rec = process_one(judge, variant, qid, q)
        with out_lock:
            if rec["valid"]:
                counts["ok"] += 1
            else:
                counts["fail"] += 1
            if rec["repaired"]:
                counts["repaired"] += 1
            total = counts["ok"] + counts["fail"]
            if total % 25 == 0:
                elapsed = time.monotonic() - t0
                rpm = total / elapsed * 60 if elapsed > 0 else 0.0
                print(
                    f"progress {total}/{len(todo)} ok={counts['ok']} fail={counts['fail']} "
                    f"repaired={counts['repaired']} elapsed_s={elapsed:.0f} rpm~={rpm:.0f}",
                    flush=True,
                )
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, item) for item in todo]
        for fut in as_completed(futures):
            fut.result()
    out_f.close()
    print(f"DONE ok={counts['ok']} fail={counts['fail']} repaired={counts['repaired']}", flush=True)

    fidelity = summarize(args.output)
    print(json.dumps(fidelity, ensure_ascii=False, indent=2))
    if args.fidelity:
        with open(args.fidelity, "w") as f:
            json.dump(fidelity, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
