#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared pieces of the rubric-free blind pairwise comparisons (library module).

The judge sees the user prompt and two anonymous responses, never a rubric,
and answers ``{"winner": "A" | "B" | "tie", "reason": ...}``. Every pair is
judged in both response orders; a stable win needs the same winner in both.

Prompts by domain (the text the judge sees is byte-identical to the one used
for the paper):

  medicine            ``eval/pairwise.py`` PROMPT ("health conversation")
  science             SCIENCE_PROMPT ("research question")
  dialogue, writing   REQUEST_PROMPT ("user request"; one prompt for both)

Request settings: temperature 0, ``max_tokens`` 3000, up to three judge
calls per verdict (a fresh call after a transport failure or an unparseable
answer). The evaluation judge (no role; paper: DeepSeek-V4-Pro) was called
with thinking disabled; the ALT judge (paper: GPT-5.6-luna) was called
without a thinking field. ``build_judge`` reproduces both.

Input contracts
---------------
questions   parquet rows with ``prompt`` (string or [{"role","content"}])
            and ``extra_info.id`` (the row id used everywhere else).
responses   JSONL, one object per row: ``id``, ``response``, optionally
            ``score`` (training-time judge score, for a sign-agreement
            column), ``data_source`` and ``error``.
ids         JSON list of ids, or a probe ``sample.json`` (see
            ``eval/appropriateness_probe/probe.py``) with ``--ids-sets``.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "reward"))

import pyarrow.parquet as pq  # noqa: E402

from judge_client import JudgeUnavailableError, judge_from_env  # noqa: E402
from pairwise import (  # noqa: E402  (package medical pairwise module)
    PROMPT as MEDICINE_PROMPT,
    bootstrap_interval,
    conversation_text,
    extra_info,
    hash_key,
    strip_fence,
)

SCIENCE_PROMPT = """You are reviewing two candidate responses to the same research question. You do not have a checklist; judge usefulness to THIS asker.

Which response would be more useful to the person who asked? Weigh, in order: whether it answers what was actually asked; whether it fits the asker's stated goal and level of expertise; whether it is scientifically accurate and does not overstate what is known; and whether it avoids unnecessary length, repetition or digression. Longer is not better by itself.

Question:
<<<
{conv}
>>>

Response A:
<<<
{a}
>>>

Response B:
<<<
{b}
>>>

Answer with a JSON object only: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""

REQUEST_PROMPT = """You are reviewing two candidate responses to the same user request. You do not have a checklist; judge usefulness to THIS user.

Which response would be more useful to the person who made the request? Weigh, in order: whether it does what was actually asked (task, format, constraints); whether it fits this user's stated goal, audience and situation; the quality and correctness of the content itself; and whether it avoids unnecessary length, repetition or digression. Longer is not better by itself.

Request:
<<<
{conv}
>>>

Response A:
<<<
{a}
>>>

Response B:
<<<
{b}
>>>

Answer with a JSON object only: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""

PROMPTS = {
    "medicine": MEDICINE_PROMPT,
    "science": SCIENCE_PROMPT,
    "dialogue": REQUEST_PROMPT,
    "writing": REQUEST_PROMPT,
}
DOMAINS = tuple(PROMPTS)

JUDGE_MAX_TOKENS = 3000
JUDGE_ATTEMPTS = 3

__all__ = [
    "DOMAINS", "PROMPTS", "SCIENCE_PROMPT", "REQUEST_PROMPT", "MEDICINE_PROMPT",
    "bootstrap_interval", "conversation_text", "hash_key", "strip_fence",
    "load_prompts", "load_arm", "parse_named_paths", "read_ids", "build_judge",
    "judge_pair", "resolve_winner",
]


def load_prompts(paths: list[str]) -> dict[str, object]:
    """id -> prompt over one or more questions parquets (id = extra_info.id)."""
    prompts: dict[str, object] = {}
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            prompts[str(extra_info(row).get("id"))] = row["prompt"]
    return prompts


def load_arm(paths: list[str], *, data_source: str | None = None, skip_error_rows: bool = False) -> dict[str, dict]:
    """id -> {"response", "score"} merged over one or more responses files.

    The first file that carries an id wins (an arm evaluated on two suites,
    e.g. Creative-v3 plus WritingBench, is given as two files). ``score`` is
    kept only from rows without an ``error``. With ``skip_error_rows`` (used
    for medicine) rows that carry an ``error`` are dropped entirely. Rows
    whose ``data_source`` is set and differs from ``data_source`` are dropped.
    """
    out: dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if data_source and row.get("data_source") not in (None, data_source):
                    continue
                if not row.get("response") or (skip_error_rows and row.get("error")):
                    continue
                entry = out.setdefault(str(row["id"]), {"response": row["response"], "score": None})
                if entry["score"] is None and not row.get("error") and row.get("score") is not None:
                    entry["score"] = row["score"]
    return out


def parse_named_paths(values: list[str]) -> dict[str, list[str]]:
    """Repeatable NAME=PATH arguments; a repeated NAME appends another file."""
    named: dict[str, list[str]] = {}
    for value in values or []:
        name, sep, path = value.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"expected NAME=PATH, got: {value!r}")
        named.setdefault(name.strip(), []).append(path.strip())
    return named


def read_ids(path: str, sets: str | None = None) -> list[str]:
    """A JSON id list, or the concatenated ``ids`` of the named sets of a probe sample.json."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [str(item) for item in data]
    if not sets:
        raise ValueError(f"{path} is a probe sample; pass --ids-sets (one of: {', '.join(data)})")
    ids: list[str] = []
    for name in sets.split(","):
        ids += [str(item) for item in data[name.strip()]["ids"]]
    return ids


def build_judge(role: str | None, *, qpm: int | None = None, workers: int | None = None):
    """Judge for ``role`` from RUBRIC_JUDGE_[<ROLE>_]* env vars.

    Thinking is forced off for every role except ALT, whose requests carried
    no thinking field (``thinking="enabled"`` makes the client omit it).
    """
    role = role or None
    thinking = "enabled" if role and role.upper() == "ALT" else "disabled"
    return judge_from_env(role, qpm=qpm, max_concurrency=workers, thinking=thinking)


def judge_pair(judge, prompt_text: str) -> tuple[str | None, str | None]:
    """Return ("A" | "B" | "tie", None) or (None, error) after up to three calls."""
    error = None
    for _ in range(JUDGE_ATTEMPTS):
        try:
            completion = judge.complete(
                [{"role": "user", "content": prompt_text}], max_tokens=JUDGE_MAX_TOKENS, temperature=0
            )
        except (JudgeUnavailableError, ValueError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:160]}"
            continue
        try:
            winner = json.loads(strip_fence(completion.raw)).get("winner")
        except Exception as exc:  # noqa: BLE001  (any unparseable answer counts as one failed attempt)
            error = f"parse:{type(exc).__name__}"
            continue
        if winner in ("A", "B", "tie"):
            return winner, None
        error = f"bad:{winner!r}"
    return None, error


def resolve_winner(letter: str, left: str, right: str) -> str:
    """Map the judge's letter back to an arm name (or "tie")."""
    return "tie" if letter == "tie" else (left if letter == "A" else right)
