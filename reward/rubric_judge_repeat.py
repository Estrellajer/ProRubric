# -*- coding: utf-8 -*-
"""Repeat-and-average rubric-judge reward: the reward manager of the Graded arm.

The Graded arm keeps ProRubric's dimensions and asks the training judge for a
0-3 grade per dimension (``RUBRIC_JUDGE_TRAIN_VERDICT_MODE=graded`` in
``judge_client.py``; the dimension score is grade / 3). This manager scores
every training response ``R`` times and averages the weighted scores
(``RUBRIC_JUDGE_REPEAT_JUDGMENTS``, 2 in the paper's runs).

The training judge runs at temperature 0, so repeating a call at the default
settings would return the identical verdict and averaging would buy nothing.
Repeats are therefore drawn at a non-zero temperature
(``RUBRIC_JUDGE_REPEAT_TEMPERATURE``, default 0.7). Validation stays on the
single greedy call so held-out scores remain comparable across reward
variants.
"""

from __future__ import annotations

import asyncio
import contextvars
import math
import os
from collections.abc import Mapping
from statistics import pstdev
from types import SimpleNamespace
from typing import Any

# The base class is aliased on purpose. verl resolves a reward manager with
# `getattr(module, reward_manager.name)`, so a config that names the base class
# while pointing at this module would silently load the base and run the plain
# reward under this arm's name. Not exporting that name turns the
# misconfiguration into an AttributeError at startup.
from rubric_judge import (
    RUBRIC_JUDGE_ROW_SCHEMA,
    GroupRewardDecision,
    JudgeCall,
    RubricJudgeRewardManager as _BaseRubricJudgeRewardManager,
    _config_node,
    _conform_extra_info,
    _result_field,
    _usage_value,
)


# Config/env readers for the two repeat settings.
def _float_setting(
    node: Mapping[str, Any],
    key: str,
    env_name: str,
    default: float,
) -> float:
    value = node.get(key)
    if value is None:
        value = os.getenv(env_name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(0.0, parsed)


def _int_setting(
    node: Mapping[str, Any],
    key: str,
    env_name: str,
    default: int,
) -> int:
    value = node.get(key)
    if value is None:
        value = os.getenv(env_name)
    if value is None:
        return default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


DEFAULT_REPEAT_JUDGMENTS = 4
DEFAULT_REPEAT_TEMPERATURE = 0.7

# Observability-only: per-step aggregates over the repeated draws. Never feed
# back into `scores`; they ride the same reward_extra_info -> events pipeline
# as `rubric_judge_calls` (see `RubricJudgeRewardManager._snapshot_extra_info`).
REPEAT_EXTRA_SCHEMA: tuple[str, ...] = (
    "repeat_judgments",
    "repeat_temperature",
    "repeat_score_spread_mean",
    "repeat_all_agree_fraction",
)
REPEAT_ROW_SCHEMA: tuple[str, ...] = RUBRIC_JUDGE_ROW_SCHEMA + REPEAT_EXTRA_SCHEMA

# Set for the duration of a training group evaluation. The base class calls
# `_score_one` from both the training and validation paths and does not pass the
# flag down, so the repeat behaviour is scoped here rather than by signature.
_REPEAT_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "rubric_judge_repeat_active", default=False
)


class RubricJudgeRepeatRewardManager(_BaseRubricJudgeRewardManager):
    """Whole-rubric reward scored ``R`` times per response and averaged."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        node = _config_node(config, "reward", "rubric_judge")
        self.repeat_judgments = _int_setting(
            node,
            "repeat_judgments",
            "RUBRIC_JUDGE_REPEAT_JUDGMENTS",
            DEFAULT_REPEAT_JUDGMENTS,
        )
        self.repeat_temperature = _float_setting(
            node,
            "repeat_temperature",
            "RUBRIC_JUDGE_REPEAT_TEMPERATURE",
            DEFAULT_REPEAT_TEMPERATURE,
        )
        self._add_metrics(repeat_samples_scored=0.0, repeat_spread_sum=0.0, repeat_all_agree=0.0)
        print(
            "[rubric_judge_repeat] repeat-and-average reward: "
            "%d draws per response at temperature %.2f"
            % (self.repeat_judgments, self.repeat_temperature),
            flush=True,
        )

    async def _evaluate_group_once(
        self, data: Any, *, validation: bool
    ) -> GroupRewardDecision:
        token = _REPEAT_ACTIVE.set(not validation)
        try:
            return await super()._evaluate_group_once(data, validation=validation)
        finally:
            _REPEAT_ACTIVE.reset(token)

    async def _score_one(
        self, payload: JudgeCall, judge: Any, *, temperature: float | None = None
    ) -> Any:
        if not _REPEAT_ACTIVE.get() or self.repeat_judgments <= 1:
            return await _BaseRubricJudgeRewardManager._score_one(
                self, payload, judge, temperature=temperature
            )

        draw_temperature = (
            self.repeat_temperature if temperature is None else temperature
        )
        results = await asyncio.gather(
            *(
                _BaseRubricJudgeRewardManager._score_one(
                    self, payload, judge, temperature=draw_temperature
                )
                for _ in range(self.repeat_judgments)
            )
        )
        return self._merge_draws(results, payload)

    def _merge_draws(self, results: list[Any], payload: JudgeCall) -> Any:
        parsed = [
            _BaseRubricJudgeRewardManager._validate_result(result, payload.rubric)
            for result in results
        ]
        scores = [score for score, _ in parsed]
        mean_score = sum(scores) / len(scores)
        # Strict majority so a split vote never reads as satisfied; the matrix
        # only feeds the negative-criteria diagnostics, not the reward itself.
        votes = zip(*(satisfied for _, satisfied in parsed))
        satisfied = tuple(2 * sum(column) > len(column) for column in votes)

        usage: dict[str, float] = {}
        for result in results:
            raw_usage = _result_field(result, "usage", {}) or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] = usage.get(key, 0.0) + _usage_value(raw_usage, key)

        self._add_metrics(
            repeat_samples_scored=1.0,
            repeat_spread_sum=pstdev(scores) if len(scores) > 1 else 0.0,
            repeat_all_agree=1.0 if len(set(scores)) == 1 else 0.0,
        )
        return SimpleNamespace(
            weighted_score=mean_score,
            satisfied=satisfied,
            raw=_result_field(results[0], "raw", ""),
            usage=usage,
            verdict_mode=_result_field(results[0], "verdict_mode", ""),
            repeat_scores=tuple(scores),
        )

    def _snapshot_extra_info(self, snapshot: Mapping[str, float]) -> dict[str, Any]:
        base = super()._snapshot_extra_info(snapshot)
        samples = snapshot.get("repeat_samples_scored", 0.0)
        base.update(
            {
                "repeat_judgments": float(self.repeat_judgments),
                "repeat_temperature": float(self.repeat_temperature),
                "repeat_score_spread_mean": (
                    snapshot.get("repeat_spread_sum", 0.0) / samples if samples else 0.0
                ),
                "repeat_all_agree_fraction": (
                    snapshot.get("repeat_all_agree", 0.0) / samples if samples else 0.0
                ),
            }
        )
        return base

    def _row_result(self, decision: GroupRewardDecision, index: int) -> dict[str, Any]:
        """Same contract as the base class, widened to `REPEAT_ROW_SCHEMA` so the
        `repeat_*` keys survive the verl reward loop's fixed-schema stacking."""

        score = float(decision.scores[index] or 0.0)
        extra = {
            **decision.reward_extra_info,
            "acc": score,
            "rubric_judge_score": score,
        }
        return {
            "reward_score": score,
            "reward_extra_info": _conform_extra_info(extra, REPEAT_ROW_SCHEMA),
        }
