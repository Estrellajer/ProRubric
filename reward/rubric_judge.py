"""Rubric-judge reward manager for fail-closed GRPO experiments.

The judge implementation lives in the sibling ``judge_client`` module and is
loaded lazily so that CPU-only tests and the fake dry-run do not need
credentials or an endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

from judge_client import RubricJudge, load_judge_env_file

try:
    from verl.experimental.reward_loop.reward_manager import register as _register
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
except ImportError:
    class RewardManagerBase:
        """Small import-time fallback used by isolated unit tests."""

        def __init__(self, config: Any, tokenizer: Any, compute_score: Any = None) -> None:
            self.config = config
            self.tokenizer = tokenizer
            self.compute_score = compute_score

    def _register(name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            return cls

        return decorator


RETRY_DISPOSITION = "retry"
ACCEPT_DISPOSITION = "accept"
SKIPPED_DISPOSITION = "skipped"
FAILED_DISPOSITION = "judge_failed"
DEFAULT_QPM = 500
DEFAULT_TPM = 5_000_000
DEFAULT_MAX_CONCURRENCY = 256
DEFAULT_TRAIN_THINKING = "disabled"
DEFAULT_VAL_THINKING = "enabled"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _config_node(config: Any, *path: str) -> Mapping[str, Any]:
    node: Any = config
    for key in path:
        if hasattr(node, "get"):
            node = node.get(key)
        else:
            node = None
        if node is None:
            return {}
    return _as_mapping(node)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, float(raw))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _config_bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _thinking_mode(value: Any, default: str) -> str:
    if value is None or not str(value).strip():
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled", "enable"}:
        return "enabled"
    if normalized in {"0", "false", "no", "off", "disabled", "disable"}:
        return "disabled"
    raise ValueError("judge thinking mode must be enabled or disabled")


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip():
            return value
    return None


def _usage_value(usage: Any, *keys: str) -> int:
    if not isinstance(usage, Mapping):
        usage = getattr(usage, "__dict__", {})
    for key in keys:
        value = usage.get(key) if isinstance(usage, Mapping) else None
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def _nested_usage_value(usage: Any, key: str) -> int:
    if not isinstance(usage, Mapping):
        return 0
    direct = _usage_value(usage, key)
    if direct:
        return direct
    nested_total = 0
    for value in usage.values():
        if isinstance(value, Mapping):
            nested_total += _nested_usage_value(value, key)
    return nested_total


def _result_field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _normalise_rubric(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("extra_info.rubric must be a list or JSON list.") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("extra_info.rubric must be a list.")
    rubric = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"extra_info.rubric[{index}] must be an object.")
        criterion = item.get("criterion")
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"extra_info.rubric[{index}].criterion must be non-empty.")
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"extra_info.rubric[{index}].weight must be numeric.") from exc
        if not math.isfinite(weight):
            raise ValueError(f"extra_info.rubric[{index}].weight must be finite.")
        # Negative weights are penalty criteria (HealthBench "points" < 0); the
        # judge client subtracts them when the undesirable behaviour is present.
        rubric.append({"criterion": criterion.strip(), "weight": weight})
    if not rubric or sum(max(0.0, item["weight"]) for item in rubric) <= 0:
        raise ValueError("extra_info.rubric must contain a positive total positive weight.")
    return rubric


def _extra_info(data_item: Any) -> dict[str, Any]:
    non_tensor_batch = getattr(data_item, "non_tensor_batch", {})
    value = non_tensor_batch.get("extra_info", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            value = {}
    return dict(value) if isinstance(value, Mapping) else {}


def _group_id(data_item: Any) -> str:
    non_tensor_batch = getattr(data_item, "non_tensor_batch", {})
    for key in ("uid", "group_uid", "prompt_id"):
        value = non_tensor_batch.get(key)
        if value is not None and str(value).strip():
            return str(value)
    extra = _extra_info(data_item)
    value = extra.get("group_uid") or extra.get("prompt_id")
    return str(value) if value is not None else "unknown"


def _problem(data_item: Any, extra_info: Mapping[str, Any]) -> str:
    value = extra_info.get("problem")
    if value is None:
        value = getattr(data_item, "non_tensor_batch", {}).get("problem")
    if value is None:
        value = getattr(data_item, "non_tensor_batch", {}).get("raw_prompt", "")
    if isinstance(value, (list, tuple)):
        text_parts = []
        for item in value:
            if isinstance(item, Mapping):
                text_parts.append(str(item.get("content", "")))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)
    return str(value)


def _reference_answer(data_item: Any, extra_info: Mapping[str, Any]) -> str | None:
    for key in ("reference_answer", "reference", "ideal_answer", "golden_answer"):
        value = extra_info.get(key)
        if value is None:
            value = getattr(data_item, "non_tensor_batch", {}).get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _decode_response(data_item: Any, tokenizer: Any) -> str:
    non_tensor_batch = getattr(data_item, "non_tensor_batch", {})
    direct_response = non_tensor_batch.get("response")
    if isinstance(direct_response, str):
        return direct_response
    if isinstance(direct_response, Sequence) and not isinstance(direct_response, (str, bytes)):
        return "\n".join(str(item) for item in direct_response)

    response_ids = data_item.batch["responses"]
    response_length = response_ids.shape[-1]
    attention_mask = data_item.batch["attention_mask"]
    valid_length = int(attention_mask[-response_length:].sum().item())
    valid_ids = response_ids[:valid_length]
    if tokenizer is None:
        return " ".join(str(item) for item in valid_ids.tolist())
    return str(tokenizer.decode(valid_ids, skip_special_tokens=True))


@dataclass(frozen=True)
class JudgeCall:
    prompt: str
    response: str
    rubric: list[dict[str, Any]]
    reference: str | None = None


@dataclass(frozen=True)
class GroupRewardDecision:
    """A group-level decision with uniform fail-closed handling."""

    group_id: str
    disposition: str
    scores: tuple[float | None, ...]
    reward_extra_info: dict[str, Any]
    judge_results: tuple[Any, ...] = ()
    satisfied_matrix: tuple[tuple[bool, ...], ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def retry(self) -> bool:
        return self.disposition == RETRY_DISPOSITION

    @property
    def judge_failed(self) -> bool:
        return self.disposition == FAILED_DISPOSITION


class RubricJudgeRetryError(RuntimeError):
    """Compatibility exception for callers that still own retry orchestration."""

    disposition = RETRY_DISPOSITION

    def __init__(self, group_id: str = "", message: str = "", *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.group_id = group_id
        self.cause = cause

    def __reduce__(self):  # keep the exception picklable across Ray workers
        return (self.__class__, (self.group_id, str(self)))


class _DryRunFakeJudge:
    """Deterministic local adapter used only by the explicit dry-run flag."""

    def __init__(self, **_: Any) -> None:
        self.calls = 0

    def score(self, prompt: str, response: str, rubric: list[dict[str, Any]]) -> Any:
        self.calls += 1
        response_lower = response.lower()
        satisfied = [criterion["criterion"].lower() in response_lower for criterion in rubric]
        total_weight = sum(float(item["weight"]) for item in rubric)
        weighted_score = (
            sum(float(item["weight"]) * int(ok) for item, ok in zip(rubric, satisfied))
            / total_weight
        )
        return SimpleNamespace(
            satisfied=satisfied,
            weighted_score=weighted_score,
            raw="dry-run-fake-judge",
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split()),
            },
        )


def _scalarize_extra_info(extra: Mapping[str, Any]) -> dict[str, Any]:
    """Verl stacks reward_extra_info values with ``np.array``; ragged sequences
    (per-prompt criterion vectors) break that, so non-scalars are JSON-encoded."""
    out: dict[str, Any] = {}
    for key, value in extra.items():
        if value is None:
            # verl's validation metrics call np.mean over non-string columns;
            # a None column would break that, so encode absence as "".
            out[key] = ""
        elif isinstance(value, (bool, int, float, str)):
            out[key] = value
        else:
            try:
                out[key] = json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                out[key] = str(value)
    return out


def _valid_response_length(data_item: Any) -> int | None:
    """Token length of the generated response, or None when the row carries no tensors."""
    batch = getattr(data_item, "batch", None)
    if batch is None:
        return None
    try:
        response_ids = batch["responses"]
        attention_mask = batch["attention_mask"]
    except (KeyError, TypeError):
        return None
    response_length = response_ids.shape[-1]
    return int(attention_mask[-response_length:].sum().item())


RUBRIC_JUDGE_ROW_SCHEMA: tuple[str, ...] = (
    "acc",
    "rubric_judge_score",
    "rubric_judge_overlong_count",
    "rubric_judge_overlong_penalty_mean",
    "rubric_judge_disposition",
    "rubric_judge_group_id",
    "judge_failed",
    "rubric_judge_skipped",
    "rubric_judge_satisfied",
    "rubric_judge_error",
    "rubric_judge_calls",
    "rubric_judge_input_tokens",
    "rubric_judge_output_tokens",
    "rubric_judge_reasoning_tokens",
    "rubric_judge_total_tokens",
    "rubric_judge_failures",
    "rubric_judge_failed_groups",
)

_STRING_EXTRA_KEYS = frozenset(
    {
        "rubric_judge_disposition",
        "rubric_judge_group_id",
        "rubric_judge_satisfied",
        "rubric_judge_error",
        "serpo_disposition",
        "serpo_group_id",
        "serpo_query_id",
        "serpo_error",
        "serpo_stats_json",
    }
)


def _conform_extra_info(extra: Mapping[str, Any], schema: Sequence[str]) -> dict[str, Any]:
    """Return ``extra`` restricted/extended to ``schema`` with typed defaults.

    Verl concatenates per-row ``reward_extra_info`` across workers and asserts
    identical key sets, so every row a manager emits must carry the same keys.
    """
    scalar = _scalarize_extra_info(extra)
    out: dict[str, Any] = {}
    for key in schema:
        if key in scalar:
            out[key] = scalar[key]
        else:
            out[key] = "" if key in _STRING_EXTRA_KEYS else 0.0
    return out


@_register("rubric_judge")
class RubricJudgeRewardManager(RewardManagerBase):
    """Async Verl reward manager backed by the W1 ``RubricJudge`` contract.

    The manager evaluates complete prompt groups when the native reward loop
    provides them. A failed group is retried once as a whole. If that retry
    also fails, every member receives the same zero score and a
    ``judge_failed=1`` marker, which gives GRPO a zero-variance advantage
    without turning an unavailable judge into a fabricated per-row score.
    """

    group_aware = True

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        compute_score: Any = None,
        reward_router_address: str | None = None,
        reward_model_tokenizer: Any = None,
        *,
        judge: Any = None,
        judge_factory: Callable[..., Any] | None = None,
        max_concurrency: int | None = None,
        dry_run_fake_judge: bool | None = None,
        **_: Any,
    ) -> None:
        del compute_score, reward_router_address, reward_model_tokenizer
        super().__init__(config, tokenizer, None)
        judge_config = _config_node(config, "reward", "rubric_judge")
        reward_kwargs = _config_node(config, "reward", "reward_kwargs")
        # DAPO-style overlong reward shaping (RISE-RL / RubricHub recipe): responses longer than
        # max_response_length - overlong_buffer_len lose up to overlong_penalty_factor linearly.
        # Disabled unless overlong_buffer_len > 0. Training rows only; validation keeps the pure score.
        data_config = _config_node(config, "data")
        self.overlong_buffer_len = int(
            judge_config.get("overlong_buffer_len") or _env_int("RUBRIC_JUDGE_OVERLONG_BUFFER_LEN", 0, minimum=0)
        )
        self.overlong_penalty_factor = float(
            judge_config.get("overlong_penalty_factor") or os.getenv("RUBRIC_JUDGE_OVERLONG_PENALTY_FACTOR") or 0.5
        )
        self.overlong_max_response_length = int(
            judge_config.get("overlong_max_response_length") or data_config.get("max_response_length") or 0
        )
        if self.overlong_buffer_len > 0 and self.overlong_max_response_length <= self.overlong_buffer_len:
            raise ValueError(
                "overlong_buffer_len must be smaller than data.max_response_length "
                f"(got {self.overlong_buffer_len} vs {self.overlong_max_response_length})"
            )
        self.max_concurrency = max(
            1,
            int(
                max_concurrency
                or judge_config.get("max_concurrency")
                or _env_int("RUBRIC_JUDGE_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
            ),
        )
        self.score_train = _config_bool(
            judge_config,
            "score_train",
            _config_bool(
                reward_kwargs,
                "score_train",
                _env_bool("RUBRIC_JUDGE_SCORE_TRAIN", True),
            ),
        )
        self.train_thinking = _thinking_mode(
            _first_nonempty(
                judge_config.get("train_thinking"),
                os.getenv("RUBRIC_JUDGE_TRAIN_THINKING"),
            ),
            DEFAULT_TRAIN_THINKING,
        )
        self.val_thinking = _thinking_mode(
            _first_nonempty(
                judge_config.get("val_thinking"),
                os.getenv("RUBRIC_JUDGE_VAL_THINKING"),
            ),
            DEFAULT_VAL_THINKING,
        )
        # Per-call wall clock *including* time spent waiting in the shared
        # QPM/TPM limiter (a 2048-call step at 300 QPM queues for ~7 min), so
        # this must be far larger than one HTTP request; the HTTP timeout is
        # a separate, smaller budget passed to the client.
        self.timeout = _env_float(
            "RUBRIC_JUDGE_TIMEOUT_SECONDS",
            float(judge_config.get("timeout_seconds", 3600.0)),
            minimum=0.001,
        )
        self.http_timeout = _env_float(
            "RUBRIC_JUDGE_HTTP_TIMEOUT_SECONDS",
            float(judge_config.get("http_timeout_seconds", 180.0)),
            minimum=0.001,
        )
        # NOTE: hydra `${oc.env:...,false}` yields the *string* "false"; a bare
        # bool() on it silently switched production runs to the fake judge.
        fake = (
            dry_run_fake_judge
            if dry_run_fake_judge is not None
            else _config_bool(judge_config, "dry_run_fake_judge", False)
            or os.getenv("RUBRIC_JUDGE_DRY_RUN_FAKE", "").lower() in {"1", "true", "yes"}
        )
        if fake:
            print("[rubric_judge] WARNING: dry-run fake judge is active; rewards are NOT real judge scores", flush=True)
        if judge is not None:
            self.judge = judge
            self._train_judge = judge
            self._train_judges = [judge]
            self._val_judge = judge
        else:
            # Separate judge models per role: a cheaper proxy for training
            # rewards and the reference judge for validation (the RGSD paper's
            # gpt-4o-mini / gpt-5.4 split). Each model has its own quota.
            self.train_use_shared_settings = _config_bool(
                judge_config, "train_use_shared_settings", False
            )
            self.train_model = (
                _first_nonempty(
                    judge_config.get("model"), os.getenv("RUBRIC_JUDGE_MODEL")
                )
                if self.train_use_shared_settings
                else _first_nonempty(
                    judge_config.get("train_model"),
                    os.getenv("RUBRIC_JUDGE_TRAIN_MODEL"),
                )
            )
            self.val_model = _first_nonempty(
                judge_config.get("val_model"), os.getenv("RUBRIC_JUDGE_VAL_MODEL")
            )
            self.train_verdict_mode = str(
                _first_nonempty(
                    judge_config.get("train_verdict_mode"),
                    judge_config.get("verdict_mode"),
                    os.getenv("RUBRIC_JUDGE_TRAIN_VERDICT_MODE"),
                    os.getenv("RUBRIC_JUDGE_VERDICT_MODE"),
                    "hard",
                )
            ).strip().lower()
            # A comma-separated train_model shards training groups across
            # several judge deployments (useful when each deployment is
            # latency-bound): every response of a prompt group is
            # scored by the same judge, chosen by a stable hash of the group
            # id, so GRPO's within-group comparison never mixes judges.
            train_models = [
                item.strip() for item in str(self.train_model or "").split(",") if item.strip()
            ] or [self.train_model]
            self._train_judges = [
                self._make_judge(
                    judge_config,
                    fake=fake,
                    judge_factory=judge_factory,
                    thinking=self.train_thinking,
                    max_concurrency=self.max_concurrency,
                    timeout=self.http_timeout,
                    model=train_model,
                    verdict_mode=self.train_verdict_mode,
                    role="train",
                )
                for train_model in train_models
            ]
            self._train_judge = self._train_judges[0]
            self._val_judge = self._make_judge(
                judge_config,
                fake=fake,
                judge_factory=judge_factory,
                thinking=self.val_thinking,
                max_concurrency=self.max_concurrency,
                timeout=self.http_timeout,
                model=self.val_model,
                verdict_mode="hard",
                role="val",
            )
            self.judge = self._train_judge
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, float] = {
            "judge_calls": 0.0,
            "judge_input_tokens": 0.0,
            "judge_output_tokens": 0.0,
            "judge_reasoning_tokens": 0.0,
            "judge_total_tokens": 0.0,
            "judge_failures": 0.0,
            "groups_retry": 0.0,
            "groups_accepted": 0.0,
            "judge_failed_groups": 0.0,
            "groups_skipped": 0.0,
        }

    @staticmethod
    def _make_judge(
        judge_config: Mapping[str, Any],
        *,
        fake: bool,
        judge_factory: Callable[..., Any] | None,
        thinking: str,
        max_concurrency: int,
        timeout: float,
        model: str | None = None,
        verdict_mode: str | None = None,
        role: str = "",
    ) -> Any:
        if fake:
            return _DryRunFakeJudge()
        factory = judge_factory or RubricJudge
        # Load the credential file before settings are resolved: a role
        # override such as RUBRIC_JUDGE_TRAIN_API_KEY that lives in the file
        # would otherwise be missed and the shared key sent to the role's
        # endpoint (which then rejects it with HTTP 401).
        try:
            load_judge_env_file()
        except OSError:
            pass

        # Per-role overrides (``train_base_url`` / ``RUBRIC_JUDGE_TRAIN_BASE_URL``
        # ...) let the training reward run on a different endpoint and quota
        # than validation; anything unset falls back to the shared setting.
        use_shared_settings = (
            role == "train"
            and _config_bool(judge_config, "train_use_shared_settings", False)
        )

        def setting(key: str) -> Any:
            env_name = "RUBRIC_JUDGE_" + key.upper()
            if role and not use_shared_settings:
                role_env = "RUBRIC_JUDGE_" + role.upper() + "_" + key.upper()
                value = _first_nonempty(
                    judge_config.get(f"{role}_{key}"), os.getenv(role_env)
                )
                if value is not None:
                    return value
            return _first_nonempty(judge_config.get(key), os.getenv(env_name))

        kwargs: dict[str, Any] = {
            "qpm": int(setting("qpm") or DEFAULT_QPM),
            "tpm": int(setting("tpm") or DEFAULT_TPM),
            "max_concurrency": max_concurrency,
            "thinking": thinking,
            "timeout": timeout,
        }
        max_tokens = setting("max_tokens")
        if max_tokens:
            kwargs["max_tokens"] = int(max_tokens)
        for key in ("base_url", "api_key", "model"):
            value = setting(key)
            if key == "model" and model:
                value = model
            if value is None and key == "model":
                role_var = f"RUBRIC_JUDGE_{role.upper()}_MODEL" if role else "RUBRIC_JUDGE_MODEL"
                raise ValueError(
                    f"no {role or 'shared'} judge model configured: set {role_var} "
                    "or RUBRIC_JUDGE_MODEL (reward.rubric_judge.model also works)"
                )
            if value is not None:
                kwargs[key] = value
        if verdict_mode:
            kwargs["verdict_mode"] = verdict_mode
        return factory(**kwargs)

    def metrics_snapshot(self) -> dict[str, float]:
        with self._metrics_lock:
            return dict(self._metrics)

    def _add_metrics(self, **values: float) -> None:
        with self._metrics_lock:
            for key, value in values.items():
                self._metrics[key] = self._metrics.get(key, 0.0) + float(value)

    def _current_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    def _judge_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        # ``asyncio.to_thread`` runs on the loop's default executor, whose pool
        # is capped at min(32, cpu+4) threads: with ~1.5 s per judge call that
        # caps every reward worker at ~1,000 calls/min no matter what
        # max_concurrency says, well below what a judge endpoint can serve at
        # 256 concurrent requests. Judge calls get their own pool sized to
        # max_concurrency.
        executor = getattr(self, "_executor", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrency, thread_name_prefix="rubric-judge"
            )
            self._executor = executor
        return executor

    _debug_payloads_left = 3
    _debug_failures_left = 20

    def _call_payload(self, data_item: Any) -> JudgeCall:
        extra_info = _extra_info(data_item)
        call = JudgeCall(
            prompt=_problem(data_item, extra_info),
            response=_decode_response(data_item, self.tokenizer),
            rubric=_normalise_rubric(extra_info.get("rubric")),
            reference=_reference_answer(data_item, extra_info),
        )
        if RubricJudgeRewardManager._debug_payloads_left > 0:
            RubricJudgeRewardManager._debug_payloads_left -= 1
            keys = sorted(getattr(data_item, "non_tensor_batch", {}).keys())
            print(
                "[rubric_judge] payload sample:"
                f" prompt={call.prompt[:160]!r} response={call.response[:200]!r}"
                f" n_criteria={len(call.rubric)} non_tensor_keys={keys}",
                flush=True,
            )
        return call

    @staticmethod
    def _is_validation(data: Any) -> bool:
        meta_info = getattr(data, "meta_info", None)
        return isinstance(meta_info, Mapping) and bool(meta_info.get("validate"))

    def _judge_for(self, validation: bool, group_id: str = "") -> Any:
        if validation:
            return self._val_judge
        judges = getattr(self, "_train_judges", None) or [self._train_judge]
        if len(judges) == 1:
            return judges[0]
        digest = hashlib.sha1(str(group_id).encode("utf-8")).hexdigest()
        return judges[int(digest, 16) % len(judges)]

    def _snapshot_extra_info(self, snapshot: Mapping[str, float]) -> dict[str, Any]:
        return {
            "rubric_judge_calls": snapshot["judge_calls"],
            "rubric_judge_input_tokens": snapshot["judge_input_tokens"],
            "rubric_judge_output_tokens": snapshot["judge_output_tokens"],
            "rubric_judge_reasoning_tokens": snapshot["judge_reasoning_tokens"],
            "rubric_judge_total_tokens": snapshot["judge_total_tokens"],
            "rubric_judge_failures": snapshot["judge_failures"],
            "rubric_judge_failed_groups": snapshot["judge_failed_groups"],
        }

    @staticmethod
    def _disposition_markers(
        *,
        judge_failed: int = 0,
        skipped: int = 0,
        satisfied: Sequence[Sequence[bool]] = (),
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "judge_failed": judge_failed,
            "rubric_judge_skipped": skipped,
            "rubric_judge_satisfied": tuple(tuple(row) for row in satisfied),
            "rubric_judge_error": error,
        }

    @staticmethod
    def _validate_result(result: Any, rubric: list[dict[str, Any]]) -> tuple[float, tuple[bool, ...]]:
        score = _result_field(result, "weighted_score")
        try:
            score = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError("RubricJudge returned a non-numeric weighted_score.") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"RubricJudge weighted_score must be in [0, 1], got {score!r}.")

        satisfied = _result_field(result, "satisfied")
        if not isinstance(satisfied, Sequence) or isinstance(satisfied, (str, bytes)):
            raise ValueError("RubricJudge returned a non-sequence satisfied field.")
        verdict_mode = str(_result_field(result, "verdict_mode", "") or "").lower()
        if len(satisfied) == 0 and verdict_mode in {"likert", "reference_likert"}:
            return score, ()
        if len(satisfied) != len(rubric) or any(not isinstance(item, bool) for item in satisfied):
            raise ValueError("RubricJudge satisfied must contain one boolean per rubric criterion.")
        return score, tuple(satisfied)

    _inflight = 0
    _inflight_peak = 0
    _inflight_lock = threading.Lock()

    async def _score_one(
        self, payload: JudgeCall, judge: Any, *, temperature: float | None = None
    ) -> Any:
        async with self._current_semaphore():
            self._add_metrics(judge_calls=1)
            with RubricJudgeRewardManager._inflight_lock:
                RubricJudgeRewardManager._inflight += 1
                RubricJudgeRewardManager._inflight_peak = max(
                    RubricJudgeRewardManager._inflight_peak, RubricJudgeRewardManager._inflight
                )
            try:
                score_method = getattr(judge, "score", None)
                if not callable(score_method):
                    raise TypeError("RubricJudge must expose score(prompt, response, rubric).")
                extra: dict[str, Any] = {}
                if payload.reference is not None:
                    extra["reference"] = payload.reference
                if temperature is not None:
                    # Only passed when a caller explicitly wants sampled draws
                    # (the Graded arm's repeat-and-average manager).
                    extra["temperature"] = temperature
                call = lambda: score_method(
                    payload.prompt, payload.response, payload.rubric, **extra
                )
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._judge_executor(), call),
                    timeout=self.timeout,
                )
                usage = _result_field(result, "usage", {})
                input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens", "prompt_token_count")
                output_tokens = _usage_value(
                    usage,
                    "completion_tokens",
                    "output_tokens",
                    "completion_token_count",
                )
                reasoning_tokens = _nested_usage_value(usage, "reasoning_tokens")
                total_tokens = _usage_value(usage, "total_tokens", "total_token_count")
                if total_tokens == 0:
                    total_tokens = input_tokens + output_tokens + reasoning_tokens
                self._add_metrics(
                    judge_input_tokens=input_tokens,
                    judge_output_tokens=output_tokens,
                    judge_reasoning_tokens=reasoning_tokens,
                    judge_total_tokens=total_tokens,
                )
                self._validate_result(result, payload.rubric)
                return result
            except Exception:
                self._add_metrics(judge_failures=1)
                raise
            finally:
                with RubricJudgeRewardManager._inflight_lock:
                    RubricJudgeRewardManager._inflight -= 1

    async def _evaluate_group_once(self, data: Any, *, validation: bool) -> GroupRewardDecision:
        """Run one complete attempt for a group."""

        if len(data) == 0:
            raise ValueError("cannot judge an empty prompt group")
        group_id = _group_id(data[0])
        payloads: list[JudgeCall] = []
        try:
            payloads = [self._call_payload(data[index]) for index in range(len(data))]
        except Exception as exc:
            self._add_metrics(judge_failures=1)
            snapshot = self.metrics_snapshot()
            return GroupRewardDecision(
                group_id=group_id,
                disposition=RETRY_DISPOSITION,
                scores=tuple(None for _ in range(len(data))),
                reward_extra_info={
                    "rubric_judge_disposition": RETRY_DISPOSITION,
                    "rubric_judge_group_id": group_id,
                    **self._disposition_markers(error=f"{type(exc).__name__}: {exc}"),
                    "rubric_judge_calls": snapshot["judge_calls"],
                    "rubric_judge_failures": snapshot["judge_failures"],
                    "rubric_judge_error": f"{type(exc).__name__}: {exc}",
                },
                metrics=snapshot,
            )

        judge = self._judge_for(validation, group_id)
        outcomes = await asyncio.gather(
            *(self._score_one(payload, judge) for payload in payloads),
            return_exceptions=True,
        )
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            snapshot = self.metrics_snapshot()
            error = failures[0]
            if RubricJudgeRewardManager._debug_failures_left > 0:
                RubricJudgeRewardManager._debug_failures_left -= 1
                print(
                    f"[rubric_judge] group {group_id} judge failure ({len(failures)}/{len(outcomes)}):"
                    f" {type(error).__name__}: {str(error)[:400]}",
                    flush=True,
                )
            return GroupRewardDecision(
                group_id=group_id,
                disposition=RETRY_DISPOSITION,
                scores=tuple(None for _ in payloads),
                reward_extra_info={
                    "rubric_judge_disposition": RETRY_DISPOSITION,
                    "rubric_judge_group_id": group_id,
                    **self._disposition_markers(error=f"{type(error).__name__}: {error}"),
                    "rubric_judge_calls": snapshot["judge_calls"],
                    "rubric_judge_failures": snapshot["judge_failures"],
                    "rubric_judge_error": f"{type(error).__name__}: {error}",
                },
                metrics=snapshot,
            )

        results = tuple(outcomes)
        validated = tuple(
            self._validate_result(result, payload.rubric)
            for result, payload in zip(results, payloads)
        )
        scores = tuple(score for score, _ in validated)
        satisfied_matrix = tuple(satisfied for _, satisfied in validated)
        overlong_info: dict[str, Any] = {}
        if not validation:
            penalties = self._overlong_penalties(data)
            penalized = [penalty for penalty in penalties if penalty < 0.0]
            if penalized:
                scores = tuple(
                    (score + penalty) if score is not None else score
                    for score, penalty in zip(scores, penalties)
                )
                self._add_metrics(overlong_rows=len(penalized), overlong_penalty_sum=-sum(penalized))
                overlong_info = {
                    "rubric_judge_overlong_count": float(len(penalized)),
                    "rubric_judge_overlong_penalty_mean": float(sum(penalized) / len(penalties)),
                }
        self._add_metrics(groups_accepted=1)
        snapshot = self.metrics_snapshot()
        return GroupRewardDecision(
            group_id=group_id,
            disposition=ACCEPT_DISPOSITION,
            scores=scores,
            reward_extra_info={
                "rubric_judge_disposition": ACCEPT_DISPOSITION,
                "rubric_judge_group_id": group_id,
                **self._disposition_markers(satisfied=satisfied_matrix),
                **self._snapshot_extra_info(snapshot),
                **overlong_info,
            },
            judge_results=results,
            satisfied_matrix=satisfied_matrix,
            metrics=snapshot,
        )

    def _overlong_penalties(self, data: Any) -> tuple[float, ...]:
        """DAPO overlong shaping: min(-(len - (max - buffer)) / buffer * factor, 0) per row."""
        n_rows = len(data)
        if self.overlong_buffer_len <= 0 or self.overlong_max_response_length <= 0:
            return tuple(0.0 for _ in range(n_rows))
        expected_len = self.overlong_max_response_length - self.overlong_buffer_len
        penalties = []
        for index in range(n_rows):
            length = _valid_response_length(data[index])
            if length is None:
                penalties.append(0.0)
                continue
            exceed = length - expected_len
            penalties.append(min(-exceed / self.overlong_buffer_len * self.overlong_penalty_factor, 0.0))
        return tuple(penalties)

    def _skipped_group(self, data: Any) -> GroupRewardDecision:
        group_id = _group_id(data[0])
        self._add_metrics(groups_skipped=1)
        snapshot = self.metrics_snapshot()
        return GroupRewardDecision(
            group_id=group_id,
            disposition=SKIPPED_DISPOSITION,
            scores=tuple(0.0 for _ in range(len(data))),
            reward_extra_info={
                "rubric_judge_disposition": SKIPPED_DISPOSITION,
                "rubric_judge_group_id": group_id,
                **self._disposition_markers(skipped=1),
                **self._snapshot_extra_info(snapshot),
            },
            metrics=snapshot,
        )

    async def evaluate_group(
        self,
        data: Any,
        *,
        validation: bool | None = None,
    ) -> GroupRewardDecision:
        """Judge a complete group, retrying it once after a failed attempt."""

        if len(data) == 0:
            raise ValueError("cannot judge an empty prompt group")
        if validation is None:
            validation = self._is_validation(data)
        if not validation and not self.score_train:
            return self._skipped_group(data)

        decision = await self._evaluate_group_once(data, validation=validation)
        if decision.disposition == ACCEPT_DISPOSITION:
            return decision

        self._add_metrics(groups_retry=1)
        decision = await self._evaluate_group_once(data, validation=validation)
        if decision.disposition == ACCEPT_DISPOSITION:
            return decision

        self._add_metrics(judge_failed_groups=1)
        snapshot = self.metrics_snapshot()
        error = decision.reward_extra_info.get("rubric_judge_error", "judge unavailable")
        return GroupRewardDecision(
            group_id=decision.group_id,
            disposition=FAILED_DISPOSITION,
            scores=tuple(0.0 for _ in range(len(data))),
            reward_extra_info={
                "rubric_judge_disposition": FAILED_DISPOSITION,
                "rubric_judge_group_id": decision.group_id,
                **self._disposition_markers(judge_failed=1, error=error),
                "rubric_judge_error": error,
                **self._snapshot_extra_info(snapshot),
            },
            metrics=snapshot,
        )

    async def evaluate_groups(self, groups: Sequence[Any]) -> list[GroupRewardDecision]:
        """Evaluate complete prompt groups concurrently."""

        return list(
            await asyncio.gather(
                *(self.evaluate_group(group) for group in groups),
            )
        )

    def _row_result(self, decision: GroupRewardDecision, index: int) -> dict[str, Any]:
        score = float(decision.scores[index] or 0.0)
        extra = {
            **decision.reward_extra_info,
            "acc": score,
            "rubric_judge_score": score,
        }
        return {
            "reward_score": score,
            "reward_extra_info": _conform_extra_info(extra, RUBRIC_JUDGE_ROW_SCHEMA),
        }

    async def run_single(self, data: Any) -> dict[str, Any]:
        decision = await self.evaluate_group(data)
        return self._row_result(decision, len(decision.scores) - 1)

    @staticmethod
    def _group_indices(data: Any) -> list[list[int]]:
        group_keys = ("uid", "group_uid", "prompt_id")
        batch_non_tensor = getattr(data, "non_tensor_batch", {})
        has_batch_group_key = any(key in batch_non_tensor for key in group_keys)
        has_row_group_key = False
        if not has_batch_group_key:
            for index in range(len(data)):
                row_non_tensor = getattr(data[index], "non_tensor_batch", {})
                if any(key in row_non_tensor for key in group_keys):
                    has_row_group_key = True
                    break
        if not has_batch_group_key and not has_row_group_key:
            return [[index] for index in range(len(data))]
        indices_by_group: dict[str, list[int]] = {}
        for index in range(len(data)):
            group_id = _group_id(data[index])
            indices_by_group.setdefault(group_id, []).append(index)
        return list(indices_by_group.values())

    @staticmethod
    def _select_rows(data: Any, indices: list[int]) -> Any:
        if len(indices) == 1:
            index = indices[0]
            return data[index : index + 1]
        if isinstance(data, list):
            return [data[index] for index in indices]
        return data[indices]

    async def run_group_batch(self, data: Any) -> list[dict[str, Any]]:
        """Return per-row outputs while preserving complete group semantics."""

        groups = self._group_indices(data)
        validation = self._is_validation(data)
        started = time.monotonic()
        calls_before = self.metrics_snapshot()["judge_calls"]
        with RubricJudgeRewardManager._inflight_lock:
            RubricJudgeRewardManager._inflight_peak = 0
        decisions = await asyncio.gather(
            *(
                self.evaluate_group(
                    self._select_rows(data, indices),
                    validation=validation,
                )
                for indices in groups
            )
        )
        elapsed = time.monotonic() - started
        snapshot = self.metrics_snapshot()
        judge = self._judge_for(validation, "")
        print(
            "[rubric_judge] batch timing: rows=%d groups=%d calls=%d elapsed=%.1fs "
            "peak_inflight=%d semaphore=%d adaptive_limit=%s limiter=%s"
            % (
                len(data),
                len(groups),
                snapshot["judge_calls"] - calls_before,
                elapsed,
                RubricJudgeRewardManager._inflight_peak,
                self.max_concurrency,
                getattr(judge, "concurrency_limit", None),
                getattr(judge, "rate_limit_snapshot", None),
            ),
            flush=True,
        )
        row_results: list[dict[str, Any] | None] = [None] * len(data)
        if len(groups) != len(decisions):
            raise RuntimeError("rubric judge group batch decision count mismatch")
        for indices, decision in zip(groups, decisions):
            for position, index in enumerate(indices):
                row_results[index] = self._row_result(decision, position)
        if any(result is None for result in row_results):
            raise RuntimeError("rubric judge group batch lost a row result")
        return [result for result in row_results if result is not None]

    def __call__(self, data: Any, return_dict: bool = False) -> Any:
        """Compatibility adapter for the legacy synchronous reward surface."""

        result = self._run_sync(self.run_single(data[-1:]))
        if return_dict:
            return result
        return result["reward_score"]

    @staticmethod
    def _run_sync(coroutine: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        raise RuntimeError("RubricJudgeRewardManager.__call__ cannot run inside an active event loop.")


def _dry_run_payload() -> dict[str, Any]:
    return {
        "extra_info": {
            "problem": "Name one criterion that appears in this response.",
            "rubric": [{"criterion": "criterion", "weight": 1.0}],
        },
        "response": "criterion",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-fake-judge", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run_fake_judge:
        parser.error("only --dry-run-fake-judge is supported by this CPU-safe path")
    judge = _DryRunFakeJudge()
    payload = _dry_run_payload()
    result = judge.score(payload["extra_info"]["problem"], payload["response"], payload["extra_info"]["rubric"])
    print(
        json.dumps(
            {
                "disposition": ACCEPT_DISPOSITION,
                "weighted_score": _result_field(result, "weighted_score"),
                "usage": _result_field(result, "usage", {}),
                "judge_calls": judge.calls,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
