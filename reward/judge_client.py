"""OpenAI-compatible rubric judge with rolling-window throttling.

Configuration is read from environment variables (constructor kwargs take
precedence):

    RUBRIC_JUDGE_BASE_URL   chat-completions endpoint, e.g.
                            https://api.openai.com/v1 (the /chat/completions
                            suffix is appended automatically)
    RUBRIC_JUDGE_API_KEY    bearer token sent as Authorization: Bearer <key>
    RUBRIC_JUDGE_MODEL      model id (required; there is no default model)
    RUBRIC_JUDGE_QPM        requests-per-minute rolling-window limit
    RUBRIC_JUDGE_TPM        tokens-per-minute rolling-window limit
    RUBRIC_JUDGE_MAX_CONCURRENCY
    RUBRIC_JUDGE_THINKING   enabled|disabled (provider-specific thinking toggle)
    RUBRIC_JUDGE_VERDICT_MODE  hard|graded|self_prob|likert|reference_likert
    RUBRIC_JUDGE_ENV_FILE   optional path to a KEY=VALUE credential file

Per-role overrides (RUBRIC_JUDGE_<ROLE>_BASE_URL / _API_KEY / _MODEL / ...)
are resolved by ``judge_from_env(role)`` and by the reward manager: a role
setting wins, anything unset falls back to the shared RUBRIC_JUDGE_* value.
Roles used in this package:

    TRAIN   training-reward judge          (paper: Doubao-mini)
    VAL     in-training validation judge   (paper: Doubao-lite)
    (none)  offline evaluation judge and dimension generator
                                           (paper: DeepSeek-V4-Pro)
    LITE    second, same-family evaluator  (paper: Doubao-lite)
    ALT     third-family evaluator         (paper: GPT-5.6-luna)
    EDIT    perturbation writer of the incentive audit (paper: Doubao-lite)
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_QPM = 500
DEFAULT_TPM = 5_000_000
DEFAULT_MAX_CONCURRENCY = 256
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120.0


class JudgeUnavailableError(RuntimeError):
    """Raised when a rubric score cannot be obtained and validated."""


class JudgeHTTPError(RuntimeError):
    """Raised for an HTTP response that cannot be used by the judge."""

    def __init__(self, status_code: int) -> None:
        self.status_code = int(status_code)
        super().__init__(f"judge HTTP status {self.status_code}")


@dataclass(frozen=True)
class JudgeResult:
    satisfied: list[bool]
    weighted_score: float
    raw: str
    usage: dict[str, Any]
    probabilities: list[float] | None = None
    rating: float | None = None
    verdict_mode: str = "hard"


@dataclass(frozen=True)
class JudgeCompletion:
    """Raw JSON completion returned by the shared judge endpoint."""

    raw: str
    usage: dict[str, Any]


@dataclass(frozen=True)
class RubricScoreInput:
    prompt: str | Sequence[Mapping[str, Any]]
    response: str
    rubric: list[dict[str, Any]]
    reference: str | None = None


class RollingTokenRateLimiter:
    """Reserve request capacity under QPM and TPM rolling windows.

    Reservations are intentionally conservative: a request reserves an
    estimated prompt plus completion budget before it is sent. This prevents a
    burst of concurrent requests from exceeding the token quota while keeping
    the implementation independent of provider-specific tokenizers.
    """

    def __init__(
        self,
        *,
        qpm: int = DEFAULT_QPM,
        tpm: int = DEFAULT_TPM,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if qpm < 1 or tpm < 1:
            raise ValueError("qpm and tpm must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.qpm = int(qpm)
        self.tpm = int(tpm)
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._sleep = sleep_fn
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()

    def _used_tokens(self) -> int:
        return sum(tokens for _, tokens in self._events)

    def reserve(self, estimated_tokens: int) -> None:
        estimated_tokens = max(1, int(estimated_tokens))
        if estimated_tokens > self.tpm:
            raise ValueError(
                f"one request reserves {estimated_tokens} tokens, above TPM limit {self.tpm}"
            )
        while True:
            delay = 0.0
            with self._lock:
                now = self._clock()
                self._prune(now)
                used_tokens = self._used_tokens()
                if len(self._events) < self.qpm and used_tokens + estimated_tokens <= self.tpm:
                    self._events.append((now, estimated_tokens))
                    return

                if self._events:
                    oldest_time = self._events[0][0]
                    delay = max(0.001, oldest_time + self.window_seconds - now)
                else:
                    delay = 0.001
            self._sleep(delay)

    @property
    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self._prune(self._clock())
            return {
                "requests": len(self._events),
                "tokens": self._used_tokens(),
                "qpm": self.qpm,
                "tpm": self.tpm,
            }


class AdaptiveConcurrency:
    """A small AIMD-style gate that reacts to throttling responses."""

    def __init__(self, maximum: int, *, initial: int | None = None) -> None:
        if maximum < 1:
            raise ValueError("maximum concurrency must be positive")
        self.maximum = int(maximum)
        self.limit = min(self.maximum, max(1, int(initial or maximum)))
        self.active = 0
        self._successes = 0
        self._condition = threading.Condition()

    def acquire(self) -> None:
        with self._condition:
            while self.active >= self.limit:
                self._condition.wait()
            self.active += 1

    def release(self, *, success: bool, throttled: bool = False) -> None:
        with self._condition:
            self.active = max(0, self.active - 1)
            if throttled:
                self.limit = max(1, self.limit // 2)
                self._successes = 0
            elif success:
                self._successes += 1
                if self._successes >= max(4, self.limit) and self.limit < self.maximum:
                    self.limit += 1
                    self._successes = 0
            self._condition.notify_all()


def _message_text(prompt: str | Sequence[Mapping[str, Any]]) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(list(prompt), ensure_ascii=False, sort_keys=True)


def load_judge_env_file(path: str | None = None, *, environ: Any = None) -> int:
    """Load ``KEY=VALUE`` lines from ``RUBRIC_JUDGE_ENV_FILE`` into the process env.

    Existing environment variables are never overridden; the number of newly
    set variables is returned. A missing path is a no-op.
    """
    env = os.environ if environ is None else environ
    target = path if path is not None else env.get("RUBRIC_JUDGE_ENV_FILE")
    if not target:
        return 0
    loaded = 0
    with open(target, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            if key and key not in env:
                env[key] = value
                loaded += 1
    return loaded


def role_setting(key: str, role: str | None = None) -> str | None:
    """``RUBRIC_JUDGE_<ROLE>_<KEY>`` if set and non-empty, else ``RUBRIC_JUDGE_<KEY>``."""
    load_judge_env_file()
    if role:
        value = os.environ.get(f"RUBRIC_JUDGE_{role.upper()}_{key.upper()}")
        if value is not None and value.strip():
            return value
    value = os.environ.get(f"RUBRIC_JUDGE_{key.upper()}")
    return value if value is not None and value.strip() else None


def judge_from_env(role: str | None = None, **kwargs: Any) -> "RubricJudge":
    """Build a ``RubricJudge`` for one role from the environment.

    Endpoint, key, model, rate limits and thinking toggle are read per role
    (see the module docstring); explicit keyword arguments win. A missing
    model raises with the name of the variable to set.
    """
    resolved: dict[str, Any] = {}
    for key in ("base_url", "api_key", "model", "thinking"):
        value = role_setting(key, role)
        if value is not None:
            resolved[key] = value
    for key in ("qpm", "tpm", "max_concurrency", "max_tokens"):
        value = role_setting(key, role)
        if value is not None:
            resolved[key] = int(value)
    if "model" not in resolved and "model" not in kwargs:
        name = f"RUBRIC_JUDGE_{role.upper()}_MODEL" if role else "RUBRIC_JUDGE_MODEL"
        raise ValueError(f"no judge model configured: set {name} (or RUBRIC_JUDGE_MODEL)")
    resolved.update({k: v for k, v in kwargs.items() if v is not None})
    return RubricJudge(**resolved)


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _normalise_thinking(value: str | None) -> str | None:
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode in {"", "enabled", "enable", "on", "true", "1"}:
        return "enabled"
    if mode in {"disabled", "disable", "off", "false", "0"}:
        return "disabled"
    raise ValueError("thinking must be enabled, disabled, or None")


def _estimate_tokens(
    prompt: str | Sequence[Mapping[str, Any]],
    response: str,
    rubric: Sequence[Mapping[str, Any]],
    max_tokens: int,
    reference: str | None = None,
) -> int:
    rubric_text = json.dumps(list(rubric), ensure_ascii=False, sort_keys=True)
    text_length = len(_message_text(prompt)) + len(response) + len(rubric_text) + len(reference or "")
    return max(1, math.ceil(text_length / 4) + max_tokens)


def _content_from_response(response: Any) -> tuple[str, dict[str, Any], str | None]:
    if isinstance(response, Mapping):
        choices = response.get("choices") or []
        usage = response.get("usage") or {}
        if not choices:
            raise ValueError("judge response has no choices")
        choice = choices[0]
        message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
        content = message.get("content") if isinstance(message, Mapping) else None
        finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        return _content_value(content), _plain_usage(usage), finish_reason

    choices = getattr(response, "choices", None) or []
    usage = getattr(response, "usage", None)
    if not choices:
        raise ValueError("judge response has no choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    return _content_value(content), _plain_usage(usage), getattr(choice, "finish_reason", None)


def _content_value(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                value = item.get("text") or item.get("content")
            else:
                value = getattr(item, "text", None) or getattr(item, "content", None)
            if value:
                parts.append(str(value))
        if parts:
            return "".join(parts)
    raise ValueError("judge response message has no text content")


def _plain_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        value = usage.model_dump()
        usage = value if isinstance(value, Mapping) else {}
    if isinstance(usage, Mapping):
        result = {str(key): value for key, value in usage.items()}
    else:
        result = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "completion_tokens_details",
        ):
            value = getattr(usage, key, None)
            if value is not None:
                result[key] = value

    details = result.get("completion_tokens_details")
    if hasattr(details, "model_dump"):
        details = details.model_dump()
        result["completion_tokens_details"] = details
    if isinstance(details, Mapping):
        reasoning_tokens = details.get("reasoning_tokens")
        if reasoning_tokens is not None:
            try:
                reasoning_tokens = int(reasoning_tokens)
            except (TypeError, ValueError):
                reasoning_tokens = 0
            if reasoning_tokens > 0:
                completion_tokens = result.get("completion_tokens", 0)
                total_tokens = result.get("total_tokens")
                try:
                    completion_tokens = int(completion_tokens)
                except (TypeError, ValueError):
                    completion_tokens = 0
                result["completion_tokens"] = completion_tokens + reasoning_tokens
                if total_tokens is not None:
                    try:
                        result["total_tokens"] = int(total_tokens) + reasoning_tokens
                    except (TypeError, ValueError):
                        result["total_tokens"] = result["completion_tokens"]
                else:
                    prompt_tokens = result.get("prompt_tokens")
                    if prompt_tokens is not None:
                        try:
                            result["total_tokens"] = int(prompt_tokens) + result["completion_tokens"]
                        except (TypeError, ValueError):
                            pass
    return result


def _is_rate_limit_error(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    return status_code == 429 or "429" in str(error)


def _retryable_status(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    return status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600)


def _urllib_http_post(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> bytes:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        raise JudgeHTTPError(exc.code) from exc


_POOL_LOCK = threading.Lock()
_POOL_CLIENTS: dict[int, Any] = {}


def _pooled_http_post(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> bytes:
    """Persistent-connection transport (one httpx client per process).

    ``urlopen`` builds a fresh TLS context per request; under a thread pool
    that serialises on the GIL at roughly 200 requests/min no matter what the
    endpoint's quota is, while a pooled client sustains thousands per minute
    against the same endpoint. Falls back to urllib when httpx is
    unavailable.
    """
    try:
        import httpx
    except ImportError:
        return _urllib_http_post(url, headers, body, timeout)
    pid = os.getpid()
    with _POOL_LOCK:
        client = _POOL_CLIENTS.get(pid)
        if client is None:
            client = httpx.Client(
                timeout=timeout,
                limits=httpx.Limits(max_connections=1024, max_keepalive_connections=512),
            )
            _POOL_CLIENTS.clear()
            _POOL_CLIENTS[pid] = client
    response = client.post(url, content=body, headers=dict(headers), timeout=timeout)
    if response.status_code >= 400:
        raise JudgeHTTPError(response.status_code)
    return response.content


def _strip_json_fence(raw: str) -> str:
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        first_line, _, rest = value.partition("\n")
        if first_line.strip().lower() in {"```", "```json"}:
            value = rest[:-3].strip()
    return value


def _parse_satisfied(raw: str, criterion_count: int) -> list[bool]:
    try:
        payload = json.loads(_strip_json_fence(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("judge output is not valid JSON") from exc
    if isinstance(payload, list):
        # Smaller judges sometimes return the bare array instead of the object.
        values: Any = payload
    elif isinstance(payload, Mapping):
        values = payload.get("satisfied")
        if values is None:
            values = payload.get("criteria_met")
    else:
        raise ValueError("judge output must be a JSON object")
    if isinstance(values, Mapping):
        # Index-keyed form ("1": true, ...): robust to judges that lose count on
        # long rubrics; every index must be present exactly once.
        indexed: dict[int, Any] = {}
        for key, value in values.items():
            try:
                index = int(str(key).strip())
            except ValueError as exc:
                raise ValueError("judge verdict keys must be criterion indices") from exc
            if index in indexed:
                raise ValueError("judge verdicts repeat a criterion index")
            indexed[index] = value
        if sorted(indexed) != list(range(1, criterion_count + 1)):
            raise ValueError(
                f"judge output must cover criterion indices 1..{criterion_count} exactly once"
            )
        values = [indexed[index] for index in range(1, criterion_count + 1)]
    if not isinstance(values, list) or len(values) != criterion_count:
        raise ValueError(
            f"judge output must contain exactly {criterion_count} boolean verdicts"
        )
    if not all(isinstance(value, bool) for value in values):
        raise ValueError("judge verdicts must be JSON booleans")
    return list(values)


def _parse_graded(raw: str, criterion_count: int) -> list[int]:
    """Parse the Graded verdict: {"grades": {"1": 0..3, ...}}, every index once."""
    try:
        payload = json.loads(_strip_json_fence(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("graded judge output is not valid JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"grades"}:
        raise ValueError("graded judge output must contain only the grades field")
    values = payload["grades"]
    if not isinstance(values, Mapping):
        raise ValueError("graded judge output grades must be an index-keyed object")
    indexed: dict[int, Any] = {}
    for key, value in values.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            index = -1
        if str(index) != str(key) or index in indexed:
            raise ValueError("graded judge verdict keys must be canonical unique criterion indices")
        indexed[index] = value
    if sorted(indexed) != list(range(1, criterion_count + 1)):
        raise ValueError(
            f"graded judge output must cover criterion indices 1..{criterion_count} exactly once"
        )
    grades: list[int] = []
    for index in range(1, criterion_count + 1):
        value = indexed[index]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or int(value) != value
            or not 0 <= int(value) <= 3
        ):
            raise ValueError("graded judge grades must be integers in 0..3")
        grades.append(int(value))
    return grades


def _parse_probabilities(raw: str, criterion_count: int) -> list[float]:
    try:
        payload = json.loads(_strip_json_fence(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("judge output is not valid JSON") from exc
    if isinstance(payload, list):
        values: Any = payload
    elif isinstance(payload, Mapping):
        values = payload.get("probabilities")
        if values is None:
            values = payload.get("criterion_probabilities")
        if values is None:
            values = payload.get("satisfied_probabilities")
    else:
        raise ValueError("judge output must be a JSON object")
    if not isinstance(values, list) or len(values) != criterion_count:
        raise ValueError(
            f"judge output must contain exactly {criterion_count} probability values"
        )
    probabilities: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("judge probabilities must be numeric, not booleans")
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("judge probabilities must be numeric") from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("judge probabilities must be in [0, 1]")
        probabilities.append(probability)
    return probabilities


def _parse_likert_rating(raw: str) -> float:
    try:
        payload = json.loads(_strip_json_fence(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("judge output is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("judge output must be a JSON object")
    value = payload.get("rating")
    if value is None:
        value = payload.get("score")
    if isinstance(value, bool):
        raise ValueError("judge rating must be numeric, not boolean")
    try:
        rating = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("judge rating must be numeric") from exc
    if not math.isfinite(rating) or not 1.0 <= rating <= 10.0:
        raise ValueError("judge rating must be in [1, 10]")
    return rating


def _normalise_likert_rating(rating: float) -> float:
    return max(0.0, min(1.0, (float(rating) - 1.0) / 9.0))


def _weighted_score(satisfied: Sequence[bool], rubric: Sequence[Mapping[str, Any]]) -> float:
    positive_total = sum(max(0.0, float(item["weight"])) for item in rubric)
    if positive_total <= 0:
        raise ValueError("rubric needs a positive total weight")
    score = sum(float(item["weight"]) * int(value) for item, value in zip(rubric, satisfied))
    return max(0.0, min(1.0, score / positive_total))


def _weighted_probability_score(
    probabilities: Sequence[float],
    rubric: Sequence[Mapping[str, Any]],
) -> float:
    positive_total = sum(max(0.0, float(item["weight"])) for item in rubric)
    if positive_total <= 0:
        raise ValueError("rubric needs a positive total weight")
    score = sum(
        float(item["weight"]) * float(value)
        for item, value in zip(rubric, probabilities)
    )
    return max(0.0, min(1.0, score / positive_total))


class RubricJudge:
    """Score all rubric criteria with one OpenAI-compatible chat completion.

    Requests go to ``<base_url>/chat/completions`` with a bearer token.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        qpm: int | None = None,
        tpm: int | None = None,
        max_concurrency: int | None = None,
        max_retries: int = 3,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_format: bool | Mapping[str, Any] = False,
        thinking: str | None = None,
        verdict_mode: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Callable[[str, Mapping[str, str], bytes, float], Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        load_judge_env_file()
        self.base_url = base_url or os.environ.get("RUBRIC_JUDGE_BASE_URL")
        self.api_key = api_key or os.environ.get("RUBRIC_JUDGE_API_KEY")
        self.model = model or os.environ.get("RUBRIC_JUDGE_MODEL")
        if not self.model:
            raise ValueError(
                "no judge model configured: set RUBRIC_JUDGE_MODEL (or the role "
                "variant, e.g. RUBRIC_JUDGE_TRAIN_MODEL) to the model id your "
                "OpenAI-compatible endpoint serves"
            )
        self.verdict_mode = (
            verdict_mode
            or os.environ.get("RUBRIC_JUDGE_VERDICT_MODE", "hard")
        ).strip().lower()
        if self.verdict_mode not in {"hard", "self_prob", "likert", "reference_likert", "graded"}:
            raise ValueError(
                "verdict_mode must be hard, self_prob, likert, reference_likert, or graded"
            )
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = float(timeout)
        self.thinking = _normalise_thinking(
            thinking if thinking is not None else os.getenv("RUBRIC_JUDGE_THINKING")
        )
        self.response_format = response_format
        self._transport = transport or _pooled_http_post
        if client is None and transport is None:
            if not self.base_url:
                raise ValueError("RUBRIC_JUDGE_BASE_URL is required")
            if not self.api_key:
                raise ValueError("RUBRIC_JUDGE_API_KEY is required")
        self._client = client
        self.max_retries = max(0, int(max_retries))
        self.max_tokens = max(1, int(max_tokens))
        self._sleep = sleep_fn
        self._random = random_fn
        self._clock = clock
        self.throttle_deadline = _env_positive_int("RUBRIC_JUDGE_THROTTLE_DEADLINE_SECONDS", 1800)
        self._limiter = RollingTokenRateLimiter(
            qpm=qpm if qpm is not None else _env_positive_int("RUBRIC_JUDGE_QPM", DEFAULT_QPM),
            tpm=tpm if tpm is not None else _env_positive_int("RUBRIC_JUDGE_TPM", DEFAULT_TPM),
            clock=clock,
            sleep_fn=sleep_fn,
        )
        self._concurrency = AdaptiveConcurrency(
            max_concurrency
            if max_concurrency is not None
            else _env_positive_int("RUBRIC_JUDGE_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
        )

    @property
    def rate_limit_snapshot(self) -> dict[str, int]:
        return self._limiter.snapshot

    @property
    def concurrency_limit(self) -> int:
        return self._concurrency.limit

    def _judge_prompt(
        self,
        prompt: str | Sequence[Mapping[str, Any]],
        response: str,
        rubric: Sequence[Mapping[str, Any]],
        *,
        reference: str | None = None,
    ) -> str:
        criteria = [
            {
                "index": index,
                "criterion": str(item["criterion"]),
                "weight": float(item["weight"]),
                "polarity": str(item.get("polarity", "positive")),
            }
            for index, item in enumerate(rubric, start=1)
        ]
        if self.verdict_mode == "self_prob":
            instruction = (
                "You are a strict rubric judge. Evaluate every criterion independently "
                "and return the probability in [0, 1] that the assistant response "
                "satisfies that criterion. For a criterion whose polarity is negative, "
                "the probability means the undesirable behavior is present, not that "
                "the response is good. These are direct self-reported probabilities, "
                "not token probabilities. Return ONLY one JSON object with exactly one "
                'field, "probabilities", whose value is a numeric array in criterion '
                "order. Do not include explanations, markdown, or extra keys."
            )
        elif self.verdict_mode == "likert":
            return (
                "You are an expert evaluator. Given a user prompt and a generated response, "
                "please rate the overall quality of the response on a scale of 1 to 10, "
                "where 1 is very poor and 10 is excellent.\n"
                "Return ONLY one JSON object with a single key \"rating\" and an integer "
                "or numeric value between 1 and 10. Do not include explanations, markdown, "
                "or extra keys.\n\n"
                "Given the following prompt, and response, please rate the overall quality "
                "of the response on a scale of 1 to 10.\n\n"
                f"<prompt>\n{_message_text(prompt)}\n</prompt>\n\n"
                f"<response>\n{response}\n</response>\n\n"
                "Your JSON Evaluation:"
            )
        elif self.verdict_mode == "reference_likert":
            if reference is None or not str(reference).strip():
                raise ValueError("reference_likert verdict_mode requires a reference answer")
            return (
                "You are an expert evaluator. Given a user prompt, a reference response, "
                "and a generated response, please rate the overall quality of the generated "
                "response on a scale of 1 to 10 based on how well it compares to the "
                "reference response.\n"
                "Consider factors such as accuracy, completeness, coherence, and helpfulness "
                "when comparing to the reference. The reference response represents a "
                "high-quality answer that you should use as a benchmark.\n"
                "Return ONLY one JSON object with a single key \"rating\" and an integer "
                "or numeric value between 1 and 10. Do not include explanations, markdown, "
                "or extra keys.\n\n"
                "Given the following prompt, reference response, and generated response, "
                "please rate the overall quality of the generated response on a scale of "
                "1 to 10 based on how well it compares to the reference.\n\n"
                f"<prompt>\n{_message_text(prompt)}\n</prompt>\n\n"
                f"<reference_response>\n{reference}\n</reference_response>\n\n"
                f"<generated_response>\n{response}\n</generated_response>\n\n"
                "Your JSON Evaluation:"
            )
        elif self.verdict_mode == "graded":
            instruction = (
                "You are a strict rubric judge. Evaluate the assistant response against "
                "every criterion independently and grade each criterion on a 0-3 scale: "
                "0 = essentially not met or the opposite is done; 1 = only a few of the "
                "criterion's substantive requirements are met; 2 = most requirements are "
                "met but a substantive gap remains; 3 = all requirements are met. "
                "Negative-weight criteria describe undesirable behavior: grade how strongly "
                "that behavior is present (3 = fully present). Return ONLY one JSON object "
                'with exactly one field, "grades", whose value is an object mapping each '
                'criterion index (as a string, e.g. "1", "2", ...) to an integer 0, 1, 2 or 3, '
                "covering every index exactly once. Do not include explanations, markdown, "
                "or extra keys."
            )
        else:
            instruction = (
                "You are a strict binary rubric judge. Evaluate the assistant response "
                "against every criterion independently. A criterion is satisfied only "
                "when the response meets all of its requirements. Negative-weight "
                "criteria describe undesirable behavior: return true when that behavior "
                "is present. Return ONLY one JSON object with exactly one field, "
                '"satisfied", whose value is an object mapping each criterion index '
                '(as a string, e.g. "1", "2", ...) to a boolean, covering every index '
                "exactly once. Do not include explanations, markdown, or extra keys."
            )
        return (
            f"{instruction}\n\n"
            f"PROMPT:\n{_message_text(prompt)}\n\n"
            f"ASSISTANT RESPONSE:\n{response}\n\n"
            f"CRITERIA:\n{json.dumps(criteria, ensure_ascii=False, indent=2)}"
        )

    def _endpoint_url(self) -> str:
        if not self.base_url:
            raise ValueError("RUBRIC_JUDGE_BASE_URL is required")
        base_url = self.base_url.rstrip("/")
        if not base_url.endswith("/chat/completions"):
            base_url += "/chat/completions"
        return base_url

    def _call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0,
    ) -> Any:
        effective_max_tokens = max(1, int(max_tokens or self.max_tokens))
        if self._client is not None:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": effective_max_tokens,
            }
            if self.thinking == "disabled":
                kwargs["thinking"] = {"type": "disabled"}
            if self.response_format:
                kwargs["response_format"] = (
                    {"type": "json_object"}
                    if self.response_format is True
                    else dict(self.response_format)
                )
            return self._client.chat.completions.create(**kwargs)

        if not self.api_key:
            raise ValueError("RUBRIC_JUDGE_API_KEY is required")
        payload: dict[str, Any] = {
            "stream": False,
            "model": self.model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
        }
        if self.thinking == "disabled":
            payload["thinking"] = {"type": "disabled"}
        if self.response_format:
            payload["response_format"] = (
                {"type": "json_object"}
                if self.response_format is True
                else dict(self.response_format)
            )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        value = self._transport(self._endpoint_url(), headers, encoded, self.timeout)
        if isinstance(value, Mapping):
            return value
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("judge HTTP response is not valid JSON") from exc
            if isinstance(decoded, Mapping):
                return decoded
        raise ValueError("judge HTTP transport returned an invalid response")

    def _score_attempt(
        self,
        prompt: str | Sequence[Mapping[str, Any]],
        response: str,
        rubric: list[dict[str, Any]],
        *,
        reference: str | None = None,
        temperature: float = 0.0,
    ) -> JudgeResult:
        estimated_tokens = _estimate_tokens(prompt, response, rubric, self.max_tokens, reference)
        self._limiter.reserve(estimated_tokens)
        self._concurrency.acquire()
        succeeded = False
        throttled = False
        try:
            raw_response = self._call(
                [
                    {
                        "role": "user",
                        "content": self._judge_prompt(
                            prompt,
                            response,
                            rubric,
                            reference=reference,
                        ),
                    }
                ],
                temperature=temperature,
            )
            raw, usage, finish_reason = _content_from_response(raw_response)
            if finish_reason == "length":
                raise ValueError("judge response was truncated (finish_reason=length)")
            probabilities: list[float] | None
            if self.verdict_mode == "self_prob":
                probabilities = _parse_probabilities(raw, len(rubric))
                satisfied = [value >= 0.5 for value in probabilities]
                weighted_score = _weighted_probability_score(probabilities, rubric)
                rating = None
            elif self.verdict_mode == "graded":
                grades = _parse_graded(raw, len(rubric))
                rating = None
                probabilities = [grade / 3.0 for grade in grades]
                satisfied = [grade >= 2 for grade in grades]
                weighted_score = _weighted_probability_score(probabilities, rubric)
            elif self.verdict_mode in {"likert", "reference_likert"}:
                rating = _parse_likert_rating(raw)
                probabilities = None
                satisfied = []
                weighted_score = _normalise_likert_rating(rating)
            else:
                rating = None
                probabilities = None
                satisfied = _parse_satisfied(raw, len(rubric))
                weighted_score = _weighted_score(satisfied, rubric)
            succeeded = True
            return JudgeResult(
                satisfied=satisfied,
                weighted_score=weighted_score,
                raw=raw,
                usage=usage,
                probabilities=probabilities,
                rating=rating,
                verdict_mode=self.verdict_mode,
            )
        except Exception as exc:
            throttled = _is_rate_limit_error(exc)
            raise
        finally:
            self._concurrency.release(success=succeeded, throttled=throttled)

    def score(
        self,
        prompt: str | Sequence[Mapping[str, Any]],
        response: str,
        rubric: list[dict[str, Any]],
        *,
        reference: str | None = None,
        temperature: float | None = None,
    ) -> JudgeResult:
        if not rubric:
            raise ValueError("rubric must contain at least one criterion")
        for item in rubric:
            if not isinstance(item, Mapping) or "criterion" not in item or "weight" not in item:
                raise ValueError("each rubric item needs criterion and weight")
        last_error: BaseException | None = None
        attempt = 0
        throttle_waits = 0
        throttle_started = self._clock()
        while True:
            try:
                # Greedy first; a malformed verdict is deterministic at
                # temperature 0, so retries sample instead of repeating it.
                return self._score_attempt(
                    prompt,
                    response,
                    rubric,
                    reference=reference,
                    temperature=(
                        (0.0 if attempt == 0 else 0.7)
                        if temperature is None
                        else temperature
                    ),
                )
            except Exception as exc:
                last_error = exc
                if _is_rate_limit_error(exc):
                    # 429 is the endpoint sharing its per-model quota among
                    # concurrent runs; it is not a judge failure. Wait with
                    # jittered backoff and retry without consuming the
                    # failure budget, up to a wall-clock cap.
                    throttle_waits += 1
                    if self._clock() - throttle_started > self.throttle_deadline:
                        break
                    delay = min(30.0, 1.0 * (2 ** min(throttle_waits, 5))) * (
                        0.75 + 0.5 * self._random()
                    )
                    self._sleep(delay)
                    continue
                if _retryable_status(exc):
                    # 5xx: server-side failure, exponential backoff against the
                    # same retry budget as other transient errors. A scoring
                    # path without this backoff once marked 12,235 rows
                    # permanently failed.
                    attempt += 1
                    if attempt > self.max_retries:
                        break
                    delay = min(30.0, 1.0 * (2 ** min(attempt, 5))) * (
                        0.75 + 0.5 * self._random()
                    )
                    self._sleep(delay)
                    continue
                attempt += 1
                if attempt > self.max_retries:
                    break
                self._sleep(min(10.0, 0.25 * (2**attempt)))
        raise JudgeUnavailableError(
            f"judge failed after {attempt} attempts and {throttle_waits} throttle waits: "
            f"{type(last_error).__name__}: {str(last_error)[:300]}"
        ) from last_error

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0,
    ) -> JudgeCompletion:
        """Run a raw completion through the same endpoint and rate limits.

        The caller owns JSON parsing.
        """

        normalized = [
            {
                "role": str(item.get("role", "user")),
                "content": str(item.get("content", "")),
            }
            for item in messages
        ]
        if not normalized:
            raise ValueError("messages must contain at least one message")
        effective_max_tokens = max(1, int(max_tokens or self.max_tokens))
        estimate = max(
            1,
            math.ceil(len(_message_text(normalized)) / 4) + effective_max_tokens,
        )
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            succeeded = False
            throttled = False
            try:
                self._limiter.reserve(estimate)
                self._concurrency.acquire()
                try:
                    raw_response = self._call(
                        normalized,
                        max_tokens=effective_max_tokens,
                        temperature=temperature,
                    )
                    raw, usage, finish_reason = _content_from_response(raw_response)
                    if finish_reason == "length":
                        raise ValueError(
                            "judge completion was truncated (finish_reason=length)"
                        )
                    succeeded = True
                    return JudgeCompletion(raw=raw, usage=usage)
                except Exception as exc:
                    throttled = _is_rate_limit_error(exc)
                    raise
                finally:
                    self._concurrency.release(
                        success=succeeded,
                        throttled=throttled,
                    )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                if _is_rate_limit_error(exc) or _retryable_status(exc):
                    delay = min(30.0, 0.5 * (2**attempt)) * (
                        0.75 + 0.5 * self._random()
                    )
                else:
                    delay = min(10.0, 0.25 * (2**attempt))
                self._sleep(delay)
        raise JudgeUnavailableError(
            f"judge completion failed after {self.max_retries + 1} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def score_batch(self, items: Sequence[RubricScoreInput | Mapping[str, Any]]) -> list[JudgeResult]:
        normalized: list[RubricScoreInput] = []
        for item in items:
            if isinstance(item, RubricScoreInput):
                normalized.append(item)
            elif isinstance(item, Mapping):
                try:
                    normalized.append(
                        RubricScoreInput(
                            prompt=item["prompt"],
                            response=str(item["response"]),
                            rubric=list(item["rubric"]),
                            reference=(
                                str(item["reference"])
                                if item.get("reference") is not None
                                else None
                            ),
                        )
                    )
                except (KeyError, TypeError) as exc:
                    raise ValueError("score_batch items need prompt, response, rubric") from exc
            else:
                raise TypeError("score_batch items must be mappings or RubricScoreInput")

        if not normalized:
            return []
        results: list[JudgeResult | None] = [None] * len(normalized)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._concurrency.maximum
        ) as executor:
            futures = {
                executor.submit(
                    self.score,
                    item.prompt,
                    item.response,
                    item.rubric,
                    reference=item.reference,
                ): index
                for index, item in enumerate(normalized)
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    results[futures[future]] = future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if any(result is None for result in results):
            raise JudgeUnavailableError("judge batch returned a missing result")
        return [result for result in results if result is not None]
