#!/usr/bin/env python3
"""Generate an evaluation responses.jsonl from a checkpoint with vLLM.

    python3 generate.py --model /path/to/checkpoint \
        --input healthbench_full.parquet --input healthbench_consensus.parquet \
        --input medqa_usmle_4opt.parquet --output responses.jsonl

Defaults are the paper's evaluation settings: Qwen3 chat template with
``enable_thinking=False``, 8,192 max new tokens (``--max-tokens 2048`` for the
equal-budget comparison), temperature 0.7, top-p 0.8, seed 42 (``--seed 43`` /
``44`` for the seed replicates), rows sharded contiguously over 8 GPUs, one
vLLM engine per GPU.

Input parquets (read in the order given and concatenated) carry ``prompt`` (a
string or ``[{"role", "content"}]`` messages) and an id in ``extra_info.id``
(then ``extra_info.sample_id`` / ``prompt_id``, then the top-level
equivalents, then the row index). The prompt is the conversation the policy
sees; no rubric is shown.

A prompt whose rendered length exceeds ``max_model_len - max_tokens - 16``
tokens is not generated; its response is the marker
``[[GENERATION_SKIPPED: prompt exceeds max_model_len]]``, which
``score_records.py`` records as a per-question failure.

Output: JSONL ``{"id", "response"}``, one row per input row, keys sorted.

vLLM (and a GPU) is needed only to generate; ``--help`` works without it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_NUM_GPUS = 8
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_SEED = 42
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85

GENERATION_SKIPPED_MARKER = "[[GENERATION_SKIPPED: prompt exceeds max_model_len]]"


# ---------------------------------------------------------------------------
# Parquet rows and prompt rendering (as in the evaluation harness)
# ---------------------------------------------------------------------------


def plain_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert Arrow scalar containers into ordinary Python containers."""

    def plain(value: Any) -> Any:
        if hasattr(value, "as_py"):
            return plain(value.as_py())
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    return [plain(row) for row in rows]


def read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return plain_rows(parquet.read_table(path).to_pylist())


def row_id(row: Mapping[str, Any], index: int) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, Mapping):
        for key in ("id", "sample_id", "prompt_id"):
            if extra.get(key) is not None:
                return str(extra[key])
    for key in ("id", "sample_id", "prompt_id"):
        if row.get(key) is not None:
            return str(row[key])
    return str(index)


def json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_prompt(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return [{"role": "user", "content": value}]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError("prompt must be a string or a sequence of chat messages")

    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError("every prompt message must be an object")
        role = str(message.get("role") or "user").strip()
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json_text(content)
        if not role:
            raise ValueError("prompt message role must not be empty")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("prompt must contain at least one message")
    return messages


def evaluation_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    return normalize_prompt(row.get("prompt"))


def apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    enable_thinking: bool = False,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        rendered = tokenizer.apply_chat_template(list(messages), **kwargs)
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template must return a string")
    return rendered


# ---------------------------------------------------------------------------
# vLLM generation for one shard
# ---------------------------------------------------------------------------


def generate_prompts_vllm(
    prompts: Sequence[Sequence[Mapping[str, Any]]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    enable_thinking: bool = False,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model),
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        # 32k fits every evaluation prompt set alongside the response budget;
        # Qwen3-4B/8B configs allow 40,960.
        max_model_len=max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    rendered_prompts = [
        apply_chat_template(tokenizer, prompt, enable_thinking=enable_thinking)
        for prompt in prompts
    ]
    # Skip over-long prompts per row (marked, so scoring records a per-row
    # failure) instead of failing the whole shard.
    budget = max_model_len - int(max_tokens) - 16
    keep_indices: list[int] = []
    skipped: dict[int, str] = {}
    for i, rendered in enumerate(rendered_prompts):
        if len(tokenizer.encode(rendered)) > budget:
            skipped[i] = GENERATION_SKIPPED_MARKER
        else:
            keep_indices.append(i)
    rendered_prompts = [rendered_prompts[i] for i in keep_indices]
    sampling_kwargs = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "seed": int(seed),
    }
    try:
        sampling_params = SamplingParams(**sampling_kwargs)
    except TypeError:
        sampling_kwargs.pop("seed")
        sampling_params = SamplingParams(**sampling_kwargs)
    results = llm.generate(
        rendered_prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    if len(results) != len(rendered_prompts):
        raise RuntimeError(
            f"vLLM returned {len(results)} outputs for {len(rendered_prompts)} prompts"
        )
    generated: list[str] = []
    for index, result in enumerate(results):
        outputs = getattr(result, "outputs", None) or []
        text = getattr(outputs[0], "text", None) if outputs else None
        if not isinstance(text, str):
            raise RuntimeError(f"vLLM returned no text for shard row {index}")
        generated.append(text)
    responses: list[str] = []
    it = iter(generated)
    for i in range(len(prompts)):
        responses.append(skipped[i] if i in skipped else next(it))
    return responses


# ---------------------------------------------------------------------------
# Sharding over GPUs: contiguous shards, one spawned process per GPU
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _shard_worker(
    shard_id: int,
    gpu_id: int,
    indexed_rows: Sequence[tuple[int, Mapping[str, Any]]],
    generation_kwargs: Mapping[str, Any],
    prompt_builder: Callable[[Mapping[str, Any]], list[dict[str, str]]],
    generation_fn: Callable[..., list[str]],
    shard_path: Path,
    result_queue: Any,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        prompts = [prompt_builder(row) for _, row in indexed_rows]
        responses = generation_fn(prompts, **generation_kwargs)
        if len(responses) != len(indexed_rows):
            raise RuntimeError(
                f"shard {shard_id} returned {len(responses)} outputs for {len(indexed_rows)} rows"
            )
        payload = {
            "rows": [
                {"index": index, "response": response}
                for (index, _), response in zip(indexed_rows, responses)
                if isinstance(response, str)
            ]
        }
        if len(payload["rows"]) != len(indexed_rows):
            raise RuntimeError(f"shard {shard_id} returned a non-string response")
        _write_json(shard_path, payload)
        result_queue.put({"shard_id": shard_id, "status": "COMPLETE"})
    except BaseException as exc:
        result_queue.put(
            {"shard_id": shard_id, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        )


def run_sharded_generation(
    rows: Sequence[Mapping[str, Any]],
    *,
    num_gpus: int,
    generation_kwargs: Mapping[str, Any],
    prompt_builder: Callable[[Mapping[str, Any]], list[dict[str, str]]] = evaluation_messages,
    generation_fn: Callable[..., list[str]] = generate_prompts_vllm,
) -> list[str]:
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    if not rows:
        return []
    temp_root = Path(tempfile.mkdtemp(prefix="eval-shards-"))
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = []
    shard_paths: dict[int, Path] = {}
    try:
        for shard_id in range(num_gpus):
            start = len(rows) * shard_id // num_gpus
            end = len(rows) * (shard_id + 1) // num_gpus
            indexed_rows = [(index, rows[index]) for index in range(start, end)]
            shard_path = temp_root / f"shard-{shard_id}.json"
            shard_paths[shard_id] = shard_path
            process = context.Process(
                target=_shard_worker,
                args=(shard_id, shard_id, indexed_rows, dict(generation_kwargs),
                      prompt_builder, generation_fn, shard_path, result_queue),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        messages = []
        for _ in processes:
            try:
                messages.append(result_queue.get(timeout=10))
            except queue.Empty:
                messages.append({"status": "FAILED", "error": "shard worker did not report completion"})
                break
        failures = [m for m in messages if m.get("status") != "COMPLETE"]
        failed_processes = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failures or failed_processes:
            raise RuntimeError(f"shard generation failed: {failures or failed_processes}")
        merged: dict[int, str] = {}
        for shard_path in shard_paths.values():
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
            for item in payload.get("rows", []):
                index = int(item["index"])
                if index in merged:
                    raise RuntimeError(f"duplicate generated row index {index}")
                merged[index] = item["response"]
        if set(merged) != set(range(len(rows))):
            raise RuntimeError(f"shard merge coverage mismatch: expected {len(rows)}, got {len(merged)}")
        return [merged[index] for index in range(len(rows))]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()
        shutil.rmtree(temp_root, ignore_errors=True)


def write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Defaults are the paper's evaluation settings; see README.md.",
    )
    parser.add_argument("--model", required=True, help="checkpoint directory (HF format) or model id")
    parser.add_argument("--input", required=True, action="append", type=Path, dest="input_paths",
                        help="suite parquet; repeat for several suites (concatenated in order)")
    parser.add_argument("--output", required=True, type=Path, help="responses JSONL to write")
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS,
                        help=f"one vLLM engine per GPU, contiguous shards (default {DEFAULT_NUM_GPUS})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="max new tokens (paper: 8192; 2048 for the equal-budget comparison)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="vLLM per-request sampling seed")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="render the chat template with enable_thinking=True (paper: off)")
    parser.add_argument("--limit", type=int, default=0, help="generate only the first N rows")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if not 0 <= args.top_p <= 1:
        raise ValueError("top_p must be in [0, 1]")
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")
    rows: list[dict[str, Any]] = []
    for path in args.input_paths:
        rows.extend(read_parquet(path.expanduser().resolve()))
    if args.limit:
        rows = rows[: args.limit]
    responses = run_sharded_generation(
        rows,
        num_gpus=args.num_gpus,
        generation_kwargs={
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "enable_thinking": args.enable_thinking,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
    )
    write_jsonl([{"id": row_id(row, i), "response": resp} for i, (row, resp) in enumerate(zip(rows, responses))],
                args.output)
    skipped = sum(r == GENERATION_SKIPPED_MARKER for r in responses)
    print(f"wrote {len(responses)} responses ({skipped} over-long prompts skipped) -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
