"""Pure OPSD teacher-prompt rendering shared by native training and audits."""

from __future__ import annotations

from typing import Any


def render_opsd_teacher_user_message(
    problem: str, solution: str, context_template: str
) -> str:
    """Render the legacy OPSD user message without tokenizing it."""

    if context_template not in ("", "default"):
        return context_template.format(problem=problem, solution=solution)
    transition_prompt = (
        "\n\nAfter reading the reference solution above, make sure you truly understand "
        "the reasoning behind each step - do not copy or paraphrase it. Now, using your "
        "own words and independent reasoning, derive the same final answer to the problem above. "
        "Think step by step, explore different approaches, and don't be afraid to backtrack "
        "or reconsider if something doesn't work out:\n"
    )
    return (
        f"Problem: {problem}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"{transition_prompt}\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )


def render_opsd_teacher_prompt_ids(
    tokenizer: Any,
    student_prompt_ids: list[int],
    *,
    render_mode: str,
    problem: str | None,
    solution: str | None,
    teacher_thinking: bool,
    context_template: str,
) -> list[int]:
    """Return native teacher prompt ids, with canonical H0 as an exact copy."""

    if not student_prompt_ids:
        raise ValueError("OPSD teacher prompt construction requires a non-empty student prompt.")
    if render_mode == "canonical_h0":
        return [int(token_id) for token_id in student_prompt_ids]
    if render_mode != "legacy_user_template":
        raise ValueError(f"unsupported OPSD render_mode: {render_mode!r}")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("OPSD legacy teacher prompt requires a non-empty problem.")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("OPSD legacy teacher prompt requires a non-empty solution.")

    messages = [
        {
            "role": "user",
            "content": render_opsd_teacher_user_message(
                problem, solution, context_template
            ),
        }
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        rendered = tokenizer.apply_chat_template(
            messages, enable_thinking=teacher_thinking, **kwargs
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    encoded = tokenizer(rendered, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or not ids:
        raise ValueError("OPSD teacher prompt tokenization returned an empty sequence.")
    return [int(token_id) for token_id in ids]
