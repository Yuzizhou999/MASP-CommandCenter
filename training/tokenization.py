from __future__ import annotations

from typing import Any


def chat_token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(encoded, (list, tuple)) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in encoded
    ):
        raise TypeError(
            "apply_chat_template 必须返回一维 token id 序列；"
            "请使用项目锁定的 transformers 4.x 环境"
        )
    return list(encoded)


def tokenize_conversation(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_length: int,
    supervise_assistant_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Tokenize a transcript without silently discarding conditioning context."""

    full_ids = chat_token_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    if len(full_ids) > max_length:
        raise ValueError(
            f"样本长度 {len(full_ids)} 超过 maxLength {max_length}，"
            "截断会丢失条件上下文；请增大 maxLength 或缩短静态 schema"
        )

    labels = [-100] * len(full_ids)
    assistant_ordinal = 0
    selected = (
        set(supervise_assistant_indices)
        if supervise_assistant_indices is not None
        else None
    )
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        should_supervise = selected is None or assistant_ordinal in selected
        if should_supervise:
            prompt_ids = chat_token_ids(
                tokenizer,
                messages[:message_index],
                add_generation_prompt=True,
            )
            through_ids = chat_token_ids(
                tokenizer,
                messages[: message_index + 1],
                add_generation_prompt=False,
            )
            start = min(len(prompt_ids), len(full_ids))
            end = min(len(through_ids), len(full_ids))
            labels[start:end] = full_ids[start:end]
        assistant_ordinal += 1

    if all(value == -100 for value in labels):
        raise ValueError("样本没有可监督的 assistant token")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
