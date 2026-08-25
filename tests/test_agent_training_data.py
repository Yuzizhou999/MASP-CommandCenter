from __future__ import annotations

import pytest

from training.intent_dataset import validate_example
from training.train_lora import tokenize_conversation


class _CharacterChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        text = "".join(
            f"<{row['role']}>{row['content']}</{row['role']}>" for row in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


def _trajectory(supervise=None):
    metadata = {
        "datasetType": "agent-trajectory",
        "expectedTerminalState": "READY",
    }
    if supervise is not None:
        metadata["superviseAssistantIndices"] = supervise
    return {
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "goal"},
            {
                "role": "assistant",
                "content": '{"action":"CALL_TOOL","tool":"search_sop","arguments":{"query":"x"}}',
            },
            {"role": "user", "content": "observation"},
            {
                "role": "assistant",
                "content": '{"action":"PROPOSE_INTENT","intent":{"intentType":"QUERY_STATUS"}}',
            },
        ],
        "metadata": metadata,
    }


def test_multi_turn_trajectory_validation_accepts_single_actions() -> None:
    result = validate_example(_trajectory())

    assert result["actionCount"] == 2
    assert result["supervisedActionCount"] == 2


def test_training_masks_non_selected_assistant_turns() -> None:
    row = _trajectory(supervise=[1])
    tokenizer = _CharacterChatTokenizer()

    encoded = tokenize_conversation(
        tokenizer,
        row["messages"],
        max_length=10000,
        supervise_assistant_indices=[1],
    )
    rendered = "".join(chr(value) for value in encoded["input_ids"])
    first_action = rendered.index('{"action":"CALL_TOOL"')
    final_action = rendered.index('{"action":"PROPOSE_INTENT"')

    assert encoded["labels"][first_action] == -100
    assert encoded["labels"][final_action] != -100


def test_trajectory_rejects_model_authored_clarification_text() -> None:
    row = _trajectory()
    row["messages"][-1]["content"] = (
        '{"action":"REQUEST_CLARIFICATION","question":"哪里？"}'
    )
    row["metadata"]["expectedTerminalState"] = "CLARIFICATION_REQUIRED"

    with pytest.raises(ValueError):
        validate_example(row)
