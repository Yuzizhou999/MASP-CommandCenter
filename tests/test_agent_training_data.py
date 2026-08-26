from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.intent_dataset import validate_example
from training.train_lora import tokenize_conversation
from training.prepare_agent_dataset_v21 import (
    _clarification_rows,
    _explanation_rows,
    _repair_rows,
    _status_rows,
)


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


def test_long_trajectory_truncation_keeps_terminal_supervision() -> None:
    row = _trajectory(supervise=[1])
    row["messages"].insert(2, {"role": "user", "content": "x" * 1000})
    tokenizer = _CharacterChatTokenizer()

    encoded = tokenize_conversation(
        tokenizer,
        row["messages"],
        max_length=128,
        supervise_assistant_indices=[1],
    )

    assert len(encoded["input_ids"]) == 128
    assert any(value != -100 for value in encoded["labels"])


def test_trajectory_rejects_model_authored_clarification_text() -> None:
    row = _trajectory()
    row["messages"][-1][
        "content"
    ] = '{"action":"REQUEST_CLARIFICATION","question":"哪里？"}'
    row["metadata"]["expectedTerminalState"] = "CLARIFICATION_REQUIRED"

    with pytest.raises(ValueError):
        validate_example(row)


def test_v21_failure_driven_rows_have_expected_terminal_actions() -> None:
    rows = [
        *_explanation_rows(),
        *_repair_rows(),
        *_clarification_rows(),
        *_status_rows(),
    ]

    assert len(rows) == 34
    assert len({row["metadata"]["exampleId"] for row in rows}) == len(rows)
    for row in rows:
        result = validate_example(row)
        assert result["valid"] is True
        assert row["metadata"]["superviseAssistantIndices"]

    clarification = [
        row
        for row in rows
        if row["metadata"]["category"] == "SOFT_AMBIGUITY_CLARIFICATION"
    ]
    assert all(
        row["metadata"]["expectedTerminalState"] == "CLARIFICATION_REQUIRED"
        for row in clarification
    )
    repair = [
        row
        for row in rows
        if row["metadata"]["category"] == "VALIDATION_REPAIR_CLOSED_LOOP"
    ]
    assert all(row["metadata"]["fixableIssueCodes"] for row in repair)


def test_v21_failure_driven_requests_do_not_copy_frozen_gold_text() -> None:
    project_root = Path(__file__).resolve().parents[1]
    suite = json.loads(
        (project_root / "evals" / "agent-trajectories-v1.json").read_text(
            encoding="utf-8"
        )
    )
    gold_requests = {case["message"] for case in suite["cases"]}
    rows = [
        *_explanation_rows(),
        *_repair_rows(),
        *_clarification_rows(),
        *_status_rows(),
    ]
    training_requests = {
        json.loads(row["messages"][1]["content"])["request"] for row in rows
    }

    assert training_requests.isdisjoint(gold_requests)
