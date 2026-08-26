from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.intent_dataset import file_sha256, validate_example, write_jsonl
from training.train_lora import tokenize_conversation
from training.tokenization_preflight import (
    inspect_dataset_tokenization,
    require_transformers_4,
)
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


def test_long_trajectory_rejects_conditioning_context_truncation() -> None:
    row = _trajectory(supervise=[1])
    row["messages"].insert(2, {"role": "user", "content": "x" * 1000})
    tokenizer = _CharacterChatTokenizer()

    with pytest.raises(ValueError, match="截断会丢失条件上下文"):
        tokenize_conversation(
            tokenizer,
            row["messages"],
            max_length=128,
            supervise_assistant_indices=[1],
        )


def test_tokenizer_mapping_output_is_rejected() -> None:
    class MappingTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return {"input_ids": [1, 2, 3]}

    with pytest.raises(TypeError, match="一维 token id 序列"):
        tokenize_conversation(
            MappingTokenizer(),
            _trajectory()["messages"],
            max_length=128,
        )


def test_transformers_major_version_is_pinned() -> None:
    require_transformers_4("4.57.6")
    with pytest.raises(RuntimeError, match="只允许项目锁定的 4.x"):
        require_transformers_4("5.12.1")


def test_dataset_tokenization_preflight_reports_lengths(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    files = {}
    for split in ("train", "valid", "test"):
        path = dataset_dir / f"agent-sft-{split}.jsonl"
        write_jsonl(path, [_trajectory(supervise=[1])])
        files[split] = {
            "path": path.name,
            "count": 1,
            "sha256": file_sha256(path),
        }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "datasetId": "preflight-test",
                "files": files,
            }
        ),
        encoding="utf-8",
    )

    report = inspect_dataset_tokenization(
        dataset_dir,
        {"maxLength": 10000},
        _CharacterChatTokenizer(),
        transformers_version="4.57.6",
        tokenizer_source="test-tokenizer",
    )

    assert report["passed"] is True
    assert report["maxObservedTokens"] > 0
    assert report["truncatedExamples"] == 0
    assert report["splitCounts"] == {"train": 1, "valid": 1, "test": 1}


def test_experiment_configs_change_only_frozen_identity_fields() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_root = project_root / "training" / "configs"
    baseline = json.loads(
        (config_root / "agent-lora-v2.json").read_text(encoding="utf-8")
    )
    repro = json.loads(
        (config_root / "agent-lora-v2-repro.json").read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (config_root / "agent-lora-v2.2.json").read_text(encoding="utf-8")
    )

    assert {key for key in baseline if baseline[key] != repro[key]} == {"modelId"}
    assert {
        key for key in baseline if baseline[key] != candidate[key]
    } == {"modelId", "version", "trainingMethod"}


def test_experiment_target_cases_are_frozen_holdout_members() -> None:
    project_root = Path(__file__).resolve().parents[1]
    experiment = json.loads(
        (project_root / "evals" / "agent-v2.2-experiment.json").read_text(
            encoding="utf-8"
        )
    )
    suite = json.loads(
        (
            project_root / "evals" / "agent-trajectories-v2.1-holdout.json"
        ).read_text(encoding="utf-8")
    )
    target_ids = set(
        experiment["directionalEvidenceCriteria"]["targetCaseIds"]
    )
    holdout_ids = {row["caseId"] for row in suite["cases"]}

    assert len(target_ids) == 6
    assert target_ids <= holdout_ids


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
