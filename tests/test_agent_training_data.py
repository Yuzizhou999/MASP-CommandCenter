from __future__ import annotations

import json
from pathlib import Path

import pytest

from command_center.agent_protocol import AgentAction
from command_center.contracts import DispatchIntent, IntentType
from command_center.knowledge import KnowledgeBase
from training.intent_dataset import (
    build_example,
    file_sha256,
    validate_example,
    write_jsonl,
)
from training.prepare_agent_dataset_v21 import (
    _clarification_rows,
    _explanation_rows,
    _repair_rows,
    _status_rows,
)
from training.prepare_agent_dataset_v23 import (
    PROTOCOL_ID,
)
from training.prepare_agent_dataset_v23 import (
    _extra_clarification_rows as _v23_clarification_rows,
)
from training.prepare_agent_dataset_v23 import (
    _intent_rows as _v23_intent_rows,
)
from training.prepare_agent_dataset_v23 import (
    _protocol_recovery_rows as _v23_protocol_recovery_rows,
)
from training.prepare_agent_dataset_v23 import (
    _schema_repair_row as _v23_schema_repair_row,
)
from training.prepare_agent_dataset_v23 import (
    _search_routing_rows as _v23_search_routing_rows,
)
from training.prepare_agent_dataset_v23 import (
    _semantic_rows as _v23_semantic_rows,
)
from training.prepare_agent_dataset_v23 import (
    _validation_repair_row as _v23_validation_repair_row,
)
from training.tokenization_preflight import (
    inspect_dataset_tokenization,
    require_transformers_4,
)
from training.train_lora import (
    _checkpoint_training_arguments,
    tokenize_conversation,
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
    with pytest.raises(RuntimeError, match=r"只允许项目锁定的 4\.x"):
        require_transformers_4("5.12.1")


def test_checkpoint_arguments_preserve_default_epoch_behavior() -> None:
    assert _checkpoint_training_arguments(None) == {
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
    }


def test_checkpoint_arguments_support_bounded_training_sessions() -> None:
    assert _checkpoint_training_arguments(20) == {
        "save_strategy": "steps",
        "save_steps": 20,
        "load_best_model_at_end": False,
    }
    with pytest.raises(ValueError, match="必须是正整数"):
        _checkpoint_training_arguments(0)


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


def test_trajectory_allows_rejected_invalid_action_only_as_unsupervised_context() -> None:
    row = _trajectory(supervise=[1])
    row["messages"][2]["content"] = '{"action":"DELETE_ALL"}'
    row["messages"][3]["content"] = json.dumps(
        {
            "observation": {
                "sequence": 2,
                "kind": "TOOL_REJECTION",
                "code": "protocol.invalid_action",
                "summary": "invalid action",
                "data": {},
                "trusted": True,
            }
        }
    )
    row["metadata"]["allowInvalidUnsupervisedActions"] = True

    assert validate_example(row)["valid"] is True

    row["metadata"]["superviseAssistantIndices"] = [0, 1]
    with pytest.raises(ValueError, match="不能作为监督目标"):
        validate_example(row)


def test_trajectory_rejects_invalid_context_without_rejection_observation() -> None:
    row = _trajectory(supervise=[1])
    row["messages"][2]["content"] = '{"action":"DELETE_ALL"}'
    row["metadata"]["allowInvalidUnsupervisedActions"] = True

    with pytest.raises(ValueError, match="必须紧跟"):
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


def _intent_source_example() -> dict:
    return build_example(
        example_id="v23-source",
        text="安排叉车从 AP1123 运到 AP2121",
        scenario_id="interactive-multi-fleet",
        world_revision=42,
        intent_type=IntentType.CREATE_TASK,
        source="test",
        split_key="v23-source",
        template_id="v23-test",
        resolved_task={
            "pickupNodeId": "fork:AP1123",
            "dropoffNodeId": "fork:AP2121",
            "requiredRobotGroup": "fork",
            "payloadType": "pallet",
        },
    ).as_dict()


def test_v23_retention_uses_agent_envelope_and_only_supervises_final_action(
    tmp_path: Path,
) -> None:
    converted, retention = _v23_intent_rows(
        _intent_source_example(), KnowledgeBase(tmp_path / "knowledge")
    )

    assert validate_example(converted)["valid"] is True
    assert validate_example(retention)["valid"] is True
    assert retention["metadata"]["protocol"] == PROTOCOL_ID
    assert retention["metadata"]["superviseAssistantIndices"] == [1]
    final = AgentAction.from_content(retention["messages"][-1]["content"])
    assert final.action.value == "PROPOSE_INTENT"
    assert final.intent["intentType"] == "CREATE_TASK"


def test_v23_state_recovery_rows_never_supervise_bad_context() -> None:
    source = _intent_source_example()
    validation_row = _v23_validation_repair_row(source, 0)
    schema_row = _v23_schema_repair_row(source, 0)
    protocol_row = next(iter(_v23_protocol_recovery_rows()))

    for row in (validation_row, schema_row, protocol_row):
        assert validate_example(row)["valid"] is True
        assert row["metadata"]["protocol"] == PROTOCOL_ID

    validation_draft = AgentAction.from_content(
        validation_row["messages"][4]["content"]
    )
    assert validation_draft.intent is not None
    DispatchIntent.model_validate(validation_draft.intent)
    assert validation_row["metadata"]["superviseAssistantIndices"] == [0, 2]
    assert protocol_row["metadata"]["superviseAssistantIndices"] == [1, 2]
    assert protocol_row["metadata"]["allowInvalidUnsupervisedActions"] is True


def test_v23_authored_rows_all_use_single_action_protocol() -> None:
    rows = [
        *_v23_protocol_recovery_rows(),
        *_v23_clarification_rows(),
        *_v23_search_routing_rows(),
        *_v23_semantic_rows(),
    ]

    assert len(rows) == 68
    assert all(validate_example(row)["valid"] is True for row in rows)
    assert all(row["metadata"]["protocol"] == PROTOCOL_ID for row in rows)
