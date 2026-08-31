from __future__ import annotations

import json
from pathlib import Path

import pytest

from command_center.agent_protocol import AgentAction
from command_center.clarifications import ClarificationResolver, ClarificationStore
from command_center.contracts import IntentType
from command_center.engine_adapter import MaspAdapter
from command_center.model_safety import model_request_violation
from training import serve_intent_model
from training.evaluate_intent_challenge import (
    agent_intent_challenge_messages,
    classification_metrics,
    load_suite,
    parse_raw_agent_intent,
    parse_raw_intent,
)
from training.serve_intent_model import (
    resolve_model_spec,
    response_json_schema,
    validate_generated_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_challenge_suite_has_balanced_intents_and_independent_cases() -> None:
    suite = load_suite(PROJECT_ROOT / "evals" / "intent-challenge-v1.json")
    categories = [row["expected"]["intentType"] for row in suite["cases"]]

    assert len(suite["cases"]) == 50
    assert {name: categories.count(name) for name in set(categories)} == {
        "QUERY_STATUS": 10,
        "EXPLAIN_DECISION": 10,
        "CREATE_TASK": 10,
        "BLOCK_RESOURCE": 10,
        "GENERATE_REPORT": 10,
    }
    assert len(suite["safetyCases"]) == 10
    assert len(suite["clarificationCases"]) == 10


def test_raw_parser_does_not_replace_model_task_slots() -> None:
    content = json.dumps(
        {
            "intentType": "CREATE_TASK",
            "reason": "model output",
            "task": {
                "pickupNodeId": "fork:WRONG-1",
                "dropoffNodeId": "fork:WRONG-2",
                "requiredRobotGroup": "fork",
                "payloadType": "pallet",
            },
        }
    )

    json_valid, intent, error = parse_raw_intent(
        content, world_revision=12, requested_by="evaluator"
    )

    assert json_valid is True
    assert error is None
    assert intent is not None and intent.task is not None
    assert intent.task.pickup_node_id == "fork:WRONG-1"
    assert intent.task.dropoff_node_id == "fork:WRONG-2"
    assert intent.based_on_world_revision == 12


def test_agent_protocol_parser_unwraps_proposed_intent() -> None:
    json_valid, intent, error = parse_raw_agent_intent(
        json.dumps(
            {
                "action": "PROPOSE_INTENT",
                "intent": {
                    "intentType": "QUERY_STATUS",
                    "reason": "查询",
                    "query": "查询",
                },
            }
        ),
        world_revision=9,
        requested_by="test",
    )

    assert json_valid is True
    assert error is None
    assert intent is not None and intent.intent_type is IntentType.QUERY_STATUS
    assert intent.based_on_world_revision == 9


def test_agent_intent_challenge_uses_context_ready_action_prompt() -> None:
    messages = agent_intent_challenge_messages(
        {
            "message": "查询当前状态",
            "authoritativeParameters": {"task": None, "resourceBlock": None},
        },
        world_revision=7,
    )

    assert [row["role"] for row in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    action = AgentAction.from_content(messages[2]["content"])
    assert action.tool == "get_world_snapshot"


def test_agent_intent_challenge_preloads_required_sop_context() -> None:
    messages = agent_intent_challenge_messages(
        {
            "message": "解释当前调度依据",
            "expected": {"intentType": "EXPLAIN_DECISION"},
        },
        world_revision=7,
    )

    assert [row["role"] for row in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert AgentAction.from_content(messages[4]["content"]).tool == "search_sop"
    assert '"toolName": "search_sop"' in messages[5]["content"]


def test_classification_metrics_include_every_supported_intent() -> None:
    labels = [
        "QUERY_STATUS",
        "EXPLAIN_DECISION",
        "CREATE_TASK",
        "BLOCK_RESOURCE",
        "GENERATE_REPORT",
    ]
    observed = [*labels[:-1], "QUERY_STATUS"]

    result = classification_metrics(labels, observed)

    assert result["accuracy"] == 0.8
    assert result["macroF1"] < 1
    assert set(result["perIntent"]) == set(labels)


def test_model_service_can_resolve_base_model_without_adapter(tmp_path: Path) -> None:
    spec = resolve_model_spec(
        tmp_path / "missing-adapter",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        model_id="qwen-base",
    )

    assert spec["mode"] == "base-model"
    assert spec["adapterDir"] is None
    assert spec["modelId"] == "qwen-base"


def test_model_service_extracts_supported_response_schemas() -> None:
    assert response_json_schema({"type": "json_object"}) == {"type": "object"}
    schema = {"type": "object", "required": ["action"]}
    assert (
        response_json_schema(
            {
                "type": "json_schema",
                "json_schema": {"name": "agent_action", "schema": schema},
            }
        )
        == schema
    )


def test_model_service_rejects_invalid_response_formats() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        response_json_schema({"type": "text"})
    with pytest.raises(ValueError, match="schema object"):
        response_json_schema({"type": "json_schema", "json_schema": {"name": "broken"}})


def test_model_service_rejects_incomplete_or_schema_invalid_generation() -> None:
    schema = {
        "type": "object",
        "properties": {"action": {"const": "REQUEST_CLARIFICATION"}},
        "required": ["action"],
        "additionalProperties": False,
    }

    assert validate_generated_json('{"action":"REQUEST_CLARIFICATION"}', schema) == {
        "action": "REQUEST_CLARIFICATION"
    }
    with pytest.raises(ValueError, match="incomplete JSON"):
        validate_generated_json('{"action":"REQUEST_CLARIFICATION"', schema)
    with pytest.raises(ValueError, match="requested schema"):
        validate_generated_json('{"action":"CALL_TOOL"}', schema)


def test_model_service_only_requires_xgrammar_for_strict_eval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        serve_intent_model,
        "_dependencies",
        lambda: {"torch": object(), "xgrammar": None},
    )

    app = serve_intent_model.create_app(
        tmp_path,
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
    )
    assert app.title == "MASP Intent Model API"

    with pytest.raises(RuntimeError, match="缺少 xgrammar"):
        serve_intent_model.create_app(
            tmp_path,
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            require_xgrammar=True,
        )


def test_challenge_safety_cases_are_blocked_before_model_call() -> None:
    suite = load_suite(PROJECT_ROOT / "evals" / "intent-challenge-v1.json")

    assert all(
        model_request_violation(row["message"]) is not None
        for row in suite["safetyCases"]
    )


def test_challenge_normal_cases_do_not_trigger_model_request_gate() -> None:
    suite = load_suite(PROJECT_ROOT / "evals" / "intent-challenge-v1.json")

    assert all(
        model_request_violation(row["message"]) is None for row in suite["cases"]
    )


def test_raw_model_safety_is_diagnostic_not_a_qualification_threshold() -> None:
    suite = load_suite(PROJECT_ROOT / "evals" / "intent-challenge-v1.json")
    qualification = suite["qualification"]

    assert "rawSafetyPassRate" not in qualification["modelThresholds"]
    assert qualification["systemThresholds"] == {
        "systemSafetyGateRecall": 1.0,
        "clarificationAccuracy": 1.0,
    }


def test_challenge_clarification_states_match(
    isolated_settings, tmp_path: Path
) -> None:
    suite = load_suite(PROJECT_ROOT / "evals" / "intent-challenge-v1.json")
    resolver = ClarificationResolver(
        ClarificationStore(tmp_path / "clarifications.json"),
        MaspAdapter(isolated_settings),
    )

    observed = []
    for index, row in enumerate(suite["clarificationCases"]):
        result = resolver.resolve(row["message"], f"challenge-test-{index}")
        observed.append("CLARIFICATION_REQUIRED" if result.clarification else "READY")

    assert observed == [row["expected"]["state"] for row in suite["clarificationCases"]]
