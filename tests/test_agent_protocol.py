from __future__ import annotations

import json

import pytest
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from pydantic import ValidationError

from command_center.agent_protocol import (
    AgentAction,
    AgentActionType,
    AgentBudgetExceeded,
    AgentBudgets,
    AgentBudgetTracker,
    agent_action_response_schema,
    classify_validation,
)
from command_center.agent_runtime import (
    AgentState,
    AgentStepLimitExceeded,
    BoundedAgentRun,
)
from command_center.contracts import IntentValidation, ValidationIssue


def test_single_action_protocol_accepts_each_legal_action() -> None:
    tool = AgentAction.from_content(
        '{"action":"CALL_TOOL","tool":"search_sop","arguments":{"query":"安全"}}'
    )
    clarify = AgentAction.from_content('{"action":"REQUEST_CLARIFICATION"}')
    proposal = AgentAction.from_content(
        '{"action":"PROPOSE_INTENT","intent":{"intentType":"QUERY_STATUS"}}'
    )

    assert tool.action is AgentActionType.CALL_TOOL
    assert clarify.model_dump(exclude_none=True) == {
        "action": AgentActionType.REQUEST_CLARIFICATION
    }
    assert proposal.intent == {"intentType": "QUERY_STATUS"}


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "CALL_TOOL"},
        {"action": "REQUEST_CLARIFICATION", "question": "哪里？"},
        {"action": "REQUEST_CLARIFICATION", "intent": {}},
        {"action": "PROPOSE_INTENT"},
        {"action": "DELETE_ALL"},
    ],
)
def test_single_action_protocol_rejects_invalid_shapes(payload: dict) -> None:
    with pytest.raises((ValidationError, ValueError)):
        AgentAction.from_content(json.dumps(payload, ensure_ascii=False))


def test_agent_generation_schema_enforces_action_and_intent_shapes() -> None:
    schema = agent_action_response_schema()
    validate_json_schema(
        {
            "action": "PROPOSE_INTENT",
            "intent": {"intentType": "QUERY_STATUS", "reason": "查询状态"},
        },
        schema,
    )
    validate_json_schema(
        {
            "action": "CALL_TOOL",
            "tool": "get_world_snapshot",
            "arguments": {},
        },
        schema,
    )

    invalid = [
        {"action": "REQUEST_CLARIFICATION", "question": "哪里？"},
        {"action": "PROPOSE_INTENT", "intent": {"intentType": "DELETE_ALL"}},
    ]
    for payload in invalid:
        with pytest.raises(JsonSchemaValidationError):
            validate_json_schema(payload, schema)


def test_agent_generation_schema_has_strict_action_branches() -> None:
    schema = agent_action_response_schema()
    branches = schema["oneOf"]
    assert len(branches) == 7
    assert {row["properties"]["action"]["const"] for row in branches} == {
        "CALL_TOOL",
        "REQUEST_CLARIFICATION",
        "PROPOSE_INTENT",
    }


def test_budget_tracker_enforces_independent_limits() -> None:
    tracker = AgentBudgetTracker(
        AgentBudgets(maxDecisions=1, maxToolCalls=1, maxRepairAttempts=0)
    )
    tracker.consume_decision(32)
    tracker.consume_tool_call()

    with pytest.raises(AgentBudgetExceeded, match="决策次数"):
        tracker.consume_decision()
    with pytest.raises(AgentBudgetExceeded, match="工具调用"):
        tracker.consume_tool_call()
    with pytest.raises(AgentBudgetExceeded, match="修复次数"):
        tracker.consume_repair()

    cost_tracker = AgentBudgetTracker(
        AgentBudgets(maxEstimatedCostUsd=0.001)
    )
    with pytest.raises(AgentBudgetExceeded, match="估算成本") as error:
        cost_tracker.consume_decision(10, estimated_cost_usd=0.0011)
    assert error.value.code == "budget.cost"
    assert cost_tracker.snapshot()["estimatedCostUsd"] == 0.0011


def test_validation_classifier_fails_closed_and_separates_fixable() -> None:
    validation = IntentValidation(
        intentId="intent-1",
        valid=False,
        riskLevel="R1_LOW",
        approvalRequired=False,
        policyCode="test",
        issues=[
            ValidationIssue(
                code="intent.task.priority.invalid",
                message="priority invalid",
                severity="error",
            ),
            ValidationIssue(
                code="future.unknown.issue",
                message="unknown",
                severity="error",
            ),
        ],
    )

    disposition = classify_validation(validation)

    assert [row.code for row in disposition.fixable] == [
        "intent.task.priority.invalid"
    ]
    assert [row.code for row in disposition.blocking] == ["future.unknown.issue"]
    assert disposition.can_repair is False


def test_reserved_terminal_step_can_end_from_observing_state() -> None:
    run = BoundedAgentRun(max_steps=6, reserve_terminal_step=True)
    run.transition(AgentState.RECEIVED, title="received", detail="received")
    run.transition(AgentState.PLANNING, title="planned", detail="planned")
    run.transition(AgentState.DECIDING, title="deciding", detail="deciding")
    run.transition(
        AgentState.CONTEXT_GATHERING,
        title="context",
        detail="context",
    )
    run.transition(AgentState.OBSERVING, title="observing", detail="observing")

    with pytest.raises(AgentStepLimitExceeded):
        run.transition(AgentState.DECIDING, title="deciding", detail="deciding")

    run.transition(
        AgentState.BUDGET_EXCEEDED,
        title="budget exceeded",
        detail="step budget exhausted",
        status="BLOCKED",
    )
    trace = run.build_trace()
    assert trace.status == "BUDGET_EXCEEDED"
    assert trace.steps[-1].state == "BUDGET_EXCEEDED"
    assert len(trace.steps) == trace.max_steps
