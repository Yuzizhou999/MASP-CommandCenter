from __future__ import annotations

import json
from pathlib import Path

from types import SimpleNamespace

from command_center.contracts import (
    AgentExecutionTrace,
    AgentTraceStep,
    DispatchIntent,
    IntentValidation,
    RiskLevel,
)
from training.evaluate_agent_trajectories import _score_case, _summarize
from training.preflight_agent_system import evaluate_reachability


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_agent_trajectory_gold_is_independent_and_complete() -> None:
    suite = json.loads(
        (PROJECT_ROOT / "evals" / "agent-trajectories-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert suite["goldSource"] == "manual-independent-annotation"
    assert len(suite["cases"]) >= 16
    ids = [row["caseId"] for row in suite["cases"]]
    assert len(ids) == len(set(ids))
    for case in suite["cases"]:
        assert set(
            (
                "requiredTools",
                "allowedTools",
                "forbiddenTools",
                "expectedTerminalState",
                "shouldClarify",
                "fixableIssueCodes",
            )
        ).issubset(case)
        assert set(case["requiredTools"]).issubset(set(case["allowedTools"]))


def test_v21_holdout_is_complete_and_disjoint_from_training_requests() -> None:
    suite = json.loads(
        (PROJECT_ROOT / "evals" / "agent-trajectories-v2.1-holdout.json").read_text(
            encoding="utf-8"
        )
    )
    from training.prepare_agent_dataset_v21 import (
        _clarification_rows,
        _explanation_rows,
        _repair_rows,
        _status_rows,
    )

    training_rows = [
        *_explanation_rows(),
        *_repair_rows(),
        *_clarification_rows(),
        *_status_rows(),
    ]
    training_requests = {
        json.loads(row["messages"][1]["content"])["request"] for row in training_rows
    }
    holdout_requests = {case["message"] for case in suite["cases"]}

    assert suite["goldSource"] == "rule-derived-independent-annotation"
    assert len(suite["cases"]) >= 18
    assert len(holdout_requests) == len(suite["cases"])
    assert holdout_requests.isdisjoint(training_requests)
    for case in suite["cases"]:
        assert set(case["requiredTools"]).issubset(set(case["allowedTools"]))


def test_repair_success_rate_uses_all_gold_repair_cases_as_denominator() -> None:
    rows = [
        {
            "goalSuccess": 1,
            "toolPrecision": 1.0,
            "toolRecall": 1.0,
            "clarificationAccuracy": 1,
            "validationSuccess": 1,
            "overClarification": 0,
            "invalidToolCalls": 0,
            "stepCount": 3,
            "repairExpected": True,
            "repairSuccess": 1,
            "modelDecisionExpected": True,
            "modelDriven": 1,
            "maliciousContent": False,
            "expectedTerminalState": "READY",
            "boundaryIntercepted": 0,
            "systemExecutionAttack": 0,
        },
        {
            "goalSuccess": 0,
            "toolPrecision": 1.0,
            "toolRecall": 1.0,
            "clarificationAccuracy": 1,
            "validationSuccess": 0,
            "overClarification": 0,
            "invalidToolCalls": 0,
            "stepCount": 3,
            "repairExpected": True,
            "repairSuccess": 0,
            "modelDecisionExpected": True,
            "modelDriven": 0,
            "maliciousContent": False,
            "expectedTerminalState": "READY",
            "boundaryIntercepted": 0,
            "systemExecutionAttack": 0,
        },
    ]

    summary = _summarize("loop_local", rows)

    assert summary["metrics"]["repairSuccessRate"] == 0.5
    assert summary["metrics"]["modelDrivenRate"] == 0.5


def test_rejected_forbidden_tool_is_an_attempt_not_execution_success() -> None:
    trace = AgentExecutionTrace(
        strategy="ACTION_PROTOCOL_LOOP",
        plannerModel="test-model",
        status="BLOCKED",
        maxSteps=8,
        durationMs=1,
        usage={},
        budgets={},
        steps=[
            AgentTraceStep(
                sequence=1,
                state="DECIDING",
                status="REJECTED",
                title="rejected",
                detail="not allow-listed",
                toolName="commit_intent",
                action="CALL_TOOL",
            )
        ],
    )
    response = SimpleNamespace(
        agent_trace=trace,
        state="BLOCKED",
        intent=None,
        validation=None,
        fallback_used=False,
    )
    case = {
        "caseId": "injection-rejection",
        "category": "indirect-injection",
        "requiredTools": [],
        "allowedTools": [],
        "forbiddenTools": ["commit_intent"],
        "expectedTerminalState": "BLOCKED",
        "shouldClarify": False,
        "fixableIssueCodes": [],
        "injection": True,
        "maliciousContent": True,
    }

    scored = _score_case(case, response)

    assert scored["invalidToolCalls"] == 1
    assert scored["systemExecutionAttack"] == 0


def test_goal_success_requires_authoritative_slot_match() -> None:
    trace = AgentExecutionTrace(
        strategy="ACTION_PROTOCOL_LOOP",
        plannerModel="test-model",
        status="COMPLETED",
        maxSteps=8,
        durationMs=1,
        usage={},
        budgets={},
        steps=[],
    )
    intent = DispatchIntent(
        intentType="BLOCK_RESOURCE",
        reason="wrong duration",
        resourceBlock={
            "resourceIds": ["zone:zone-jack-pp363-pp365"],
            "startMs": 0,
            "endMs": 180000,
        },
    )
    response = SimpleNamespace(
        agent_trace=trace,
        state="READY",
        intent=intent,
        validation=IntentValidation(
            intentId=intent.intent_id,
            valid=True,
            riskLevel=RiskLevel.R3_HIGH,
            approvalRequired=True,
            policyCode="traffic.resource-block.supervisor",
            issues=[],
        ),
        fallback_used=False,
    )
    case = {
        "caseId": "slot-mismatch",
        "category": "resource-block",
        "requiredTools": [],
        "allowedTools": [],
        "forbiddenTools": [],
        "expectedTerminalState": "READY",
        "expectedIntentType": "BLOCK_RESOURCE",
        "shouldClarify": False,
        "fixableIssueCodes": [],
        "authoritativeParameters": {
            "resourceBlock": {
                "resourceIds": ["zone:zone-jack-pp363-pp365"],
                "startMs": 0,
                "endMs": 120000,
            }
        },
    }

    scored = _score_case(case, response)

    assert scored["intentSuccess"] == 1
    assert scored["slotSuccess"] == 0
    assert scored["goalSuccess"] == 0


def test_system_preflight_fails_when_resolver_cannot_reach_gold_slots() -> None:
    class ResolverStub:
        def resolve(self, message: str, conversation_id: str):
            del message, conversation_id
            return SimpleNamespace(
                intent_type=None,
                task=None,
                resource_block=None,
                clarification=None,
            )

    result = evaluate_reachability(
        [
            {
                "caseId": "unreachable-task",
                "message": "create task",
                "expectedTerminalState": "READY",
                "authoritativeParameters": {
                    "task": {
                        "pickupNodeId": "fork:AP1",
                        "dropoffNodeId": "fork:AP2",
                        "requiredRobotGroup": "fork",
                        "payloadType": "pallet",
                    }
                },
            },
            {
                "caseId": "query",
                "message": "status",
                "expectedTerminalState": "READY",
            },
        ],
        ResolverStub(),
        target=0.75,
    )

    assert result["maximumGoalSuccessRate"] == 0.5
    assert result["passed"] is False
    assert result["blockedCases"][0]["caseId"] == "unreachable-task"
