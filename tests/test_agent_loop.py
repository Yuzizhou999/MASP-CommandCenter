from __future__ import annotations

from command_center.agent_protocol import AgentAction, AgentActionType, AgentBudgets
from command_center.audit import AuditStore
from command_center.contracts import (
    ChatRequest,
    IntentValidation,
    RiskLevel,
    ValidationIssue,
)
from command_center.engine_adapter import MaspAdapter
from command_center.knowledge import KnowledgeBase
from command_center.orchestrator import DispatchOrchestrator
from command_center.provider import AgentDecisionResult, DeepSeekProvider


def _loop_orchestrator(
    isolated_settings, provider=None, budgets=None
) -> DispatchOrchestrator:
    engine = MaspAdapter(isolated_settings)
    return DispatchOrchestrator(
        engine=engine,
        provider=provider or DeepSeekProvider(isolated_settings),
        knowledge=KnowledgeBase(isolated_settings.root / "knowledge"),
        audit=AuditStore(isolated_settings.data_dir / "audit.jsonl"),
        runtime_mode="loop",
        budgets=budgets,
    )


def test_deterministic_loop_observes_tool_results_before_completion(
    isolated_settings,
) -> None:
    response = _loop_orchestrator(isolated_settings).chat(
        ChatRequest(
            message="当前车辆和任务状态怎么样？",
            scenarioId="interactive-multi-fleet",
            conversationId="loop-observe",
        )
    )

    assert response.state == "READY"
    assert response.agent_trace is not None
    assert response.agent_trace.strategy == "ACTION_PROTOCOL_LOOP"
    assert response.agent_trace.usage["decisions"] == 3
    assert response.agent_trace.usage["toolCalls"] == 2
    actions = [row.action for row in response.agent_trace.steps if row.action]
    assert actions == ["CALL_TOOL", "CALL_TOOL", "PROPOSE_INTENT"]
    snapshot_tool_index = next(
        index
        for index, row in enumerate(response.agent_trace.steps)
        if row.tool_name == "get_world_snapshot"
    )
    next_decision_index = next(
        index
        for index, row in enumerate(response.agent_trace.steps)
        if index > snapshot_tool_index and row.state == "DECIDING"
    )
    assert next_decision_index > snapshot_tool_index


def test_hard_missing_fields_override_model_policy(isolated_settings) -> None:
    response = _loop_orchestrator(isolated_settings).chat(
        ChatRequest(
            message="创建一个紧急叉车任务",
            scenarioId="interactive-multi-fleet",
            conversationId="loop-hard-clarification",
        )
    )

    assert response.state == "CLARIFICATION_REQUIRED"
    assert response.agent_trace is not None
    assert response.agent_trace.usage["decisions"] == 0
    assert response.agent_trace.terminal_reason == "policy.hard_required_fields"


class _MaliciousDriver(DeepSeekProvider):
    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.calls = 0

    def status(self):
        return {"model": "malicious-test-driver"}

    def decide_agent_action(self, text, tool_definitions, *, observations, authoritative_parameters, action_history=None):
        del text, tool_definitions, observations, authoritative_parameters, action_history
        self.calls += 1
        if self.calls == 1:
            action = AgentAction(
                action=AgentActionType.CALL_TOOL,
                tool="get_world_snapshot",
                arguments={},
            )
        else:
            action = AgentAction(
                action=AgentActionType.PROPOSE_INTENT,
                intent={
                    "intentType": "CREATE_TASK",
                    "reason": "poisoned",
                    "task": {
                        "pickupNodeId": "fork:AP1123",
                        "dropoffNodeId": "fork:AP2121",
                        "requiredRobotGroup": "jack",
                        "payloadType": "pallet",
                    },
                },
            )
        return AgentDecisionResult(
            action=action,
            model="malicious-test-driver",
            fallback_used=False,
        )


def test_authority_mismatch_is_non_fixable_block(isolated_settings) -> None:
    driver = _MaliciousDriver(isolated_settings)
    response = _loop_orchestrator(isolated_settings, driver).chat(
        ChatRequest(
            message="创建紧急叉车任务，从 AP1123 运到 AP2121",
            scenarioId="interactive-multi-fleet",
            conversationId="loop-authority",
        )
    )

    assert response.state == "BLOCKED"
    assert response.agent_trace is not None
    assert response.agent_trace.terminal_reason == "intent.task.authority-mismatch"
    assert response.agent_trace.usage["repairAttempts"] == 0


class _UngroundedTaskDriver(_MaliciousDriver):
    def decide_agent_action(self, text, tool_definitions, *, observations, authoritative_parameters, action_history=None):
        del text, tool_definitions, observations, authoritative_parameters, action_history
        self.calls += 1
        if self.calls == 1:
            action = AgentAction(
                action=AgentActionType.CALL_TOOL,
                tool="get_world_snapshot",
                arguments={},
            )
        else:
            action = AgentAction(
                action=AgentActionType.PROPOSE_INTENT,
                intent={
                    "intentType": "CREATE_TASK",
                    "reason": "fabricated task without resolver authority",
                    "task": {
                        "pickupNodeId": "fork:AP1123",
                        "dropoffNodeId": "fork:AP2121",
                        "requiredRobotGroup": "fork",
                        "payloadType": "pallet",
                    },
                },
            )
        return AgentDecisionResult(
            action=action,
            model="ungrounded-test-driver",
            fallback_used=False,
        )


def test_ungrounded_intent_requests_clarification_without_repair_budget(
    isolated_settings,
) -> None:
    driver = _UngroundedTaskDriver(isolated_settings)
    response = _loop_orchestrator(isolated_settings, driver).chat(
        ChatRequest(
            message="当前状态？",
            scenarioId="interactive-multi-fleet",
            conversationId="loop-ungrounded",
        )
    )

    assert response.state == "CLARIFICATION_REQUIRED"
    assert response.clarification is not None
    assert set(response.clarification.missing_fields) == {
        "pickupNodeId",
        "dropoffNodeId",
        "requiredRobotGroup",
    }
    assert response.agent_trace is not None
    assert response.agent_trace.terminal_reason == "intent.task.ungrounded"
    assert response.agent_trace.usage["repairAttempts"] == 0


class _InvalidToolDriver(DeepSeekProvider):
    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.observation_codes: list[list[str]] = []
        self.calls = 0

    def status(self):
        return {"model": "invalid-tool-test-driver"}

    def decide_agent_action(self, text, tool_definitions, *, observations, authoritative_parameters, action_history=None):
        del action_history
        self.observation_codes.append([row.code for row in observations])
        self.calls += 1
        if self.calls == 1:
            action = AgentAction(
                action=AgentActionType.CALL_TOOL,
                tool="delete_all",
                arguments={},
            )
        else:
            return self._deterministic_agent_action(
                text,
                tool_definitions,
                observations=observations,
                authoritative_parameters=authoritative_parameters,
            )
        return AgentDecisionResult(
            action=action,
            model="invalid-tool-test-driver",
            fallback_used=False,
        )


def test_invalid_tool_rejection_is_returned_as_next_observation(
    isolated_settings,
) -> None:
    driver = _InvalidToolDriver(isolated_settings)
    response = _loop_orchestrator(isolated_settings, driver).chat(
        ChatRequest(
            message="当前状态？",
            scenarioId="interactive-multi-fleet",
            conversationId="loop-tool-rejection",
        )
    )

    assert response.state == "READY"
    assert "tool.rejected" in driver.observation_codes[1]
    assert response.agent_trace is not None
    assert any(
        row.status == "REJECTED" and row.tool_name == "delete_all"
        for row in response.agent_trace.steps
    )


class _RepairingDriver(DeepSeekProvider):
    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.calls = 0
        self.validation_observations: list[dict] = []

    def status(self):
        return {"model": "repair-test-driver"}

    def decide_agent_action(self, text, tool_definitions, *, observations, authoritative_parameters, action_history=None):
        del text, tool_definitions, authoritative_parameters, action_history
        self.calls += 1
        if self.calls == 1:
            action = AgentAction(
                action=AgentActionType.CALL_TOOL,
                tool="get_world_snapshot",
                arguments={},
            )
        else:
            validation_rows = [
                row.model_dump(by_alias=True, mode="json")
                for row in observations
                if row.code == "validation.fixable"
            ]
            self.validation_observations.extend(validation_rows)
            action = AgentAction(
                action=AgentActionType.PROPOSE_INTENT,
                intent={
                    "intentType": "CREATE_TASK",
                    "reason": "repaired after verifier feedback" if validation_rows else "initial draft",
                },
            )
        return AgentDecisionResult(
            action=action,
            model="repair-test-driver",
            fallback_used=False,
        )


def test_fixable_masp_issue_is_observed_repaired_and_revalidated(
    isolated_settings,
    monkeypatch,
) -> None:
    driver = _RepairingDriver(isolated_settings)
    orchestrator = _loop_orchestrator(isolated_settings, driver)
    validations = 0

    def validate_twice(intent, scenario_id):
        nonlocal validations
        del scenario_id
        validations += 1
        if validations == 1:
            return IntentValidation(
                intentId=intent.intent_id,
                valid=False,
                riskLevel=RiskLevel.R1_LOW,
                approvalRequired=False,
                policyCode="task.single.simulation",
                issues=[
                    ValidationIssue(
                        code="intent.task.priority.invalid",
                        message="priorityClass 必须位于 1 到 4",
                        severity="error",
                    )
                ],
            )
        return IntentValidation(
            intentId=intent.intent_id,
            valid=True,
            riskLevel=RiskLevel.R1_LOW,
            approvalRequired=False,
            policyCode="task.single.simulation",
            issues=[],
        )

    monkeypatch.setattr(orchestrator.engine, "validate_intent", validate_twice)
    response = orchestrator.chat(
        ChatRequest(
            message="创建紧急叉车任务，从 AP1123 运到 AP2121",
            scenarioId="interactive-multi-fleet",
            conversationId="loop-verifier-repair",
        )
    )

    assert response.state == "READY"
    assert validations == 2
    assert driver.validation_observations == [
        {
            "sequence": 3,
            "kind": "VALIDATION_ISSUES",
            "code": "validation.fixable",
            "summary": "MASP 返回可修复问题",
            "data": {
                "attempt": 1,
                "issues": [
                    {
                        "code": "intent.task.priority.invalid",
                        "message": "priorityClass 必须位于 1 到 4",
                        "severity": "error",
                    }
                ],
            },
            "toolName": None,
            "trusted": True,
        }
    ]
    assert response.intent is not None
    assert response.intent.reason == "repaired after verifier feedback"
    assert response.agent_trace is not None
    assert response.agent_trace.usage["repairAttempts"] == 1
    repair_steps = [
        row for row in response.agent_trace.steps if row.state == "REPAIRING"
    ]
    assert len(repair_steps) == 1
    assert repair_steps[0].attempt == 1
    assert repair_steps[0].observation_code == "validation.fixable"


class _NeverValidDriver(DeepSeekProvider):
    def status(self):
        return {"model": "step-budget-test-driver"}

    def decide_agent_action(self, text, tool_definitions, *, observations, authoritative_parameters, action_history=None):
        del text, tool_definitions, observations, authoritative_parameters, action_history
        return AgentDecisionResult(
            action=None,
            model="step-budget-test-driver",
            fallback_used=False,
            error_code="protocol.invalid_action",
            error_message="deliberate invalid action",
        )


def test_step_limit_returns_structured_budget_terminal(isolated_settings) -> None:
    response = _loop_orchestrator(
        isolated_settings,
        _NeverValidDriver(isolated_settings),
        AgentBudgets(maxSteps=8, maxDecisions=32),
    ).chat(
        ChatRequest(
            message="当前状态？",
            scenarioId="interactive-multi-fleet",
            conversationId="loop-step-budget",
        )
    )

    assert response.state == "BUDGET_EXCEEDED"
    assert response.agent_trace is not None
    assert response.agent_trace.status == "BUDGET_EXCEEDED"
    assert response.agent_trace.terminal_reason == "budget.steps"
    assert len(response.agent_trace.steps) == 8
    assert response.agent_trace.steps[-1].state == "BUDGET_EXCEEDED"


class _CostlyDriver(DeepSeekProvider):
    def status(self):
        return {"model": "cost-budget-test-driver"}

    def decide_agent_action(self, text, tool_definitions, *, observations, authoritative_parameters, action_history=None):
        del text, tool_definitions, observations, authoritative_parameters, action_history
        return AgentDecisionResult(
            action=AgentAction(
                action=AgentActionType.CALL_TOOL,
                tool="get_world_snapshot",
                arguments={},
            ),
            model="cost-budget-test-driver",
            fallback_used=False,
            prompt_tokens=100,
            completion_tokens=10,
            estimated_cost_usd=0.02,
        )


def test_estimated_cost_limit_is_a_hard_runtime_budget(isolated_settings) -> None:
    response = _loop_orchestrator(
        isolated_settings,
        _CostlyDriver(isolated_settings),
        AgentBudgets(maxEstimatedCostUsd=0.01),
    ).chat(
        ChatRequest(
            message="当前状态？",
            scenarioId="interactive-multi-fleet",
            conversationId="loop-cost-budget",
        )
    )

    assert response.state == "BUDGET_EXCEEDED"
    assert response.agent_trace is not None
    assert response.agent_trace.terminal_reason == "budget.cost"
    assert response.agent_trace.usage["estimatedCostUsd"] == 0.02
