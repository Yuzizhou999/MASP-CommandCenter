from __future__ import annotations

import json

import pytest

from command_center.audit import AuditStore
from command_center.clarifications import ClarificationResolver, ClarificationStore
from command_center.contracts import (
    ChatRequest,
    PlanExplanationRequest,
    SimulationRequest,
)
from command_center.engine_adapter import MaspAdapter
from command_center.explanations import PlanExplanationService
from command_center.knowledge import KnowledgeBase
from command_center.orchestrator import DispatchOrchestrator
from command_center.provider import DeepSeekProvider


def _orchestrator(isolated_settings):
    engine = MaspAdapter(isolated_settings)
    audit = AuditStore(isolated_settings.data_dir / "audit.jsonl")
    resolver = ClarificationResolver(
        ClarificationStore(isolated_settings.data_dir / "clarifications.json"), engine
    )
    return (
        engine,
        audit,
        DispatchOrchestrator(
            engine=engine,
            provider=DeepSeekProvider(isolated_settings),
            knowledge=KnowledgeBase(isolated_settings.root / "knowledge"),
            audit=audit,
            clarifications=resolver,
        ),
    )


def test_missing_task_parameters_are_clarified_without_default_ids(
    isolated_settings,
) -> None:
    _, _, orchestrator = _orchestrator(isolated_settings)
    response = orchestrator.chat(
        ChatRequest(
            message="创建一个紧急叉车任务",
            scenarioId="interactive-multi-fleet",
            requestedBy="tester",
            conversationId="conversation-clarify",
        )
    )

    assert response.state == "CLARIFICATION_REQUIRED"
    assert response.intent is None
    assert response.validation is None
    assert response.clarification is not None
    assert set(response.clarification.missing_fields) == {
        "pickupNodeId",
        "dropoffNodeId",
    }
    collected = json.dumps(
        response.clarification.collected_parameters, ensure_ascii=False
    )
    assert "fork:AP1123" not in collected
    assert "fork:AP2121" not in collected


def test_clarification_follow_up_completes_exact_task_parameters(
    isolated_settings,
) -> None:
    _, audit, orchestrator = _orchestrator(isolated_settings)
    conversation_id = "conversation-follow-up"
    first = orchestrator.chat(
        ChatRequest(
            message="创建一个紧急叉车任务",
            scenarioId="interactive-multi-fleet",
            conversationId=conversation_id,
        )
    )
    assert first.state == "CLARIFICATION_REQUIRED"

    second = orchestrator.chat(
        ChatRequest(
            message="从 AP1123 运到 AP2121",
            scenarioId="interactive-multi-fleet",
            conversationId=conversation_id,
        )
    )

    assert second.state == "READY"
    assert second.intent is not None and second.intent.task is not None
    assert second.intent.task.pickup_node_id == "fork:AP1123"
    assert second.intent.task.dropoff_node_id == "fork:AP2121"
    assert second.intent.task.required_robot_group == "fork"
    event_types = [row.event_type for row in audit.latest(10)]
    assert "AGENT_CLARIFICATION_REQUESTED" in event_types
    assert "AGENT_INTENT_PARSED" in event_types


def test_semantic_dropoff_is_not_reused_as_missing_pickup(isolated_settings) -> None:
    _, _, orchestrator = _orchestrator(isolated_settings)
    response = orchestrator.chat(
        ChatRequest(
            message="创建叉车任务，送到 AP2121",
            scenarioId="interactive-multi-fleet",
            conversationId="conversation-one-endpoint",
        )
    )

    assert response.clarification is not None
    assert response.clarification.missing_fields == ["pickupNodeId"]
    assert response.clarification.collected_parameters["dropoffNodeId"] == "fork:AP2121"


def test_generic_resource_block_requires_target(isolated_settings) -> None:
    _, _, orchestrator = _orchestrator(isolated_settings)
    response = orchestrator.chat(
        ChatRequest(
            message="请封路三分钟",
            scenarioId="interactive-multi-fleet",
            conversationId="conversation-block",
        )
    )

    assert response.state == "CLARIFICATION_REQUIRED"
    assert response.intent is None
    assert response.clarification is not None
    assert response.clarification.missing_fields == ["resourceIds"]
    assert response.clarification.collected_parameters["durationMs"] == 180000


@pytest.mark.parametrize(
    ("message", "missing_fields"),
    [
        (
            "新增一个紧急运输任务",
            {"pickupNodeId", "dropoffNodeId", "requiredRobotGroup"},
        ),
        ("帮我把货送过去", {"pickupNodeId", "dropoffNodeId", "requiredRobotGroup"}),
        ("临时停用一条通道", {"resourceIds"}),
    ],
)
def test_expanded_dispatch_wording_enters_clarification(
    isolated_settings, message: str, missing_fields: set[str]
) -> None:
    _, _, orchestrator = _orchestrator(isolated_settings)
    response = orchestrator.chat(
        ChatRequest(
            message=message,
            scenarioId="interactive-multi-fleet",
            conversationId=f"wording-{len(message)}",
        )
    )

    assert response.state == "CLARIFICATION_REQUIRED"
    assert response.clarification is not None
    assert set(response.clarification.missing_fields) == missing_fields


def test_holdout_task_wording_resolves_authoritative_entities(
    isolated_settings,
) -> None:
    _, _, orchestrator = _orchestrator(isolated_settings)
    response = orchestrator.chat(
        ChatRequest(
            message="把 AP1123 的托盘紧急送往 AP2121，使用叉车",
            scenarioId="interactive-multi-fleet",
            conversationId="holdout-task-wording",
        )
    )

    assert response.state == "READY"
    assert response.intent is not None and response.intent.task is not None
    assert response.intent.task.pickup_node_id == "fork:AP1123"
    assert response.intent.task.dropoff_node_id == "fork:AP2121"


def test_holdout_resource_wording_resolves_target_and_chinese_duration(
    isolated_settings,
) -> None:
    _, _, orchestrator = _orchestrator(isolated_settings)
    response = orchestrator.chat(
        ChatRequest(
            message="将 zone:zone-jack-pp363-pp365 暂停开放两分钟",
            scenarioId="interactive-multi-fleet",
            conversationId="holdout-resource-wording",
        )
    )

    assert response.state == "READY"
    assert response.intent is not None and response.intent.resource_block is not None
    assert response.intent.resource_block.resource_ids == ["zone:zone-jack-pp363-pp365"]
    assert response.intent.resource_block.end_ms == 120000


def test_plan_explanation_cites_persisted_masp_evidence(isolated_settings) -> None:
    engine = MaspAdapter(isolated_settings)
    audit = AuditStore(isolated_settings.data_dir / "audit.jsonl")
    summary = engine.simulate(
        SimulationRequest(
            scenarioId="interactive-multi-fleet",
            label="规划解释测试",
            policy="congestion",
            seed=3,
        )
    )
    service = PlanExplanationService(
        engine=engine,
        provider=DeepSeekProvider(isolated_settings),
        audit=audit,
    )

    report = service.explain(
        summary.run_id,
        PlanExplanationRequest(question="为什么这样分配并安排等待？"),
    )

    evidence_ids = {row.evidence_id for row in report.evidence}
    assert report.fallback_used is True
    assert report.model == "deterministic-evidence-explainer"
    assert evidence_ids
    assert any(row.category == "ASSIGNMENT" for row in report.evidence)
    assert any(row.category == "ROUTE" for row in report.evidence)
    assert all(set(row.evidence_ids).issubset(evidence_ids) for row in report.findings)
    assert all(row.classification in {"FACT", "INFERENCE"} for row in report.findings)
    assert audit.latest(1)[0].event_type == "PLAN_EXPLAINED"


def test_plan_explanation_rejects_unknown_filter(isolated_settings) -> None:
    engine = MaspAdapter(isolated_settings)
    audit = AuditStore(isolated_settings.data_dir / "audit.jsonl")
    summary = engine.simulate(
        SimulationRequest(
            scenarioId="explicit-single-vehicle",
            label="解释过滤测试",
        )
    )
    service = PlanExplanationService(
        engine=engine,
        provider=DeepSeekProvider(isolated_settings),
        audit=audit,
    )

    with pytest.raises(ValueError, match="不存在车辆"):
        service.explain(
            summary.run_id,
            PlanExplanationRequest(question="解释车辆", vehicleId="invented-vehicle"),
        )
