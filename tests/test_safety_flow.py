from __future__ import annotations

from pathlib import Path

import pytest

from command_center.approvals import ApprovalStore
from command_center.audit import AuditStore
from command_center.contracts import (
    ApprovalDecision,
    ChatRequest,
    DispatchIntent,
    IntentType,
    ResourceBlockDraft,
    SimulationRequest,
)
from command_center.engine_adapter import MaspAdapter
from command_center.intent_store import IntentStore
from command_center.knowledge import KnowledgeBase
from command_center.orchestrator import DispatchOrchestrator
from command_center.provider import DeepSeekProvider


def test_unknown_resource_is_rejected(isolated_settings) -> None:
    engine = MaspAdapter(isolated_settings)
    scenario_id = "interactive-multi-fleet"
    intent = DispatchIntent(
        intentType=IntentType.BLOCK_RESOURCE,
        basedOnWorldRevision=engine.world_revision(scenario_id),
        resourceBlock=ResourceBlockDraft(
            resourceIds=["zone:model-invented-this"],
            startMs=0,
            endMs=10000,
        ),
    )
    validation = engine.validate_intent(intent, scenario_id)
    assert validation.valid is False
    assert validation.approval_required is True
    assert any(issue.code == "intent.resource.unknown" for issue in validation.issues)


def test_r3_flow_requires_simulation_and_human_approval(isolated_settings) -> None:
    project_root = Path(__file__).resolve().parents[1]
    engine = MaspAdapter(isolated_settings)
    audit = AuditStore(isolated_settings.data_dir / "audit.jsonl")
    orchestrator = DispatchOrchestrator(
        engine=engine,
        provider=DeepSeekProvider(isolated_settings),
        knowledge=KnowledgeBase(project_root / "knowledge"),
        audit=audit,
    )
    scenario_id = "interactive-multi-fleet"
    response = orchestrator.chat(
        ChatRequest(
            message="共享窄路检修，请封路三分钟",
            scenarioId=scenario_id,
            requestedBy="integration-tester",
        )
    )
    assert response.validation is not None
    assert response.validation.valid is True
    assert response.validation.approval_required is True
    assert response.intent is not None

    summary = engine.simulate(
        SimulationRequest(
            scenarioId=scenario_id,
            label="R3 integration",
            intent=response.intent,
        )
    )
    assert summary.status == "COMPLETED"
    assert summary.safety["conflictFree"] is True
    assert summary.metrics["reservationConflictRejections"] == 0

    approvals = ApprovalStore(isolated_settings.data_dir / "approvals.json")
    request = approvals.create(response.intent, response.validation, [summary.run_id])
    store = IntentStore(isolated_settings.data_dir / "intents.json")
    with pytest.raises(ValueError, match="尚未批准"):
        store.commit(
            response.intent,
            current_world_revision=engine.world_revision(scenario_id),
            approval=request,
            actor="integration-tester",
        )

    approved = approvals.decide(
        request.approval_id,
        ApprovalDecision(approved=True, decidedBy="supervisor", reason="仿真无冲突"),
    )
    committed = store.commit(
        response.intent,
        current_world_revision=engine.world_revision(scenario_id),
        approval=approved,
        actor="integration-tester",
    )
    assert committed["status"] == "SIMULATION_COMMITTED"
    assert committed["environment"] == "simulation"


def test_commit_rejects_stale_world_revision(isolated_settings) -> None:
    store = IntentStore(isolated_settings.data_dir / "intents.json")
    intent = DispatchIntent(
        intentType=IntentType.QUERY_STATUS,
        basedOnWorldRevision=10,
        query="status",
    )
    with pytest.raises(ValueError, match="世界状态已经变化"):
        store.commit(intent, current_world_revision=11, approval=None, actor="tester")


def test_commit_rejects_non_simulation_environment(isolated_settings) -> None:
    store = IntentStore(isolated_settings.data_dir / "intents.json")
    intent = DispatchIntent(
        intentType=IntentType.QUERY_STATUS,
        environment="production",
        basedOnWorldRevision=10,
        query="status",
    )
    with pytest.raises(ValueError, match="只允许提交到simulation"):
        store.commit(intent, current_world_revision=10, approval=None, actor="tester")
