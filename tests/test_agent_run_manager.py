from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic, sleep

from command_center.agent_run_manager import AgentRunManager
from command_center.approvals import ApprovalStore
from command_center.audit import AuditStore
from command_center.contracts import (
    AgentRunCreateRequest,
    AgentRunRecord,
    AgentRunResumeRequest,
    AgentWorkflowRecommendation,
)
from command_center.dispatch_workflow import DispatchWorkflowService
from command_center.engine_adapter import MaspAdapter
from command_center.intent_store import IntentStore
from command_center.knowledge import KnowledgeBase
from command_center.orchestrator import DispatchOrchestrator
from command_center.provider import DeepSeekProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _manager(isolated_settings) -> AgentRunManager:
    provider = DeepSeekProvider(isolated_settings)
    engine = MaspAdapter(isolated_settings)
    audit = AuditStore(isolated_settings.data_dir / "audit.jsonl")
    orchestrator = DispatchOrchestrator(
        engine=engine,
        provider=provider,
        knowledge=KnowledgeBase(PROJECT_ROOT / "knowledge"),
        audit=audit,
    )
    workflow = DispatchWorkflowService(
        engine=engine,
        approvals=ApprovalStore(isolated_settings.data_dir / "approvals.json"),
        intents=IntentStore(isolated_settings.data_dir / "committed-intents.json"),
        audit=audit,
    )
    return AgentRunManager(
        isolated_settings.data_dir / "agent-runs.json",
        orchestrator=orchestrator,
        provider=provider,
        workflow=workflow,
    )


def _wait_for(manager: AgentRunManager, run_id: str, statuses: set[str]):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        record = manager.get(run_id)
        if record.status in statuses:
            return record
        sleep(0.02)
    raise AssertionError(f"Agent run did not reach {statuses}: {manager.get(run_id).status}")


def test_sqlite_store_completes_50_concurrent_runs(isolated_settings) -> None:
    manager = _manager(isolated_settings)

    def create(index: int):
        return manager.create(
            AgentRunCreateRequest(
                message="当前车辆和任务状态怎么样？",
                scenarioId="interactive-multi-fleet",
                conversationId=f"sqlite-concurrent-{index}",
                timeoutSeconds=30,
            ),
            idempotency_key=f"sqlite-concurrent-{index}",
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        created = list(executor.map(create, range(50)))
    deadline = monotonic() + 30
    pending = {row.run_id for row in created}
    while pending and monotonic() < deadline:
        pending = {
            run_id
            for run_id in pending
            if manager.get(run_id).status not in {"COMPLETED", "FAILED"}
        }
        if pending:
            sleep(0.02)

    assert not pending
    records = [manager.get(row.run_id) for row in created]
    assert len({row.run_id for row in records}) == 50
    assert all(row.status == "COMPLETED" for row in records)
    assert all(
        [event.event_id for event in row.events]
        == list(range(1, len(row.events) + 1))
        for row in records
    )
    manager.shutdown()


def test_async_agent_run_is_idempotent_persistent_and_evaluated(
    isolated_settings,
) -> None:
    manager = _manager(isolated_settings)
    request = AgentRunCreateRequest(
        message="当前车辆和任务状态怎么样？",
        scenarioId="interactive-multi-fleet",
        conversationId="async-query",
        timeoutSeconds=10,
    )

    created = manager.create(request, idempotency_key="query-1")
    repeated = manager.create(request, idempotency_key="query-1")
    completed = _wait_for(manager, created.run_id, {"COMPLETED", "FAILED"})

    assert repeated.run_id == created.run_id
    assert completed.status == "COMPLETED"
    assert completed.response is not None
    assert completed.evaluation is not None and completed.evaluation.passed
    assert completed.provider_usage["fallbackCount"] >= 1
    assert [step.sequence for step in completed.trace_steps] == list(
        range(1, len(completed.trace_steps) + 1)
    )
    assert any(event.event_type == "trace_step" for event in completed.events)

    reloaded = _manager(isolated_settings).get(created.run_id)
    assert reloaded.status == "COMPLETED"
    assert reloaded.response is not None


def test_high_risk_run_pauses_and_resumes_after_human_approval(
    isolated_settings,
) -> None:
    manager = _manager(isolated_settings)
    created = manager.create(
        AgentRunCreateRequest(
            message="共享窄路需要检修，请封闭三分钟并评估任务影响",
            scenarioId="interactive-multi-fleet",
            conversationId="async-approval",
            timeoutSeconds=10,
        )
    )

    waiting = _wait_for(manager, created.run_id, {"WAITING_APPROVAL", "FAILED"})
    assert waiting.status == "WAITING_APPROVAL"
    assert waiting.approval is not None
    assert waiting.approval["validation"]["riskLevel"] == "R3_HIGH"

    manager.resume(
        created.run_id,
        AgentRunResumeRequest(
            approved=True,
            decidedBy="shift-supervisor",
            reason="允许形成仿真草案",
        ),
    )
    completed = _wait_for(manager, created.run_id, {"COMPLETED", "FAILED"})

    assert completed.status == "COMPLETED"
    assert completed.response is not None
    assert any(event.event_type == "run_resumed" for event in completed.events)


def test_waiting_agent_run_can_be_cancelled(isolated_settings) -> None:
    manager = _manager(isolated_settings)
    created = manager.create(
        AgentRunCreateRequest(
            message="共享窄路需要检修，请封闭三分钟并评估任务影响",
            scenarioId="interactive-multi-fleet",
            conversationId="async-cancel",
            timeoutSeconds=10,
        )
    )
    _wait_for(manager, created.run_id, {"WAITING_APPROVAL", "FAILED"})

    cancelled = manager.cancel(created.run_id)

    assert cancelled.status == "CANCELLED"
    assert cancelled.cancel_requested is True


def test_service_start_recovers_persisted_queued_run(isolated_settings) -> None:
    manager = _manager(isolated_settings)
    now = datetime.now(UTC)
    run_id = "agent-run-recovery"
    seeded = AgentRunRecord(
        runId=run_id,
        status="QUEUED",
        request=AgentRunCreateRequest(
            message="当前车辆和任务状态怎么样？",
            scenarioId="interactive-multi-fleet",
            conversationId="async-recovery",
            timeoutSeconds=10,
        ),
        deadlineAt=now + timedelta(seconds=10),
        createdAt=now,
        updatedAt=now,
    )
    manager._write(
        {
            "schemaVersion": 1,
            "runs": {
                run_id: seeded.model_dump(by_alias=True, mode="json"),
            },
        }
    )

    recovered_count = manager.recover()
    completed = _wait_for(manager, run_id, {"COMPLETED", "FAILED"})

    assert recovered_count == 1
    assert completed.status == "COMPLETED"
    assert completed.recovered is True
    assert any(event.event_type == "run_recovered" for event in completed.events)


def test_shutdown_releases_worker_without_losing_approval_checkpoint(
    isolated_settings,
) -> None:
    manager = _manager(isolated_settings)
    created = manager.create(
        AgentRunCreateRequest(
            message="共享窄路需要检修，请封闭三分钟并评估任务影响",
            scenarioId="interactive-multi-fleet",
            conversationId="async-shutdown",
            timeoutSeconds=10,
        )
    )
    _wait_for(manager, created.run_id, {"WAITING_APPROVAL", "FAILED"})
    worker = manager._futures[created.run_id]

    manager.shutdown()
    worker.result(timeout=2)

    persisted = manager.get(created.run_id)
    assert persisted.status == "WAITING_APPROVAL"
    assert persisted.approval is not None

    restarted = _manager(isolated_settings)
    restarted.start()
    restarted.resume(
        created.run_id,
        AgentRunResumeRequest(
            approved=True,
            decidedBy="restart-supervisor",
            reason="服务恢复后批准",
        ),
    )
    completed = _wait_for(restarted, created.run_id, {"COMPLETED", "FAILED"})

    assert completed.status == "COMPLETED"
    assert completed.recovered is True
    assert completed.attempt == 2
    assert any(event.event_type == "approval_reused" for event in completed.events)


def test_goal_execution_simulates_and_commits_low_risk_task(
    isolated_settings,
) -> None:
    manager = _manager(isolated_settings)
    created = manager.create(
        AgentRunCreateRequest(
            message="创建紧急叉车任务，从 AP1123 运到 AP2121",
            scenarioId="interactive-multi-fleet",
            conversationId="goal-low-risk",
            timeoutSeconds=30,
            executionMode="GOAL_EXECUTION",
        )
    )

    completed = _wait_for(manager, created.run_id, {"COMPLETED", "FAILED"})

    assert completed.status == "COMPLETED"
    assert completed.workflow is not None
    assert completed.workflow.phase == "COMPLETED"
    assert completed.workflow.simulation is not None
    assert completed.workflow.recommendation is not None
    assert completed.workflow.recommendation.decision == "PROCEED"
    assert completed.workflow.approval_request is None
    assert completed.workflow.commitment is not None
    assert [step.action for step in completed.workflow.steps] == [
        "SIMULATE",
        "COMMIT",
    ]


def test_goal_execution_pauses_after_simulation_and_approval_survives_restart(
    isolated_settings,
) -> None:
    manager = _manager(isolated_settings)
    created = manager.create(
        AgentRunCreateRequest(
            message="共享窄路需要检修，请封闭三分钟并评估任务影响",
            scenarioId="interactive-multi-fleet",
            conversationId="goal-high-risk",
            timeoutSeconds=30,
            executionMode="GOAL_EXECUTION",
        )
    )
    waiting = _wait_for(manager, created.run_id, {"WAITING_APPROVAL", "FAILED"})

    assert waiting.status == "WAITING_APPROVAL"
    assert waiting.workflow is not None
    assert waiting.workflow.simulation is not None
    assert waiting.workflow.approval_request is not None
    simulation_run_id = waiting.workflow.simulation["runId"]
    approval_id = waiting.workflow.approval_request.approval_id
    assert waiting.approval is not None
    assert waiting.approval["stage"] == "POST_SIMULATION"

    worker = manager._futures[created.run_id]
    manager.shutdown()
    worker.result(timeout=2)
    restarted = _manager(isolated_settings)
    restarted.start()
    restarted.resume(
        created.run_id,
        AgentRunResumeRequest(
            approved=True,
            decidedBy="goal-supervisor",
            reason="仿真结果满足安全要求",
        ),
    )
    completed = _wait_for(restarted, created.run_id, {"COMPLETED", "FAILED"})

    assert completed.status == "COMPLETED"
    assert completed.workflow is not None
    assert completed.workflow.phase == "COMPLETED"
    assert completed.workflow.simulation["runId"] == simulation_run_id
    assert completed.workflow.approval_request.approval_id == approval_id
    assert completed.workflow.approval_request.status.value == "APPROVED"
    assert completed.workflow.commitment is not None
    assert len(IntentStore(isolated_settings.data_dir / "committed-intents.json").list()) == 1


def test_waiting_approval_survives_restart_after_original_deadline(
    isolated_settings,
) -> None:
    manager = _manager(isolated_settings)
    created = manager.create(
        AgentRunCreateRequest(
            message="共享窄路需要检修，请封闭三分钟并评估任务影响",
            scenarioId="interactive-multi-fleet",
            conversationId="goal-expired-approval",
            timeoutSeconds=30,
            executionMode="GOAL_EXECUTION",
        )
    )
    _wait_for(manager, created.run_id, {"WAITING_APPROVAL", "FAILED"})
    worker = manager._futures[created.run_id]
    manager.shutdown()
    worker.result(timeout=2)

    now = datetime.now(UTC)
    data = manager._read()
    row = data["runs"][created.run_id]
    row["approval"]["requestedAt"] = (now - timedelta(minutes=5)).isoformat()
    row["deadlineAt"] = (now - timedelta(minutes=4)).isoformat()
    manager._write(data)

    restarted = _manager(isolated_settings)
    restarted.start()
    recovered = restarted.get(created.run_id)
    assert recovered.status == "WAITING_APPROVAL"

    restarted.resume(
        created.run_id,
        AgentRunResumeRequest(
            approved=True,
            decidedBy="late-supervisor",
            reason="审批等待不计入执行时限",
        ),
    )
    completed = _wait_for(restarted, created.run_id, {"COMPLETED", "FAILED"})

    assert completed.status == "COMPLETED"
    assert completed.deadline_at > now


def test_goal_execution_blocks_commit_when_simulation_gate_fails(
    isolated_settings, monkeypatch
) -> None:
    manager = _manager(isolated_settings)
    monkeypatch.setattr(
        manager.workflow,
        "recommend",
        lambda summary: AgentWorkflowRecommendation(
            decision="BLOCK",
            reasons=["测试安全门槛阻断"],
            safetyChecks={"conflictFree": False},
        ),
    )
    created = manager.create(
        AgentRunCreateRequest(
            message="创建紧急叉车任务，从 AP1123 运到 AP2121",
            scenarioId="interactive-multi-fleet",
            conversationId="goal-blocked",
            timeoutSeconds=30,
            executionMode="GOAL_EXECUTION",
        )
    )

    completed = _wait_for(manager, created.run_id, {"COMPLETED", "FAILED"})

    assert completed.status == "COMPLETED"
    assert completed.workflow is not None
    assert completed.workflow.phase == "BLOCKED"
    assert completed.workflow.commitment is None
    assert [step.action for step in completed.workflow.steps] == ["SIMULATE"]


def test_goal_execution_marks_active_step_failed_when_workflow_raises(
    isolated_settings, monkeypatch
) -> None:
    manager = _manager(isolated_settings)

    def fail_simulation(request):
        raise RuntimeError("数字孪生不可用")

    monkeypatch.setattr(manager.workflow, "simulate", fail_simulation)
    created = manager.create(
        AgentRunCreateRequest(
            message="创建紧急叉车任务，从 AP1123 运到 AP2121",
            scenarioId="interactive-multi-fleet",
            conversationId="goal-workflow-failure",
            timeoutSeconds=30,
            executionMode="GOAL_EXECUTION",
        )
    )

    failed = _wait_for(manager, created.run_id, {"COMPLETED", "FAILED"})

    assert failed.status == "FAILED"
    assert failed.workflow is not None
    assert failed.workflow.phase == "BLOCKED"
    assert failed.workflow.steps[-1].action == "SIMULATE"
    assert failed.workflow.steps[-1].status == "FAILED"
    assert failed.workflow.steps[-1].detail == "数字孪生不可用"
    assert any(event.event_type == "workflow_action_failed" for event in failed.events)
