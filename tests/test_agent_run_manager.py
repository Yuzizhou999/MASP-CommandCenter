from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic, sleep

from command_center.agent_run_manager import AgentRunManager
from command_center.audit import AuditStore
from command_center.contracts import (
    AgentRunCreateRequest,
    AgentRunRecord,
    AgentRunResumeRequest,
)
from command_center.engine_adapter import MaspAdapter
from command_center.knowledge import KnowledgeBase
from command_center.orchestrator import DispatchOrchestrator
from command_center.provider import DeepSeekProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _manager(isolated_settings) -> AgentRunManager:
    provider = DeepSeekProvider(isolated_settings)
    orchestrator = DispatchOrchestrator(
        engine=MaspAdapter(isolated_settings),
        provider=provider,
        knowledge=KnowledgeBase(PROJECT_ROOT / "knowledge"),
        audit=AuditStore(isolated_settings.data_dir / "audit.jsonl"),
    )
    return AgentRunManager(
        isolated_settings.data_dir / "agent-runs.json",
        orchestrator=orchestrator,
        provider=provider,
    )


def _wait_for(manager: AgentRunManager, run_id: str, statuses: set[str]):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        record = manager.get(run_id)
        if record.status in statuses:
            return record
        sleep(0.02)
    raise AssertionError(f"Agent run did not reach {statuses}: {manager.get(run_id).status}")


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
    now = datetime.now(timezone.utc)
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
