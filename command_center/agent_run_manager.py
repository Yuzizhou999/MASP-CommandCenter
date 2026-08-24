from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Condition, RLock
from typing import Any, Callable

from .contracts import (
    AgentRunCreateRequest,
    AgentRunEvaluation,
    AgentRunRecord,
    AgentRunResumeRequest,
    AgentTraceStep,
    ChatRequest,
    ChatResponse,
    DispatchIntent,
    IntentValidation,
    new_id,
)
from .orchestrator import DispatchOrchestrator
from .provider import DeepSeekProvider


TERMINAL_AGENT_RUN_STATUSES = {
    "COMPLETED",
    "REJECTED",
    "CANCELLED",
    "TIMED_OUT",
    "FAILED",
}


class AgentRunCancelled(RuntimeError):
    pass


class AgentRunTimedOut(RuntimeError):
    pass


class AgentRunRejected(RuntimeError):
    pass


class AgentRunStopping(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat()


class AgentRunManager:
    """Durable execution wrapper around the synchronous dispatch orchestrator."""

    def __init__(
        self,
        path: Path,
        *,
        orchestrator: DispatchOrchestrator,
        provider: DeepSeekProvider,
        max_workers: int = 4,
    ) -> None:
        self.path = path
        self.orchestrator = orchestrator
        self.provider = provider
        self._lock = RLock()
        self._conditions: dict[str, Condition] = {}
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = self._new_executor()
        self._futures: dict[str, Future[None]] = {}
        self._recovery_started = False
        self._stopping = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"schemaVersion": 1, "runs": {}})

    def start(self) -> int:
        with self._lock:
            if not self._stopping and self._recovery_started:
                return 0
            if self._stopping:
                self._recovery_started = False
                self._futures = {}
            if self._executor is None:
                self._executor = self._new_executor()
            self._stopping = False
        return self.recover()

    def shutdown(self) -> None:
        with self._lock:
            self._stopping = True
            executor = self._executor
            self._executor = None
            conditions = list(self._conditions.values())
        for condition in conditions:
            with condition:
                condition.notify_all()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def create(
        self,
        request: AgentRunCreateRequest,
        *,
        idempotency_key: str | None = None,
    ) -> AgentRunRecord:
        normalized_key = idempotency_key.strip()[:200] if idempotency_key else None
        with self._lock:
            data = self._read()
            if normalized_key:
                for existing in data["runs"].values():
                    if existing.get("idempotencyKey") != normalized_key:
                        continue
                    submitted = request.model_dump(by_alias=True, mode="json")
                    if existing.get("request") != submitted:
                        raise ValueError("同一 Idempotency-Key 不能用于不同的 Agent 请求")
                    return AgentRunRecord.model_validate(existing)

            now = _utc_now()
            run_id = new_id("agent-run")
            row: dict[str, Any] = {
                "runId": run_id,
                "status": "QUEUED",
                "request": request.model_dump(by_alias=True, mode="json"),
                "idempotencyKey": normalized_key,
                "attempt": 0,
                "recovered": False,
                "cancelRequested": False,
                "traceSteps": [],
                "response": None,
                "approval": None,
                "evaluation": None,
                "providerUsage": {},
                "error": None,
                "events": [],
                "createdAt": _iso(now),
                "updatedAt": _iso(now),
                "deadlineAt": _iso(now + timedelta(seconds=request.timeout_seconds)),
                "startedAt": None,
                "completedAt": None,
            }
            self._append_event(row, "run_queued", {"status": "QUEUED"})
            data["runs"][run_id] = row
            self._write(data)
            record = AgentRunRecord.model_validate(row)
        self._submit(run_id)
        return record

    def get(self, run_id: str) -> AgentRunRecord:
        with self._lock:
            row = self._read()["runs"].get(run_id)
            if row is None:
                raise KeyError(f"未知 Agent run：{run_id}")
            return AgentRunRecord.model_validate(row)

    def events_after(self, run_id: str, event_id: int) -> list[dict[str, Any]]:
        record = self.get(run_id)
        return [
            event.model_dump(by_alias=True, mode="json")
            for event in record.events
            if event.event_id > event_id
        ]

    def resume(
        self, run_id: str, decision: AgentRunResumeRequest
    ) -> AgentRunRecord:
        should_submit = False

        def update(row: dict[str, Any]) -> None:
            nonlocal should_submit
            if row["status"] != "WAITING_APPROVAL":
                raise ValueError("只有 WAITING_APPROVAL 状态可以恢复")
            approval = dict(row.get("approval") or {})
            if approval.get("decision") is not None:
                raise ValueError("当前 Agent run 已经完成审批")
            approval["decision"] = {
                **decision.model_dump(by_alias=True, mode="json"),
                "decidedAt": _iso(),
            }
            row["approval"] = approval
            self._append_event(
                row,
                "approval_decided",
                {
                    "approved": decision.approved,
                    "decidedBy": decision.decided_by,
                    "reason": decision.reason,
                },
            )
            future = self._futures.get(run_id)
            if future is None or future.done():
                if decision.approved:
                    row["status"] = "QUEUED"
                    row["recovered"] = True
                    row["traceSteps"] = []
                    self._append_event(
                        row,
                        "run_recovered",
                        {"reason": "resume_after_process_restart"},
                    )
                    should_submit = True
                else:
                    self._set_terminal(
                        row,
                        "REJECTED",
                        "主管拒绝了高风险 Agent 草案",
                    )

        record = self._update(run_id, update)
        with self._lock:
            condition = self._conditions.get(run_id)
        if condition is not None:
            with condition:
                condition.notify_all()
        if should_submit:
            self._submit(run_id)
        return record

    def cancel(self, run_id: str) -> AgentRunRecord:
        def update(row: dict[str, Any]) -> None:
            if row["status"] in TERMINAL_AGENT_RUN_STATUSES:
                return
            row["cancelRequested"] = True
            self._set_terminal(row, "CANCELLED", "用户取消了 Agent run")

        record = self._update(run_id, update)
        with self._lock:
            condition = self._conditions.get(run_id)
        if condition is not None:
            with condition:
                condition.notify_all()
        return record

    def recover(self) -> int:
        with self._lock:
            if self._recovery_started:
                return 0
            self._recovery_started = True
            data = self._read()
            run_ids: list[str] = []
            changed = False
            now = _utc_now()
            for run_id, row in data["runs"].items():
                status = row.get("status")
                if status not in {"QUEUED", "RUNNING", "WAITING_APPROVAL"}:
                    continue
                deadline = datetime.fromisoformat(row["deadlineAt"])
                if now >= deadline:
                    self._set_terminal(row, "TIMED_OUT", "Agent run 在服务恢复前已超时")
                    changed = True
                    continue
                if status in {"QUEUED", "RUNNING"}:
                    row["status"] = "QUEUED"
                    row["recovered"] = True
                    row["traceSteps"] = []
                    row["updatedAt"] = _iso()
                    self._append_event(
                        row,
                        "run_recovered",
                        {"reason": "service_restart"},
                    )
                    run_ids.append(run_id)
                    changed = True
            if changed:
                self._write(data)
        for run_id in run_ids:
            self._submit(run_id)
        return len(run_ids)

    def _submit(self, run_id: str) -> None:
        with self._lock:
            if self._stopping:
                return
            future = self._futures.get(run_id)
            if future is not None and not future.done():
                return
            if self._executor is None:
                self._executor = self._new_executor()
            self._futures[run_id] = self._executor.submit(self._execute, run_id)

    def _execute(self, run_id: str) -> None:
        try:
            self._check_control(run_id)

            def start(row: dict[str, Any]) -> None:
                row["status"] = "RUNNING"
                row["attempt"] = int(row.get("attempt", 0)) + 1
                row["traceSteps"] = []
                row["error"] = None
                row["startedAt"] = row.get("startedAt") or _iso()
                self._append_event(
                    row,
                    "run_started",
                    {"attempt": row["attempt"], "recovered": row.get("recovered", False)},
                )

            record = self._update(run_id, start)
            request = ChatRequest(
                message=record.request.message,
                scenarioId=record.request.scenario_id,
                requestedBy=record.request.requested_by,
                conversationId=record.request.conversation_id,
            )
            with self.provider.telemetry_scope(
                run_id,
                control=lambda: self._check_control(run_id),
            ):
                response = self.orchestrator.chat(
                    request,
                    on_step=lambda step: self._record_step(run_id, step),
                    approval_gate=lambda intent, validation: self._await_approval(
                        run_id, intent, validation
                    ),
                )
            usage = self.provider.telemetry(run_id)
            evaluation = self._evaluate(response, record.request.timeout_seconds)

            def complete(row: dict[str, Any]) -> None:
                if row["status"] == "CANCELLED":
                    return
                row["status"] = "COMPLETED"
                row["response"] = response.model_dump(by_alias=True, mode="json")
                row["providerUsage"] = usage
                row["evaluation"] = evaluation.model_dump(
                    by_alias=True, mode="json"
                )
                row["completedAt"] = _iso()
                self._append_event(
                    row,
                    "run_completed",
                    {
                        "status": "COMPLETED",
                        "evaluationPassed": evaluation.passed,
                        "score": evaluation.score,
                    },
                )

            self._update(run_id, complete)
        except AgentRunCancelled as error:
            self._finish_error(run_id, "CANCELLED", str(error))
        except AgentRunTimedOut as error:
            self._finish_error(run_id, "TIMED_OUT", str(error))
        except AgentRunRejected as error:
            self._finish_error(run_id, "REJECTED", str(error))
        except AgentRunStopping:
            return
        except Exception as error:
            self._finish_error(run_id, "FAILED", str(error))
        finally:
            with self._lock:
                self._conditions.pop(run_id, None)

    def _record_step(self, run_id: str, step: AgentTraceStep) -> None:
        self._check_control(run_id)
        serialized = step.model_dump(by_alias=True, mode="json")

        def update(row: dict[str, Any]) -> None:
            row["traceSteps"].append(serialized)
            self._append_event(row, "trace_step", serialized)

        self._update(run_id, update)

    def _await_approval(
        self,
        run_id: str,
        intent: DispatchIntent,
        validation: IntentValidation,
    ) -> None:
        self._check_control(run_id)

        def pause(row: dict[str, Any]) -> None:
            existing = row.get("approval") or {}
            if (existing.get("decision") or {}).get("approved") is True:
                self._append_event(
                    row,
                    "approval_reused",
                    {"reason": "approved_before_recovery"},
                )
                return
            row["status"] = "WAITING_APPROVAL"
            row["approval"] = {
                "intent": intent.model_dump(by_alias=True, mode="json"),
                "validation": validation.model_dump(by_alias=True, mode="json"),
                "requestedAt": existing.get("requestedAt") or _iso(),
                "decision": existing.get("decision"),
            }
            self._append_event(
                row,
                "approval_required",
                {
                    "intentId": intent.intent_id,
                    "riskLevel": validation.risk_level.value,
                },
            )

        record = self._update(run_id, pause)
        prior_decision = (record.approval or {}).get("decision") or {}
        if prior_decision.get("approved") is True:
            return

        with self._lock:
            condition = self._conditions.setdefault(run_id, Condition())
        while True:
            with condition:
                condition.wait(timeout=0.25)
            self._check_control(run_id)
            record = self.get(run_id)
            decision = (record.approval or {}).get("decision")
            if decision is None:
                continue
            if not decision.get("approved"):
                raise AgentRunRejected("主管拒绝了高风险 Agent 草案")

            def continue_run(row: dict[str, Any]) -> None:
                row["status"] = "RUNNING"
                self._append_event(
                    row,
                    "run_resumed",
                    {"decidedBy": decision.get("decidedBy")},
                )

            self._update(run_id, continue_run)
            return

    def _check_control(self, run_id: str) -> None:
        with self._lock:
            if self._stopping:
                raise AgentRunStopping("Agent run manager is stopping")
        record = self.get(run_id)
        if record.cancel_requested or record.status == "CANCELLED":
            raise AgentRunCancelled("用户取消了 Agent run")
        if _utc_now() >= record.deadline_at:
            raise AgentRunTimedOut(
                f"Agent run 超过 {record.request.timeout_seconds} 秒执行时限"
            )

    @staticmethod
    def _evaluate(
        response: ChatResponse, timeout_seconds: int
    ) -> AgentRunEvaluation:
        trace = response.agent_trace
        steps = trace.steps if trace else []
        sequences = [step.sequence for step in steps]
        checks = {
            "boundedSteps": bool(trace and len(steps) <= trace.max_steps),
            "sequentialTrace": sequences == list(range(1, len(steps) + 1)),
            "readOnlyTools": all(
                step.read_only is not False for step in steps if step.tool_name
            ),
            "deterministicBoundary": bool(
                response.state == "CLARIFICATION_REQUIRED"
                or any(step.state == "SAFETY_VALIDATION" for step in steps)
            ),
            "terminalState": bool(
                trace
                and trace.status in {"COMPLETED", "CLARIFICATION_REQUIRED"}
            ),
            "withinTimeout": bool(
                trace and trace.duration_ms <= timeout_seconds * 1000
            ),
        }
        passed_count = sum(checks.values())
        notes = [name for name, passed in checks.items() if not passed]
        return AgentRunEvaluation(
            passed=passed_count == len(checks),
            score=round(passed_count / len(checks), 3),
            checks=checks,
            notes=notes,
        )

    def _finish_error(self, run_id: str, status: str, message: str) -> None:
        usage = self.provider.telemetry(run_id)

        def update(row: dict[str, Any]) -> None:
            if row["status"] in TERMINAL_AGENT_RUN_STATUSES:
                if row["status"] == "CANCELLED":
                    return
            self._set_terminal(row, status, message)
            row["providerUsage"] = usage

        self._update(run_id, update)

    def _set_terminal(
        self, row: dict[str, Any], status: str, message: str
    ) -> None:
        row["status"] = status
        row["error"] = message
        row["completedAt"] = _iso()
        self._append_event(
            row,
            "run_terminal",
            {"status": status, "error": message},
        )

    def _update(
        self,
        run_id: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> AgentRunRecord:
        with self._lock:
            data = self._read()
            row = data["runs"].get(run_id)
            if row is None:
                raise KeyError(f"未知 Agent run：{run_id}")
            callback(row)
            row["updatedAt"] = _iso()
            self._write(data)
            return AgentRunRecord.model_validate(row)

    @staticmethod
    def _append_event(
        row: dict[str, Any], event_type: str, payload: dict[str, Any]
    ) -> None:
        events = row.setdefault("events", [])
        events.append(
            {
                "eventId": len(events) + 1,
                "eventType": event_type,
                "payload": payload,
                "createdAt": _iso(),
            }
        )

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schemaVersion": 1, "runs": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _new_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="agent-run",
        )
