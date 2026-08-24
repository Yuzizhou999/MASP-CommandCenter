from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Condition, RLock
from time import perf_counter
from typing import Any, Callable

from .contracts import (
    AgentRunCreateRequest,
    AgentRunEvaluation,
    AgentRunRecord,
    AgentRunResumeRequest,
    AgentTraceStep,
    AgentWorkflowStep,
    ApprovalDecision,
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
    DispatchIntent,
    IntentType,
    IntentValidation,
    SimulationRequest,
    SimulationSummary,
    new_id,
)
from .dispatch_workflow import DispatchWorkflowService
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
        workflow: DispatchWorkflowService | None = None,
        max_workers: int = 4,
    ) -> None:
        self.path = path
        self.orchestrator = orchestrator
        self.provider = provider
        self.workflow = workflow
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
                "workflow": None,
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
            paused_at = approval.get("requestedAt")
            if paused_at:
                paused_duration = _utc_now() - datetime.fromisoformat(paused_at)
                row["deadlineAt"] = _iso(
                    datetime.fromisoformat(row["deadlineAt"]) + paused_duration
                )
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
                if status != "WAITING_APPROVAL" and now >= deadline:
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
            goal_execution = (
                record.request.execution_mode == "GOAL_EXECUTION"
                and self.workflow is not None
            )
            if goal_execution and record.response is not None and record.workflow:
                response = record.response
                usage = record.provider_usage

                def reuse_checkpoint(row: dict[str, Any]) -> None:
                    trace = response.agent_trace
                    row["traceSteps"] = (
                        [
                            step.model_dump(by_alias=True, mode="json")
                            for step in trace.steps
                        ]
                        if trace is not None
                        else []
                    )
                    self._append_event(
                        row,
                        "workflow_checkpoint_reused",
                        {"intentId": response.intent.intent_id if response.intent else None},
                    )

                self._update(run_id, reuse_checkpoint)
            else:
                with self.provider.telemetry_scope(
                    run_id,
                    control=lambda: self._check_control(run_id),
                ):
                    response = self.orchestrator.chat(
                        request,
                        on_step=lambda step: self._record_step(run_id, step),
                        approval_gate=(
                            None
                            if goal_execution
                            else lambda intent, validation: self._await_advisory_approval(
                                run_id, intent, validation
                            )
                        ),
                    )
                usage = self.provider.telemetry(run_id)
                if goal_execution:

                    def save_checkpoint(row: dict[str, Any]) -> None:
                        row["response"] = response.model_dump(
                            by_alias=True, mode="json"
                        )
                        row["providerUsage"] = usage
                        self._append_event(
                            row,
                            "workflow_checkpoint_saved",
                            {
                                "intentId": (
                                    response.intent.intent_id
                                    if response.intent is not None
                                    else None
                                )
                            },
                        )

                    self._update(run_id, save_checkpoint)
            if goal_execution:
                self._execute_goal_workflow(run_id, response)
            workflow_record = self.get(run_id).workflow
            workflow_phase = workflow_record.phase if workflow_record else None
            evaluation = self._evaluate(
                response,
                record.request.timeout_seconds,
                workflow_phase=workflow_phase,
            )

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

    def _execute_goal_workflow(
        self, run_id: str, response: ChatResponse
    ) -> None:
        if self.workflow is None:
            return
        intent = response.intent
        validation = response.validation
        actionable = bool(
            response.state == "READY"
            and intent is not None
            and validation is not None
            and validation.valid
            and intent.intent_type in {IntentType.CREATE_TASK, IntentType.BLOCK_RESOURCE}
        )
        if not actionable or intent is None or validation is None:
            reason = (
                "请求需要补充参数，未进入目标执行"
                if response.state == "CLARIFICATION_REQUIRED"
                else "当前意图属于只读或不可执行类型"
            )

            def skip(row: dict[str, Any]) -> None:
                row["workflow"] = {
                    "phase": "NOT_APPLICABLE",
                    "intentId": intent.intent_id if intent else None,
                    "simulation": None,
                    "recommendation": None,
                    "approvalRequest": None,
                    "commitment": None,
                    "steps": [],
                }
                self._append_event(
                    row, "workflow_skipped", {"reason": reason}
                )

            self._update(run_id, skip)
            return

        current = self.get(run_id)
        existing = current.workflow
        if existing is not None and existing.intent_id != intent.intent_id:
            existing = None
        summary: SimulationSummary
        recommendation = None
        if existing is not None and existing.simulation is not None:
            summary = SimulationSummary.model_validate(existing.simulation)
            recommendation = existing.recommendation
            self._record_workflow_event(
                run_id,
                "workflow_simulation_reused",
                {"simulationRunId": summary.run_id},
            )
        else:
            started = self._begin_workflow_step(
                run_id,
                intent_id=intent.intent_id,
                action="SIMULATE",
                phase="SIMULATING",
                title="运行 MASP 数字孪生",
                detail="使用确定性规划、资源预约和冲突检测评估调度意图",
            )
            self._check_control(run_id)
            summary = self.workflow.simulate(
                SimulationRequest(
                    scenarioId=current.request.scenario_id,
                    label=(
                        "Agent 闭环 | 通道封闭"
                        if intent.intent_type is IntentType.BLOCK_RESOURCE
                        else "Agent 闭环 | 紧急插单"
                    ),
                    intent=intent,
                )
            )
            recommendation = self.workflow.recommend(summary)

            def save_simulation(row: dict[str, Any]) -> None:
                workflow = self._workflow_payload(row, intent.intent_id)
                workflow["simulation"] = summary.model_dump(
                    by_alias=True, mode="json"
                )
                workflow["recommendation"] = recommendation.model_dump(
                    by_alias=True, mode="json"
                )
                row["workflow"] = workflow

            self._update(run_id, save_simulation)
            self._finish_workflow_step(
                run_id,
                action="SIMULATE",
                status="COMPLETED",
                detail=(
                    f"仿真 {summary.run_id} 完成，推进建议 {recommendation.decision}"
                ),
                output_ref=summary.run_id,
                duration_ms=(perf_counter() - started) * 1000,
            )

        if recommendation is None:
            recommendation = self.workflow.recommend(summary)
        if recommendation.decision == "BLOCK":

            def block(row: dict[str, Any]) -> None:
                workflow = self._workflow_payload(row, intent.intent_id)
                workflow["phase"] = "BLOCKED"
                row["workflow"] = workflow
                self._append_event(
                    row,
                    "workflow_blocked",
                    {"reasons": recommendation.reasons},
                )

            self._update(run_id, block)
            return

        approval: ApprovalRequest | None = None
        current = self.get(run_id)
        if validation.approval_required:
            if (
                current.workflow is not None
                and current.workflow.approval_request is not None
            ):
                approval = current.workflow.approval_request
                self._record_workflow_event(
                    run_id,
                    "workflow_approval_reused",
                    {"approvalId": approval.approval_id},
                )
            else:
                started = self._begin_workflow_step(
                    run_id,
                    intent_id=intent.intent_id,
                    action="REQUEST_APPROVAL",
                    phase="WAITING_APPROVAL",
                    title="创建仿真关联审批",
                    detail="高风险意图必须关联已通过安全门槛的仿真结果",
                )
                approval = self.workflow.create_approval(
                    intent,
                    scenario_id=current.request.scenario_id,
                    run_ids=[summary.run_id],
                )

                def save_approval(row: dict[str, Any]) -> None:
                    workflow = self._workflow_payload(row, intent.intent_id)
                    workflow["approvalRequest"] = approval.model_dump(
                        by_alias=True, mode="json"
                    )
                    row["workflow"] = workflow

                self._update(run_id, save_approval)
                self._finish_workflow_step(
                    run_id,
                    action="REQUEST_APPROVAL",
                    status="COMPLETED",
                    detail=f"审批单 {approval.approval_id} 已创建",
                    output_ref=approval.approval_id,
                    duration_ms=(perf_counter() - started) * 1000,
                )

            decision = self._wait_for_approval(
                run_id,
                intent,
                validation,
                approval_request=approval,
                simulation=summary,
            )
            decided = self.workflow.decide_approval(
                approval.approval_id,
                ApprovalDecision(
                    approved=decision.approved,
                    decidedBy=decision.decided_by,
                    reason=decision.reason,
                ),
            )

            def save_decision(row: dict[str, Any]) -> None:
                workflow = self._workflow_payload(row, intent.intent_id)
                workflow["approvalRequest"] = decided.model_dump(
                    by_alias=True, mode="json"
                )
                row["workflow"] = workflow

            self._update(run_id, save_decision)
            if not decision.approved:
                raise AgentRunRejected("主管拒绝了仿真后的高风险方案")

        current = self.get(run_id)
        if current.workflow is not None and current.workflow.commitment is not None:
            self._record_workflow_event(
                run_id,
                "workflow_commit_reused",
                {"commitId": current.workflow.commitment.get("commitId")},
            )
            return
        started = self._begin_workflow_step(
            run_id,
            intent_id=intent.intent_id,
            action="COMMIT",
            phase="COMMITTING",
            title="提交到仿真环境",
            detail="写入仿真意图存储，不向 WMS、RCS 或真实车辆下发",
        )
        self._check_control(run_id)
        commitment = self.workflow.commit(
            intent,
            scenario_id=current.request.scenario_id,
            approval_id=approval.approval_id if approval is not None else None,
        )

        def save_commitment(row: dict[str, Any]) -> None:
            workflow = self._workflow_payload(row, intent.intent_id)
            workflow["phase"] = "COMPLETED"
            workflow["commitment"] = commitment
            row["workflow"] = workflow

        self._update(run_id, save_commitment)
        self._finish_workflow_step(
            run_id,
            action="COMMIT",
            status="COMPLETED",
            detail=f"仿真提交 {commitment['commitId']} 已完成",
            output_ref=str(commitment["commitId"]),
            duration_ms=(perf_counter() - started) * 1000,
        )

    def _begin_workflow_step(
        self,
        run_id: str,
        *,
        intent_id: str,
        action: str,
        phase: str,
        title: str,
        detail: str,
    ) -> float:
        started = perf_counter()

        def update(row: dict[str, Any]) -> None:
            workflow = self._workflow_payload(row, intent_id)
            workflow["phase"] = phase
            step = AgentWorkflowStep(
                sequence=len(workflow["steps"]) + 1,
                action=action,
                status="RUNNING",
                title=title,
                detail=detail,
            ).model_dump(by_alias=True, mode="json")
            workflow["steps"].append(step)
            row["workflow"] = workflow
            self._append_event(row, "workflow_action_started", step)

        self._update(run_id, update)
        return started

    def _finish_workflow_step(
        self,
        run_id: str,
        *,
        action: str,
        status: str,
        detail: str,
        output_ref: str | None,
        duration_ms: float,
    ) -> None:
        def update(row: dict[str, Any]) -> None:
            workflow = self._workflow_payload(row, None)
            for step in reversed(workflow["steps"]):
                if step["action"] == action and step["status"] == "RUNNING":
                    step.update(
                        {
                            "status": status,
                            "detail": detail,
                            "outputRef": output_ref,
                            "durationMs": round(max(0, duration_ms), 3),
                        }
                    )
                    self._append_event(row, "workflow_action_completed", step)
                    break
            row["workflow"] = workflow

        self._update(run_id, update)

    def _record_workflow_event(
        self, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        self._update(
            run_id,
            lambda row: self._append_event(row, event_type, payload),
        )

    @staticmethod
    def _workflow_payload(
        row: dict[str, Any], intent_id: str | None
    ) -> dict[str, Any]:
        return dict(
            row.get("workflow")
            or {
                "phase": "PENDING",
                "intentId": intent_id,
                "simulation": None,
                "recommendation": None,
                "approvalRequest": None,
                "commitment": None,
                "steps": [],
            }
        )

    def _record_step(self, run_id: str, step: AgentTraceStep) -> None:
        self._check_control(run_id)
        serialized = step.model_dump(by_alias=True, mode="json")

        def update(row: dict[str, Any]) -> None:
            row["traceSteps"].append(serialized)
            self._append_event(row, "trace_step", serialized)

        self._update(run_id, update)

    def _await_advisory_approval(
        self,
        run_id: str,
        intent: DispatchIntent,
        validation: IntentValidation,
    ) -> None:
        decision = self._wait_for_approval(run_id, intent, validation)
        if not decision.approved:
            raise AgentRunRejected("主管拒绝了高风险 Agent 草案")

    def _wait_for_approval(
        self,
        run_id: str,
        intent: DispatchIntent,
        validation: IntentValidation,
        *,
        approval_request: ApprovalRequest | None = None,
        simulation: SimulationSummary | None = None,
    ) -> AgentRunResumeRequest:
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
                "stage": (
                    "POST_SIMULATION" if approval_request is not None else "INTENT_DRAFT"
                ),
                "approvalId": (
                    approval_request.approval_id if approval_request is not None else None
                ),
                "simulationRunId": simulation.run_id if simulation is not None else None,
            }
            if row.get("workflow") is not None and approval_request is not None:
                row["workflow"]["phase"] = "WAITING_APPROVAL"
            self._append_event(
                row,
                "approval_required",
                {
                    "intentId": intent.intent_id,
                    "riskLevel": validation.risk_level.value,
                    "stage": row["approval"]["stage"],
                    "approvalId": row["approval"]["approvalId"],
                    "simulationRunId": row["approval"]["simulationRunId"],
                },
            )

        record = self._update(run_id, pause)
        prior_decision = (record.approval or {}).get("decision") or {}
        if prior_decision:
            return self._parse_resume_decision(prior_decision)

        with self._lock:
            condition = self._conditions.setdefault(run_id, Condition())
        while True:
            with condition:
                condition.wait(timeout=0.25)
            self._check_control(run_id, include_deadline=False)
            record = self.get(run_id)
            decision = (record.approval or {}).get("decision")
            if decision is None:
                continue
            parsed_decision = self._parse_resume_decision(decision)
            if not parsed_decision.approved:
                return parsed_decision

            def continue_run(row: dict[str, Any]) -> None:
                row["status"] = "RUNNING"
                self._append_event(
                    row,
                    "run_resumed",
                    {"decidedBy": decision.get("decidedBy")},
                )

            self._update(run_id, continue_run)
            return parsed_decision

    @staticmethod
    def _parse_resume_decision(payload: dict[str, Any]) -> AgentRunResumeRequest:
        return AgentRunResumeRequest(
            approved=bool(payload["approved"]),
            decidedBy=str(payload["decidedBy"]),
            reason=str(payload["reason"]),
        )

    def _check_control(self, run_id: str, *, include_deadline: bool = True) -> None:
        with self._lock:
            if self._stopping:
                raise AgentRunStopping("Agent run manager is stopping")
        record = self.get(run_id)
        if record.cancel_requested or record.status == "CANCELLED":
            raise AgentRunCancelled("用户取消了 Agent run")
        if include_deadline and _utc_now() >= record.deadline_at:
            raise AgentRunTimedOut(
                f"Agent run 超过 {record.request.timeout_seconds} 秒执行时限"
            )

    @staticmethod
    def _evaluate(
        response: ChatResponse,
        timeout_seconds: int,
        *,
        workflow_phase: str | None = None,
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
        if workflow_phase is not None:
            checks["goalWorkflowTerminal"] = workflow_phase in {
                "COMPLETED",
                "BLOCKED",
                "NOT_APPLICABLE",
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
            workflow = row.get("workflow")
            if workflow is not None and workflow.get("phase") not in {
                "COMPLETED",
                "BLOCKED",
                "NOT_APPLICABLE",
            }:
                workflow["phase"] = "BLOCKED"
                for step in reversed(workflow.get("steps", [])):
                    if step.get("status") == "RUNNING":
                        step.update(
                            {
                                "status": "FAILED",
                                "detail": message,
                            }
                        )
                        self._append_event(row, "workflow_action_failed", step)
                        break
                row["workflow"] = workflow
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
