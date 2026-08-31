from __future__ import annotations

from typing import Any

from .approvals import ApprovalStore
from .audit import AuditStore
from .contracts import (
    AgentWorkflowRecommendation,
    ApprovalDecision,
    ApprovalRequest,
    DispatchIntent,
    IntentValidation,
    SimulationRequest,
    SimulationSummary,
    new_id,
)
from .engine_adapter import MaspAdapter
from .intent_store import IntentStore


class DispatchWorkflowService:
    """Shared safety-governed simulation, approval, and commit operations."""

    def __init__(
        self,
        *,
        engine: MaspAdapter,
        approvals: ApprovalStore,
        intents: IntentStore,
        audit: AuditStore,
    ) -> None:
        self.engine = engine
        self.approvals = approvals
        self.intents = intents
        self.audit = audit

    def simulate(self, request: SimulationRequest) -> SimulationSummary:
        trace_id = new_id("trace")
        actor = request.intent.requested_by if request.intent else "demo-operator"
        try:
            summary = self.engine.simulate(request)
        except Exception as error:
            self.audit.append(
                trace_id=trace_id,
                event_type="SIMULATION_REJECTED",
                actor=actor,
                payload={
                    "error": str(error),
                    "request": request.model_dump(by_alias=True, mode="json"),
                },
            )
            raise
        self.audit.append(
            trace_id=trace_id,
            event_type="SIMULATION_COMPLETED",
            actor=actor,
            payload=summary.model_dump(by_alias=True, mode="json"),
        )
        return summary

    def recommend(self, summary: SimulationSummary) -> AgentWorkflowRecommendation:
        checks = {
            "simulationCompleted": summary.status == "COMPLETED",
            "conflictFree": bool(summary.safety.get("conflictFree", False)),
            "allTasksPlanned": int(summary.safety.get("unplannedTaskCount", 0)) == 0,
            "simulationOnly": bool(summary.safety.get("simulationOnly", False)),
        }
        reasons: list[str] = []
        if not checks["simulationCompleted"]:
            reasons.append("数字孪生未成功完成")
        if not checks["conflictFree"]:
            reasons.append("仿真检测到资源预约冲突")
        if not checks["allTasksPlanned"]:
            reasons.append(
                f"仍有 {int(summary.safety.get('unplannedTaskCount', 0))} 个任务未规划"
            )
        if not checks["simulationOnly"]:
            reasons.append("运行结果未标记为仿真环境")
        if all(checks.values()):
            reasons.append("仿真完成、无资源冲突且所有任务均已规划")
        return AgentWorkflowRecommendation(
            decision="PROCEED" if all(checks.values()) else "BLOCK",
            reasons=reasons,
            safetyChecks=checks,
        )

    def create_approval(
        self,
        intent: DispatchIntent,
        *,
        scenario_id: str,
        run_ids: list[str],
    ) -> ApprovalRequest:
        validation = self.engine.validate_intent(intent, scenario_id)
        if not validation.valid:
            raise ValueError("意图未通过确定性校验。")
        self._validate_simulation_evidence(intent, validation, run_ids)
        request = self.approvals.create(intent, validation, run_ids)
        self.audit.append(
            trace_id=new_id("trace"),
            event_type="APPROVAL_CREATED",
            actor=intent.requested_by,
            payload=request.model_dump(by_alias=True, mode="json"),
        )
        return request

    def decide_approval(
        self,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        authenticated: bool = False,
    ) -> ApprovalRequest:
        current = self.approvals.get(approval_id)
        if current.status.value != "PENDING":
            expected = "APPROVED" if decision.approved else "REJECTED"
            if current.status.value == expected:
                return current
        result = self.approvals.decide(approval_id, decision)
        self.audit.append(
            trace_id=new_id("trace"),
            event_type="APPROVAL_DECIDED",
            actor=decision.decided_by,
            payload={
                **result.model_dump(by_alias=True, mode="json"),
                # 记录该决策是否经过 token 鉴权，否则演示模式和鉴权模式的审计
                # 事件无法区分，事后无法判断审批人身份是否可信。
                "authenticated": authenticated,
            },
        )
        return result

    def commit(
        self,
        intent: DispatchIntent,
        *,
        scenario_id: str,
        approval_id: str | None = None,
    ) -> dict[str, Any]:
        validation = self.engine.validate_intent(intent, scenario_id)
        if not validation.valid:
            raise ValueError("意图未通过确定性校验。")
        approval = None
        if validation.approval_required:
            if approval_id is None:
                raise PermissionError("该意图需要主管审批。")
            approval = self.approvals.get(approval_id)
        record = self.intents.commit(
            intent,
            current_world_revision=self.engine.world_revision(scenario_id),
            approval=approval,
            actor=intent.requested_by,
        )
        self.audit.append(
            trace_id=new_id("trace"),
            event_type="INTENT_SIMULATION_COMMITTED",
            actor=intent.requested_by,
            payload=record,
        )
        return record

    def _validate_simulation_evidence(
        self,
        intent: DispatchIntent,
        validation: IntentValidation,
        run_ids: list[str],
    ) -> None:
        if not validation.approval_required:
            return
        if not run_ids:
            raise ValueError("高风险意图必须先完成数字孪生仿真。")
        for run_id in run_ids:
            run = self.engine.get_run(run_id)
            if run.status != "COMPLETED":
                raise ValueError(f"仿真 {run_id} 尚未成功完成。")
            if run.intent_id != intent.intent_id:
                raise ValueError(f"仿真 {run_id} 与审批意图不匹配。")
            recommendation = self.recommend(run)
            if recommendation.decision != "PROCEED":
                raise ValueError(f"仿真 {run_id} 未通过确定性推进门槛。")
