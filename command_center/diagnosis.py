from __future__ import annotations

from .contracts import (
    DiagnosisReport,
    IncidentRecord,
    IncidentRecommendation,
    RiskLevel,
    RootCauseCandidate,
)


ALLOWED_INCIDENT_ACTIONS = {
    "WAIT_RECOVERY",
    "ISOLATE_REASSIGN",
    "SAFETY_STOP",
}


def deterministic_diagnosis(
    incident: IncidentRecord,
    *,
    model: str = "deterministic-fallback",
) -> DiagnosisReport:
    """Build an evidence-linked report without relying on an external model."""

    vehicle_id = incident.vehicle_ids[0] if incident.vehicle_ids else "unknown"
    load_note = (
        "车辆已取货，任务不可自动退回公共队列。"
        if incident.load_state == "loaded"
        else "车辆尚未载货，可在仿真验证后考虑重派任务。"
    )
    evidence_ids = {row.evidence_id for row in incident.evidence}

    def existing(*candidates: str) -> list[str]:
        selected = [row for row in candidates if row in evidence_ids]
        if selected:
            return selected
        return [incident.evidence[0].evidence_id] if incident.evidence else []

    causes = [
        RootCauseCandidate(
            code="VEHICLE_CONTROLLER_FAULT",
            title="车辆控制器报告故障",
            explanation=(
                "故障码已由本次注入事件确认；物理部件的最终失效原因仍需真实遥测或检修记录佐证。"
            ),
            confidence=0.98,
            evidenceIds=existing("EV-001", "EV-002"),
            classification="FACT",
        ),
        RootCauseCandidate(
            code="RESOURCE_OCCUPANCY_AFTER_FAULT",
            title="故障位置资源持续占用",
            explanation=(
                "停止车辆可能持续占用节点及邻接冲突资源，后续车辆需要等待或重新规划。"
            ),
            confidence=0.82,
            evidenceIds=existing("EV-002", "EV-004"),
            classification="INFERENCE",
        ),
    ]
    recommendations = [
        IncidentRecommendation(
            actionCode="WAIT_RECOVERY",
            action="冻结故障点资源并等待车辆恢复",
            rationale="保留安全占用，评估短时恢复窗口对任务完成和排队的影响。",
            riskLevel=RiskLevel.R3_HIGH,
            evidenceIds=existing("EV-001", "EV-004"),
        ),
        IncidentRecommendation(
            actionCode="ISOLATE_REASSIGN",
            action="隔离故障车辆并重派允许重派的任务",
            rationale=(
                "将故障车辆从候选运力中移除，并通过 MASP 验证剩余车队能力；"
                "载货任务仍需人工转运。"
            ),
            riskLevel=RiskLevel.R3_HIGH,
            evidenceIds=existing("EV-003", "EV-004"),
        ),
        IncidentRecommendation(
            actionCode="SAFETY_STOP",
            action="维持故障区域安全停车",
            rationale="在现场状态未知或风险不可接受时，持续冻结受影响资源并等待人工处置。",
            riskLevel=RiskLevel.R3_HIGH,
            evidenceIds=existing("EV-002", "EV-004"),
        ),
    ]
    return DiagnosisReport(
        summary=(
            f"车辆 {vehicle_id} 在 {incident.fault_at_ms}ms 上报 "
            f"{incident.fault_code or '未知故障'}，故障点位于 "
            f"{incident.location_node_id or '未知节点'}。{load_note}"
        ),
        confirmedFacts=[
            row.detail
            for row in incident.deterministic_findings
            if row.certainty == "CONFIRMED"
        ],
        rootCauseCandidates=causes,
        affectedVehicleIds=incident.vehicle_ids,
        affectedTaskIds=incident.task_ids,
        recommendations=recommendations,
        uncertainties=[
            "缺少真实电机温度、电流和控制器诊断帧，不能判断物理故障的最终根因。",
            "What-if 是基于已完成运行构造的安全节点分支，不代表真实车辆已经执行恢复动作。",
        ],
        model=model,
        fallbackUsed=True,
    )
