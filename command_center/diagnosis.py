from __future__ import annotations

from .contracts import (
    DiagnosisReport,
    IncidentRecommendation,
    IncidentRecord,
    IncidentType,
    RiskLevel,
    RootCauseCandidate,
)

INCIDENT_ACTIONS: dict[IncidentType, set[str]] = {
    IncidentType.VEHICLE_FAULT: {
        "WAIT_RECOVERY",
        "ISOLATE_REASSIGN",
        "SAFETY_STOP",
    },
    IncidentType.WORKSTATION_DISABLED: {
        "WAIT_RECOVERY",
        "SUSPEND_AFFECTED_TASKS",
        "SAFETY_STOP",
    },
    IncidentType.DEADLOCK_RISK: {
        "CONTROLLED_REVERSE",
        "SAFETY_STOP",
    },
}
ALLOWED_INCIDENT_ACTIONS = set().union(*INCIDENT_ACTIONS.values())


def allowed_actions_for(incident: IncidentRecord) -> set[str]:
    actions = set(INCIDENT_ACTIONS.get(incident.incident_type, {"SAFETY_STOP"}))
    if (
        incident.incident_type is IncidentType.DEADLOCK_RISK
        and not incident.event_attributes.get("recoveryAvailable", False)
    ):
        actions.discard("CONTROLLED_REVERSE")
    return actions


def deterministic_diagnosis(
    incident: IncidentRecord,
    *,
    model: str = "deterministic-fallback",
) -> DiagnosisReport:
    """Build an evidence-linked report without relying on an external model."""

    evidence_ids = {row.evidence_id for row in incident.evidence}

    def existing(*candidates: str) -> list[str]:
        selected = [row for row in candidates if row in evidence_ids]
        if selected:
            return selected
        return [incident.evidence[0].evidence_id] if incident.evidence else []

    if incident.incident_type is IncidentType.WORKSTATION_DISABLED:
        return _workstation_diagnosis(incident, model, existing)
    if incident.incident_type is IncidentType.DEADLOCK_RISK:
        return _deadlock_diagnosis(incident, model, existing)
    return _vehicle_diagnosis(incident, model, existing)


def _vehicle_diagnosis(incident, model, existing) -> DiagnosisReport:
    vehicle_id = incident.vehicle_ids[0] if incident.vehicle_ids else "unknown"
    load_note = (
        "车辆已取货，任务不可自动退回公共队列。"
        if incident.load_state == "loaded"
        else "车辆尚未载货，可在仿真验证后考虑重派任务。"
    )
    causes = [
        RootCauseCandidate(
            code="VEHICLE_CONTROLLER_FAULT",
            title="车辆控制器报告故障",
            explanation="故障码已由注入事件确认；物理部件失效原因仍需真实遥测或检修记录佐证。",
            confidence=0.98,
            evidenceIds=existing("EV-001", "EV-002"),
            classification="FACT",
        ),
        RootCauseCandidate(
            code="RESOURCE_OCCUPANCY_AFTER_FAULT",
            title="故障位置资源持续占用",
            explanation="停止车辆可能持续占用节点及邻接冲突资源，后续车辆需要等待或重新规划。",
            confidence=0.82,
            evidenceIds=existing("EV-002", "EV-004"),
            classification="INFERENCE",
        ),
    ]
    recommendations = [
        _recommendation(
            "WAIT_RECOVERY",
            "冻结故障点资源并等待车辆恢复",
            "保留安全占用，评估短时恢复窗口对任务完成和排队的影响。",
            existing("EV-001", "EV-004"),
        ),
        _recommendation(
            "ISOLATE_REASSIGN",
            "隔离故障车辆并重派允许重派的任务",
            "将故障车辆从候选运力中移除并由 MASP 验证剩余车队能力；载货任务仍需人工转运。",
            existing("EV-003", "EV-004"),
        ),
        _recommendation(
            "SAFETY_STOP",
            "维持故障区域安全停车",
            "在现场状态未知或风险不可接受时，持续冻结受影响资源并等待人工处置。",
            existing("EV-002", "EV-004"),
        ),
    ]
    return DiagnosisReport(
        summary=(
            f"车辆 {vehicle_id} 在 {incident.fault_at_ms}ms 上报 "
            f"{incident.fault_code or '未知故障'}，故障点位于 "
            f"{incident.location_node_id or '未知节点'}。{load_note}"
        ),
        confirmedFacts=_confirmed_facts(incident),
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


def _workstation_diagnosis(incident, model, existing) -> DiagnosisReport:
    station = incident.workstation_id or incident.location_node_id or "unknown"
    causes = [
        RootCauseCandidate(
            code="WORKSTATION_DISABLED_SIGNAL",
            title="工位停用事件已确认",
            explanation="停用对象、起始时刻和冻结资源来自结构化事件，不由模型推测。",
            confidence=0.99,
            evidenceIds=existing("EV-001", "EV-002"),
            classification="FACT",
        ),
        RootCauseCandidate(
            code="WORKSTATION_CAPACITY_LOSS",
            title="作业能力下降影响关联任务",
            explanation="以该工位为取货点或放货点的任务在停用窗口内无法完成对应服务，可能形成等待或积压。",
            confidence=0.9,
            evidenceIds=existing("EV-002", "EV-003", "EV-004"),
            classification="INFERENCE",
        ),
    ]
    recommendations = [
        _recommendation(
            "WAIT_RECOVERY",
            "冻结工位并等待恢复",
            "保留原任务，使用 MASP 评估停用窗口带来的排队和完工影响。",
            existing("EV-001", "EV-003"),
        ),
        _recommendation(
            "SUSPEND_AFFECTED_TASKS",
            "暂停受影响任务并重算其余任务",
            "将涉及停用工位的任务从本次分支暂存，避免系统自行改写目的工位。",
            existing("EV-002", "EV-003"),
        ),
        _recommendation(
            "SAFETY_STOP",
            "持续封锁工位及相邻节点",
            "恢复时间未知时维持资源冻结，防止车辆进入不可作业区域。",
            existing("EV-001", "EV-002"),
        ),
    ]
    return DiagnosisReport(
        summary=(
            f"工位 {station} 自 {incident.fault_at_ms}ms 起停用，"
            f"当前识别到 {len(incident.task_ids)} 个关联任务和 {len(incident.vehicle_ids)} 辆关联车辆。"
        ),
        confirmedFacts=_confirmed_facts(incident),
        rootCauseCandidates=causes,
        affectedVehicleIds=incident.vehicle_ids,
        affectedTaskIds=incident.task_ids,
        recommendations=recommendations,
        uncertainties=[
            "事件未包含现场检修结论和实际恢复时刻，不能判断工位设备的物理根因。",
            "系统没有获得替代工位业务映射，因此不会自动改写任务起终点。",
        ],
        model=model,
        fallbackUsed=True,
    )


def _deadlock_diagnosis(incident, model, existing) -> DiagnosisReport:
    cycle_length = int(incident.event_attributes.get("maxCycleLength", 0))
    recovery_available = bool(incident.event_attributes.get("recoveryAvailable", False))
    causes = [
        RootCauseCandidate(
            code="WAIT_GRAPH_CYCLE",
            title="等待图存在循环依赖",
            explanation=f"MASP 等待图监督器确认了长度为 {cycle_length} 的强连通等待环。",
            confidence=1.0,
            evidenceIds=existing("EV-001", "EV-002"),
            classification="FACT",
        ),
        RootCauseCandidate(
            code="RECOVERY_FEASIBILITY",
            title="恢复可行性已由确定性控制器判定",
            explanation=(
                "MASP 已生成满足倒退距离、时间和资源预约约束的恢复计划。"
                if recovery_available
                else "MASP 未找到满足约束的倒退计划，当前确定性决策为安全停车。"
            ),
            confidence=1.0,
            evidenceIds=existing("EV-003", "EV-004"),
            classification="FACT",
        ),
    ]
    recommendations = []
    if recovery_available:
        recommendations.append(
            _recommendation(
                "CONTROLLED_REVERSE",
                "提交 MASP 受控倒退恢复方案",
                "只使用确定性恢复控制器已经生成并预约的倒退计划，审批通过后仍需复核世界版本。",
                existing("EV-003", "EV-004"),
            )
        )
    recommendations.append(
        _recommendation(
            "SAFETY_STOP",
            "维持循环车辆和资源安全停车",
            "恢复计划不可用或审批未通过时，不解除冻结资源并请求现场人工处置。",
            existing("EV-001", "EV-003"),
        )
    )
    return DiagnosisReport(
        summary=(
            f"MASP 在 {incident.fault_at_ms}ms 检测到 {cycle_length} 车等待环；"
            + (
                "存在受控倒退候选。"
                if recovery_available
                else "无合法倒退方案，已保持安全停车。"
            )
        ),
        confirmedFacts=_confirmed_facts(incident),
        rootCauseCandidates=causes,
        affectedVehicleIds=incident.vehicle_ids,
        affectedTaskIds=incident.task_ids,
        recommendations=recommendations,
        uncertainties=[
            "演示等待环来自版本锁定的 MASP 仓储拓扑和预约快照，未接入现场车辆实时定位。",
            "恢复执行前仍需确认通道净空、车辆状态和预约版本未发生变化。",
        ],
        model=model,
        fallbackUsed=True,
    )


def _recommendation(code: str, action: str, rationale: str, evidence_ids: list[str]):
    return IncidentRecommendation(
        actionCode=code,
        action=action,
        rationale=rationale,
        riskLevel=RiskLevel.R3_HIGH,
        evidenceIds=evidence_ids,
    )


def _confirmed_facts(incident: IncidentRecord) -> list[str]:
    return [
        row.detail
        for row in incident.deterministic_findings
        if row.certainty == "CONFIRMED"
    ]
