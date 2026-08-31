from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from .audit import AuditStore
from .contracts import (
    DeadlockInjectionRequest,
    DeterministicFinding,
    DiagnosisReport,
    FaultInjectionRequest,
    IncidentEvidence,
    IncidentRecord,
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
    WhatIfMode,
    WorkstationInjectionRequest,
    utc_now,
)
from .diagnosis import allowed_actions_for, deterministic_diagnosis
from .engine_adapter import MaspAdapter
from .knowledge import KnowledgeBase
from .model_safety import diagnosis_violation
from .provider import DeepSeekProvider


class IncidentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._items = self._load()

    def _load(self) -> dict[str, IncidentRecord]:
        if not self.path.exists():
            return {}
        rows = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            item.incident_id: item
            for item in (IncidentRecord.model_validate(row) for row in rows)
        }

    def _save(self) -> None:
        rows = [
            item.model_dump(by_alias=True, mode="json")
            for item in sorted(self._items.values(), key=lambda row: row.created_at)
        ]
        self.path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def put(self, incident: IncidentRecord) -> IncidentRecord:
        with self._lock:
            self._items[incident.incident_id] = incident
            self._save()
        return incident

    def get(self, incident_id: str) -> IncidentRecord:
        item = self._items.get(incident_id)
        if item is None:
            raise KeyError(incident_id)
        return item

    def list(self) -> list[IncidentRecord]:
        return sorted(self._items.values(), key=lambda row: row.created_at, reverse=True)


class IncidentService:
    def __init__(
        self,
        *,
        store: IncidentStore,
        engine: MaspAdapter,
        provider: DeepSeekProvider,
        knowledge: KnowledgeBase,
        audit: AuditStore,
    ) -> None:
        self.store = store
        self.engine = engine
        self.provider = provider
        self.knowledge = knowledge
        self.audit = audit

    def inject_vehicle_fault(self, request: FaultInjectionRequest) -> IncidentRecord:
        detail = self.engine.get_run_detail(request.run_id)
        if detail["summary"]["status"] != "COMPLETED":
            raise ValueError("只能在已完成且证据完整的仿真运行上注入故障。")
        plans = detail["scenario"].get("plans", [])
        vehicles = detail["scenario"].get("vehicles", [])
        known_vehicle_ids = {row["vehicleId"] for row in vehicles}
        vehicle_id = request.vehicle_id or self._default_vehicle(plans)
        if vehicle_id not in known_vehicle_ids:
            raise ValueError(f"运行中不存在车辆 {vehicle_id!r}。")

        vehicle_plans = [row for row in plans if row["vehicleId"] == vehicle_id]
        if not vehicle_plans:
            raise ValueError(f"车辆 {vehicle_id!r} 在该运行中没有可分析的计划。")
        target_ms = request.requested_at_ms
        if target_ms is None:
            target_ms = max(1, int(detail["scenario"]["endTimeMs"] * 0.18))
        segment, plan = self._safe_fault_segment(vehicle_plans, target_ms)
        fault_at_ms = int(segment["endMs"])
        location_node_id = segment.get("endNodeId") or segment.get("startNodeId")
        if not location_node_id:
            raise ValueError("无法为故障选择确定的安全节点。")
        resource_ids = sorted(
            resource_id
            for resource_id in set(segment.get("resourceIds", []))
            if resource_id.startswith(("node:", "edge:", "edge-conflict:"))
        )
        task_id = plan.get("taskId")
        load_state = segment.get("expectedLoadState", "unknown")
        recent_events = [
            row
            for row in detail["result"].get("eventLog", [])
            if row.get("payload", {}).get("vehicleId") == vehicle_id
            and fault_at_ms - 30000 <= int(row.get("timeMs", 0)) <= fault_at_ms
        ][-8:]

        evidence: list[IncidentEvidence] = [
            IncidentEvidence(
                evidenceId="EV-001",
                evidenceType="FAULT_SIGNAL",
                fact=f"车辆 {vehicle_id} 上报故障码 {request.fault_code}。",
                source=f"injection:{request.run_id}",
                observedAtMs=fault_at_ms,
                attributes={"faultCode": request.fault_code, "vehicleId": vehicle_id},
            ),
            IncidentEvidence(
                evidenceId="EV-002",
                evidenceType="VEHICLE_POSITION",
                fact=f"故障在安全节点 {location_node_id} 注入，未截停在边内。",
                source=f"MASP:{request.run_id}:planned-scenario",
                observedAtMs=fault_at_ms,
                attributes={
                    "nodeId": location_node_id,
                    "precedingEdgeId": segment.get("edgeId"),
                    "loadState": load_state,
                },
            ),
            IncidentEvidence(
                evidenceId="EV-003",
                evidenceType="ACTIVE_TASK",
                fact=f"故障时车辆关联任务为 {task_id}，载荷状态为 {load_state}。",
                source=f"MASP:{request.run_id}:plan",
                observedAtMs=fault_at_ms,
                attributes={"taskId": task_id, "planId": plan.get("id")},
            ),
            IncidentEvidence(
                evidenceId="EV-004",
                evidenceType="RESOURCE_OCCUPANCY",
                fact=f"故障节点及相邻计划段涉及 {len(resource_ids)} 个确定性资源。",
                source=f"MASP:{request.run_id}:reservations",
                observedAtMs=fault_at_ms,
                attributes={"resourceIds": resource_ids},
            ),
            IncidentEvidence(
                evidenceId="EV-005",
                evidenceType="PLANNING_METRICS",
                fact=(
                    f"基线完成 {detail['summary']['metrics'].get('completedTaskCount', 0)} 个任务，"
                    f"资源冲突拒绝 {detail['summary']['metrics'].get('reservationConflictRejections', 0)} 次。"
                ),
                source=f"MASP:{request.run_id}:result",
                attributes={"metrics": detail["summary"]["metrics"]},
            ),
        ]
        for index, row in enumerate(recent_events, start=6):
            evidence.append(
                IncidentEvidence(
                    evidenceId=f"EV-{index:03d}",
                    evidenceType="RECENT_EVENT",
                    fact=(
                        f"{int(row.get('timeMs', 0))}ms 发生 {row.get('type')}，"
                        f"车辆状态 {row.get('vehicleState', 'unknown')}。"
                    ),
                    source=f"MASP:{request.run_id}:eventLog",
                    observedAtMs=int(row.get("timeMs", 0)),
                    attributes={"event": row},
                )
            )

        findings = [
            DeterministicFinding(
                code="vehicle.fault.reported",
                title="车辆控制器故障",
                detail=f"故障码 {request.fault_code} 已作为本次演示故障的确定事实写入。",
                certainty="CONFIRMED",
                evidenceIds=["EV-001"],
            ),
            DeterministicFinding(
                code="vehicle.safe_node.stop",
                title="故障点位于安全节点",
                detail="故障在完成当前移动段后注入，未伪造边内急停轨迹。",
                certainty="CONFIRMED",
                evidenceIds=["EV-002"],
            ),
            DeterministicFinding(
                code="task.assignment.affected",
                title="当前任务可能受影响",
                detail=f"车辆故障时仍关联任务 {task_id}，需要根据载荷阶段决定重派或人工转运。",
                certainty="INFERRED",
                evidenceIds=["EV-003"],
            ),
        ]
        if resource_ids:
            findings.append(
                DeterministicFinding(
                    code="resource.occupancy.risk",
                    title="故障位置可能形成资源阻塞",
                    detail="处置仿真必须冻结故障点资源，不能假设车辆立即消失。",
                    certainty="INFERRED",
                    evidenceIds=["EV-004"],
                )
            )
        if load_state == "loaded":
            findings.append(
                DeterministicFinding(
                    code="payload.manual_transfer.required",
                    title="已取货任务不可自动重派",
                    detail="故障车辆为载货状态，必须保留任务归属并请求人工转运确认。",
                    certainty="CONFIRMED",
                    evidenceIds=["EV-003"],
                )
            )

        incident = IncidentRecord(
            incidentType=IncidentType.VEHICLE_FAULT,
            severity=(
                IncidentSeverity.CRITICAL if load_state == "loaded" else IncidentSeverity.HIGH
            ),
            scenarioId=detail["summary"]["scenarioId"],
            runId=request.run_id,
            vehicleIds=[vehicle_id],
            taskIds=[task_id] if task_id else [],
            resourceIds=resource_ids,
            faultCode=request.fault_code,
            faultAtMs=fault_at_ms,
            recoveryDurationMs=request.recovery_duration_ms,
            locationNodeId=location_node_id,
            locationEdgeId=segment.get("edgeId"),
            loadState=load_state,
            evidence=evidence,
            deterministicFindings=findings,
            createdBy=request.requested_by,
        )
        self.store.put(incident)
        self.audit.append(
            trace_id=incident.incident_id,
            event_type="INCIDENT_INJECTED",
            actor=request.requested_by,
            payload=incident.model_dump(by_alias=True, mode="json"),
        )
        return incident

    def inject_workstation_outage(
        self, request: WorkstationInjectionRequest
    ) -> IncidentRecord:
        detail = self._completed_run_detail(request.run_id)
        scenario = detail["scenario"]
        workstations = {
            row["nodeId"]: row for row in self.engine.workstation_catalog()
        }
        task_counts: dict[str, int] = {}
        for task in scenario.get("tasks", []):
            for node_id in (task["pickupNodeId"], task["dropoffNodeId"]):
                if node_id in workstations:
                    task_counts[node_id] = task_counts.get(node_id, 0) + 1
        node_id = request.workstation_node_id
        if node_id is None:
            if not task_counts:
                raise ValueError("运行中没有可用于停用演示的任务工位。")
            node_id = max(sorted(task_counts), key=task_counts.get)
        station = workstations.get(node_id)
        if station is None:
            raise ValueError(f"运行中不存在工位节点 {node_id!r}。")

        event_at_ms = request.requested_at_ms
        if event_at_ms is None:
            event_at_ms = max(1, int(scenario["endTimeMs"] * 0.18))
        if event_at_ms >= int(scenario["endTimeMs"]):
            raise ValueError("工位停用时刻必须早于运行结束时间。")
        outage_end_ms = min(
            int(scenario["endTimeMs"]),
            event_at_ms + request.recovery_duration_ms,
        )
        affected_tasks = [
            task
            for task in scenario.get("tasks", [])
            if node_id in {task["pickupNodeId"], task["dropoffNodeId"]}
            and int(task.get("releaseTimeMs", 0)) < outage_end_ms
        ]
        affected_task_ids = [row["taskId"] for row in affected_tasks]
        affected_vehicle_ids = sorted(
            {
                plan["vehicleId"]
                for plan in scenario.get("plans", [])
                if plan.get("taskId") in set(affected_task_ids)
            }
        )
        station_resource = f"workstation:{station['id']}"
        resource_ids = [station_resource, f"node:{node_id}"]
        evidence = [
            IncidentEvidence(
                evidenceId="EV-001",
                evidenceType="WORKSTATION_OUTAGE",
                fact=(
                    f"工位 {station['id']} 自 {event_at_ms}ms 起停用，"
                    f"演示恢复窗口结束于 {outage_end_ms}ms。"
                ),
                source=f"injection:{request.run_id}",
                observedAtMs=event_at_ms,
                attributes={
                    "workstationId": station["id"],
                    "nodeId": node_id,
                    "outageEndMs": outage_end_ms,
                },
            ),
            IncidentEvidence(
                evidenceId="EV-002",
                evidenceType="WORKSTATION_DEFINITION",
                fact=(
                    f"工位位于 {node_id}，容量 {station['capacity']}，"
                    f"允许车型 {', '.join(station['allowedRobotGroups'])}。"
                ),
                source="MASP:xiate-workstations.json",
                attributes={"workstation": station, "resourceIds": resource_ids},
            ),
            IncidentEvidence(
                evidenceId="EV-003",
                evidenceType="AFFECTED_TASKS",
                fact=f"停用窗口涉及 {len(affected_task_ids)} 个取货或放货任务。",
                source=f"MASP:{request.run_id}:planned-scenario",
                observedAtMs=event_at_ms,
                attributes={
                    "taskIds": affected_task_ids,
                    "vehicleIds": affected_vehicle_ids,
                },
            ),
            IncidentEvidence(
                evidenceId="EV-004",
                evidenceType="PLANNING_METRICS",
                fact=(
                    f"基线完成 {detail['summary']['metrics'].get('completedTaskCount', 0)} 个任务，"
                    f"插入等待 {detail['summary']['planning'].get('insertedWaitMs', 0)}ms。"
                ),
                source=f"MASP:{request.run_id}:result",
                attributes={"metrics": detail["summary"]["metrics"]},
            ),
        ]
        findings = [
            DeterministicFinding(
                code="workstation.outage.reported",
                title="工位停用事件已登记",
                detail=f"停用对象已解析为工位 {station['id']} 和节点 {node_id}。",
                certainty="CONFIRMED",
                evidenceIds=["EV-001", "EV-002"],
            ),
            DeterministicFinding(
                code="workstation.tasks.affected",
                title="关联任务已经确定",
                detail=f"共有 {len(affected_task_ids)} 个任务直接使用该工位。",
                certainty="CONFIRMED",
                evidenceIds=["EV-003"],
            ),
            DeterministicFinding(
                code="workstation.queue.risk",
                title="工位能力损失可能形成积压",
                detail="实际等待和未规划任务数量必须以各 What-if 分支的 MASP 结果为准。",
                certainty="INFERRED",
                evidenceIds=["EV-003", "EV-004"],
            ),
        ]
        incident = IncidentRecord(
            incidentType=IncidentType.WORKSTATION_DISABLED,
            severity=IncidentSeverity.HIGH,
            scenarioId=detail["summary"]["scenarioId"],
            runId=request.run_id,
            vehicleIds=affected_vehicle_ids,
            taskIds=affected_task_ids,
            resourceIds=resource_ids,
            faultCode="WORKSTATION_DISABLED",
            faultAtMs=event_at_ms,
            recoveryDurationMs=request.recovery_duration_ms,
            locationNodeId=node_id,
            workstationId=station["id"],
            eventAttributes={
                "outageEndMs": outage_end_ms,
                "availableWhatIfModes": [
                    WhatIfMode.WAIT_RECOVERY.value,
                    WhatIfMode.SUSPEND_AFFECTED_TASKS.value,
                    WhatIfMode.SAFETY_STOP.value,
                ],
            },
            evidence=evidence,
            deterministicFindings=findings,
            createdBy=request.requested_by,
        )
        return self._record_injection(incident, request.requested_by)

    def inject_deadlock(self, request: DeadlockInjectionRequest) -> IncidentRecord:
        detail = self._completed_run_detail(request.run_id)
        recovery = self.engine.deadlock_recovery_evidence(request.deadlock_case)
        case = recovery["case"]
        scenario_case = recovery["scenarioCase"]
        wait_graph = case["waitGraph"]
        decision = case["decision"]
        plan = decision.get("plan")
        vehicle_ids = list(wait_graph.get("blockedVehicleIds", []))
        resource_ids = sorted(
            {
                row["resourceId"]
                for row in wait_graph.get("dependencies", [])
            }
            | set(decision.get("frozenResourceIds", []))
        )
        location_node_id = next(
            (
                row.get("currentNodeId")
                for row in scenario_case.get("recoveryVehicles", [])
                if row.get("currentNodeId")
            ),
            plan.get("recoveryNodeId") if plan else None,
        )
        evidence = [
            IncidentEvidence(
                evidenceId="EV-001",
                evidenceType="WAIT_GRAPH_CYCLE",
                fact=(
                    f"MASP 等待图在 {wait_graph['analyzedAtMs']}ms 检测到 "
                    f"{len(wait_graph['cycles'])} 个循环，最大长度 {wait_graph['maxCycleLength']}。"
                ),
                source="MASP:deadlock-supervisor",
                observedAtMs=wait_graph["analyzedAtMs"],
                attributes={"cycles": wait_graph["cycles"]},
            ),
            IncidentEvidence(
                evidenceId="EV-002",
                evidenceType="WAIT_DEPENDENCIES",
                fact=f"循环车辆之间存在 {len(wait_graph['dependencies'])} 条确定性资源依赖。",
                source="MASP:reservation-table",
                observedAtMs=wait_graph["analyzedAtMs"],
                attributes={
                    "dependencies": wait_graph["dependencies"],
                    "priorityAgeMs": wait_graph["priorityAgeMs"],
                },
            ),
            IncidentEvidence(
                evidenceId="EV-003",
                evidenceType="RECOVERY_DECISION",
                fact=(
                    f"MASP 恢复控制器给出 {decision['action']} 决策，"
                    f"原因码为 {decision['reasonCode']}。"
                ),
                source="MASP:recovery-controller",
                observedAtMs=wait_graph["analyzedAtMs"],
                attributes={"decision": decision},
            ),
            IncidentEvidence(
                evidenceId="EV-004",
                evidenceType="RECOVERY_PLAN" if plan else "SAFETY_STOP",
                fact=(
                    f"受控倒退车辆为 {plan['vehicleId']}，距离 {plan['totalDistanceM']}m，"
                    f"预约 {plan['reservationCount']} 个资源。"
                    if plan
                    else f"无合法倒退计划，已冻结 {len(decision['freezeReservationIds'])} 条预约并安全停车。"
                ),
                source="MASP:recovery-controller",
                observedAtMs=wait_graph["analyzedAtMs"],
                attributes={"plan": plan, "freezeReservationIds": decision["freezeReservationIds"]},
            ),
            IncidentEvidence(
                evidenceId="EV-005",
                evidenceType="RECOVERY_ACCEPTANCE",
                fact="MASP 死锁场景的等待环、倒退预约和安全停车验收项全部通过。",
                source="MASP:recovery-scenario",
                attributes={"checks": recovery["result"]["checks"]},
            ),
        ]
        recovery_available = plan is not None
        findings = [
            DeterministicFinding(
                code="deadlock.wait_graph.cycle",
                title="等待图循环依赖已确认",
                detail=f"循环包含 {wait_graph['maxCycleLength']} 辆车，依赖边来自预约表。",
                certainty="CONFIRMED",
                evidenceIds=["EV-001", "EV-002"],
            ),
            DeterministicFinding(
                code="deadlock.recovery.decision",
                title="恢复可行性已确定",
                detail=(
                    "MASP 已生成并预约受控倒退计划。"
                    if recovery_available
                    else "MASP 未找到合法倒退计划，确定性安全停车已经生效。"
                ),
                certainty="CONFIRMED",
                evidenceIds=["EV-003", "EV-004", "EV-005"],
            ),
        ]
        modes = [WhatIfMode.SAFETY_STOP.value]
        if recovery_available:
            modes.insert(0, WhatIfMode.CONTROLLED_REVERSE.value)
        incident = IncidentRecord(
            incidentType=IncidentType.DEADLOCK_RISK,
            severity=(IncidentSeverity.HIGH if recovery_available else IncidentSeverity.CRITICAL),
            scenarioId=detail["summary"]["scenarioId"],
            runId=request.run_id,
            vehicleIds=vehicle_ids,
            resourceIds=resource_ids,
            faultCode=("WAIT_GRAPH_CYCLE" if recovery_available else "DEADLOCK_SAFETY_STOP"),
            faultAtMs=wait_graph["analyzedAtMs"],
            recoveryDurationMs=max(0, recovery["scenario"]["endTimeMs"] - wait_graph["analyzedAtMs"]),
            locationNodeId=location_node_id,
            locationEdgeId=(scenario_case.get("evidenceEdgeIds") or [None])[0],
            eventAttributes={
                "deadlockCase": request.deadlock_case,
                "maxCycleLength": wait_graph["maxCycleLength"],
                "recoveryAvailable": recovery_available,
                "recoveryDecision": decision,
                "availableWhatIfModes": modes,
            },
            evidence=evidence,
            deterministicFindings=findings,
            createdBy=request.requested_by,
        )
        return self._record_injection(incident, request.requested_by)

    def diagnose(self, incident_id: str, actor: str = "demo-operator") -> IncidentRecord:
        incident = self.store.get(incident_id)
        query = {
            IncidentType.VEHICLE_FAULT: f"车辆故障 {incident.fault_code or ''} 安全停车 任务重派",
            IncidentType.WORKSTATION_DISABLED: "工位停用 作业能力 任务暂停 安全封锁",
            IncidentType.DEADLOCK_RISK: "等待图 死锁 受控倒退 安全停车",
        }.get(incident.incident_type, "仓储异常 安全处置")
        sop_rows = self.knowledge.search(query, limit=3)
        evidence = [row for row in incident.evidence if row.evidence_type != "SOP"]
        for row in sop_rows:
            evidence.append(
                IncidentEvidence(
                    evidenceId=f"EV-{len(evidence) + 1:03d}",
                    evidenceType="SOP",
                    fact=row.detail,
                    source=row.source,
                    attributes={"title": row.title},
                )
            )
        candidate = incident.model_copy(update={"evidence": evidence})
        diagnosis = self.provider.diagnose_incident(candidate)
        diagnosis = self._validate_diagnosis(candidate, diagnosis)
        updated = candidate.model_copy(
            update={
                "diagnosis": diagnosis,
                "status": IncidentStatus.DIAGNOSED,
                "updated_at": utc_now(),
            }
        )
        self.store.put(updated)
        self.audit.append(
            trace_id=incident_id,
            event_type="INCIDENT_DIAGNOSED",
            actor=actor,
            payload=diagnosis.model_dump(by_alias=True, mode="json"),
        )
        return updated

    def run_what_if(
        self, incident_id: str, mode: WhatIfMode, actor: str = "demo-operator"
    ) -> IncidentRecord:
        incident = self.store.get(incident_id)
        if mode.value not in allowed_actions_for(incident):
            raise ValueError(f"{incident.incident_type.value} 不支持处置模式 {mode.value}。")
        summary = self.engine.simulate_incident_option(incident, mode)
        run_ids = dict(incident.what_if_run_ids)
        run_ids[mode.value] = summary.run_id
        updated = incident.model_copy(
            update={
                "what_if_run_ids": run_ids,
                "status": IncidentStatus.MITIGATING,
                "updated_at": utc_now(),
            }
        )
        self.store.put(updated)
        self.audit.append(
            trace_id=incident_id,
            event_type="INCIDENT_WHAT_IF_COMPLETED",
            actor=actor,
            payload={
                "incidentId": incident_id,
                "mode": mode.value,
                "run": summary.model_dump(by_alias=True, mode="json"),
            },
        )
        return updated

    def link_approval(
        self, incident_id: str, mode: WhatIfMode, approval_id: str, actor: str
    ) -> IncidentRecord:
        incident = self.store.get(incident_id)
        approval_ids = dict(incident.approval_ids)
        approval_ids[mode.value] = approval_id
        updated = incident.model_copy(
            update={"approval_ids": approval_ids, "updated_at": utc_now()}
        )
        self.store.put(updated)
        self.audit.append(
            trace_id=incident_id,
            event_type="INCIDENT_APPROVAL_CREATED",
            actor=actor,
            payload={
                "incidentId": incident_id,
                "mode": mode.value,
                "approvalId": approval_id,
            },
        )
        return updated

    def _completed_run_detail(self, run_id: str) -> dict[str, Any]:
        detail = self.engine.get_run_detail(run_id)
        if detail["summary"]["status"] != "COMPLETED":
            raise ValueError("只能在已完成且证据完整的仿真运行上注入事件。")
        return detail

    def _record_injection(self, incident: IncidentRecord, actor: str) -> IncidentRecord:
        self.store.put(incident)
        self.audit.append(
            trace_id=incident.incident_id,
            event_type="INCIDENT_INJECTED",
            actor=actor,
            payload=incident.model_dump(by_alias=True, mode="json"),
        )
        return incident

    @staticmethod
    def _default_vehicle(plans: list[dict[str, Any]]) -> str:
        counts: dict[str, int] = {}
        for plan in plans:
            counts[plan["vehicleId"]] = counts.get(plan["vehicleId"], 0) + len(
                plan.get("segments", [])
            )
        if not counts:
            raise ValueError("运行中没有车辆计划。")
        return max(sorted(counts), key=counts.get)

    @staticmethod
    def _safe_fault_segment(
        plans: list[dict[str, Any]], target_ms: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidates = [
            (segment, plan)
            for plan in plans
            for segment in plan.get("segments", [])
            if segment.get("endNodeId") and segment.get("kind") == "traverse"
        ]
        if not candidates:
            raise ValueError("车辆计划中不存在可以安全注入故障的移动段。")
        return min(candidates, key=lambda item: (abs(int(item[0]["endMs"]) - target_ms), int(item[0]["endMs"])))

    @staticmethod
    def _validate_diagnosis(
        incident: IncidentRecord, diagnosis: DiagnosisReport
    ) -> DiagnosisReport:
        violation = diagnosis_violation(
            diagnosis,
            evidence_ids=(row.evidence_id for row in incident.evidence),
            vehicle_ids=incident.vehicle_ids,
            task_ids=incident.task_ids,
            allowed_actions=allowed_actions_for(incident),
        )
        if violation is not None:
            return IncidentService._fallback_diagnosis(
                incident, model=f"{diagnosis.model}:{violation}"
            )
        return diagnosis

    @staticmethod
    def _fallback_diagnosis(
        incident: IncidentRecord, model: str = "deterministic-fallback"
    ) -> DiagnosisReport:
        return deterministic_diagnosis(incident, model=model)
