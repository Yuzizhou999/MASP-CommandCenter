from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from .audit import AuditStore
from .contracts import (
    DeterministicFinding,
    DiagnosisReport,
    FaultInjectionRequest,
    IncidentEvidence,
    IncidentRecord,
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
    WhatIfMode,
    utc_now,
)
from .diagnosis import ALLOWED_INCIDENT_ACTIONS, deterministic_diagnosis
from .engine_adapter import MaspAdapter
from .knowledge import KnowledgeBase
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

    def diagnose(self, incident_id: str, actor: str = "demo-operator") -> IncidentRecord:
        incident = self.store.get(incident_id)
        sop_rows = self.knowledge.search(
            f"车辆故障 {incident.fault_code or ''} 安全停车 任务重派", limit=3
        )
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
        evidence_ids = {row.evidence_id for row in incident.evidence}
        referenced = {
            evidence_id
            for row in diagnosis.root_cause_candidates
            for evidence_id in row.evidence_ids
        }
        referenced.update(
            evidence_id
            for row in diagnosis.recommendations
            for evidence_id in row.evidence_ids
        )
        if not referenced.issubset(evidence_ids):
            return IncidentService._fallback_diagnosis(
                incident, model=f"{diagnosis.model}:invalid-evidence"
            )
        if not set(diagnosis.affected_vehicle_ids).issubset(set(incident.vehicle_ids)):
            return IncidentService._fallback_diagnosis(
                incident, model=f"{diagnosis.model}:invalid-vehicle"
            )
        if not set(diagnosis.affected_task_ids).issubset(set(incident.task_ids)):
            return IncidentService._fallback_diagnosis(
                incident, model=f"{diagnosis.model}:invalid-task"
            )
        if any(
            row.action_code not in ALLOWED_INCIDENT_ACTIONS
            or row.risk_level.value != "R3_HIGH"
            or not row.requires_simulation
            or not row.requires_approval
            for row in diagnosis.recommendations
        ):
            return IncidentService._fallback_diagnosis(
                incident, model=f"{diagnosis.model}:invalid-action"
            )
        return diagnosis

    @staticmethod
    def _fallback_diagnosis(
        incident: IncidentRecord, model: str = "deterministic-fallback"
    ) -> DiagnosisReport:
        return deterministic_diagnosis(incident, model=model)
