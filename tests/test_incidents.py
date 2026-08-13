from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import command_center.api as api_module
from command_center.audit import AuditStore
from command_center.contracts import (
    DiagnosisReport,
    FaultInjectionRequest,
    IncidentRecommendation,
    RiskLevel,
    RootCauseCandidate,
    SimulationRequest,
    WhatIfMode,
)
from command_center.engine_adapter import MaspAdapter
from command_center.incidents import IncidentService, IncidentStore
from command_center.knowledge import KnowledgeBase
from command_center.provider import DeepSeekProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def incident_system(isolated_settings):
    engine = MaspAdapter(isolated_settings)
    store = IncidentStore(isolated_settings.data_dir / "incidents.json")
    service = IncidentService(
        store=store,
        engine=engine,
        provider=DeepSeekProvider(isolated_settings),
        knowledge=KnowledgeBase(PROJECT_ROOT / "knowledge"),
        audit=AuditStore(isolated_settings.data_dir / "audit.jsonl"),
    )
    baseline = engine.simulate(
        SimulationRequest(
            scenarioId="interactive-multi-fleet",
            label="incident integration baseline",
        )
    )
    return engine, store, service, baseline


def test_fault_is_injected_at_completed_traverse_node_and_evidence_is_grounded(
    incident_system,
) -> None:
    engine, _, service, baseline = incident_system
    incident = service.inject_vehicle_fault(
        FaultInjectionRequest(runId=baseline.run_id, requestedAtMs=60000)
    )
    detail = engine.get_run_detail(baseline.run_id)
    matching_segments = [
        segment
        for plan in detail["scenario"]["plans"]
        if plan["vehicleId"] == incident.vehicle_ids[0]
        for segment in plan["segments"]
        if segment["kind"] == "traverse"
        and segment["endMs"] == incident.fault_at_ms
        and segment["endNodeId"] == incident.location_node_id
    ]
    assert matching_segments
    assert incident.location_node_id is not None
    assert incident.location_edge_id is not None
    evidence_ids = [row.evidence_id for row in incident.evidence]
    assert len(evidence_ids) == len(set(evidence_ids))
    assert set(incident.vehicle_ids).issubset(
        {row["vehicleId"] for row in detail["scenario"]["vehicles"]}
    )
    assert set(incident.task_ids).issubset(
        {row["taskId"] for row in detail["scenario"]["tasks"]}
    )


def test_unknown_vehicle_and_incomplete_run_are_rejected(incident_system) -> None:
    _, _, service, baseline = incident_system
    with pytest.raises(ValueError, match="不存在车辆"):
        service.inject_vehicle_fault(
            FaultInjectionRequest(runId=baseline.run_id, vehicleId="model-invented")
        )

    original = service.engine.get_run_detail
    service.engine.get_run_detail = lambda _: {"summary": {"status": "FAILED"}}  # type: ignore[method-assign]
    try:
        with pytest.raises(ValueError, match="已完成"):
            service.inject_vehicle_fault(FaultInjectionRequest(runId="failed-run"))
    finally:
        service.engine.get_run_detail = original  # type: ignore[method-assign]


def test_unconfigured_model_and_hallucinated_evidence_use_rule_fallback(
    incident_system,
) -> None:
    _, _, service, baseline = incident_system
    incident = service.inject_vehicle_fault(FaultInjectionRequest(runId=baseline.run_id))
    diagnosed = service.diagnose(incident.incident_id)
    assert diagnosed.diagnosis is not None
    assert diagnosed.diagnosis.fallback_used is True
    assert all(
        evidence_id in {row.evidence_id for row in diagnosed.evidence}
        for cause in diagnosed.diagnosis.root_cause_candidates
        for evidence_id in cause.evidence_ids
    )

    fabricated = DiagnosisReport(
        summary="fabricated",
        confirmedFacts=[],
        rootCauseCandidates=[
            RootCauseCandidate(
                code="FAKE",
                title="fake",
                explanation="fake",
                confidence=1,
                evidenceIds=["EV-999"],
            )
        ],
        affectedVehicleIds=diagnosed.vehicle_ids,
        affectedTaskIds=diagnosed.task_ids,
        recommendations=[
            IncidentRecommendation(
                actionCode="WAIT_RECOVERY",
                action="fake",
                rationale="fake",
                riskLevel=RiskLevel.R3_HIGH,
                evidenceIds=["EV-999"],
            )
        ],
        model="test-model",
        fallbackUsed=False,
    )
    validated = service._validate_diagnosis(diagnosed, fabricated)
    assert validated.fallback_used is True
    assert validated.model.endswith(":invalid-evidence")


def test_what_if_runs_create_real_evidence_and_loaded_task_is_not_reassigned(
    incident_system,
) -> None:
    engine, _, service, baseline = incident_system
    detail = engine.get_run_detail(baseline.run_id)
    loaded = next(
        (plan, segment)
        for plan in detail["scenario"]["plans"]
        for segment in plan["segments"]
        if segment["kind"] == "traverse"
        and segment.get("expectedLoadState") == "loaded"
    )
    plan, segment = loaded
    incident = service.inject_vehicle_fault(
        FaultInjectionRequest(
            runId=baseline.run_id,
            vehicleId=plan["vehicleId"],
            requestedAtMs=segment["endMs"],
        )
    )
    assert incident.load_state == "loaded"

    service.run_what_if(incident.incident_id, WhatIfMode.WAIT_RECOVERY)
    incident = service.run_what_if(
        incident.incident_id,
        WhatIfMode.ISOLATE_REASSIGN,
    )
    summaries = [engine.get_run(run_id) for run_id in incident.what_if_run_ids.values()]
    assert all(row.status == "COMPLETED" for row in summaries)
    assert all(row.safety["incidentId"] == incident.incident_id for row in summaries)
    assert all(row.safety["simulationOnly"] is True for row in summaries)
    assert engine.compare(list(incident.what_if_run_ids.values())).runs

    isolate_run_id = incident.what_if_run_ids[WhatIfMode.ISOLATE_REASSIGN.value]
    isolate_root = isolated_root = Path(engine.get_run(isolate_run_id).manifest_path).parent
    context = json.loads((isolate_root / "incident-context.json").read_text(encoding="utf-8"))
    scenario = json.loads((isolated_root / "input-scenario.json").read_text(encoding="utf-8"))
    assert context["manualTransferRequired"] is True
    assert context["removedTaskIds"] == incident.task_ids
    assert not set(incident.task_ids).intersection(row["taskId"] for row in scenario["tasks"])


def test_incident_api_endpoints_use_isolated_service(monkeypatch, incident_system) -> None:
    engine, store, service, baseline = incident_system
    monkeypatch.setattr(api_module, "engine", engine)
    monkeypatch.setattr(api_module, "incident_store", store)
    monkeypatch.setattr(api_module, "incident_service", service)
    client = TestClient(api_module.app)

    injected = client.post(
        "/api/v1/incidents/inject",
        json={"runId": baseline.run_id, "faultCode": "TEST_FAULT"},
    )
    assert injected.status_code == 200
    incident_id = injected.json()["incidentId"]
    diagnosed = client.post(f"/api/v1/incidents/{incident_id}/diagnose")
    assert diagnosed.status_code == 200
    what_if = client.post(
        f"/api/v1/incidents/{incident_id}/what-if",
        json={"mode": "WAIT_RECOVERY"},
    )
    assert what_if.status_code == 200
    report = client.get(f"/api/v1/incidents/{incident_id}/report")
    assert report.status_code == 200
    assert report.json()["mode"] == "simulation-only"
