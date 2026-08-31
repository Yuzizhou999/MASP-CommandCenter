from __future__ import annotations

import json
import zipfile

from command_center.approvals import ApprovalStore
from command_center.audit import AuditStore
from command_center.benchmark import BenchmarkRunner, BenchmarkScenarioFactory
from command_center.contracts import BenchmarkRequest, DatasetExportRequest
from command_center.dataset_exports import DatasetExporter
from command_center.engine_adapter import MaspAdapter
from command_center.incidents import IncidentStore
from command_center.intent_store import IntentStore


def test_benchmark_factory_scales_to_100_unique_vehicles(isolated_settings) -> None:
    engine = MaspAdapter(isolated_settings)
    factory = BenchmarkScenarioFactory(engine, "rhpp-long-distance-conflict")

    first = factory.build(
        vehicle_count=100,
        arrival_profile="high",
        fleet_mix="mixed",
        seed=7,
        horizon_ms=900000,
    )
    repeated = factory.build(
        vehicle_count=100,
        arrival_profile="high",
        fleet_mix="mixed",
        seed=7,
        horizon_ms=900000,
    )

    assert first == repeated
    assert len(first["vehicles"]) == 100
    assert len({row["vehicleId"] for row in first["vehicles"]}) == 100
    assert len({row["initialNodeId"] for row in first["vehicles"]}) == 100
    assert len(first["tasks"]) == 300
    assert max(row["releaseTimeMs"] for row in first["tasks"]) <= 720000


def test_full_benchmark_matrix_is_representable() -> None:
    request = BenchmarkRequest(
        vehicleCounts=[14, 30, 50, 100],
        arrivalProfiles=["low", "medium", "high"],
        fleetMixes=["mixed", "fork", "jack"],
        policies=["task_age", "shortest_remaining", "congestion", "top_k", "rl"],
        seeds=list(range(10)),
    )
    assert request.case_count == 1800


def test_benchmark_runner_writes_reproducible_statistics(isolated_settings) -> None:
    engine = MaspAdapter(isolated_settings)
    audit = AuditStore(isolated_settings.data_dir / "audit.jsonl")
    runner = BenchmarkRunner(isolated_settings.data_dir, engine, audit)

    report = runner.run(
        BenchmarkRequest(
            suiteName="自动化评测",
            baseScenarioId="interactive-multi-fleet",
            vehicleCounts=[4],
            arrivalProfiles=["medium"],
            fleetMixes=["mixed"],
            policies=["top_k"],
            seeds=[0, 1],
            horizonMs=300000,
        )
    )

    assert report["caseCount"] == 2
    assert report["completedCaseCount"] == 2
    assert report["safetyGate"]["passed"] is True
    assert report["safetyGate"]["conflictCaseCount"] == 0
    aggregate = report["aggregates"][0]
    assert aggregate["metrics"]["completedTaskCount"]["count"] == 2
    assert aggregate["metrics"]["completedTaskCount"]["ci95Low"] is not None
    report_path = (
        isolated_settings.data_dir
        / "evaluations"
        / report["benchmarkId"]
        / "report.json"
    )
    assert (
        json.loads(report_path.read_text(encoding="utf-8"))["benchmarkId"]
        == report["benchmarkId"]
    )
    assert runner.get(report["benchmarkId"])["caseCount"] == 2
    assert runner.list()[0]["benchmarkId"] == report["benchmarkId"]


def test_planning_timeout_fails_benchmark_safety_gate(
    isolated_settings, monkeypatch
) -> None:
    engine = MaspAdapter(isolated_settings)
    runner = BenchmarkRunner(isolated_settings.data_dir, engine)

    monkeypatch.setattr(
        engine,
        "evaluate_scenario_document",
        lambda *args, **kwargs: {
            "status": "COMPLETED",
            "durationMs": 1.0,
            "metrics": {"reservationConflictRejections": 0},
            "planning": {"planningTimeoutCount": 1},
            "safety": {"reservationConflictRejections": 0},
        },
    )
    report = runner.run(
        BenchmarkRequest(
            baseScenarioId="interactive-multi-fleet",
            vehicleCounts=[4],
            arrivalProfiles=["low"],
            fleetMixes=["mixed"],
            policies=["top_k"],
            seeds=[0],
            horizonMs=300000,
        )
    )

    assert report["safetyGate"]["passed"] is False
    assert report["safetyGate"]["planningTimeoutCaseCount"] == 1


def test_dataset_export_removes_free_text_actors_paths_and_secrets(
    isolated_settings,
) -> None:
    engine = MaspAdapter(isolated_settings)
    audit = AuditStore(isolated_settings.data_dir / "audit.jsonl")
    audit.append(
        trace_id="trace-export-test",
        event_type="TEST_EVENT",
        actor="operator@example.test",
        payload={
            "request": "联系电话 13800138000，请立即执行",
            "apiKey": "must-not-leak",
            "manifestPath": "C:/private/run.json",
        },
    )
    exporter = DatasetExporter(
        isolated_settings.data_dir,
        engine=engine,
        audit=audit,
        approvals=ApprovalStore(isolated_settings.data_dir / "approvals.json"),
        intents=IntentStore(isolated_settings.data_dir / "committed-intents.json"),
        incidents=IncidentStore(isolated_settings.data_dir / "incidents.json"),
    )

    manifest = exporter.create(
        DatasetExportRequest(name="自动化脱敏导出", requestedBy="data-owner")
    )

    assert manifest["quality"]["passed"] is True
    assert manifest["recordCount"] == 1
    root = isolated_settings.data_dir / "dataset-exports" / manifest["exportId"]
    dataset_text = (root / "dataset.jsonl").read_text(encoding="utf-8")
    assert "operator@example.test" not in dataset_text
    assert "13800138000" not in dataset_text
    assert "must-not-leak" not in dataset_text
    assert "C:/private" not in dataset_text
    assert "[TEXT_REMOVED]" in dataset_text
    with zipfile.ZipFile(root / "dataset-bundle.zip") as bundle:
        assert set(bundle.namelist()) == {
            "manifest.json",
            "quality-report.json",
            "dataset.jsonl",
        }
