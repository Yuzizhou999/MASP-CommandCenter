from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from command_center.masp.scenario_package import (
    ScenarioPackage,
    compile_scenario_package,
    package_from_assets,
    validate_scenario_package_document,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT.parent / "MASP"


def read_json(path: str) -> dict:
    return json.loads((ENGINE_ROOT / path).read_text(encoding="utf-8"))


def build_package() -> ScenarioPackage:
    return package_from_assets(
        package_id="test-realistic",
        version="1.0.0",
        map_document=read_json("generated/xiate-unified-map-model.json"),
        profile_document=read_json("config/robot-profiles.json"),
        workstation_document=read_json("generated/xiate-workstations.json"),
        vehicle_document=read_json("config/initial-vehicles.json"),
        traffic_document=read_json("config/traffic-zones.json"),
        scenario_document=read_json("scenarios/realistic-multi-fleet.json"),
        conflict_document=read_json("generated/xiate-conflict-resources.json"),
    )


def package_document(package: ScenarioPackage) -> dict:
    return {
        "schemaVersion": 1,
        "packageId": package.package_id,
        "version": package.version,
        "status": package.status,
        "metadata": dict(package.metadata),
        "warehouseScene": {
            "sceneId": package.warehouse_scene.scene_id,
            "name": package.warehouse_scene.name,
            "bounds": dict(package.warehouse_scene.bounds),
            "robotProfiles": package.warehouse_scene.robot_profiles,
            "nodes": list(package.warehouse_scene.nodes),
            "edges": list(package.warehouse_scene.edges),
            "workstations": list(package.warehouse_scene.workstations),
            "vehicles": list(package.warehouse_scene.vehicles),
            "recoveryNodes": list(package.warehouse_scene.recovery_nodes),
            "trafficZones": list(package.warehouse_scene.traffic_zones),
            "safety": dict(package.warehouse_scene.safety),
        },
        "taskStream": {
            "streamId": package.task_stream.stream_id,
            "seed": package.task_stream.seed,
            "endTimeMs": package.task_stream.end_time_ms,
            "tasks": list(package.task_stream.tasks),
            "events": list(package.task_stream.events),
        },
    }


def test_existing_runtime_assets_migrate_to_a_valid_package() -> None:
    package = build_package()
    report = package.validate()
    assert report.valid, report.to_dict()
    assert report.stats["nodeCount"] == 552
    assert report.stats["edgeCount"] == 1204
    assert report.stats["taskCount"] == 32


def test_package_schema_accepts_migrated_document() -> None:
    validate_scenario_package_document(
        package_document(build_package()),
        ROOT / "schemas" / "scenario-package.schema.json",
    )


def test_invalid_package_reports_unreachable_task_and_duplicate_start() -> None:
    package = build_package()
    document = package_document(package)
    document["warehouseScene"]["vehicles"].append(
        deepcopy(document["warehouseScene"]["vehicles"][0])
    )
    document["warehouseScene"]["vehicles"][-1]["vehicleId"] = "duplicate-start"
    document["taskStream"]["tasks"][0]["dropoffNodeId"] = "fork:PP1171"
    broken = ScenarioPackage.from_dict(document)
    report = broken.validate()
    codes = {item.code for item in report.issues}
    assert "scenario.vehicle.start_duplicate" in codes
    assert "scenario.task.workstation" in codes or "scenario.task.dropoff_unreachable" in codes


def test_compile_keeps_existing_runtime_contracts(tmp_path: Path) -> None:
    compiled = compile_scenario_package(
        build_package(),
        scheduler_template=read_json("config/scheduler.json"),
    )
    paths = compiled.write_to(tmp_path)
    assert set(paths) == {
        "map-model.json",
        "conflict-resources.json",
        "workstations.json",
        "robot-profiles.json",
        "scheduler.json",
        "initial-vehicles.json",
        "traffic-zones.json",
        "dispatch-scenario.json",
        "task-stream.json",
        "validation-report.json",
        "manifest.json",
    }
    scenario = json.loads((tmp_path / "dispatch-scenario.json").read_text(encoding="utf-8"))
    assert scenario["scenarioId"] == "test-realistic@1.0.0"
    assert len(scenario["tasks"]) == 32
    assert compiled.validation.valid
