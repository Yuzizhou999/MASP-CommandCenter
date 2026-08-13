from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from command_center.audit import AuditStore
from command_center.engine_adapter import MaspAdapter
from command_center.scenario_drafts import ScenarioDraftConflict, ScenarioDraftStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = PROJECT_ROOT.parent / "MASP"


def read(path: str) -> dict:
    return json.loads((ENGINE_ROOT / path).read_text(encoding="utf-8"))


def package_document() -> dict:
    map_document = read("generated/xiate-unified-map-model.json")
    profile = read("config/robot-profiles.json")
    workstations = read("generated/xiate-workstations.json")
    vehicles = read("config/initial-vehicles.json")
    traffic = read("config/traffic-zones.json")
    scenario = read("scenarios/realistic-multi-fleet.json")
    conflicts = read("generated/xiate-conflict-resources.json")
    bounds = map_document["metadata"]["bounds"]
    nodes = []
    for row in map_document["nodes"]:
        nodes.append({
            "id": row["id"], "type": row["type"], "x": row["x"], "y": row["y"],
            "allowedRobotGroups": row["allowedRobotGroups"],
            "waitAllowedByGroup": row.get("allowWaitByGroup", {}),
            "positionsByGroup": row.get("positions", {}), "headings": row.get("headings", {}),
            "propertiesByGroup": row.get("propertiesByGroup", {}), "capacity": row.get("capacity", 1),
        })
    edges = []
    for row in map_document["edges"]:
        edges.append({
            "id": row["id"], "name": row.get("name", row["id"]),
            "startNodeId": row["start"], "endNodeId": row["end"],
            "controlPoints": [row[key] for key in ("p0", "p1", "p2", "p3")],
            "lengthM": row["length"], "motionDirection": row.get("motionDirection", 0),
            "moveStyle": row.get("moveStyle", 0), "maxSpeedMps": row.get("maxSpeed"),
            "loadedMaxSpeedMps": row.get("loadMaxSpeed"), "robotGroup": row["robotGroup"],
        })
    return {
        "schemaVersion": 1, "packageId": "draft-api-test", "version": "1.0.0", "status": "draft",
        "metadata": {"createdBy": "test"},
        "warehouseScene": {
            "sceneId": "draft-api-scene", "name": "测试场景", "bounds": bounds,
            "robotProfiles": profile["robotGroups"], "nodes": nodes, "edges": edges,
            "workstations": workstations["workstations"], "vehicles": vehicles["vehicles"],
            "recoveryNodes": traffic.get("recoveryNodes", []), "trafficZones": traffic.get("zones", []),
            "safety": {"footprintMarginM": profile.get("simulationSafety", {}).get("footprintMargin", 0), "conflictSampleSpacingM": conflicts.get("metadata", {}).get("sampleSpacing", 0.25), "localizationErrorM": None, "communicationLatencyMs": None, "provisional": True},
        },
        "taskStream": {"streamId": "draft-api-stream", "seed": scenario["seed"], "endTimeMs": scenario["endTimeMs"], "tasks": scenario["tasks"], "events": []},
    }


def store(isolated_settings):
    engine = MaspAdapter(isolated_settings)
    return ScenarioDraftStore(isolated_settings.data_dir, engine, AuditStore(isolated_settings.data_dir / "audit.jsonl"))


def test_draft_lifecycle_and_revision_guard(isolated_settings) -> None:
    drafts = store(isolated_settings)
    created = drafts.create(package_document(), "operator")
    assert created["revision"] == 1
    assert drafts.validate("draft-api-test", "operator")["valid"]

    changed = deepcopy(package_document())
    changed["version"] = "1.0.1"
    updated = drafts.update("draft-api-test", changed, 1, "operator")
    assert updated["revision"] == 2
    with pytest.raises(ScenarioDraftConflict):
        drafts.update("draft-api-test", changed, 1, "operator")


def test_task_generation_compile_and_publish(isolated_settings) -> None:
    drafts = store(isolated_settings)
    drafts.create(package_document(), "operator")
    current = drafts.generate_tasks(
        "draft-api-test",
        {
            "streamId": "draft-generated", "seed": 9, "endTimeMs": 120000, "maxTasks": 3,
            "arrival": {"mode": "fixed_interval", "intervalMs": 30000},
            "odPairs": [{"pickupNodeId": "fork:AP1123", "dropoffNodeId": "fork:AP1119", "requiredRobotGroup": "fork", "payloadType": "pallet", "weight": 1}],
        },
        1,
        "operator",
    )
    assert current["revision"] == 2
    compiled = drafts.compile("draft-api-test", "operator")
    assert compiled["compiled"] is True
    published = drafts.publish("draft-api-test", "supervisor")
    assert published["status"] == "published"
    assert json.loads((isolated_settings.data_dir / "scenario-builds" / "draft-api-test" / "1.0.0" / "published" / "manifest.json").read_text(encoding="utf-8"))["status"] == "published"


def test_runtime_import_uses_selected_scenario_vehicles(isolated_settings) -> None:
    drafts = store(isolated_settings)
    document = drafts.engine.scenario_package_from_runtime(
        "explicit-single-vehicle",
        package_id="explicit-import",
        version="0.1.0",
        created_by="operator",
    )
    assert len(document["warehouseScene"]["vehicles"]) == len(
        read("scenarios/explicit-single-vehicle.json")["vehicles"]
    )
    assert document["warehouseScene"]["vehicles"][0]["vehicleId"] == read(
        "scenarios/explicit-single-vehicle.json"
    )["vehicles"][0]["vehicleId"]
