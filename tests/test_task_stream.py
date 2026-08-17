from __future__ import annotations

import json
from pathlib import Path

import pytest

from command_center.masp.domain import DomainError
from command_center.masp.scenario_package import package_from_assets
from command_center.masp.task_stream import (
    generate_task_stream,
    load_task_stream_generation,
    task_stream_to_dict,
    validate_task_stream_generation_document,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT.parent / "MASP"


def read_json(path: str) -> dict:
    return json.loads((ENGINE_ROOT / path).read_text(encoding="utf-8"))


def scene():
    package = package_from_assets(
        package_id="task-stream-test",
        version="1.0.0",
        map_document=read_json("generated/xiate-unified-map-model.json"),
        profile_document=read_json("config/robot-profiles.json"),
        workstation_document=read_json("generated/xiate-workstations.json"),
        vehicle_document=read_json("config/initial-vehicles.json"),
        traffic_document=read_json("config/traffic-zones.json"),
        scenario_document=read_json("scenarios/realistic-multi-fleet.json"),
        conflict_document=read_json("generated/xiate-conflict-resources.json"),
    )
    return package.warehouse_scene


def generation(**overrides):
    value = {
        "streamId": "generated-demo",
        "seed": 17,
        "endTimeMs": 120000,
        "maxTasks": 8,
        "arrival": {"mode": "fixed_interval", "intervalMs": 10000},
        "odPairs": [
            {
                "pickupNodeId": "fork:AP1123",
                "dropoffNodeId": "fork:AP1119",
                "requiredRobotGroup": "fork",
                "payloadType": "pallet",
                "weight": 1,
            },
            {
                "pickupNodeId": "jack:AP146",
                "dropoffNodeId": "jack:AP189",
                "requiredRobotGroup": "jack",
                "payloadType": "shelf",
                "weight": 1,
            },
        ],
        "priorityDistribution": [
            {"priorityClass": 0, "weight": 3},
            {"priorityClass": 2, "weight": 1},
        ],
        "dueTimePolicy": {"mode": "relative", "offsetMs": 60000},
    }
    value.update(overrides)
    return value


def test_same_seed_produces_identical_stream() -> None:
    first = task_stream_to_dict(generate_task_stream(scene(), generation()))
    second = task_stream_to_dict(generate_task_stream(scene(), generation()))
    assert first == second
    assert len(first["tasks"]) == 8
    assert first["tasks"][0]["taskId"] == "generated-demo-task-000001"


def test_poisson_and_time_windows_are_supported() -> None:
    poisson = generate_task_stream(
        scene(),
        generation(
            arrival={"mode": "poisson", "meanIntervalMs": 10000},
            maxTasks=20,
        ),
    )
    assert 1 <= len(poisson.tasks) <= 20
    assert all(0 <= int(task["releaseTimeMs"]) < 120000 for task in poisson.tasks)

    windows = generate_task_stream(
        scene(),
        generation(
            arrival={
                "mode": "time_windows",
                "windows": [
                    {"startTimeMs": 0, "endTimeMs": 30000, "mode": "fixed_interval", "intervalMs": 15000},
                    {"startTimeMs": 60000, "endTimeMs": 90000, "mode": "fixed_interval", "intervalMs": 10000},
                ],
            },
            maxTasks=10,
        ),
    )
    assert [task["releaseTimeMs"] for task in windows.tasks] == [0, 15000, 60000, 70000, 80000]


def test_unreachable_or_incompatible_od_is_filtered_and_empty_is_rejected() -> None:
    generated = generate_task_stream(
        scene(),
        generation(
            odPairs=[
                {
                    "pickupNodeId": "fork:AP1123",
                    "dropoffNodeId": "fork:PP1171",
                    "requiredRobotGroup": "fork",
                    "payloadType": "pallet",
                    "weight": 100,
                },
                generation()["odPairs"][0],
            ]
        ),
    )
    assert all(task["dropoffNodeId"] == "fork:AP1119" for task in generated.tasks)

    with pytest.raises(DomainError, match="no configured OD pair"):
        generate_task_stream(
            scene(),
            generation(
                odPairs=[
                    {
                        "pickupNodeId": "fork:AP1123",
                        "dropoffNodeId": "fork:PP1171",
                        "requiredRobotGroup": "fork",
                        "payloadType": "pallet",
                        "weight": 1,
                    }
                ]
            ),
        )


def test_generation_schema_is_checked_before_generation() -> None:
    schema = ROOT / "schemas" / "task-stream-generation.schema.json"
    validate_task_stream_generation_document(generation(), schema)
    loaded = load_task_stream_generation(
        ROOT / "tests" / "fixtures" / "task-stream-generation.json",
        schema_path=schema,
    )
    assert loaded["streamId"] == "fixture-stream"

    with pytest.raises(DomainError, match="intervalMs"):
        validate_task_stream_generation_document(
            generation(arrival={"mode": "fixed_interval", "intervalMs": 0}),
            schema,
        )
