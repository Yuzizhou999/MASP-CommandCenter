from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import networkx as nx
from jsonschema import Draft202012Validator

from .domain import DomainError
from .scenario_package import TaskStreamSpec, WarehouseSceneSpec


def validate_task_stream_generation_document(
    document: Mapping[str, Any], schema_path: Path
) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda item: [str(part) for part in item.absolute_path],
    )
    if not errors:
        return
    error = errors[0]
    path = "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in error.absolute_path
    )
    raise DomainError(
        "task_stream.generation.schema.invalid",
        f"task stream generation {path}: {error.message}",
    )


def load_task_stream_generation(
    path: Path, *, schema_path: Path | None = None
) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if schema_path is not None:
        validate_task_stream_generation_document(document, schema_path)
    return document


def _weighted_choice(
    rng: random.Random, values: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    total = sum(float(item["weight"]) for item in values)
    point = rng.random() * total
    cumulative = 0.0
    for item in values:
        cumulative += float(item["weight"])
        if point < cumulative:
            return item
    return values[-1]


def _graphs(scene: WarehouseSceneSpec) -> dict[str, nx.DiGraph]:
    graphs = {group: nx.DiGraph() for group in scene.robot_profiles}
    for node in scene.nodes:
        for group in node["allowedRobotGroups"]:
            if group in graphs:
                graphs[group].add_node(str(node["id"]))
    for edge in scene.edges:
        group = str(edge["robotGroup"])
        if group in graphs:
            graphs[group].add_edge(
                str(edge["startNodeId"]), str(edge["endNodeId"])
            )
    return graphs


def _eligible_od_pairs(
    scene: WarehouseSceneSpec, configured: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    stations = {str(item["nodeId"]): item for item in scene.workstations}
    fleet_groups = {str(item["robotGroup"]) for item in scene.vehicles}
    graphs = _graphs(scene)
    eligible: list[Mapping[str, Any]] = []
    for pair in configured:
        group = str(pair["requiredRobotGroup"])
        pickup_id = str(pair["pickupNodeId"])
        dropoff_id = str(pair["dropoffNodeId"])
        pickup = stations.get(pickup_id)
        dropoff = stations.get(dropoff_id)
        graph = graphs.get(group)
        if (
            group not in fleet_groups
            or pickup is None
            or dropoff is None
            or group not in pickup["allowedRobotGroups"]
            or group not in dropoff["allowedRobotGroups"]
            or "pickup" not in pickup["capabilities"]
            or "dropoff" not in dropoff["capabilities"]
            or graph is None
            or pickup_id not in graph
            or dropoff_id not in graph
            or not nx.has_path(graph, pickup_id, dropoff_id)
        ):
            continue
        eligible.append(pair)
    if not eligible:
        raise DomainError(
            "task_stream.od_pair.unavailable",
            "no configured OD pair is compatible with the fleet and directed map",
        )
    return eligible


def _fixed_arrivals(
    *, start_ms: int, end_ms: int, interval_ms: int, limit: int
) -> list[int]:
    return list(range(start_ms, end_ms, interval_ms))[:limit]


def _poisson_arrivals(
    rng: random.Random,
    *,
    start_ms: int,
    end_ms: int,
    mean_interval_ms: float,
    limit: int,
) -> list[int]:
    arrivals: list[int] = []
    current = float(start_ms)
    while len(arrivals) < limit:
        current += rng.expovariate(1.0 / mean_interval_ms)
        rounded = round(current)
        if rounded >= end_ms:
            break
        arrivals.append(max(start_ms, rounded))
    return arrivals


def _arrival_times(
    rng: random.Random, arrival: Mapping[str, Any], end_time_ms: int, limit: int
) -> list[int]:
    mode = str(arrival["mode"])
    start_ms = int(arrival.get("startTimeMs", 0))
    if mode == "fixed_interval":
        return _fixed_arrivals(
            start_ms=start_ms,
            end_ms=end_time_ms,
            interval_ms=int(arrival["intervalMs"]),
            limit=limit,
        )
    if mode == "poisson":
        return _poisson_arrivals(
            rng,
            start_ms=start_ms,
            end_ms=end_time_ms,
            mean_interval_ms=float(arrival["meanIntervalMs"]),
            limit=limit,
        )
    arrivals: list[int] = []
    for window in arrival["windows"]:
        remaining = limit - len(arrivals)
        if remaining <= 0:
            break
        window_start = int(window["startTimeMs"])
        window_end = min(int(window["endTimeMs"]), end_time_ms)
        if window["mode"] == "fixed_interval":
            rows = _fixed_arrivals(
                start_ms=window_start,
                end_ms=window_end,
                interval_ms=int(window["intervalMs"]),
                limit=remaining,
            )
        else:
            rows = _poisson_arrivals(
                rng,
                start_ms=window_start,
                end_ms=window_end,
                mean_interval_ms=float(window["meanIntervalMs"]),
                limit=remaining,
            )
        arrivals.extend(rows)
    return sorted(arrivals)[:limit]


def _service_time(
    policy: Mapping[str, Any], station: Mapping[str, Any], field: str
) -> int:
    if policy["mode"] == "fixed":
        return int(policy[field])
    return int(station[field])


def _due_time(
    rng: random.Random, policy: Mapping[str, Any], release_time_ms: int
) -> int | None:
    mode = policy["mode"]
    if mode == "none":
        return None
    if mode == "relative":
        offset = int(policy["offsetMs"])
    else:
        offset = rng.randint(int(policy["minOffsetMs"]), int(policy["maxOffsetMs"]))
    return release_time_ms + offset


def generate_task_stream(
    scene: WarehouseSceneSpec, generation: Mapping[str, Any]
) -> TaskStreamSpec:
    """Generate a reproducible task stream against an editable warehouse scene."""
    seed = int(generation["seed"])
    end_time_ms = int(generation["endTimeMs"])
    rng = random.Random(seed)
    arrivals = _arrival_times(
        rng,
        generation["arrival"],
        end_time_ms,
        int(generation["maxTasks"]),
    )
    if not arrivals:
        raise DomainError(
            "task_stream.arrival.empty",
            "arrival settings produce no task inside the simulation window",
        )

    od_pairs = _eligible_od_pairs(scene, generation["odPairs"])
    priorities = generation.get(
        "priorityDistribution", [{"priorityClass": 0, "weight": 1.0}]
    )
    service_policy = generation.get("serviceTimePolicy", {"mode": "workstation_defaults"})
    due_policy = generation.get("dueTimePolicy", {"mode": "none"})
    stations = {str(item["nodeId"]): item for item in scene.workstations}
    stream_id = str(generation["streamId"])
    tasks: list[dict[str, Any]] = []
    for index, release_time_ms in enumerate(arrivals, start=1):
        pair = _weighted_choice(rng, od_pairs)
        priority = _weighted_choice(rng, priorities)
        pickup_id = str(pair["pickupNodeId"])
        dropoff_id = str(pair["dropoffNodeId"])
        payload_type = str(pair["payloadType"])
        task: dict[str, Any] = {
            "taskId": f"{stream_id}-task-{index:06d}",
            "releaseTimeMs": release_time_ms,
            "pickupNodeId": pickup_id,
            "dropoffNodeId": dropoff_id,
            "requiredRobotGroup": str(pair["requiredRobotGroup"]),
            "payloadType": payload_type,
            "payloadId": f"{stream_id}-{payload_type}-{index:06d}",
            "pickupServiceMs": _service_time(
                service_policy, stations[pickup_id], "pickupServiceMs"
            ),
            "dropoffServiceMs": _service_time(
                service_policy, stations[dropoff_id], "dropoffServiceMs"
            ),
            "priorityClass": int(priority["priorityClass"]),
            "dueTimeMs": _due_time(rng, due_policy, release_time_ms),
        }
        tasks.append(task)

    return TaskStreamSpec(
        stream_id=stream_id,
        seed=seed,
        end_time_ms=end_time_ms,
        tasks=tuple(tasks),
        events=tuple(deepcopy(generation.get("events", []))),
    )


def task_stream_to_dict(stream: TaskStreamSpec) -> dict[str, Any]:
    return {
        "streamId": stream.stream_id,
        "seed": stream.seed,
        "endTimeMs": stream.end_time_ms,
        "tasks": [deepcopy(item) for item in stream.tasks],
        "events": [deepcopy(item) for item in stream.events],
    }
