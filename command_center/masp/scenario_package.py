from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
from jsonschema import Draft202012Validator
from shapely import affinity
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .domain import DomainError


ROBOT_GROUPS = frozenset({"fork", "jack"})
NODE_TYPES = frozenset({"LM", "AP", "PP", "CP"})
WAITABLE_NODE_TYPES = frozenset({"PP", "CP"})


@dataclass(frozen=True)
class PackageValidationIssue:
    severity: str
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True)
class PackageValidationReport:
    issues: tuple[PackageValidationIssue, ...]
    stats: Mapping[str, int]

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [item.to_dict() for item in self.issues],
            "stats": dict(self.stats),
        }

    def raise_for_errors(self) -> None:
        error = next((item for item in self.issues if item.severity == "error"), None)
        if error is not None:
            raise DomainError(error.code, f"{error.path}: {error.message}")


@dataclass(frozen=True)
class WarehouseSceneSpec:
    scene_id: str
    name: str
    bounds: Mapping[str, float]
    robot_profiles: Mapping[str, Mapping[str, Any]]
    nodes: tuple[Mapping[str, Any], ...]
    edges: tuple[Mapping[str, Any], ...]
    workstations: tuple[Mapping[str, Any], ...]
    vehicles: tuple[Mapping[str, Any], ...]
    recovery_nodes: tuple[Mapping[str, Any], ...]
    traffic_zones: tuple[Mapping[str, Any], ...]
    safety: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WarehouseSceneSpec:
        return cls(
            scene_id=str(value["sceneId"]),
            name=str(value["name"]),
            bounds=deepcopy(value["bounds"]),
            robot_profiles=deepcopy(value["robotProfiles"]),
            nodes=tuple(deepcopy(value["nodes"])),
            edges=tuple(deepcopy(value["edges"])),
            workstations=tuple(deepcopy(value["workstations"])),
            vehicles=tuple(deepcopy(value["vehicles"])),
            recovery_nodes=tuple(deepcopy(value.get("recoveryNodes", []))),
            traffic_zones=tuple(deepcopy(value.get("trafficZones", []))),
            safety=deepcopy(value["safety"]),
        )


@dataclass(frozen=True)
class TaskStreamSpec:
    stream_id: str
    seed: int
    end_time_ms: int
    tasks: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskStreamSpec:
        return cls(
            stream_id=str(value["streamId"]),
            seed=int(value["seed"]),
            end_time_ms=int(value["endTimeMs"]),
            tasks=tuple(deepcopy(value["tasks"])),
            events=tuple(deepcopy(value.get("events", []))),
        )


@dataclass(frozen=True)
class ScenarioPackage:
    package_id: str
    version: str
    status: str
    warehouse_scene: WarehouseSceneSpec
    task_stream: TaskStreamSpec
    metadata: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScenarioPackage:
        return cls(
            package_id=str(value["packageId"]),
            version=str(value["version"]),
            status=str(value["status"]),
            warehouse_scene=WarehouseSceneSpec.from_dict(value["warehouseScene"]),
            task_stream=TaskStreamSpec.from_dict(value["taskStream"]),
            metadata=deepcopy(value.get("metadata", {})),
        )

    def validate(self) -> PackageValidationReport:
        return validate_package(self)


@dataclass(frozen=True)
class CompiledScenarioPackage:
    documents: Mapping[str, Mapping[str, Any]]
    validation: PackageValidationReport

    def write_to(self, output_dir: Path) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for name, document in self.documents.items():
            path = output_dir / name
            path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            paths[name] = str(path)
        return paths


def load_scenario_package(
    path: Path,
    *,
    schema_path: Path | None = None,
) -> ScenarioPackage:
    document = json.loads(path.read_text(encoding="utf-8"))
    if schema_path is not None:
        validate_scenario_package_document(document, schema_path)
    return ScenarioPackage.from_dict(document)


def package_from_assets(
    *,
    package_id: str,
    version: str,
    map_document: Mapping[str, Any],
    profile_document: Mapping[str, Any],
    workstation_document: Mapping[str, Any],
    vehicle_document: Mapping[str, Any],
    traffic_document: Mapping[str, Any],
    scenario_document: Mapping[str, Any],
    conflict_document: Mapping[str, Any] | None = None,
    created_by: str = "migration-tool",
) -> ScenarioPackage:
    """Convert the fixed MASP runtime assets into the editable package contract."""
    metadata = map_document.get("metadata", {})
    bounds = metadata.get("bounds")
    if bounds is None:
        coordinates = [
            (float(node["x"]), float(node["y"])) for node in map_document["nodes"]
        ]
        bounds = {
            "minX": min(point[0] for point in coordinates),
            "maxX": max(point[0] for point in coordinates),
            "minY": min(point[1] for point in coordinates),
            "maxY": max(point[1] for point in coordinates),
        }
    nodes = []
    for source in map_document["nodes"]:
        nodes.append(
            {
                "id": source["id"],
                "type": source["type"],
                "x": source["x"],
                "y": source["y"],
                "allowedRobotGroups": list(source["allowedRobotGroups"]),
                "aliases": deepcopy(source.get("aliases", {})),
                "positionsByGroup": deepcopy(source.get("positions", {})),
                "headings": deepcopy(source.get("headings", {})),
                "propertiesByGroup": deepcopy(source.get("propertiesByGroup", {})),
                "waitAllowedByGroup": {
                    group: bool(value)
                    for group, value in source.get("allowWaitByGroup", {}).items()
                },
                "capacity": int(source.get("capacity", 1)),
            }
        )
    edges = []
    for source in map_document["edges"]:
        edges.append(
            {
                "id": source["id"],
                "name": source.get("name", source["id"]),
                "startNodeId": source["start"],
                "endNodeId": source["end"],
                "controlPoints": [
                    list(source[point]) for point in ("p0", "p1", "p2", "p3")
                ],
                "lengthM": source["length"],
                "motionDirection": source.get("motionDirection", 0),
                "moveStyle": source.get("moveStyle", 0),
                "maxSpeedMps": source.get("maxSpeed"),
                "loadedMaxSpeedMps": source.get("loadMaxSpeed"),
                "robotGroup": source["robotGroup"],
            }
        )
    sample_spacing = float(
        (conflict_document or {}).get("metadata", {}).get("sampleSpacing", 0.25)
    )
    safety_source = profile_document.get("simulationSafety", {})
    safety = {
        "footprintMarginM": float(safety_source.get("footprintMargin", 0.0)),
        "conflictSampleSpacingM": sample_spacing,
        "localizationErrorM": safety_source.get("localizationError"),
        "communicationLatencyMs": safety_source.get("communicationLatency"),
        "provisional": bool(safety_source.get("provisional", True)),
    }
    scene = WarehouseSceneSpec(
        scene_id=str(scenario_document.get("scenarioId", f"{package_id}-scene")),
        name=str(metadata.get("modelType", "MASP仓储场景")),
        bounds=deepcopy(bounds),
        robot_profiles=deepcopy(profile_document["robotGroups"]),
        nodes=tuple(nodes),
        edges=tuple(edges),
        workstations=tuple(deepcopy(workstation_document["workstations"])),
        vehicles=tuple(deepcopy(vehicle_document["vehicles"])),
        recovery_nodes=tuple(deepcopy(traffic_document.get("recoveryNodes", []))),
        traffic_zones=tuple(deepcopy(traffic_document.get("zones", []))),
        safety=safety,
    )
    stream = TaskStreamSpec(
        stream_id=f"{package_id}-stream",
        seed=int(scenario_document["seed"]),
        end_time_ms=int(scenario_document["endTimeMs"]),
        tasks=tuple(deepcopy(scenario_document["tasks"])),
        events=tuple(),
    )
    return ScenarioPackage(
        package_id=package_id,
        version=version,
        status="draft",
        warehouse_scene=scene,
        task_stream=stream,
        metadata={
            "createdBy": created_by,
            "source": "MASP runtime assets",
            "sourceScenarioId": scenario_document.get("scenarioId"),
        },
    )


def validate_scenario_package_document(
    document: Mapping[str, Any],
    schema_path: Path,
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
        "scenario.package.schema.invalid",
        f"scenario package {path}: {error.message}",
    )


def _duplicates(values: Iterable[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _issue(
    issues: list[PackageValidationIssue],
    code: str,
    path: str,
    message: str,
    severity: str = "error",
) -> None:
    issues.append(PackageValidationIssue(severity, code, path, message))


def _point(value: Sequence[Any]) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _edge_points(
    edge: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[float, float], ...]:
    control_points = edge.get("controlPoints")
    if control_points:
        return tuple(_point(item) for item in control_points)
    start = nodes[str(edge["startNodeId"])]
    end = nodes[str(edge["endNodeId"])]
    group = str(edge.get("robotGroup", ""))
    p0 = _point(start.get("positionsByGroup", {}).get(group, (start["x"], start["y"])))
    p3 = _point(end.get("positionsByGroup", {}).get(group, (end["x"], end["y"])))
    return (
        p0,
        (p0[0] * 2.0 / 3.0 + p3[0] / 3.0, p0[1] * 2.0 / 3.0 + p3[1] / 3.0),
        (p0[0] / 3.0 + p3[0] * 2.0 / 3.0, p0[1] / 3.0 + p3[1] * 2.0 / 3.0),
        p3,
    )


def _cubic_point(points: Sequence[Sequence[float]], t: float) -> tuple[float, float]:
    p0, p1, p2, p3 = points
    u = 1.0 - t
    return (
        u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
    )


def _cubic_heading(points: Sequence[Sequence[float]], t: float) -> float:
    p0, p1, p2, p3 = points
    u = 1.0 - t
    dx = (
        3 * u * u * (p1[0] - p0[0])
        + 6 * u * t * (p2[0] - p1[0])
        + 3 * t * t * (p3[0] - p2[0])
    )
    dy = (
        3 * u * u * (p1[1] - p0[1])
        + 6 * u * t * (p2[1] - p1[1])
        + 3 * t * t * (p3[1] - p2[1])
    )
    if abs(dx) + abs(dy) < 1e-12:
        dx, dy = p3[0] - p0[0], p3[1] - p0[1]
    return math.degrees(math.atan2(dy, dx))


def _curve_length(points: Sequence[Sequence[float]], samples: int = 32) -> float:
    positions = [_cubic_point(points, index / samples) for index in range(samples + 1)]
    return sum(
        math.dist(positions[index - 1], positions[index])
        for index in range(1, len(positions))
    )


def validate_package(package: ScenarioPackage) -> PackageValidationReport:
    scene = package.warehouse_scene
    stream = package.task_stream
    issues: list[PackageValidationIssue] = []
    nodes = {str(item["id"]): item for item in scene.nodes}
    edges = {str(item["id"]): item for item in scene.edges}
    groups = set(scene.robot_profiles)

    if package.status == "published" and not package.metadata.get("createdBy"):
        _issue(
            issues,
            "scenario.package.owner.missing",
            "$.metadata.createdBy",
            "published packages require an owner",
        )
    if not groups or not groups <= ROBOT_GROUPS:
        _issue(
            issues,
            "scenario.robot_group.invalid",
            "$.warehouseScene.robotProfiles",
            f"robot groups must be a non-empty subset of {sorted(ROBOT_GROUPS)!r}",
        )

    for duplicate in _duplicates(str(item["id"]) for item in scene.nodes):
        _issue(issues, "scenario.node.duplicate", "$.warehouseScene.nodes", f"duplicate node {duplicate!r}")
    for duplicate in _duplicates(str(item["id"]) for item in scene.edges):
        _issue(issues, "scenario.edge.duplicate", "$.warehouseScene.edges", f"duplicate edge {duplicate!r}")

    bounds = scene.bounds
    for index, node in enumerate(scene.nodes):
        path = f"$.warehouseScene.nodes[{index}]"
        node_id = str(node["id"])
        node_groups = set(node["allowedRobotGroups"])
        if node["type"] not in NODE_TYPES:
            _issue(issues, "scenario.node.type", f"{path}.type", f"node {node_id!r} has an unsupported type")
        if not node_groups or not node_groups <= groups:
            _issue(issues, "scenario.node.groups", f"{path}.allowedRobotGroups", f"node {node_id!r} has invalid groups")
        x, y = float(node["x"]), float(node["y"])
        if not (float(bounds["minX"]) <= x <= float(bounds["maxX"])) or not (
            float(bounds["minY"]) <= y <= float(bounds["maxY"])
        ):
            _issue(issues, "scenario.node.out_of_bounds", path, f"node {node_id!r} is outside scene bounds")
        wait_flags = node.get("waitAllowedByGroup", {})
        if set(wait_flags) != node_groups:
            _issue(issues, "scenario.node.wait_policy", f"{path}.waitAllowedByGroup", f"node {node_id!r} requires one wait flag per group")

    for index, edge in enumerate(scene.edges):
        path = f"$.warehouseScene.edges[{index}]"
        edge_id = str(edge["id"])
        start_id = str(edge["startNodeId"])
        end_id = str(edge["endNodeId"])
        group = str(edge["robotGroup"])
        if start_id == end_id:
            _issue(issues, "scenario.edge.self_loop", path, f"edge {edge_id!r} is a self loop")
        if start_id not in nodes or end_id not in nodes:
            _issue(issues, "scenario.edge.endpoint", path, f"edge {edge_id!r} references an unknown endpoint")
            continue
        if group not in groups:
            _issue(issues, "scenario.edge.group", f"{path}.robotGroup", f"edge {edge_id!r} has an unknown group")
        elif group not in nodes[start_id]["allowedRobotGroups"] or group not in nodes[end_id]["allowedRobotGroups"]:
            _issue(issues, "scenario.edge.endpoint_group", path, f"edge {edge_id!r} group is not allowed by both endpoints")
        points = _edge_points(edge, nodes)
        expected_start = nodes[start_id].get("positionsByGroup", {}).get(group, (nodes[start_id]["x"], nodes[start_id]["y"]))
        expected_end = nodes[end_id].get("positionsByGroup", {}).get(group, (nodes[end_id]["x"], nodes[end_id]["y"]))
        if math.dist(points[0], _point(expected_start)) > 1e-6:
            _issue(issues, "scenario.edge.geometry_start", f"{path}.controlPoints[0]", f"edge {edge_id!r} does not begin at its start node")
        if math.dist(points[-1], _point(expected_end)) > 1e-6:
            _issue(issues, "scenario.edge.geometry_end", f"{path}.controlPoints[3]", f"edge {edge_id!r} does not end at its end node")
        if _curve_length(points) <= 1e-6:
            _issue(issues, "scenario.edge.length", path, f"edge {edge_id!r} has zero length")

    workstations_by_node: dict[str, Mapping[str, Any]] = {}
    for duplicate in _duplicates(str(item["id"]) for item in scene.workstations):
        _issue(issues, "scenario.workstation.duplicate", "$.warehouseScene.workstations", f"duplicate workstation {duplicate!r}")
    for index, station in enumerate(scene.workstations):
        path = f"$.warehouseScene.workstations[{index}]"
        node_id = str(station["nodeId"])
        if node_id in workstations_by_node:
            _issue(issues, "scenario.workstation.node_duplicate", f"{path}.nodeId", f"node {node_id!r} has multiple workstations")
        workstations_by_node[node_id] = station
        node = nodes.get(node_id)
        if node is None or node.get("type") != "AP":
            _issue(issues, "scenario.workstation.node", f"{path}.nodeId", f"workstation node {node_id!r} is not an AP")
        elif not set(station["allowedRobotGroups"]) <= set(node["allowedRobotGroups"]):
            _issue(issues, "scenario.workstation.groups", f"{path}.allowedRobotGroups", f"workstation {station['id']!r} has incompatible groups")
    ap_ids = {node_id for node_id, node in nodes.items() if node["type"] == "AP"}
    if set(workstations_by_node) != ap_ids:
        missing = sorted(ap_ids - set(workstations_by_node))
        extra = sorted(set(workstations_by_node) - ap_ids)
        _issue(issues, "scenario.workstation.coverage", "$.warehouseScene.workstations", f"workstations must cover every AP; missing={missing!r}, extra={extra!r}")

    vehicle_groups: Counter[str] = Counter()
    vehicle_start_nodes: set[str] = set()
    for duplicate in _duplicates(str(item["vehicleId"]) for item in scene.vehicles):
        _issue(issues, "scenario.vehicle.duplicate", "$.warehouseScene.vehicles", f"duplicate vehicle {duplicate!r}")
    for index, vehicle in enumerate(scene.vehicles):
        path = f"$.warehouseScene.vehicles[{index}]"
        node_id = str(vehicle["initialNodeId"])
        group = str(vehicle["robotGroup"])
        vehicle_groups[group] += 1
        node = nodes.get(node_id)
        if node is None or group not in node.get("allowedRobotGroups", []):
            _issue(issues, "scenario.vehicle.start", f"{path}.initialNodeId", f"vehicle {vehicle['vehicleId']!r} has an incompatible start node")
        elif not bool(node.get("waitAllowedByGroup", {}).get(group)):
            _issue(issues, "scenario.vehicle.start_wait", f"{path}.initialNodeId", f"vehicle {vehicle['vehicleId']!r} must start at a waitable node")
        if node_id in vehicle_start_nodes:
            _issue(issues, "scenario.vehicle.start_duplicate", f"{path}.initialNodeId", f"multiple vehicles start at {node_id!r}")
        vehicle_start_nodes.add(node_id)

    graph_by_group = {group: nx.DiGraph() for group in groups}
    for node_id, node in nodes.items():
        for group in set(node["allowedRobotGroups"]) & groups:
            graph_by_group[group].add_node(node_id)
    for edge in scene.edges:
        group = str(edge["robotGroup"])
        if group in graph_by_group and edge["startNodeId"] in nodes and edge["endNodeId"] in nodes:
            graph_by_group[group].add_edge(str(edge["startNodeId"]), str(edge["endNodeId"]))

    recovery_by_group: dict[str, set[str]] = defaultdict(set)
    recovery_ids: set[str] = set()
    for index, recovery in enumerate(scene.recovery_nodes):
        path = f"$.warehouseScene.recoveryNodes[{index}]"
        node_id = str(recovery["nodeId"])
        if node_id in recovery_ids:
            _issue(issues, "scenario.recovery.duplicate", f"{path}.nodeId", f"duplicate recovery node {node_id!r}")
        recovery_ids.add(node_id)
        node = nodes.get(node_id)
        if node is None:
            _issue(issues, "scenario.recovery.node", f"{path}.nodeId", f"unknown recovery node {node_id!r}")
            continue
        for group in recovery["allowedRobotGroups"]:
            if group not in node["allowedRobotGroups"] or not node["waitAllowedByGroup"].get(group, False):
                _issue(issues, "scenario.recovery.group", path, f"recovery node {node_id!r} is not waitable for {group!r}")
            else:
                recovery_by_group[group].add(node_id)

    claimed_zone_nodes: set[str] = set()
    for index, zone in enumerate(scene.traffic_zones):
        path = f"$.warehouseScene.trafficZones[{index}]"
        members = set(zone["memberNodeIds"])
        if not members or not members <= set(nodes):
            _issue(issues, "scenario.zone.nodes", f"{path}.memberNodeIds", f"zone {zone['id']!r} has missing or unknown member nodes")
        overlap = members & claimed_zone_nodes
        if overlap:
            _issue(issues, "scenario.zone.overlap", f"{path}.memberNodeIds", f"zone {zone['id']!r} overlaps nodes {sorted(overlap)!r}")
        claimed_zone_nodes.update(members)
        if not set(zone["recoveryNodeIds"]) <= recovery_ids:
            _issue(issues, "scenario.zone.recovery", f"{path}.recoveryNodeIds", f"zone {zone['id']!r} references undeclared recovery nodes")

    for duplicate in _duplicates(str(item["taskId"]) for item in stream.tasks):
        _issue(issues, "scenario.task.duplicate", "$.taskStream.tasks", f"duplicate task {duplicate!r}")
    for index, task in enumerate(stream.tasks):
        path = f"$.taskStream.tasks[{index}]"
        group = str(task["requiredRobotGroup"])
        pickup = str(task["pickupNodeId"])
        dropoff = str(task["dropoffNodeId"])
        release = int(task["releaseTimeMs"])
        due = task.get("dueTimeMs")
        if release >= stream.end_time_ms:
            _issue(issues, "scenario.task.release", f"{path}.releaseTimeMs", f"task {task['taskId']!r} releases outside the simulation window")
        if due is not None and int(due) < release:
            _issue(issues, "scenario.task.due_time", f"{path}.dueTimeMs", f"task {task['taskId']!r} is due before release")
        if group not in vehicle_groups:
            _issue(issues, "scenario.task.fleet", f"{path}.requiredRobotGroup", f"task {task['taskId']!r} has no compatible vehicle")
        for field, node_id in (("pickupNodeId", pickup), ("dropoffNodeId", dropoff)):
            node = nodes.get(node_id)
            station = workstations_by_node.get(node_id)
            if node is None or node.get("type") != "AP" or station is None or group not in station.get("allowedRobotGroups", []):
                _issue(issues, "scenario.task.workstation", f"{path}.{field}", f"task {task['taskId']!r} references an incompatible workstation")
        graph = graph_by_group.get(group)
        if graph is not None and pickup in graph and dropoff in graph:
            starts = [str(item["initialNodeId"]) for item in scene.vehicles if item["robotGroup"] == group]
            if starts and not any(nx.has_path(graph, start, pickup) for start in starts if start in graph):
                _issue(issues, "scenario.task.pickup_unreachable", f"{path}.pickupNodeId", f"no {group!r} vehicle can reach pickup {pickup!r}")
            if not nx.has_path(graph, pickup, dropoff):
                _issue(issues, "scenario.task.dropoff_unreachable", path, f"task {task['taskId']!r} has no directed pickup-to-dropoff route")
            recoveries = recovery_by_group.get(group, set())
            if not recoveries or not any(nx.has_path(graph, dropoff, node_id) for node_id in recoveries if node_id in graph):
                _issue(issues, "scenario.task.recovery_unreachable", f"{path}.dropoffNodeId", f"task {task['taskId']!r} has no reachable recovery node")

    for index, event in enumerate(stream.events):
        if int(event["atMs"]) > stream.end_time_ms:
            _issue(issues, "scenario.event.time", f"$.taskStream.events[{index}].atMs", f"event {event['eventId']!r} occurs after the simulation window")

    if float(scene.safety["footprintMarginM"]) == 0:
        _issue(
            issues,
            "scenario.safety.provisional",
            "$.warehouseScene.safety.footprintMarginM",
            "zero footprint margin is suitable for simulation only",
            severity="warning",
        )

    stats = {
        "nodeCount": len(scene.nodes),
        "edgeCount": len(scene.edges),
        "workstationCount": len(scene.workstations),
        "vehicleCount": len(scene.vehicles),
        "taskCount": len(stream.tasks),
        "eventCount": len(stream.events),
        "trafficZoneCount": len(scene.traffic_zones),
        "recoveryNodeCount": len(scene.recovery_nodes),
    }
    return PackageValidationReport(tuple(issues), stats)


def _compiled_node(node: Mapping[str, Any], max_wait_ms: int) -> dict[str, Any]:
    groups = list(node["allowedRobotGroups"])
    wait_flags = node["waitAllowedByGroup"]
    return {
        "id": node["id"],
        "type": node["type"],
        "x": float(node["x"]),
        "y": float(node["y"]),
        "allowedRobotGroups": groups,
        "aliases": deepcopy(node.get("aliases", {})),
        "positions": {
            group: list(node.get("positionsByGroup", {}).get(group, (node["x"], node["y"])))
            for group in groups
        },
        "headings": {group: float(node.get("headings", {}).get(group, 0.0)) for group in groups},
        "propertiesByGroup": {group: deepcopy(node.get("propertiesByGroup", {}).get(group, {})) for group in groups},
        "allowWaitByGroup": {group: bool(wait_flags[group]) for group in groups},
        "waitPolicyByGroup": {
            group: {
                "allowed": bool(wait_flags[group]),
                "maxWaitMs": max_wait_ms if wait_flags[group] else 0,
                "source": "scenario-package",
            }
            for group in groups
        },
        "capacity": int(node.get("capacity", 1)),
        "resourceIds": [f"node:{node['id']}"],
    }


def _compiled_edge(edge: Mapping[str, Any], nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    points = _edge_points(edge, nodes)
    return {
        "id": edge["id"],
        "name": edge.get("name", edge["id"]),
        "pathClass": "CubicBezier",
        "start": edge["startNodeId"],
        "end": edge["endNodeId"],
        "p0": list(points[0]),
        "p1": list(points[1]),
        "p2": list(points[2]),
        "p3": list(points[3]),
        "length": round(float(edge.get("lengthM", _curve_length(points))), 6),
        "motionDirection": int(edge.get("motionDirection", 0)),
        "moveStyle": int(edge.get("moveStyle", 0)),
        "maxSpeed": edge.get("maxSpeedMps"),
        "loadMaxSpeed": edge.get("loadedMaxSpeedMps"),
        "robotGroup": edge["robotGroup"],
        "localStart": edge["startNodeId"],
        "localEnd": edge["endNodeId"],
        "allowedRobotGroups": [edge["robotGroup"]],
    }


def _swept_polygon(
    edge: Mapping[str, Any],
    *,
    length: float,
    width: float,
    margin: float,
    sample_spacing: float,
) -> Any:
    footprint = box(
        -(length / 2.0 + margin),
        -(width / 2.0 + margin),
        length / 2.0 + margin,
        width / 2.0 + margin,
    )
    points = (edge["p0"], edge["p1"], edge["p2"], edge["p3"])
    samples = max(3, math.ceil(float(edge["length"]) / sample_spacing) + 1)
    placements = []
    for index in range(samples):
        t = index / (samples - 1)
        x, y = _cubic_point(points, t)
        rotated = affinity.rotate(footprint, _cubic_heading(points, t), origin=(0, 0), use_radians=False)
        placements.append(affinity.translate(rotated, xoff=x, yoff=y))
    return unary_union(placements)


def _compile_conflicts(
    model: Mapping[str, Any],
    profiles: Mapping[str, Any],
    *,
    margin: float,
    sample_spacing: float,
) -> dict[str, Any]:
    edges = model["edges"]
    polygons = []
    for edge in edges:
        dimensions = profiles["robotGroups"][edge["robotGroup"]]["dimensions"]
        polygons.append(
            _swept_polygon(
                edge,
                length=float(dimensions["length"]),
                width=float(dimensions["width"]),
                margin=margin,
                sample_spacing=sample_spacing,
            )
        )
    tree = STRtree(polygons)
    resources_by_edge: dict[str, list[str]] = defaultdict(list)
    pair_rows: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    for left_index, left_polygon in enumerate(polygons):
        for candidate in tree.query(left_polygon, predicate="intersects"):
            right_index = int(candidate)
            if right_index <= left_index:
                continue
            intersection = left_polygon.intersection(polygons[right_index])
            if intersection.is_empty:
                continue
            left, right = edges[left_index], edges[right_index]
            resource_id = f"edge-conflict:{len(pair_rows)}"
            resources_by_edge[left["id"]].append(resource_id)
            resources_by_edge[right["id"]].append(resource_id)
            type_counts["-".join(sorted((left["robotGroup"], right["robotGroup"])))] += 1
            pair_rows.append(
                {
                    "resourceId": resource_id,
                    "edgeA": left["id"],
                    "edgeB": right["id"],
                    "groupA": left["robotGroup"],
                    "groupB": right["robotGroup"],
                    "sharedCanonicalEndpoint": bool({left["start"], left["end"]} & {right["start"], right["end"]}),
                    "intersectionArea": round(float(intersection.area), 6),
                }
            )
    return {
        "metadata": {
            "map": "map-model.json",
            "profiles": "robot-profiles.json",
            "sampleSpacing": sample_spacing,
            "footprintMargin": margin,
            "baseGeometryOnly": margin == 0.0,
        },
        "stats": {
            "edgeCount": len(edges),
            "conflictPairCount": len(pair_rows),
            "conflictTypeCounts": dict(sorted(type_counts.items())),
            "nodeResourceCount": len(model["nodes"]),
        },
        "nodeResources": [
            {
                "resourceId": f"node:{node['id']}",
                "nodeId": node["id"],
                "allowedRobotGroups": node["allowedRobotGroups"],
            }
            for node in model["nodes"]
        ],
        "edgeResources": [
            {
                "edgeId": edge["id"],
                "robotGroup": edge["robotGroup"],
                "ownResource": f"edge:{edge['id']}",
                "conflictResources": sorted(resources_by_edge[edge["id"]]),
            }
            for edge in edges
        ],
        "conflictPairs": pair_rows,
    }


def _compile_traffic_zones(scene: WarehouseSceneSpec, edges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    zones = []
    for source in scene.traffic_zones:
        members = set(source["memberNodeIds"])
        zones.append(
            {
                "id": source["id"],
                "memberNodeIds": sorted(members),
                "memberEdgeIds": sorted(edge["id"] for edge in edges if edge["start"] in members and edge["end"] in members),
                "entryEdgeIds": sorted(edge["id"] for edge in edges if edge["start"] not in members and edge["end"] in members),
                "exitEdgeIds": sorted(edge["id"] for edge in edges if edge["start"] in members and edge["end"] not in members),
                "capacity": 1,
                "passingAllowed": False,
                "directionalMode": "single_direction_at_a_time",
                "recoveryNodeIds": list(source["recoveryNodeIds"]),
            }
        )
    return {
        "schemaVersion": 1,
        "recoveryNodes": [deepcopy(item) for item in scene.recovery_nodes],
        "zones": zones,
    }


def _digest(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compile_scenario_package(
    package: ScenarioPackage,
    *,
    scheduler_template: Mapping[str, Any],
) -> CompiledScenarioPackage:
    validation = package.validate()
    validation.raise_for_errors()
    scene = package.warehouse_scene
    stream = package.task_stream
    source_nodes = {str(item["id"]): item for item in scene.nodes}
    scheduler = deepcopy(scheduler_template)
    scheduler["mode"] = "simulation"
    scheduler["fleet"]["fixedDuringRun"] = True
    scheduler["fleet"]["counts"] = dict(sorted(Counter(item["robotGroup"] for item in scene.vehicles).items()))
    scheduler["fleet"]["initialVehiclesFile"] = "initial-vehicles.json"
    scheduler["safety"]["footprintMarginM"] = float(scene.safety["footprintMarginM"])
    max_wait_ms = int(scheduler["traffic"]["wait"]["maxPlannedWaitMs"])
    nodes = [_compiled_node(item, max_wait_ms) for item in scene.nodes]
    edges = [_compiled_edge(item, source_nodes) for item in scene.edges]
    model = {
        "metadata": {
            "modelType": "scenario-package-map",
            "sceneId": scene.scene_id,
            "packageId": package.package_id,
            "packageVersion": package.version,
            "bounds": dict(scene.bounds),
        },
        "stats": {
            "canonicalNodeCount": len(nodes),
            "sharedNodeCount": sum(1 for node in nodes if len(node["allowedRobotGroups"]) > 1),
            "forkOnlyNodeCount": sum(node["allowedRobotGroups"] == ["fork"] for node in nodes),
            "jackOnlyNodeCount": sum(node["allowedRobotGroups"] == ["jack"] for node in nodes),
            "edgeCount": len(edges),
            "forkEdgeCount": sum(edge["robotGroup"] == "fork" for edge in edges),
            "jackEdgeCount": sum(edge["robotGroup"] == "jack" for edge in edges),
            "sharedPathMatchCount": 0,
        },
        "nodeMatches": [],
        "nodes": nodes,
        "edges": edges,
        "sharedPathMatches": [],
    }
    profiles = {
        "schemaVersion": 1,
        "units": {
            "linearSpeed": "m/s",
            "linearAcceleration": "m/s^2",
            "angularSpeed": "deg/s",
            "angularAcceleration": "deg/s^2",
            "dimensions": "m",
        },
        "robotGroups": deepcopy(scene.robot_profiles),
        "simulationSafety": {
            "footprintMargin": float(scene.safety["footprintMarginM"]),
            "localizationError": scene.safety.get("localizationErrorM"),
            "communicationLatency": scene.safety.get("communicationLatencyMs"),
            "provisional": bool(scene.safety.get("provisional", True)),
            "note": "Compiled from ScenarioPackage safety settings.",
        },
    }
    conflicts = _compile_conflicts(
        model,
        profiles,
        margin=float(scene.safety["footprintMarginM"]),
        sample_spacing=float(scene.safety["conflictSampleSpacingM"]),
    )
    workstations = {
        "schemaVersion": 1,
        "map": "map-model.json",
        "workstations": [
            {
                "id": item["id"],
                "nodeId": item["nodeId"],
                "capabilities": list(item["capabilities"]),
                "allowedRobotGroups": list(item["allowedRobotGroups"]),
                "capacity": int(item.get("capacity", 1)),
                "pickupServiceMs": int(item["pickupServiceMs"]),
                "dropoffServiceMs": int(item["dropoffServiceMs"]),
                "blocksTransitDuringService": bool(item.get("blocksTransitDuringService", True)),
                "propertiesByGroup": deepcopy(source_nodes[item["nodeId"]].get("propertiesByGroup", {})),
            }
            for item in scene.workstations
        ],
    }
    vehicles = {
        "schemaVersion": 1,
        "fixedDuringRun": True,
        "vehicles": [deepcopy(item) for item in scene.vehicles],
    }
    traffic_zones = _compile_traffic_zones(scene, edges)
    scenario_id = f"{package.package_id}@{package.version}"
    dispatch_scenario = {
        "schemaVersion": 1,
        "scenarioId": scenario_id,
        "seed": stream.seed,
        "endTimeMs": stream.end_time_ms,
        "vehicles": [deepcopy(item) for item in scene.vehicles],
        "tasks": [deepcopy(item) for item in stream.tasks],
    }
    task_stream = {
        "schemaVersion": 1,
        "streamId": stream.stream_id,
        "seed": stream.seed,
        "endTimeMs": stream.end_time_ms,
        "tasks": [deepcopy(item) for item in stream.tasks],
        "events": [deepcopy(item) for item in stream.events],
    }
    artifacts: dict[str, Mapping[str, Any]] = {
        "map-model.json": model,
        "conflict-resources.json": conflicts,
        "workstations.json": workstations,
        "robot-profiles.json": profiles,
        "scheduler.json": scheduler,
        "initial-vehicles.json": vehicles,
        "traffic-zones.json": traffic_zones,
        "dispatch-scenario.json": dispatch_scenario,
        "task-stream.json": task_stream,
        "validation-report.json": validation.to_dict(),
    }
    manifest = {
        "schemaVersion": 1,
        "packageId": package.package_id,
        "packageVersion": package.version,
        "status": package.status,
        "sceneId": scene.scene_id,
        "streamId": stream.stream_id,
        "scenarioId": scenario_id,
        "compiler": "command_center.masp.scenario_package/v1",
        "artifacts": [
            {"file": name, "sha256": _digest(document)}
            for name, document in artifacts.items()
        ],
        "stats": validation.to_dict()["stats"],
    }
    return CompiledScenarioPackage({**artifacts, "manifest.json": manifest}, validation)
