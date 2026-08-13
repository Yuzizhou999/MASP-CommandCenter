from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

from .contracts import (
    ComparisonResult,
    DispatchIntent,
    IncidentRecord,
    IntentType,
    IntentValidation,
    RiskLevel,
    SimulationRequest,
    SimulationSummary,
    ValidationIssue,
    WhatIfMode,
    new_id,
)
from .settings import Settings


class EngineVersionError(RuntimeError):
    pass


class MaspAdapter:
    """The only boundary allowed to import and call the MASP engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.engine_root
        self._simulation_lock = Lock()
        self._modules: dict[str, Any] | None = None
        self.settings.runs_dir.mkdir(parents=True, exist_ok=True)
        self._assert_layout()

    def _assert_layout(self) -> None:
        required = (
            self.root / "masp" / "online.py",
            self.root / "generated" / "xiate-unified-map-model.json",
            self.root / "scenarios" / "interactive-multi-fleet.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise EngineVersionError(f"MASP engine is incomplete: {missing}")

    @staticmethod
    def _git(root: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def engine_status(self) -> dict[str, Any]:
        current = self._git(self.root, "rev-parse", "HEAD")
        dirty_rows = self._git(self.root, "status", "--porcelain=v1").splitlines()
        matches = current == self.settings.engine_commit
        dirty = bool(dirty_rows)
        allowed = matches and (
            not dirty
            or not self.settings.is_production
            and self.settings.allow_dirty_development
        )
        return {
            "root": str(self.root),
            "expectedCommit": self.settings.engine_commit,
            "currentCommit": current,
            "commitMatches": matches,
            "dirty": dirty,
            "dirtyFileCount": len(dirty_rows),
            "allowed": allowed,
            "environment": self.settings.app_env,
            "warning": (
                "开发模式正在使用包含未提交修改的MASP工作区，比赛发布前必须固定干净版本。"
                if dirty and allowed
                else None
            ),
        }

    def _require_engine(self) -> None:
        status = self.engine_status()
        if not status["allowed"]:
            raise EngineVersionError(
                "MASP engine does not match engine.lock.json or is dirty in production"
            )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _assets(self) -> dict[str, dict[str, Any]]:
        return {
            "model": self._read_json(
                self.root / "generated" / "xiate-unified-map-model.json"
            ),
            "conflicts": self._read_json(
                self.root / "generated" / "xiate-conflict-resources.json"
            ),
            "workstations": self._read_json(
                self.root / "generated" / "xiate-workstations.json"
            ),
            "profiles": self._read_json(self.root / "config" / "robot-profiles.json"),
            "scheduler": self._read_json(self.root / "config" / "scheduler.json"),
            "zones": self._read_json(self.root / "config" / "traffic-zones.json"),
        }

    def _engine_modules(self) -> dict[str, Any]:
        if self._modules is not None:
            return self._modules
        root_text = str(self.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from masp.domain import TransportTask, Vehicle
        from masp.online import OnlineDispatchRuntime
        from masp.scenario_package import (
            ScenarioPackage,
            compile_scenario_package,
            package_from_assets,
            validate_scenario_package_document,
        )
        from masp.task_stream import generate_task_stream
        from masp.task_stream import validate_task_stream_generation_document
        from masp.topology import MapTopology

        self._modules = {
            "TransportTask": TransportTask,
            "Vehicle": Vehicle,
            "OnlineDispatchRuntime": OnlineDispatchRuntime,
            "MapTopology": MapTopology,
            "ScenarioPackage": ScenarioPackage,
            "compile_scenario_package": compile_scenario_package,
            "package_from_assets": package_from_assets,
            "validate_scenario_package_document": validate_scenario_package_document,
            "generate_task_stream": generate_task_stream,
            "validate_task_stream_generation_document": validate_task_stream_generation_document,
        }
        return self._modules

    def validate_scenario_package(self, document: dict[str, Any]) -> dict[str, Any]:
        """Validate an editable package without making it a runtime scenario."""
        modules = self._engine_modules()
        modules["validate_scenario_package_document"](
            document,
            self.root / "schemas" / "scenario-package.schema.json",
        )
        package = modules["ScenarioPackage"].from_dict(document)
        return package.validate().to_dict()

    def compile_scenario_package(
        self, document: dict[str, Any], output_dir: Path
    ) -> dict[str, Any]:
        """Compile a validated package into an isolated, simulation-only asset set."""
        self._require_engine()
        modules = self._engine_modules()
        modules["validate_scenario_package_document"](
            document,
            self.root / "schemas" / "scenario-package.schema.json",
        )
        package = modules["ScenarioPackage"].from_dict(document)
        compiled = modules["compile_scenario_package"](
            package,
            scheduler_template=self._read_json(self.root / "config" / "scheduler.json"),
        )
        return {
            "paths": compiled.write_to(output_dir),
            "validation": compiled.validation.to_dict(),
            "manifest": compiled.documents["manifest.json"],
        }

    def scenario_package_from_runtime(
        self, scenario_id: str, *, package_id: str, version: str, created_by: str
    ) -> dict[str, Any]:
        """Create an editable package from a fixed MASP scenario and runtime assets."""
        self._require_engine()
        modules = self._engine_modules()
        assets = self._assets()
        runtime_scenario = self.load_scenario(scenario_id)
        package = modules["package_from_assets"](
            package_id=package_id,
            version=version,
            map_document=assets["model"],
            profile_document=assets["profiles"],
            workstation_document=assets["workstations"],
            vehicle_document={"vehicles": runtime_scenario["vehicles"]},
            traffic_document=assets["zones"],
            scenario_document=runtime_scenario,
            conflict_document=assets["conflicts"],
            created_by=created_by,
        )
        document = {
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
        self.validate_scenario_package(document)
        return document

    def generate_scenario_tasks(
        self, document: dict[str, Any], generation: dict[str, Any]
    ) -> dict[str, Any]:
        modules = self._engine_modules()
        modules["validate_task_stream_generation_document"](
            generation,
            self.root / "schemas" / "task-stream-generation.schema.json",
        )
        modules["validate_scenario_package_document"](
            document,
            self.root / "schemas" / "scenario-package.schema.json",
        )
        package = modules["ScenarioPackage"].from_dict(document)
        stream = modules["generate_task_stream"](package.warehouse_scene, generation)
        return {
            "streamId": stream.stream_id,
            "seed": stream.seed,
            "endTimeMs": stream.end_time_ms,
            "tasks": [dict(item) for item in stream.tasks],
            "events": [dict(item) for item in stream.events],
        }

    def scenarios(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted((self.root / "scenarios").glob("*.json")):
            raw = self._read_json(path)
            if "vehicles" not in raw or "tasks" not in raw:
                continue
            rows.append(
                {
                    "scenarioId": raw["scenarioId"],
                    "file": path.name,
                    "vehicleCount": len(raw["vehicles"]),
                    "taskCount": len(raw["tasks"]),
                    "endTimeMs": int(raw["endTimeMs"]),
                }
            )
        return rows

    def _scenario_path(self, scenario_id: str) -> Path:
        for row in self.scenarios():
            if row["scenarioId"] == scenario_id:
                return self.root / "scenarios" / row["file"]
        raise KeyError(f"unknown scenario {scenario_id!r}")

    def load_scenario(self, scenario_id: str) -> dict[str, Any]:
        return self._read_json(self._scenario_path(scenario_id))

    def world_revision(self, scenario_id: str) -> int:
        scenario_bytes = self._scenario_path(scenario_id).read_bytes()
        value = hashlib.sha256(
            scenario_bytes + self.settings.engine_commit.encode("ascii")
        ).hexdigest()
        return int(value[:8], 16)

    def world_snapshot(self, scenario_id: str) -> dict[str, Any]:
        scenario = self.load_scenario(scenario_id)
        assets = self._assets()
        groups: dict[str, int] = {}
        for vehicle in scenario["vehicles"]:
            groups[vehicle["robotGroup"]] = groups.get(vehicle["robotGroup"], 0) + 1
        tasks_by_group: dict[str, int] = {}
        for task in scenario["tasks"]:
            group = task["requiredRobotGroup"]
            tasks_by_group[group] = tasks_by_group.get(group, 0) + 1
        return {
            "scenarioId": scenario_id,
            "worldRevision": self.world_revision(scenario_id),
            "mode": "simulation",
            "endTimeMs": int(scenario["endTimeMs"]),
            "counts": {
                "vehicles": len(scenario["vehicles"]),
                "tasks": len(scenario["tasks"]),
                "nodes": len(assets["model"]["nodes"]),
                "edges": len(assets["model"]["edges"]),
                "conflictPairs": len(assets["conflicts"]["conflictPairs"]),
                "workstations": len(assets["workstations"]["workstations"]),
            },
            "groups": groups,
            "taskGroups": tasks_by_group,
            "vehicles": [
                {
                    "vehicleId": row["vehicleId"],
                    "robotGroup": row["robotGroup"],
                    "currentNodeId": row["initialNodeId"],
                    "state": "IDLE",
                    "loadState": row["initialLoadState"],
                }
                for row in scenario["vehicles"]
            ],
            "tasks": [
                {
                    "taskId": row["taskId"],
                    "pickupNodeId": row["pickupNodeId"],
                    "dropoffNodeId": row["dropoffNodeId"],
                    "requiredRobotGroup": row["requiredRobotGroup"],
                    "priorityClass": int(row.get("priorityClass", 0)),
                    "releaseTimeMs": int(row["releaseTimeMs"]),
                    "dueTimeMs": row.get("dueTimeMs"),
                    "state": "QUEUED",
                }
                for row in scenario["tasks"]
            ],
            "zones": assets["zones"].get("zones", []),
            "engine": self.engine_status(),
        }

    def map_model(self) -> dict[str, Any]:
        scene_path = self.root / "generated" / "xiate-unified-scene-model.json"
        scene = self._read_json(scene_path)
        return {
            "bounds": scene.get("metadata", {}).get("bounds", {}),
            "stats": scene.get("stats", {}),
            "nodes": [
                {
                    "id": row["id"],
                    "type": row.get("type", "LM"),
                    "x": row["x"],
                    "y": row["y"],
                    "groups": row.get("groups", row.get("allowedRobotGroups", [])),
                }
                for row in scene.get("nodes", [])
            ],
            "edges": [
                {
                    "id": row["id"],
                    "group": row.get("group", "shared"),
                    "start": row["start"],
                    "end": row["end"],
                    "p0": row["p0"],
                    "p1": row["p1"],
                    "p2": row["p2"],
                    "p3": row["p3"],
                    "shared": bool(row.get("sharedMatch")),
                }
                for row in scene.get("edges", [])
            ],
            "sharedOverlays": scene.get("sharedOverlays", []),
        }

    def validate_intent(
        self, intent: DispatchIntent, scenario_id: str
    ) -> IntentValidation:
        issues: list[ValidationIssue] = []
        revision = self.world_revision(scenario_id)
        if intent.based_on_world_revision not in {0, revision}:
            issues.append(
                ValidationIssue(
                    code="intent.world_revision.stale",
                    message="调度意图基于过期的世界状态，请重新获取状态后再试。",
                    severity="error",
                )
            )
        elif intent.based_on_world_revision == 0:
            issues.append(
                ValidationIssue(
                    code="intent.world_revision.unspecified",
                    message="未指定世界版本，执行前必须重新绑定当前版本。",
                    severity="warning",
                )
            )

        risk = RiskLevel.R0_READ_ONLY
        policy = "intent.read-only"
        approval_required = False
        assets = self._assets()
        modules = self._engine_modules()
        if intent.intent_type is IntentType.CREATE_TASK:
            risk = RiskLevel.R1_LOW
            policy = "task.single.simulation"
            if intent.task is not None:
                try:
                    defaults = assets["scheduler"]["serviceDefaults"]
                    task = modules["TransportTask"].from_dict(
                        intent.task.model_dump(by_alias=True),
                        int(defaults["pickupServiceMs"]),
                        int(defaults["dropoffServiceMs"]),
                    )
                    topology = modules["MapTopology"](
                        assets["model"],
                        assets["conflicts"],
                        assets["workstations"],
                        assets["zones"],
                    )
                    topology.validate_task(task)
                    existing_ids = {
                        row["taskId"] for row in self.load_scenario(scenario_id)["tasks"]
                    }
                    if task.task_id in existing_ids:
                        raise ValueError(f"任务ID {task.task_id} 已存在")
                except (ValueError, KeyError) as error:
                    issues.append(
                        ValidationIssue(
                            code="intent.task.invalid",
                            message=str(error),
                            severity="error",
                        )
                    )
        elif intent.intent_type is IntentType.BLOCK_RESOURCE:
            risk = RiskLevel.R3_HIGH
            policy = "traffic.resource-block.supervisor"
            approval_required = True
            valid_resources = {
                f"zone:{row['id']}" for row in assets["zones"].get("zones", [])
            }
            valid_resources.update(
                row["ownResource"] for row in assets["conflicts"]["edgeResources"]
            )
            for resource_id in (intent.resource_block.resource_ids if intent.resource_block else []):
                if resource_id not in valid_resources:
                    issues.append(
                        ValidationIssue(
                            code="intent.resource.unknown",
                            message=f"资源 {resource_id} 不存在或不允许由智能体封锁。",
                            severity="error",
                        )
                    )
        elif intent.intent_type in {
            IntentType.REPORT_VEHICLE_FAULT,
            IntentType.REQUEST_RECOVERY,
        }:
            risk = RiskLevel.R3_HIGH
            policy = "incident.supervisor"
            approval_required = True
            issues.append(
                ValidationIssue(
                    code="intent.not_implemented",
                    message="该意图当前仅支持诊断，不开放执行工具。",
                    severity="warning",
                )
            )

        return IntentValidation(
            intentId=intent.intent_id,
            valid=not any(row.severity == "error" for row in issues),
            riskLevel=risk,
            approvalRequired=approval_required,
            policyCode=policy,
            issues=issues,
        )

    def simulate(self, request: SimulationRequest) -> SimulationSummary:
        self._require_engine()
        validation = (
            self.validate_intent(request.intent, request.scenario_id)
            if request.intent is not None
            else None
        )
        if validation is not None and not validation.valid:
            raise ValueError("调度意图未通过校验，不能进入仿真。")
        with self._simulation_lock:
            return self._simulate_locked(request)

    def _simulate_locked(self, request: SimulationRequest) -> SimulationSummary:
        started = time.perf_counter()
        run_id = new_id("run")
        output_dir = self.settings.runs_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            scenario = deepcopy(self.load_scenario(request.scenario_id))
            scenario["seed"] = request.seed
            if request.intent and request.intent.intent_type is IntentType.CREATE_TASK:
                task_doc = request.intent.task.model_dump(by_alias=True, exclude_none=True)
                scenario["tasks"].append(task_doc)
            runtime = self._run_scenario(
                scenario,
                policy=request.policy,
                resource_block=(
                    request.intent.resource_block
                    if request.intent
                    and request.intent.intent_type is IntentType.BLOCK_RESOURCE
                    else None
                ),
            )
            planning = runtime.planning_result()
            planning_summary = planning.summary()
            result = runtime.result()
            planned = runtime.planned_scenario(request.scenario_id, request.seed)
            self._write_json(output_dir / "input-scenario.json", scenario)
            self._write_json(output_dir / "planned-scenario.json", planned)
            self._write_json(output_dir / "planning-summary.json", planning_summary)
            self._write_json(output_dir / "result.json", result)
            manifest = {
                "schemaVersion": 1,
                "runId": run_id,
                "scenarioId": request.scenario_id,
                "policy": request.policy,
                "seed": request.seed,
                "engine": self.engine_status(),
                "intent": (
                    request.intent.model_dump(by_alias=True, mode="json")
                    if request.intent
                    else None
                ),
                "resultDigest": result.get("eventDigestSha256"),
            }
            self._write_json(output_dir / "manifest.json", manifest)
            metrics = result["metrics"]
            summary = SimulationSummary(
                runId=run_id,
                scenarioId=request.scenario_id,
                label=request.label,
                policy=request.policy,
                seed=request.seed,
                status="COMPLETED",
                durationMs=round((time.perf_counter() - started) * 1000, 3),
                metrics=metrics,
                planning={
                    key: planning_summary.get(key)
                    for key in (
                        "plannedTaskCount",
                        "unplannedTaskCount",
                        "insertedWaitMs",
                        "routeCombinationsTried",
                        "scheduleAttempts",
                        "planningLatencyMs",
                        "planningTimeoutCount",
                        "planningPeriodMissCount",
                    )
                },
                safety={
                    "conflictFree": metrics.get("reservationConflictRejections", 0) == 0,
                    "reservationConflictRejections": metrics.get(
                        "reservationConflictRejections", 0
                    ),
                    "unplannedTaskCount": len(planning.unplanned_task_ids),
                    "simulationOnly": True,
                },
                intentId=request.intent.intent_id if request.intent else None,
                manifestPath=str(output_dir / "manifest.json"),
            )
            self._write_json(
                output_dir / "command-center-summary.json",
                summary.model_dump(by_alias=True, mode="json"),
            )
            return summary
        except Exception as error:
            summary = SimulationSummary(
                runId=run_id,
                scenarioId=request.scenario_id,
                label=request.label,
                policy=request.policy,
                seed=request.seed,
                status="FAILED",
                durationMs=round((time.perf_counter() - started) * 1000, 3),
                metrics={},
                planning={},
                safety={"conflictFree": False, "simulationOnly": True},
                intentId=request.intent.intent_id if request.intent else None,
                manifestPath=str(output_dir / "manifest.json"),
                error=str(error),
            )
            self._write_json(
                output_dir / "command-center-summary.json",
                summary.model_dump(by_alias=True, mode="json"),
            )
            raise

    def _run_scenario(
        self,
        scenario: dict[str, Any],
        *,
        policy: str,
        resource_block: Any | None,
        unavailable_until: dict[str, int] | None = None,
    ) -> Any:
        modules = self._engine_modules()
        assets = self._assets()
        defaults = assets["scheduler"]["serviceDefaults"]
        task_rows = sorted(
            scenario["tasks"],
            key=lambda row: (int(row["releaseTimeMs"]), row["taskId"]),
        )
        runtime_vehicles = [
            modules["Vehicle"].from_dict(row) for row in scenario["vehicles"]
        ]
        for vehicle in runtime_vehicles:
            vehicle.available_at_ms = int(
                (unavailable_until or {}).get(vehicle.vehicle_id, 0)
            )
        runtime = modules["OnlineDispatchRuntime"](
            topology=modules["MapTopology"](
                assets["model"],
                assets["conflicts"],
                assets["workstations"],
                assets["zones"],
            ),
            model=assets["model"],
            profiles=assets["profiles"],
            scheduler=assets["scheduler"],
            traffic_zones=assets["zones"],
            vehicles=runtime_vehicles,
            end_time_ms=int(scenario["endTimeMs"]),
            policy=policy,
            seed=int(scenario["seed"]),
        )
        if resource_block is not None:
            for table_name, table in (
                ("planner", runtime.reservations),
                ("simulator", runtime.simulator.reservations),
            ):
                table.freeze_resources(
                    freeze_id=(
                        f"command-center:{table_name}:{resource_block.start_ms}:"
                        f"{resource_block.end_ms}"
                    ),
                    resource_ids=resource_block.resource_ids,
                    start_ms=resource_block.start_ms,
                    end_ms=min(resource_block.end_ms, runtime.end_time_ms),
                )

        planning_period_ms = int(assets["scheduler"]["planner"]["planningPeriodMs"])
        next_cycle_ms = 0
        task_index = 0
        while runtime.now_ms < runtime.end_time_ms:
            next_release_ms = (
                int(task_rows[task_index]["releaseTimeMs"])
                if task_index < len(task_rows)
                else runtime.end_time_ms
            )
            next_time_ms = min(next_cycle_ms, next_release_ms, runtime.end_time_ms)
            runtime.advance_to(next_time_ms)
            submitted = False
            while (
                task_index < len(task_rows)
                and int(task_rows[task_index]["releaseTimeMs"]) == next_time_ms
            ):
                runtime.submit_task(
                    modules["TransportTask"].from_dict(
                        task_rows[task_index],
                        int(defaults["pickupServiceMs"]),
                        int(defaults["dropoffServiceMs"]),
                    )
                )
                task_index += 1
                submitted = True
            if submitted:
                runtime.advance_to(next_time_ms)
            if next_time_ms == next_cycle_ms or submitted:
                for proposal in runtime.plan_cycle():
                    runtime.acknowledge_plan(
                        proposal.proposal_id,
                        proposal.plan.revision,
                        accepted=True,
                    )
                runtime.advance_to(next_time_ms)
            if next_time_ms == next_cycle_ms:
                next_cycle_ms = min(
                    runtime.end_time_ms,
                    next_cycle_ms + planning_period_ms,
                )
        return runtime

    def simulate_incident_option(
        self,
        incident: IncidentRecord,
        mode: WhatIfMode,
    ) -> SimulationSummary:
        """Run a conservative, evidence-backed counterfactual without mutating MASP."""

        self._require_engine()
        with self._simulation_lock:
            return self._simulate_incident_option_locked(incident, mode)

    def _simulate_incident_option_locked(
        self,
        incident: IncidentRecord,
        mode: WhatIfMode,
    ) -> SimulationSummary:
        started = time.perf_counter()
        run_id = new_id("run")
        output_dir = self.settings.runs_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        source_dir = self.settings.runs_dir / incident.run_id
        source_path = source_dir / "input-scenario.json"
        if not source_path.exists():
            raise KeyError(incident.run_id)

        scenario = deepcopy(self._read_json(source_path))
        scenario["seed"] = int(scenario.get("seed", 0))
        end_time_ms = int(scenario["endTimeMs"])
        fault_vehicle_id = incident.vehicle_ids[0]
        known_vehicle_ids = {row["vehicleId"] for row in scenario["vehicles"]}
        if fault_vehicle_id not in known_vehicle_ids:
            raise ValueError("源场景中不存在故障车辆，无法建立分支仿真。")
        if incident.fault_at_ms >= end_time_ms:
            raise ValueError("故障时刻必须早于源场景结束时间。")

        manual_transfer_required = (
            incident.load_state == "loaded"
            and mode in {WhatIfMode.ISOLATE_REASSIGN, WhatIfMode.SAFETY_STOP}
        )
        removed_task_ids: list[str] = []
        unavailable_until: dict[str, int] = {}
        freeze_end_ms = min(
            end_time_ms,
            incident.fault_at_ms + incident.recovery_duration_ms,
        )

        if mode is WhatIfMode.WAIT_RECOVERY:
            # MASP has no public mid-run fault API. Delaying availability from the
            # beginning is a conservative bound and is disclosed in every artifact.
            unavailable_until[fault_vehicle_id] = freeze_end_ms
        else:
            scenario["vehicles"] = [
                row
                for row in scenario["vehicles"]
                if row["vehicleId"] != fault_vehicle_id
            ]
            if manual_transfer_required:
                removed_task_ids = list(incident.task_ids)
                scenario["tasks"] = [
                    row
                    for row in scenario["tasks"]
                    if row["taskId"] not in set(removed_task_ids)
                ]
            if mode is WhatIfMode.SAFETY_STOP:
                freeze_end_ms = end_time_ms

        resource_ids = list(incident.resource_ids)
        if not resource_ids and incident.location_node_id:
            resource_ids = [f"node:{incident.location_node_id}"]
        resource_block = None
        if resource_ids and freeze_end_ms > incident.fault_at_ms:
            resource_block = type(
                "IncidentResourceBlock",
                (),
                {
                    "resource_ids": resource_ids,
                    "start_ms": incident.fault_at_ms,
                    "end_ms": freeze_end_ms,
                },
            )()

        mode_labels = {
            WhatIfMode.WAIT_RECOVERY: "等待恢复",
            WhatIfMode.ISOLATE_REASSIGN: "隔离与重派",
            WhatIfMode.SAFETY_STOP: "安全停车",
        }
        branch_context = {
            "schemaVersion": 1,
            "incidentId": incident.incident_id,
            "sourceRunId": incident.run_id,
            "whatIfMode": mode.value,
            "faultVehicleId": fault_vehicle_id,
            "faultAtMs": incident.fault_at_ms,
            "safeFaultNodeId": incident.location_node_id,
            "resourceIds": resource_ids,
            "resourceFreezeEndMs": freeze_end_ms,
            "vehicleUnavailableUntilMs": unavailable_until.get(fault_vehicle_id),
            "removedTaskIds": removed_task_ids,
            "manualTransferRequired": manual_transfer_required,
            "branchModel": "conservative-whole-scenario-counterfactual",
            "limitations": [
                "MASP 当前没有公开的中途车辆故障续跑接口。",
                "分支使用同一源场景做保守反事实推演，不表示真实车辆已经执行处置。",
                "故障点来自已完成移动段的安全节点，未构造边内急停轨迹。",
            ],
        }
        try:
            runtime = self._run_scenario(
                scenario,
                policy="top_k",
                resource_block=resource_block,
                unavailable_until=unavailable_until,
            )
            planning = runtime.planning_result()
            planning_summary = planning.summary()
            result = runtime.result()
            planned = runtime.planned_scenario(incident.scenario_id, scenario["seed"])
            self._write_json(output_dir / "input-scenario.json", scenario)
            self._write_json(output_dir / "planned-scenario.json", planned)
            self._write_json(output_dir / "planning-summary.json", planning_summary)
            self._write_json(output_dir / "result.json", result)
            self._write_json(output_dir / "incident-context.json", branch_context)
            manifest = {
                "schemaVersion": 1,
                "runId": run_id,
                "scenarioId": incident.scenario_id,
                "policy": "top_k",
                "seed": scenario["seed"],
                "engine": self.engine_status(),
                "incident": branch_context,
                "resultDigest": result.get("eventDigestSha256"),
            }
            self._write_json(output_dir / "manifest.json", manifest)
            metrics = dict(result["metrics"])
            metrics["manualTransferTaskCount"] = len(removed_task_ids)
            summary = SimulationSummary(
                runId=run_id,
                scenarioId=incident.scenario_id,
                label=f"故障推演 | {mode_labels[mode]}",
                policy="top_k",
                seed=scenario["seed"],
                status="COMPLETED",
                durationMs=round((time.perf_counter() - started) * 1000, 3),
                metrics=metrics,
                planning={
                    key: planning_summary.get(key)
                    for key in (
                        "plannedTaskCount",
                        "unplannedTaskCount",
                        "insertedWaitMs",
                        "routeCombinationsTried",
                        "scheduleAttempts",
                        "planningLatencyMs",
                        "planningTimeoutCount",
                        "planningPeriodMissCount",
                    )
                },
                safety={
                    "conflictFree": metrics.get("reservationConflictRejections", 0) == 0,
                    "reservationConflictRejections": metrics.get(
                        "reservationConflictRejections", 0
                    ),
                    "unplannedTaskCount": len(planning.unplanned_task_ids),
                    "simulationOnly": True,
                    "incidentId": incident.incident_id,
                    "sourceRunId": incident.run_id,
                    "whatIfMode": mode.value,
                    "faultVehicleId": fault_vehicle_id,
                    "faultAtMs": incident.fault_at_ms,
                    "faultNodeId": incident.location_node_id,
                    "resourceFreezeEndMs": freeze_end_ms,
                    "manualTransferRequired": manual_transfer_required,
                    "branchModel": branch_context["branchModel"],
                    "requiresApproval": True,
                },
                manifestPath=str(output_dir / "manifest.json"),
            )
            self._write_json(
                output_dir / "command-center-summary.json",
                summary.model_dump(by_alias=True, mode="json"),
            )
            return summary
        except Exception as error:
            self._write_json(output_dir / "incident-context.json", branch_context)
            summary = SimulationSummary(
                runId=run_id,
                scenarioId=incident.scenario_id,
                label=f"故障推演 | {mode_labels[mode]}",
                policy="top_k",
                seed=scenario["seed"],
                status="FAILED",
                durationMs=round((time.perf_counter() - started) * 1000, 3),
                metrics={},
                planning={},
                safety={
                    "conflictFree": False,
                    "simulationOnly": True,
                    "incidentId": incident.incident_id,
                    "whatIfMode": mode.value,
                    "manualTransferRequired": manual_transfer_required,
                },
                manifestPath=str(output_dir / "manifest.json"),
                error=str(error),
            )
            self._write_json(
                output_dir / "command-center-summary.json",
                summary.model_dump(by_alias=True, mode="json"),
            )
            raise

    def list_runs(self) -> list[SimulationSummary]:
        rows: list[SimulationSummary] = []
        for path in self.settings.runs_dir.glob("*/command-center-summary.json"):
            try:
                rows.append(SimulationSummary.model_validate(self._read_json(path)))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row.created_at, reverse=True)

    def get_run(self, run_id: str) -> SimulationSummary:
        path = self.settings.runs_dir / run_id / "command-center-summary.json"
        if not path.exists():
            raise KeyError(run_id)
        return SimulationSummary.model_validate(self._read_json(path))

    def get_run_detail(self, run_id: str) -> dict[str, Any]:
        root = self.settings.runs_dir / run_id
        if not root.exists():
            raise KeyError(run_id)
        return {
            "summary": self._read_json(root / "command-center-summary.json"),
            "scenario": self._read_json(root / "planned-scenario.json"),
            "result": self._read_json(root / "result.json"),
            "planning": self._read_json(root / "planning-summary.json"),
        }

    def compare(self, run_ids: list[str]) -> ComparisonResult:
        runs = [self.get_run(run_id) for run_id in run_ids]
        successful = [row for row in runs if row.status == "COMPLETED"]
        if not successful:
            raise ValueError("没有可比较的成功仿真。")

        def score(row: SimulationSummary) -> tuple[float, float, float, float]:
            metrics = row.metrics
            conflict_penalty = float(row.safety.get("reservationConflictRejections", 0))
            completed = float(metrics.get("completedTaskCount", 0))
            cycle = float(metrics.get("meanTaskCycleTimeMs") or 10**12)
            queue = float(metrics.get("meanTaskQueueTimeMs") or 10**12)
            return (conflict_penalty, -completed, cycle, queue)

        recommended = min(successful, key=score)
        base = runs[0]
        rationale = [
            "所有候选方案首先按资源冲突为零进行硬筛选。",
            f"推荐方案完成 {recommended.metrics.get('completedTaskCount', 0)} 个任务。",
        ]
        if recommended.metrics.get("meanTaskCycleTimeMs") is not None:
            rationale.append(
                f"平均任务周期为 {recommended.metrics['meanTaskCycleTimeMs'] / 1000:.1f} 秒。"
            )
        if recommended.run_id != base.run_id:
            rationale.append("推荐结果优于列表中的首个基线方案。")
        return ComparisonResult(
            runs=runs,
            recommendedRunId=recommended.run_id,
            rationale=rationale,
        )
