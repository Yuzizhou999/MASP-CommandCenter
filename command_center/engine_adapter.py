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
    AgentModelStatus,
    AgentPolicyEvidence,
    AgentPolicyOptions,
    ComparisonResult,
    DispatchIntent,
    IncidentRecord,
    IncidentType,
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

    @staticmethod
    def _schema_path(name: str) -> Path:
        return Path(__file__).resolve().parents[1] / "schemas" / name

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
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    def _bundle_status(self) -> dict[str, Any] | None:
        manifest_path = self.root / "engine.bundle.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = self._read_json(manifest_path)
            if manifest.get("schemaVersion") != 1:
                raise ValueError("unsupported schema version")
            commit = str(manifest["commit"])
            hashes = dict(manifest["files"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            return {
                "currentCommit": "",
                "commitMatches": False,
                "dirty": True,
                "dirtyFileCount": 1,
                "bundleVerified": False,
                "warning": f"MASP 离线版本证明无效：{error}",
            }

        mismatches: list[str] = []
        for relative, expected_hash in hashes.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                mismatches.append(str(relative))
                continue
            candidate = (self.root / relative_path).resolve()
            if not candidate.is_file() or self._file_sha256(candidate) != expected_hash:
                mismatches.append(str(relative))

        required = {
            "masp/online.py",
            "generated/xiate-unified-map-model.json",
            "scenarios/interactive-multi-fleet.json",
        }
        normalized = {str(Path(path)).replace("\\", "/") for path in hashes}
        missing_required = sorted(required - normalized)
        mismatches.extend(missing_required)
        verified = not mismatches
        return {
            "currentCommit": commit,
            "commitMatches": commit == self.settings.engine_commit,
            "dirty": not verified,
            "dirtyFileCount": len(set(mismatches)),
            "bundleVerified": verified,
            "warning": (
                None
                if verified
                else f"MASP 离线文件校验失败，共 {len(set(mismatches))} 项不一致。"
            ),
        }

    def engine_status(self) -> dict[str, Any]:
        git_top_level = self._git(self.root, "rev-parse", "--show-toplevel")
        is_engine_worktree = bool(git_top_level) and (
            Path(git_top_level).resolve() == self.root.resolve()
        )
        current = self._git(self.root, "rev-parse", "HEAD") if is_engine_worktree else ""
        if not current:
            bundle = self._bundle_status()
            if bundle is not None:
                matches = bool(bundle["commitMatches"])
                verified = bool(bundle["bundleVerified"])
                return {
                    "root": str(self.root),
                    "expectedCommit": self.settings.engine_commit,
                    **bundle,
                    "allowed": matches and verified,
                    "environment": self.settings.app_env,
                    "source": "verified-bundle",
                }
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
            "source": "git-worktree",
            "bundleVerified": False,
            "warning": (
                "开发模式正在使用包含未提交修改的 MASP 工作区，发布前必须固定干净版本。"
                if dirty and allowed
                else None
            ),
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def agent_model_status(self) -> AgentModelStatus:
        checkpoint = self.settings.agent_checkpoint
        checkpoint_present = bool(checkpoint and checkpoint.is_file())
        checkpoint_sha256 = (
            self._file_sha256(checkpoint)
            if checkpoint is not None and checkpoint_present
            else None
        )
        if checkpoint_present:
            notice = "已检测到模型权重，运行前将校验版本、观测和动作契约。"
        elif checkpoint is not None:
            notice = "配置的模型权重不存在，智能体请求将使用规则基线。"
        else:
            notice = "未配置模型权重，智能体请求将使用规则基线。"
        return AgentModelStatus(
            modelId=self.settings.agent_model_id,
            modelVersion=self.settings.agent_model_version,
            algorithm="Actor-Critic/PPO priority policy",
            mode="LEARNED" if checkpoint_present else "BASELINE",
            configured=checkpoint is not None,
            checkpointPresent=checkpoint_present,
            checkpointName=checkpoint.name if checkpoint is not None else None,
            checkpointSha256=checkpoint_sha256,
            device=self.settings.agent_device,
            safetyController="MASP Top-K guardian + SIPP + resource reservation",
            notice=notice,
        )

    def _prepare_agent_policy(
        self, options: AgentPolicyOptions | None, seed: int
    ) -> dict[str, Any]:
        requested = options or AgentPolicyOptions()
        status = self.agent_model_status()
        if requested.model_id not in (None, status.model_id):
            raise ValueError(
                f"未登记智能体模型 {requested.model_id!r}，当前仅允许 {status.model_id!r}。"
            )

        reasons: list[str] = []
        metadata: dict[str, Any] = {}
        deviation_enabled = False
        checkpoint_path: Path | None = None
        if not requested.allow_deviation:
            reasons.append("本次运行未授权学习策略改变规则候选顺序。")
        elif self.settings.agent_checkpoint is None:
            reasons.append("服务端未配置智能体模型权重。")
        elif not self.settings.agent_checkpoint.is_file():
            reasons.append("服务端配置的智能体模型权重不存在。")
        else:
            try:
                root_text = str(self.root)
                if root_text not in sys.path:
                    sys.path.insert(0, root_text)
                import torch

                torch.set_num_threads(self.settings.agent_torch_threads)
                try:
                    torch.set_num_interop_threads(
                        self.settings.agent_torch_threads
                    )
                except RuntimeError:
                    # PyTorch only permits setting inter-op threads before work starts.
                    pass
                from masp.rl_priority import load_checkpoint

                payload = load_checkpoint(
                    self.settings.agent_checkpoint,
                    device=self.settings.agent_device,
                )
                raw_metadata = dict(payload.get("metadata", {}))
                metadata = {
                    key: raw_metadata[key]
                    for key in (
                        "observation_version",
                        "action_mode",
                        "reward_version",
                        "candidate_count",
                        "priority_prefix_count",
                    )
                    if key in raw_metadata
                }
                deviation_enabled = True
                checkpoint_path = self.settings.agent_checkpoint
            except Exception as error:
                reasons.append(
                    "模型权重兼容性校验失败，已禁止学习策略输出："
                    f"{type(error).__name__}: {error}"
                )

        return {
            "modelId": status.model_id,
            "modelVersion": status.model_version,
            "checkpointSha256": status.checkpoint_sha256,
            "checkpointPath": checkpoint_path,
            "candidateCount": requested.candidate_count,
            "deviationRequested": requested.allow_deviation,
            "deviationEnabled": deviation_enabled,
            "fallbackReasons": reasons,
            "checkpointMetadata": metadata,
            "seed": seed,
        }

    def validate_agent_model(self) -> dict[str, Any]:
        """Validate the registered checkpoint without running a simulation."""
        prepared = self._prepare_agent_policy(
            AgentPolicyOptions(allowDeviation=True), seed=0
        )
        return {
            "ok": bool(prepared["deviationEnabled"]),
            "mode": "LEARNED" if prepared["deviationEnabled"] else "BASELINE",
            "modelId": prepared["modelId"],
            "modelVersion": prepared["modelVersion"],
            "checkpointSha256": prepared["checkpointSha256"],
            "checkpointMetadata": prepared["checkpointMetadata"],
            "fallbackReasons": prepared["fallbackReasons"],
        }

    @staticmethod
    def _agent_policy_evidence(
        planning_summary: dict[str, Any],
        prepared: dict[str, Any],
        evidence_path: Path | None,
    ) -> AgentPolicyEvidence:
        cycles = list(planning_summary.get("cycles", []))
        agent_candidate_count = 0
        selected_agent_candidate_count = 0
        for cycle in cycles:
            candidates = {
                row.get("candidateId"): row
                for row in cycle.get("candidates", [])
            }
            agent_candidate_count += sum(
                row.get("strategy") == "rl" for row in candidates.values()
            )
            selected_agent_candidate_count += sum(
                candidates.get(candidate_id, {}).get("strategy") == "rl"
                for candidate_id in cycle.get("selectedCandidateIds", [])
            )

        inference_count = int(planning_summary.get("rlInferenceCount", 0))
        runtime_fallback_count = int(planning_summary.get("rlFallbackCount", 0))
        decision_cycle_count = int(planning_summary.get("decisionCycleCount", 0))
        fallback_count = (
            runtime_fallback_count
            if prepared["deviationEnabled"]
            else decision_cycle_count
        )
        safety_fallback_count = int(
            planning_summary.get("rlSafetyFallbackCount", 0)
        )
        guardian_override_count = int(
            planning_summary.get("rlGuardianOverrideCount", 0)
        )
        reasons = list(prepared["fallbackReasons"])
        if runtime_fallback_count:
            reasons.append(
                f"{runtime_fallback_count} 次推理未产生有效候选，已使用确定性候选。"
            )
        if safety_fallback_count:
            reasons.append(
                f"{safety_fallback_count} 次学习候选未通过安全选优，已采用规则接管。"
            )
        if guardian_override_count:
            reasons.append(
                f"Top-K guardian 以更优安全评分覆盖学习候选 {guardian_override_count} 次。"
            )
        notes: list[str] = []
        if prepared["deviationEnabled"] and inference_count == 0:
            notes.append("本次场景未形成需要学习策略排序的耦合冲突分量。")
        if not prepared["deviationEnabled"] and decision_cycle_count:
            notes.append(f"{decision_cycle_count} 个决策周期由规则基线接管。")
        if selected_agent_candidate_count:
            notes.append(
                f"安全校验后共有 {selected_agent_candidate_count} 个学习候选被采用。"
            )
        return AgentPolicyEvidence(
            mode="LEARNED" if prepared["deviationEnabled"] else "BASELINE",
            modelId=prepared["modelId"],
            modelVersion=prepared["modelVersion"],
            checkpointSha256=prepared["checkpointSha256"],
            candidateCount=prepared["candidateCount"],
            deviationRequested=prepared["deviationRequested"],
            deviationEnabled=prepared["deviationEnabled"],
            inferenceCount=inference_count,
            inferenceMs=float(planning_summary.get("rlInferenceMs", 0.0)),
            fallbackCount=fallback_count,
            safetyFallbackCount=safety_fallback_count,
            guardianCandidateCount=int(
                planning_summary.get("rlGuardianCandidateCount", 0)
            ),
            guardianOverrideCount=guardian_override_count,
            agentCandidateCount=agent_candidate_count,
            selectedAgentCandidateCount=selected_agent_candidate_count,
            decisionCycleCount=decision_cycle_count,
            fallbackReasons=list(dict.fromkeys(reasons)),
            notes=notes,
            evidencePath=str(evidence_path) if evidence_path is not None else None,
        )

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
        from .masp.scenario_package import (
            ScenarioPackage,
            compile_scenario_package,
            package_from_assets,
            validate_scenario_package_document,
        )
        from .masp.task_stream import (
            generate_task_stream,
            validate_task_stream_generation_document,
        )
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

    def workstation_catalog(self) -> list[dict[str, Any]]:
        """Return the version-locked MASP workstation catalog."""

        self._require_engine()
        return deepcopy(self._assets()["workstations"]["workstations"])

    def deadlock_recovery_evidence(
        self, deadlock_case: str = "RECOVERABLE"
    ) -> dict[str, Any]:
        """Run MASP's deterministic wait-graph and recovery acceptance scenario."""

        self._require_engine()
        with self._simulation_lock:
            return self._deadlock_recovery_evidence_locked(deadlock_case)

    def _deadlock_recovery_evidence_locked(
        self, deadlock_case: str
    ) -> dict[str, Any]:
        root_text = str(self.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from masp.recovery_scenario import run_recovery_scenario

        key_by_case = {
            "RECOVERABLE": ("recoverableDeadlock", "recoverableDeadlock"),
            "UNRECOVERABLE": ("unrecoverableDeadlock", "unrecoverableDeadlock"),
        }
        if deadlock_case not in key_by_case:
            raise ValueError(f"未知死锁演示类型 {deadlock_case!r}。")
        result_key, scenario_key = key_by_case[deadlock_case]
        scenario = self._read_json(self.root / "scenarios" / "deadlock-recovery.json")
        assets = self._assets()
        result = run_recovery_scenario(
            scenario,
            assets["model"],
            assets["conflicts"],
            assets["workstations"],
            assets["profiles"],
            assets["scheduler"],
            assets["zones"],
            self.root / "schemas",
        ).to_dict()
        if not result.get("accepted"):
            raise ValueError("MASP 死锁恢复验收场景未通过全部安全检查。")
        return {
            "scenario": scenario,
            "scenarioCase": deepcopy(scenario[scenario_key]),
            "result": result,
            "case": deepcopy(result[result_key]),
        }

    def validate_scenario_package(self, document: dict[str, Any]) -> dict[str, Any]:
        """Validate an editable package without making it a runtime scenario."""
        modules = self._engine_modules()
        modules["validate_scenario_package_document"](
            document,
            self._schema_path("scenario-package.schema.json"),
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
            self._schema_path("scenario-package.schema.json"),
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
            self._schema_path("task-stream-generation.schema.json"),
        )
        modules["validate_scenario_package_document"](
            document,
            self._schema_path("scenario-package.schema.json"),
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

    def benchmark_source(self, scenario_id: str) -> dict[str, Any]:
        """Return only immutable inputs approved for generated benchmark cases."""
        assets = self._assets()
        return {
            "scenario": deepcopy(self.load_scenario(scenario_id)),
            "nodes": deepcopy(assets["model"].get("nodes", [])),
        }

    def resolve_node_reference(
        self, reference: str, robot_group: str | None = None
    ) -> list[str]:
        """Resolve a user-provided node alias against the approved map catalog."""
        normalized = reference.strip()
        if not normalized:
            return []
        lowered = normalized.lower()
        rows = self._assets()["model"].get("nodes", [])
        matches: list[str] = []
        for row in rows:
            node_id = str(row["id"])
            group, _, alias = node_id.partition(":")
            if robot_group is not None and group != robot_group:
                continue
            aliases = {str(value).lower() for value in row.get("aliases", {}).values()}
            if lowered in {node_id.lower(), alias.lower(), *aliases}:
                matches.append(node_id)
        return sorted(set(matches))

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
            valid_resources.update(
                f"node:{row['id']}" for row in assets["model"]["nodes"]
            )
            valid_resources.update(
                f"workstation:{row['id']}"
                for row in assets["workstations"]["workstations"]
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

    def evaluate_scenario_document(
        self,
        scenario: dict[str, Any],
        *,
        policy: str,
        seed: int,
        agent_policy: AgentPolicyOptions | None = None,
    ) -> dict[str, Any]:
        """Evaluate an isolated generated scenario without adding it to normal runs."""
        self._require_engine()
        document = deepcopy(scenario)
        document["seed"] = int(seed)
        prepared_agent = (
            self._prepare_agent_policy(agent_policy, seed) if policy == "rl" else None
        )
        started = time.perf_counter()
        with self._simulation_lock:
            runtime = self._run_scenario(
                document,
                policy=policy,
                resource_block=None,
                agent_runtime=prepared_agent,
            )
        planning = runtime.planning_result()
        planning_summary = planning.summary()
        result = runtime.result()
        metrics = result["metrics"]
        agent_evidence = (
            self._agent_policy_evidence(planning_summary, prepared_agent, None)
            if prepared_agent is not None
            else None
        )
        return {
            "status": "COMPLETED",
            "policy": policy,
            "seed": seed,
            "durationMs": round((time.perf_counter() - started) * 1000, 3),
            "metrics": metrics,
            "planning": planning_summary,
            "safety": {
                "conflictFree": metrics.get("reservationConflictRejections", 0) == 0,
                "reservationConflictRejections": metrics.get(
                    "reservationConflictRejections", 0
                ),
                "unplannedTaskCount": len(planning.unplanned_task_ids),
                "simulationOnly": True,
            },
            "agentPolicy": (
                agent_evidence.model_dump(by_alias=True, mode="json")
                if agent_evidence is not None
                else None
            ),
            "resultDigest": result.get("eventDigestSha256"),
        }

    def _simulate_locked(self, request: SimulationRequest) -> SimulationSummary:
        started = time.perf_counter()
        run_id = new_id("run")
        output_dir = self.settings.runs_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        prepared_agent: dict[str, Any] | None = None
        try:
            scenario = deepcopy(self.load_scenario(request.scenario_id))
            scenario["seed"] = request.seed
            if request.intent and request.intent.intent_type is IntentType.CREATE_TASK:
                task_doc = request.intent.task.model_dump(by_alias=True, exclude_none=True)
                scenario["tasks"].append(task_doc)
            if request.policy == "rl":
                prepared_agent = self._prepare_agent_policy(
                    request.agent_policy, request.seed
                )
            runtime = self._run_scenario(
                scenario,
                policy=request.policy,
                agent_runtime=prepared_agent,
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
            agent_evidence: AgentPolicyEvidence | None = None
            if prepared_agent is not None:
                evidence_path = output_dir / "agent-policy-evidence.json"
                agent_evidence = self._agent_policy_evidence(
                    planning_summary, prepared_agent, evidence_path
                )
                self._write_json(
                    evidence_path,
                    {
                        "schemaVersion": 1,
                        "runId": run_id,
                        "seed": prepared_agent["seed"],
                        "model": {
                            "modelId": prepared_agent["modelId"],
                            "modelVersion": prepared_agent["modelVersion"],
                            "checkpointSha256": prepared_agent["checkpointSha256"],
                            "checkpointMetadata": prepared_agent[
                                "checkpointMetadata"
                            ],
                        },
                        "execution": agent_evidence.model_dump(
                            by_alias=True, mode="json"
                        ),
                        "safetyBoundary": {
                            "candidateOutputOnly": True,
                            "deterministicValidationRequired": True,
                            "guardianStrategy": "Top-K congestion guardian",
                            "trajectoryPlanner": "continuous-time SIPP",
                            "fieldExecutionEnabled": False,
                        },
                        "decisionCycles": planning_summary.get("cycles", []),
                    },
                )
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
                "agentPolicy": (
                    agent_evidence.model_dump(by_alias=True, mode="json")
                    if agent_evidence is not None
                    else None
                ),
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
                agentPolicy=agent_evidence,
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
                agentPolicy=(
                    AgentPolicyEvidence(
                        mode="BASELINE",
                        modelId=prepared_agent["modelId"],
                        modelVersion=prepared_agent["modelVersion"],
                        checkpointSha256=prepared_agent["checkpointSha256"],
                        candidateCount=prepared_agent["candidateCount"],
                        deviationRequested=prepared_agent["deviationRequested"],
                        deviationEnabled=False,
                        fallbackReasons=[
                            *prepared_agent["fallbackReasons"],
                            f"智能体仿真失败：{type(error).__name__}: {error}",
                        ],
                    )
                    if prepared_agent is not None
                    else None
                ),
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
        agent_runtime: dict[str, Any] | None = None,
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
            rl_checkpoint=(
                str(agent_runtime["checkpointPath"])
                if agent_runtime is not None
                and agent_runtime["checkpointPath"] is not None
                else None
            ),
            rl_candidate_count=(
                int(agent_runtime["candidateCount"])
                if agent_runtime is not None
                else None
            ),
            rl_allow_deviation=(
                bool(agent_runtime["deviationEnabled"])
                if agent_runtime is not None
                else False
            ),
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

        if incident.incident_type is IncidentType.DEADLOCK_RISK:
            return self._simulate_deadlock_option_locked(
                incident=incident,
                mode=mode,
                run_id=run_id,
                output_dir=output_dir,
                source_dir=source_dir,
                started=started,
            )

        scenario = deepcopy(self._read_json(source_path))
        scenario["seed"] = int(scenario.get("seed", 0))
        end_time_ms = int(scenario["endTimeMs"])
        is_workstation_outage = (
            incident.incident_type is IncidentType.WORKSTATION_DISABLED
        )
        fault_vehicle_id = incident.vehicle_ids[0] if incident.vehicle_ids else None
        if not is_workstation_outage:
            known_vehicle_ids = {row["vehicleId"] for row in scenario["vehicles"]}
            if fault_vehicle_id not in known_vehicle_ids:
                raise ValueError("源场景中不存在故障车辆，无法建立分支仿真。")
        if incident.fault_at_ms >= end_time_ms:
            raise ValueError("事件时刻必须早于源场景结束时间。")

        manual_transfer_required = (
            not is_workstation_outage
            and incident.load_state == "loaded"
            and mode in {WhatIfMode.ISOLATE_REASSIGN, WhatIfMode.SAFETY_STOP}
        )
        removed_task_ids: list[str] = []
        unavailable_until: dict[str, int] = {}
        freeze_end_ms = min(
            end_time_ms,
            incident.fault_at_ms + incident.recovery_duration_ms,
        )

        if is_workstation_outage:
            if mode is WhatIfMode.SUSPEND_AFFECTED_TASKS:
                removed_task_ids = list(incident.task_ids)
                removed = set(removed_task_ids)
                scenario["tasks"] = [
                    row for row in scenario["tasks"] if row["taskId"] not in removed
                ]
            elif mode is WhatIfMode.SAFETY_STOP:
                freeze_end_ms = end_time_ms
        elif mode is WhatIfMode.WAIT_RECOVERY:
            # MASP has no public mid-run fault API. Delaying availability from the
            # beginning is a conservative bound and is disclosed in every artifact.
            assert fault_vehicle_id is not None
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
            WhatIfMode.SUSPEND_AFFECTED_TASKS: "暂停受影响任务",
            WhatIfMode.SAFETY_STOP: "安全停车",
        }
        incident_label = "工位停用推演" if is_workstation_outage else "故障推演"
        branch_context = {
            "schemaVersion": 1,
            "incidentId": incident.incident_id,
            "sourceRunId": incident.run_id,
            "whatIfMode": mode.value,
            "incidentType": incident.incident_type.value,
            "faultVehicleId": fault_vehicle_id,
            "faultAtMs": incident.fault_at_ms,
            "safeFaultNodeId": incident.location_node_id,
            "resourceIds": resource_ids,
            "resourceFreezeEndMs": freeze_end_ms,
            "vehicleUnavailableUntilMs": unavailable_until.get(fault_vehicle_id),
            "removedTaskIds": removed_task_ids,
            "manualTransferRequired": manual_transfer_required,
            "workstationId": incident.workstation_id,
            "suspendedTaskIds": removed_task_ids if is_workstation_outage else [],
            "branchModel": "conservative-whole-scenario-counterfactual",
            "limitations": [
                (
                    "工位停用分支冻结真实工位和节点资源；暂停任务分支不会自行改写任务起终点。"
                    if is_workstation_outage
                    else "MASP 当前没有公开的中途车辆故障续跑接口。"
                ),
                "分支使用同一源场景做保守反事实推演，不表示真实车辆已经执行处置。",
                (
                    "工位对象来自 MASP 工位目录，停用分支没有构造替代工位。"
                    if is_workstation_outage
                    else "故障点来自已完成移动段的安全节点，未构造边内急停轨迹。"
                ),
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
            metrics["manualTransferTaskCount"] = (
                len(removed_task_ids) if manual_transfer_required else 0
            )
            metrics["suspendedTaskCount"] = (
                len(removed_task_ids) if is_workstation_outage else 0
            )
            summary = SimulationSummary(
                runId=run_id,
                scenarioId=incident.scenario_id,
                label=f"{incident_label} | {mode_labels[mode]}",
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
                    "incidentType": incident.incident_type.value,
                    "faultVehicleId": fault_vehicle_id,
                    "faultAtMs": incident.fault_at_ms,
                    "faultNodeId": incident.location_node_id,
                    "resourceFreezeEndMs": freeze_end_ms,
                    "manualTransferRequired": manual_transfer_required,
                    "workstationId": incident.workstation_id,
                    "suspendedTaskIds": (
                        removed_task_ids if is_workstation_outage else []
                    ),
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
                label=f"{incident_label} | {mode_labels[mode]}",
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
                    "incidentType": incident.incident_type.value,
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

    def _simulate_deadlock_option_locked(
        self,
        *,
        incident: IncidentRecord,
        mode: WhatIfMode,
        run_id: str,
        output_dir: Path,
        source_dir: Path,
        started: float,
    ) -> SimulationSummary:
        deadlock_case = str(incident.event_attributes.get("deadlockCase", "RECOVERABLE"))
        recovery = self._deadlock_recovery_evidence_locked(deadlock_case)
        case = recovery["case"]
        decision = case["decision"]
        plan = decision.get("plan")
        if mode is WhatIfMode.CONTROLLED_REVERSE and plan is None:
            raise ValueError("当前等待环没有通过 MASP 校验的受控倒退计划。")
        if mode not in {WhatIfMode.CONTROLLED_REVERSE, WhatIfMode.SAFETY_STOP}:
            raise ValueError(f"死锁事件不支持处置模式 {mode.value}。")

        source_input = self._read_json(source_dir / "input-scenario.json")
        source_planned = self._read_json(source_dir / "planned-scenario.json")
        source_result = self._read_json(source_dir / "result.json")
        source_planning = self._read_json(source_dir / "planning-summary.json")
        recovery_metrics = dict(recovery["result"]["metrics"])
        if mode is WhatIfMode.SAFETY_STOP and plan is not None:
            recovery_metrics.update(
                {
                    "recoverySuccessCount": 0,
                    "reverseRecoveryCount": 0,
                    "reverseDistanceM": 0.0,
                    "safeStopCount": 1,
                }
            )
        metrics = dict(source_result.get("metrics", {}))
        metrics.update(recovery_metrics)
        action = "reverse" if mode is WhatIfMode.CONTROLLED_REVERSE else "safety_stop"
        branch_context = {
            "schemaVersion": 1,
            "incidentId": incident.incident_id,
            "incidentType": incident.incident_type.value,
            "sourceRunId": incident.run_id,
            "whatIfMode": mode.value,
            "deadlockCase": deadlock_case,
            "waitGraph": case["waitGraph"],
            "maspDecision": decision,
            "selectedAction": action,
            "branchModel": "masp-deterministic-deadlock-recovery",
            "limitations": [
                "等待环使用版本锁定的 MASP 仓储拓扑和预约快照进行确定性重放。",
                "受控倒退计划来自 MASP 恢复控制器，大模型没有生成或修改路线。",
                "所有恢复候选仍需主管审批和执行前世界版本复核。",
            ],
        }
        planned = deepcopy(source_planned)
        planned["deadlockRecovery"] = branch_context
        result = deepcopy(source_result)
        result["metrics"] = metrics
        result["deadlockRecovery"] = recovery["result"]
        planning = deepcopy(source_planning)
        planning["deadlockRecovery"] = branch_context
        planning.update(recovery_metrics)
        self._write_json(output_dir / "input-scenario.json", source_input)
        self._write_json(output_dir / "planned-scenario.json", planned)
        self._write_json(output_dir / "planning-summary.json", planning)
        self._write_json(output_dir / "result.json", result)
        self._write_json(output_dir / "incident-context.json", branch_context)
        self._write_json(output_dir / "recovery-result.json", recovery["result"])
        manifest = {
            "schemaVersion": 1,
            "runId": run_id,
            "scenarioId": incident.scenario_id,
            "policy": "deterministic_recovery",
            "seed": int(source_input.get("seed", 0)),
            "engine": self.engine_status(),
            "incident": branch_context,
            "recoveryChecks": recovery["result"]["checks"],
        }
        self._write_json(output_dir / "manifest.json", manifest)
        summary = SimulationSummary(
            runId=run_id,
            scenarioId=incident.scenario_id,
            label=(
                "等待环推演 | 受控倒退"
                if mode is WhatIfMode.CONTROLLED_REVERSE
                else "等待环推演 | 安全停车"
            ),
            policy="deterministic_recovery",
            seed=int(source_input.get("seed", 0)),
            status="COMPLETED",
            durationMs=round((time.perf_counter() - started) * 1000, 3),
            metrics=metrics,
            planning={
                "waitGraphCycleCount": recovery_metrics.get("waitGraphCycleCount", 0),
                "maxWaitGraphCycleLength": recovery_metrics.get(
                    "maxWaitGraphCycleLength", 0
                ),
                "recoverySuccessCount": recovery_metrics.get("recoverySuccessCount", 0),
                "reverseDistanceM": recovery_metrics.get("reverseDistanceM", 0),
            },
            safety={
                "conflictFree": True,
                "simulationOnly": True,
                "incidentId": incident.incident_id,
                "incidentType": incident.incident_type.value,
                "sourceRunId": incident.run_id,
                "whatIfMode": mode.value,
                "selectedAction": action,
                "maspReasonCode": decision["reasonCode"],
                "recoveryPlanId": plan.get("id") if plan else None,
                "requiresApproval": True,
                "branchModel": branch_context["branchModel"],
            },
            manifestPath=str(output_dir / "manifest.json"),
        )
        self._write_json(
            output_dir / "command-center-summary.json",
            summary.model_dump(by_alias=True, mode="json"),
        )
        return summary

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
        detail = {
            "summary": self._read_json(root / "command-center-summary.json"),
            "scenario": self._read_json(root / "planned-scenario.json"),
            "result": self._read_json(root / "result.json"),
            "planning": self._read_json(root / "planning-summary.json"),
        }
        evidence_path = root / "agent-policy-evidence.json"
        if evidence_path.exists():
            detail["agentEvidence"] = self._read_json(evidence_path)
        return detail

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
