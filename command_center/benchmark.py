from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from .audit import AuditStore
from .contracts import BenchmarkRequest, new_id
from .engine_adapter import MaspAdapter

ARRIVAL_TASKS_PER_VEHICLE = {
    "low": 0.75,
    "medium": 1.5,
    "high": 3.0,
}

METRIC_PATHS = {
    "completedTaskCount": ("metrics", "completedTaskCount"),
    "completedDropoffsPerHour": ("metrics", "completedDropoffsPerHour"),
    "meanTaskCycleTimeMs": ("metrics", "meanTaskCycleTimeMs"),
    "meanTaskQueueTimeMs": ("metrics", "meanTaskQueueTimeMs"),
    "planningLatencyP95Ms": ("planning", "planningLatencyMs", "p95"),
    "planningTimeoutCount": ("planning", "planningTimeoutCount"),
    "planningPeriodMissCount": ("planning", "planningPeriodMissCount"),
    "reservationConflictRejections": (
        "safety",
        "reservationConflictRejections",
    ),
    "unplannedTaskCount": ("safety", "unplannedTaskCount"),
    "wallClockDurationMs": ("durationMs",),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_digest(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_path(document: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class BenchmarkScenarioFactory:
    """Build deterministic, scalable scenarios from approved MASP fixtures."""

    def __init__(self, engine: MaspAdapter, base_scenario_id: str) -> None:
        source = engine.benchmark_source(base_scenario_id)
        self.base = source["scenario"]
        self.nodes = source["nodes"]
        self.tasks_by_group = {
            group: [
                row
                for row in self.base["tasks"]
                if row["requiredRobotGroup"] == group
            ]
            for group in ("fork", "jack")
        }
        for group, tasks in self.tasks_by_group.items():
            if not tasks:
                raise ValueError(
                    f"base scenario {base_scenario_id!r} has no {group} task templates"
                )

    @staticmethod
    def _group_counts(vehicle_count: int, fleet_mix: str) -> dict[str, int]:
        if fleet_mix == "fork":
            return {"fork": vehicle_count, "jack": 0}
        if fleet_mix == "jack":
            return {"fork": 0, "jack": vehicle_count}
        if vehicle_count == 1:
            return {"fork": 1, "jack": 0}
        fork_count = min(vehicle_count - 1, max(1, round(vehicle_count * 3 / 7)))
        return {"fork": fork_count, "jack": vehicle_count - fork_count}

    def _initial_nodes(
        self,
        group: str,
        count: int,
        seed: int,
        used_node_ids: set[str],
        used_resources: set[str],
    ) -> list[dict[str, Any]]:
        candidates = [
            node
            for node in self.nodes
            if group in node.get("allowedRobotGroups", [])
            and node.get("type") in {"PP", "CP", "LM"}
        ]
        type_order = {"PP": 0, "CP": 1, "LM": 2}
        candidates.sort(key=lambda row: (type_order.get(row.get("type"), 9), row["id"]))
        preferred = [row for row in candidates if row.get("type") in {"PP", "CP"}]
        remaining = [row for row in candidates if row.get("type") == "LM"]
        rng = random.Random(f"benchmark-nodes:{seed}:{group}:{count}")
        rng.shuffle(preferred)
        rng.shuffle(remaining)

        selected: list[dict[str, Any]] = []
        for node in [*preferred, *remaining]:
            resources = set(node.get("resourceIds", []))
            if node["id"] in used_node_ids or resources & used_resources:
                continue
            selected.append(node)
            used_node_ids.add(node["id"])
            used_resources.update(resources)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(
                f"warehouse model has only {len(selected)} independent {group} start nodes; "
                f"{count} requested"
            )
        return selected

    @staticmethod
    def _release_times(count: int, horizon_ms: int, rng: random.Random) -> list[int]:
        if count <= 1:
            return [0] * count
        cumulative = [0.0]
        for _ in range(1, count):
            cumulative.append(cumulative[-1] + rng.expovariate(1.0))
        scale = (horizon_ms * 0.8) / max(cumulative[-1], 1.0)
        return [int(value * scale) for value in cumulative]

    def build(
        self,
        *,
        vehicle_count: int,
        arrival_profile: str,
        fleet_mix: str,
        seed: int,
        horizon_ms: int,
    ) -> dict[str, Any]:
        group_counts = self._group_counts(vehicle_count, fleet_mix)
        vehicles: list[dict[str, Any]] = []
        used_node_ids: set[str] = set()
        used_resources: set[str] = set()
        for group in ("fork", "jack"):
            nodes = self._initial_nodes(
                group,
                group_counts[group],
                seed,
                used_node_ids,
                used_resources,
            )
            for index, node in enumerate(nodes, start=1):
                vehicles.append(
                    {
                        "vehicleId": f"{group}-benchmark-{index:03d}",
                        "robotGroup": group,
                        "initialNodeId": node["id"],
                        "initialHeadingRad": float(
                            node.get("headings", {}).get(group, 0.0)
                        ),
                        "initialLoadState": "empty",
                    }
                )

        total_tasks = max(
            1,
            math.ceil(
                vehicle_count * ARRIVAL_TASKS_PER_VEHICLE[arrival_profile]
            ),
        )
        active_groups = [group for group, count in group_counts.items() if count]
        task_counts: dict[str, int] = {}
        remaining_tasks = total_tasks
        for index, group in enumerate(active_groups):
            if index == len(active_groups) - 1:
                task_counts[group] = remaining_tasks
            else:
                group_tasks = max(
                    1,
                    round(total_tasks * group_counts[group] / vehicle_count),
                )
                task_counts[group] = group_tasks
                remaining_tasks -= group_tasks

        tasks: list[dict[str, Any]] = []
        for group in active_groups:
            rng = random.Random(
                f"benchmark-tasks:{seed}:{group}:{arrival_profile}:{vehicle_count}"
            )
            release_times = self._release_times(
                task_counts[group], horizon_ms, rng
            )
            templates = self.tasks_by_group[group]
            offset = rng.randrange(len(templates))
            for index, release_time in enumerate(release_times, start=1):
                template = templates[(offset + index - 1) % len(templates)]
                task = {
                    key: value
                    for key, value in template.items()
                    if key
                    not in {
                        "taskId",
                        "payloadId",
                        "releaseTimeMs",
                        "dueTimeMs",
                        "state",
                        "assignedVehicleId",
                    }
                }
                task["taskId"] = f"benchmark-{group}-{index:04d}"
                task["payloadId"] = f"benchmark-payload-{group}-{index:04d}"
                task["releaseTimeMs"] = release_time
                if index % 10 == 0:
                    task["priorityClass"] = 0
                template_due = template.get("dueTimeMs")
                due_delta = max(
                    180000,
                    int(
                        template_due
                        if template_due is not None
                        else int(template.get("releaseTimeMs", 0)) + horizon_ms // 2
                    )
                    - int(template.get("releaseTimeMs", 0)),
                )
                task["dueTimeMs"] = release_time + due_delta
                tasks.append(task)

        tasks.sort(key=lambda row: (row["releaseTimeMs"], row["taskId"]))
        return {
            "schemaVersion": 1,
            "scenarioId": (
                f"benchmark-{vehicle_count}-{arrival_profile}-{fleet_mix}-s{seed}"
            ),
            "seed": seed,
            "endTimeMs": horizon_ms,
            "vehicles": vehicles,
            "tasks": tasks,
        }


class BenchmarkRunner:
    def __init__(
        self, root: Path, engine: MaspAdapter, audit: AuditStore | None = None
    ) -> None:
        self.root = root / "evaluations"
        self.engine = engine
        self.audit = audit
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path: Path, document: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _safe_id(benchmark_id: str) -> str:
        if not re.fullmatch(r"benchmark-[a-z0-9]+", benchmark_id):
            raise ValueError("invalid benchmarkId")
        return benchmark_id

    @staticmethod
    def _case_summary(
        case_id: str,
        scenario: dict[str, Any],
        policy: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        planning = result.get("planning", {})
        return {
            "caseId": case_id,
            "scenarioId": scenario["scenarioId"],
            "scenarioDigest": _canonical_digest(scenario),
            "vehicleCount": len(scenario["vehicles"]),
            "taskCount": len(scenario["tasks"]),
            "policy": policy,
            "seed": scenario["seed"],
            "status": result["status"],
            "durationMs": result.get("durationMs"),
            "metrics": result.get("metrics", {}),
            "planning": {
                key: planning.get(key)
                for key in (
                    "plannedTaskCount",
                    "unplannedTaskCount",
                    "planningLatencyMs",
                    "planningTimeoutCount",
                    "planningPeriodMissCount",
                    "reservationConflictRejections",
                    "rlInferenceCount",
                    "rlFallbackCount",
                    "rlSafetyFallbackCount",
                    "rlGuardianOverrideCount",
                )
            },
            "safety": result.get("safety", {}),
            "agentPolicy": result.get("agentPolicy"),
            "resultDigest": result.get("resultDigest"),
            "error": result.get("error"),
        }

    @staticmethod
    def _statistics(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "count": 0,
                "mean": None,
                "stddev": None,
                "ci95Low": None,
                "ci95High": None,
            }
        mean = statistics.fmean(values)
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        margin = 1.96 * stddev / math.sqrt(len(values))
        return {
            "count": len(values),
            "mean": round(mean, 6),
            "stddev": round(stddev, 6),
            "ci95Low": round(mean - margin, 6),
            "ci95High": round(mean + margin, 6),
        }

    def _aggregate(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in cases:
            key = (
                row["vehicleCount"],
                row["arrivalProfile"],
                row["fleetMix"],
                row["policy"],
            )
            grouped.setdefault(key, []).append(row)
        aggregates = []
        for key, rows in sorted(grouped.items()):
            successful = [row for row in rows if row["status"] == "COMPLETED"]
            metrics = {
                name: self._statistics(
                    [
                        value
                        for row in successful
                        if (value := _read_path(row, path)) is not None
                    ]
                )
                for name, path in METRIC_PATHS.items()
            }
            aggregates.append(
                {
                    "vehicleCount": key[0],
                    "arrivalProfile": key[1],
                    "fleetMix": key[2],
                    "policy": key[3],
                    "caseCount": len(rows),
                    "successfulCaseCount": len(successful),
                    "failedCaseCount": len(rows) - len(successful),
                    "metrics": metrics,
                }
            )
        return aggregates

    @staticmethod
    def _markdown(report: dict[str, Any]) -> str:
        lines = [
            f"# {report['suiteName']}",
            "",
            f"- 评测编号：`{report['benchmarkId']}`",
            f"- 生成时间：{report['createdAt']}",
            f"- 用例：{report['completedCaseCount']}/{report['caseCount']} 完成",
            f"- 安全门槛：{'通过' if report['safetyGate']['passed'] else '未通过'}",
            "",
            "| 车辆 | 到达强度 | 车型 | 策略 | 成功/总数 | 完成任务均值 | 吞吐均值 | 周期均值(ms) | 冲突拒绝 |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in report["aggregates"]:
            metric = row["metrics"]

            def mean(name: str, metric: dict[str, Any] = metric) -> str:
                value = metric[name]["mean"]
                return "-" if value is None else f"{value:.3f}"

            lines.append(
                "| {vehicleCount} | {arrivalProfile} | {fleetMix} | {policy} | "
                "{successfulCaseCount}/{caseCount} | {completed} | {throughput} | "
                "{cycle} | {conflicts} |".format(
                    **row,
                    completed=mean("completedTaskCount"),
                    throughput=mean("completedDropoffsPerHour"),
                    cycle=mean("meanTaskCycleTimeMs"),
                    conflicts=mean("reservationConflictRejections"),
                )
            )
        lines.extend(
            [
                "",
                "## 安全门槛",
                "",
                f"- 资源冲突失败用例：{report['safetyGate']['conflictCaseCount']}",
                f"- 规划超时用例：{report['safetyGate']['planningTimeoutCaseCount']}",
                f"- 仿真失败用例：{report['safetyGate']['failedCaseCount']}",
                "",
                "报告数据来自固定输入、策略、种子和锁定的 MASP 引擎；置信区间使用正态近似。",
            ]
        )
        return "\n".join(lines) + "\n"

    def run(self, request: BenchmarkRequest) -> dict[str, Any]:
        benchmark_id = new_id("benchmark")
        output_dir = self.root / benchmark_id
        inputs_dir = output_dir / "inputs"
        results_dir = output_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=False)
        self._write_json(
            output_dir / "request.json",
            request.model_dump(by_alias=True, mode="json"),
        )
        factory = BenchmarkScenarioFactory(self.engine, request.base_scenario_id)
        cases: list[dict[str, Any]] = []
        started = time.perf_counter()
        matrix = product(
            request.vehicle_counts,
            request.arrival_profiles,
            request.fleet_mixes,
            request.policies,
            request.seeds,
        )
        for vehicle_count, arrival_profile, fleet_mix, policy, seed in matrix:
            case_id = (
                f"v{vehicle_count}-{arrival_profile}-{fleet_mix}-{policy}-s{seed}"
            )
            scenario: dict[str, Any] = {
                "scenarioId": case_id,
                "vehicles": [],
                "tasks": [],
                "seed": seed,
            }
            try:
                scenario = factory.build(
                    vehicle_count=vehicle_count,
                    arrival_profile=arrival_profile,
                    fleet_mix=fleet_mix,
                    seed=seed,
                    horizon_ms=request.horizon_ms,
                )
                self._write_json(inputs_dir / f"{case_id}.json", scenario)
                result = self.engine.evaluate_scenario_document(
                    scenario,
                    policy=policy,
                    seed=seed,
                    agent_policy=request.agent_policy if policy == "rl" else None,
                )
            except Exception as error:
                result = {
                    "status": "FAILED",
                    "policy": policy,
                    "seed": seed,
                    "durationMs": None,
                    "metrics": {},
                    "planning": {},
                    "safety": {
                        "conflictFree": False,
                        "simulationOnly": True,
                    },
                    "error": f"{type(error).__name__}: {error}",
                }
            summary = self._case_summary(case_id, scenario, policy, result)
            summary["vehicleCount"] = vehicle_count
            summary["arrivalProfile"] = arrival_profile
            summary["fleetMix"] = fleet_mix
            cases.append(summary)
            self._write_json(results_dir / f"{case_id}.json", summary)

        failed = [row for row in cases if row["status"] != "COMPLETED"]
        conflict_cases = [
            row
            for row in cases
            if float(row.get("safety", {}).get("reservationConflictRejections", 0))
            > 0
        ]
        timeout_cases = [
            row
            for row in cases
            if float(row.get("planning", {}).get("planningTimeoutCount") or 0) > 0
        ]
        report = {
            "schemaVersion": 1,
            "benchmarkId": benchmark_id,
            "suiteName": request.suite_name,
            "status": "COMPLETED_WITH_FAILURES" if failed else "COMPLETED",
            "createdAt": _utc_now(),
            "durationMs": round((time.perf_counter() - started) * 1000, 3),
            "caseCount": len(cases),
            "completedCaseCount": len(cases) - len(failed),
            "coverage": {
                "baseScenarioId": request.base_scenario_id,
                "vehicleCounts": request.vehicle_counts,
                "arrivalProfiles": request.arrival_profiles,
                "fleetMixes": request.fleet_mixes,
                "policies": request.policies,
                "seeds": request.seeds,
                "horizonMs": request.horizon_ms,
            },
            "engine": self.engine.engine_status(),
            "safetyGate": {
                "passed": not conflict_cases and not timeout_cases and not failed,
                "conflictCaseCount": len(conflict_cases),
                "planningTimeoutCaseCount": len(timeout_cases),
                "failedCaseCount": len(failed),
                "fieldExecutionEnabled": False,
            },
            "aggregates": self._aggregate(cases),
            "failureCases": [
                {
                    "caseId": row["caseId"],
                    "error": row.get("error"),
                }
                for row in failed
            ],
            "cases": cases,
            "artifacts": {
                "directory": str(output_dir),
                "request": str(output_dir / "request.json"),
                "inputs": str(inputs_dir),
                "results": str(results_dir),
                "jsonReport": str(output_dir / "report.json"),
                "markdownReport": str(output_dir / "report.md"),
            },
        }
        self._write_json(output_dir / "report.json", report)
        (output_dir / "report.md").write_text(
            self._markdown(report), encoding="utf-8"
        )
        if self.audit is not None:
            self.audit.append(
                trace_id=new_id("trace"),
                event_type="BENCHMARK_COMPLETED",
                actor=request.requested_by,
                payload={
                    "benchmarkId": benchmark_id,
                    "caseCount": len(cases),
                    "safetyGate": report["safetyGate"],
                    "coverage": report["coverage"],
                },
            )
        return report

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in self.root.glob("benchmark-*/report.json"):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(
                {
                    key: report[key]
                    for key in (
                        "benchmarkId",
                        "suiteName",
                        "status",
                        "createdAt",
                        "durationMs",
                        "caseCount",
                        "completedCaseCount",
                        "coverage",
                        "safetyGate",
                    )
                }
            )
        return sorted(rows, key=lambda row: row["createdAt"], reverse=True)

    def get(self, benchmark_id: str) -> dict[str, Any]:
        path = self.root / self._safe_id(benchmark_id) / "report.json"
        if not path.exists():
            raise KeyError(benchmark_id)
        return json.loads(path.read_text(encoding="utf-8"))
