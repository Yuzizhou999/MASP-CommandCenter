from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from command_center.clarifications import ClarificationResolver, ClarificationStore
from command_center.contracts import DispatchIntent, IntentType
from command_center.engine_adapter import MaspAdapter
from command_center.settings import Settings

TASK_AUTHORITY_FIELDS = (
    "pickupNodeId",
    "dropoffNodeId",
    "requiredRobotGroup",
    "payloadType",
)
RESOURCE_AUTHORITY_FIELDS = ("resourceIds", "startMs", "endMs")


def suite_sha256(suite: dict[str, Any]) -> str:
    encoded = json.dumps(
        suite, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def suite_quality(suite: dict[str, Any]) -> dict[str, Any]:
    cases = list(suite.get("cases") or [])
    design = suite.get("evaluationDesign") or {}
    counts = Counter(str(row.get("stratum") or row.get("category")) for row in cases)
    configured = bool(design)
    minimum_cases = int(design.get("minimumCaseCount", 0))
    minimum_per_stratum = int(design.get("minimumCasesPerStratum", 0))
    required_strata = [str(value) for value in design.get("requiredStrata") or []]
    goal_rule = (
        (suite.get("qualification") or {})
        .get("trajectoryThresholds", {})
        .get("goalSuccessRate", {})
    )
    goal_target = float(goal_rule.get("value", 0.0))
    minimum_passing_count = math.ceil(goal_target * len(cases)) if cases else 0
    checks = {
        "minimumCaseCount": {
            "actual": len(cases),
            "minimum": minimum_cases,
            "passed": not configured or len(cases) >= minimum_cases,
        },
        "minimumCasesPerStratum": {
            "actual": min((counts.get(name, 0) for name in required_strata), default=0),
            "minimum": minimum_per_stratum,
            "passed": not configured
            or all(counts.get(name, 0) >= minimum_per_stratum for name in required_strata),
        },
        "requiredStrata": {
            "actual": sorted(counts),
            "required": sorted(required_strata),
            "passed": not configured or set(required_strata).issubset(counts),
        },
        "declaredCaseCount": {
            "actual": len(cases),
            "declared": design.get("caseCount"),
            "passed": not configured or design.get("caseCount") == len(cases),
        },
    }
    return {
        "configured": configured,
        "suiteSha256": suite_sha256(suite),
        "caseCount": len(cases),
        "singleCaseRateImpact": round(1 / len(cases), 6) if cases else None,
        "goalSuccessGate": {
            "target": goal_target,
            "minimumPassingCount": minimum_passing_count,
            "maximumAllowedFailures": len(cases) - minimum_passing_count,
        },
        "stratumCounts": dict(sorted(counts.items())),
        "checks": checks,
        "passed": all(row["passed"] for row in checks.values()),
    }


def expected_authority(case: dict[str, Any]) -> dict[str, Any] | None:
    parameters = case.get("authoritativeParameters") or {}
    if isinstance(parameters.get("task"), dict):
        return {
            "intentType": IntentType.CREATE_TASK.value,
            "task": {
                field: parameters["task"].get(field)
                for field in TASK_AUTHORITY_FIELDS
            },
        }
    if isinstance(parameters.get("resourceBlock"), dict):
        return {
            "intentType": IntentType.BLOCK_RESOURCE.value,
            "resourceBlock": {
                field: parameters["resourceBlock"].get(field)
                for field in RESOURCE_AUTHORITY_FIELDS
            },
        }
    return None


def observed_intent_authority(intent: DispatchIntent | None) -> dict[str, Any] | None:
    if intent is None:
        return None
    if intent.intent_type is IntentType.CREATE_TASK and intent.task is not None:
        task = intent.task.model_dump(by_alias=True, mode="json")
        return {
            "intentType": intent.intent_type.value,
            "task": {field: task.get(field) for field in TASK_AUTHORITY_FIELDS},
        }
    if (
        intent.intent_type is IntentType.BLOCK_RESOURCE
        and intent.resource_block is not None
    ):
        resource = intent.resource_block.model_dump(by_alias=True, mode="json")
        return {
            "intentType": intent.intent_type.value,
            "resourceBlock": {
                field: resource.get(field) for field in RESOURCE_AUTHORITY_FIELDS
            },
        }
    return {"intentType": intent.intent_type.value}


def authority_matches(
    expected: dict[str, Any] | None, observed: dict[str, Any] | None
) -> bool:
    return expected is None or expected == observed


def evaluate_reachability(
    cases: list[dict[str, Any]],
    resolver: ClarificationResolver,
    *,
    target: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        expected = expected_authority(case)
        resolved = resolver.resolve(
            str(case["message"]), f"preflight-{index}-{case['caseId']}"
        )
        observed: dict[str, Any] | None = None
        if resolved.task is not None:
            observed = {
                "intentType": IntentType.CREATE_TASK.value,
                "task": {
                    field: resolved.task.get(field) for field in TASK_AUTHORITY_FIELDS
                },
            }
        elif resolved.resource_block is not None:
            observed = {
                "intentType": IntentType.BLOCK_RESOURCE.value,
                "resourceBlock": {
                    field: resolved.resource_block.get(field)
                    for field in RESOURCE_AUTHORITY_FIELDS
                },
            }

        reachable = authority_matches(expected, observed)
        if expected is None:
            reason = "no-deterministic-authority-precondition"
        elif reachable:
            reason = "authoritative-parameters-resolved"
        elif resolved.clarification is not None:
            reason = "resolver-requested-clarification-for-ready-case"
        elif resolved.intent_type is None:
            reason = "resolver-did-not-recognize-authoritative-intent"
        else:
            reason = "resolver-authority-mismatch"
        rows.append(
            {
                "caseId": case["caseId"],
                "expectedTerminalState": case["expectedTerminalState"],
                "hardReachable": reachable,
                "reason": reason,
                "expectedAuthority": expected,
                "observedAuthority": observed,
            }
        )

    reachable_count = sum(row["hardReachable"] for row in rows)
    ceiling = reachable_count / max(1, len(rows))
    return {
        "caseCount": len(rows),
        "hardReachableCount": reachable_count,
        "maximumGoalSuccessRate": round(ceiling, 6),
        "requiredGoalSuccessRate": target,
        "passed": ceiling >= target,
        "blockedCases": [row for row in rows if not row["hardReachable"]],
        "cases": rows,
    }


def preflight_suite(
    suite: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    target = float(
        suite["qualification"]["trajectoryThresholds"]["goalSuccessRate"][
            "value"
        ]
    )
    engine = MaspAdapter(settings)
    with tempfile.TemporaryDirectory(prefix="masp-agent-system-preflight-") as root:
        resolver = ClarificationResolver(
            ClarificationStore(Path(root) / "clarifications.json"), engine
        )
        result = evaluate_reachability(
            list(suite["cases"]), resolver, target=target
        )
        result["suiteQuality"] = suite_quality(suite)
        return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练前检查 Agent 轨迹门槛是否受确定性解析器硬上限阻断"
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("evals/agent-trajectories-v2.1-holdout.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-unreachable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    suite = json.loads(args.suite.resolve().read_text(encoding="utf-8"))
    result = preflight_suite(suite, Settings.load())
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["passed"] and not args.allow_unreachable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
