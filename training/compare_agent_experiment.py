from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _system(report: dict[str, Any], mode: str) -> dict[str, Any]:
    for row in report["systems"]:
        if row["mode"] == mode:
            return row
    raise ValueError(f"轨迹报告缺少 mode={mode}")


def _cases(report: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    return {row["caseId"]: row for row in report["cases"][mode]}


def _metric(system: dict[str, Any], name: str) -> float:
    value = system["metrics"][name]
    return float(value["mean"] if isinstance(value, dict) else value)


def _paired_rows(
    control: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    target_ids: set[str],
) -> list[dict[str, Any]]:
    if set(control) != set(candidate):
        raise ValueError("控制组与候选组的轨迹 caseId 不一致")
    rows = []
    for case_id in sorted(control):
        control_success = bool(control[case_id]["goalSuccess"])
        candidate_success = bool(candidate[case_id]["goalSuccess"])
        outcome = (
            "WIN"
            if candidate_success and not control_success
            else "REGRESSION"
            if control_success and not candidate_success
            else "TIE"
        )
        rows.append(
            {
                "caseId": case_id,
                "category": candidate[case_id]["category"],
                "target": case_id in target_ids,
                "controlGoalSuccess": control_success,
                "candidateGoalSuccess": candidate_success,
                "outcome": outcome,
                "controlTerminalState": control[case_id]["actualTerminalState"],
                "candidateTerminalState": candidate[case_id]["actualTerminalState"],
            }
        )
    return rows


def compare_experiment(
    spec: dict[str, Any],
    *,
    existing_intent: dict[str, Any],
    control_intent: dict[str, Any],
    existing_trajectory: dict[str, Any],
    control_trajectory: dict[str, Any],
    candidate_trajectory: dict[str, Any],
    mode: str = "loop_local",
) -> dict[str, Any]:
    expected_suite = spec["sharedEvaluation"]["trajectorySuite"].split("/")[-1]
    expected_suite_id = Path(expected_suite).stem
    trajectory_reports = (
        existing_trajectory,
        control_trajectory,
        candidate_trajectory,
    )
    suite_ids = {row["suiteId"] for row in trajectory_reports}
    if len(suite_ids) != 1:
        raise ValueError("三个轨迹报告不是同一套件")
    if existing_intent["suiteId"] != control_intent["suiteId"]:
        raise ValueError("现存 v2 与控制组的 intent challenge 不一致")

    criteria = spec["reproductionCriteria"]
    existing_system = _system(existing_trajectory, mode)
    control_system = _system(control_trajectory, mode)
    candidate_system = _system(candidate_trajectory, mode)
    existing_cases = _cases(existing_trajectory, mode)
    control_cases = _cases(control_trajectory, mode)
    if set(existing_cases) != set(control_cases):
        raise ValueError("现存 v2 与控制组的轨迹 caseId 不一致")
    target_ids = set(spec["directionalEvidenceCriteria"]["targetCaseIds"])
    paired = _paired_rows(
        control_cases,
        _cases(candidate_trajectory, mode),
        target_ids,
    )

    goal_regressions = sum(
        bool(existing_cases[case_id]["goalSuccess"])
        and not bool(control_cases[case_id]["goalSuccess"])
        for case_id in existing_cases
    )
    metric_checks = {}
    for name in criteria["metricsWithOneCaseTolerance"]:
        existing_value = _metric(existing_system, name)
        control_value = _metric(control_system, name)
        degradation = existing_value - control_value
        metric_checks[name] = {
            "existingV2": existing_value,
            "control": control_value,
            "degradation": round(degradation, 6),
            "passed": degradation <= float(criteria["maximumMetricDegradation"]),
        }
    reproduction_checks = {
        "rawSchemaValidRate": {
            "actual": float(control_intent["metrics"]["rawSchemaValidRate"]),
            "minimum": float(criteria["minimumRawSchemaValidRate"]),
            "passed": float(control_intent["metrics"]["rawSchemaValidRate"])
            >= float(criteria["minimumRawSchemaValidRate"]),
        },
        "goalSuccessRegressions": {
            "actual": goal_regressions,
            "maximum": int(criteria["maximumGoalSuccessRegressions"]),
            "passed": goal_regressions
            <= int(criteria["maximumGoalSuccessRegressions"]),
        },
        "boundaryInterceptionRecall": {
            "actual": _metric(control_system, "boundaryInterceptionRecall"),
            "minimum": float(criteria["minimumBoundaryInterceptionRecall"]),
            "passed": _metric(control_system, "boundaryInterceptionRecall")
            >= float(criteria["minimumBoundaryInterceptionRecall"]),
        },
        "systemExecutionAttackRate": {
            "actual": _metric(control_system, "systemExecutionAttackRate"),
            "required": float(criteria["requiredSystemExecutionAttackRate"]),
            "passed": _metric(control_system, "systemExecutionAttackRate")
            == float(criteria["requiredSystemExecutionAttackRate"]),
        },
        "pairedMetrics": {
            "checks": metric_checks,
            "passed": all(row["passed"] for row in metric_checks.values()),
        },
    }
    reproduction_passed = all(row["passed"] for row in reproduction_checks.values())

    target_rows = [row for row in paired if row["target"]]
    non_target_rows = [row for row in paired if not row["target"]]
    directional_criteria = spec["directionalEvidenceCriteria"]
    target_wins = sum(row["outcome"] == "WIN" for row in target_rows)
    target_regressions = sum(row["outcome"] == "REGRESSION" for row in target_rows)
    non_target_regressions = sum(
        row["outcome"] == "REGRESSION" for row in non_target_rows
    )
    candidate_attack_rate = _metric(candidate_system, "systemExecutionAttackRate")
    candidate_boundary_recall = _metric(candidate_system, "boundaryInterceptionRecall")
    directional_checks = {
        "targetWins": {
            "actual": target_wins,
            "minimum": int(directional_criteria["minimumTargetWins"]),
            "passed": target_wins >= int(directional_criteria["minimumTargetWins"]),
        },
        "targetRegressions": {
            "actual": target_regressions,
            "maximum": int(directional_criteria["maximumTargetRegressions"]),
            "passed": target_regressions
            <= int(directional_criteria["maximumTargetRegressions"]),
        },
        "nonTargetRegressions": {
            "actual": non_target_regressions,
            "maximum": int(directional_criteria["maximumNonTargetRegressions"]),
            "passed": non_target_regressions
            <= int(directional_criteria["maximumNonTargetRegressions"]),
        },
        "systemExecutionAttackRate": {
            "actual": candidate_attack_rate,
            "required": float(
                directional_criteria["requiredSystemExecutionAttackRate"]
            ),
            "passed": candidate_attack_rate
            == float(directional_criteria["requiredSystemExecutionAttackRate"]),
        },
        "boundaryInterceptionRecall": {
            "actual": candidate_boundary_recall,
            "required": float(
                directional_criteria["requiredBoundaryInterceptionRecall"]
            ),
            "passed": candidate_boundary_recall
            == float(directional_criteria["requiredBoundaryInterceptionRecall"]),
        },
    }
    directional_passed = all(row["passed"] for row in directional_checks.values())

    return {
        "schemaVersion": 1,
        "experimentId": spec["experimentId"],
        "generatedAt": datetime.now(UTC).isoformat(),
        "suiteId": next(iter(suite_ids)),
        "expectedSuiteFileStem": expected_suite_id,
        "reproduction": {
            "checks": reproduction_checks,
            "passed": reproduction_passed,
        },
        "directionalEvidence": {
            "targetSuccesses": sum(row["candidateGoalSuccess"] for row in target_rows),
            "targetCaseCount": len(target_rows),
            "targetWins": target_wins,
            "targetRegressions": target_regressions,
            "targetTies": len(target_rows) - target_wins - target_regressions,
            "nonTargetRegressions": non_target_regressions,
            "nonTargetCaseCount": len(non_target_rows),
            "checks": directional_checks,
            "passed": directional_passed,
            "claim": directional_criteria["claim"],
        },
        "pairedCases": paired,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Agent v2.2 同 case 配对报告")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--existing-intent", type=Path, required=True)
    parser.add_argument("--control-intent", type=Path, required=True)
    parser.add_argument("--existing-trajectory", type=Path, required=True)
    parser.add_argument("--control-trajectory", type=Path, required=True)
    parser.add_argument("--candidate-trajectory", type=Path, required=True)
    parser.add_argument("--mode", default="loop_local")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    report = compare_experiment(
        _read(args.spec),
        existing_intent=_read(args.existing_intent),
        control_intent=_read(args.control_intent),
        existing_trajectory=_read(args.existing_trajectory),
        control_trajectory=_read(args.control_trajectory),
        candidate_trajectory=_read(args.candidate_trajectory),
        mode=args.mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
