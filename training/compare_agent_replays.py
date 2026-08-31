from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对两个 Agent 轨迹评测做 paired diff")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-mode", default="linear_v1")
    parser.add_argument("--candidate-mode", default="loop_local")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _paired_interval(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values) if values else 0.0
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "meanDifference": round(mean, 6),
        "stdDifference": round(std, 6),
        "ci95Low": round(mean - margin, 6),
        "ci95High": round(mean + margin, 6),
    }


def _rows(report: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    values = report.get("cases", {}).get(mode)
    if not isinstance(values, list) or not values:
        raise ValueError(f"评测报告不包含模式 {mode} 的逐 case 结果")
    return {str(row["caseId"]): row for row in values}


def _prompt_sha(report: dict[str, Any], mode: str) -> str | None:
    by_mode = report.get("promptSha256ByMode") or {}
    return by_mode.get(mode) or report.get("promptSha256")


def main() -> None:
    args = _arguments()
    baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate.read_text(encoding="utf-8"))
    if baseline_report["suiteId"] != candidate_report["suiteId"]:
        raise ValueError("paired replay 必须使用同一冻结 suite")
    baseline = _rows(baseline_report, args.baseline_mode)
    candidate = _rows(candidate_report, args.candidate_mode)
    if baseline.keys() != candidate.keys():
        raise ValueError("paired replay 的 caseId 集合不一致")

    pairs: list[dict[str, Any]] = []
    goal_differences: list[float] = []
    regressions: list[str] = []
    improvements: list[str] = []
    for case_id in sorted(baseline):
        left = baseline[case_id]
        right = candidate[case_id]
        difference = float(right["goalSuccess"]) - float(left["goalSuccess"])
        goal_differences.append(difference)
        if difference < 0:
            regressions.append(case_id)
        elif difference > 0:
            improvements.append(case_id)
        pairs.append(
            {
                "caseId": case_id,
                "goalDifference": difference,
                "baseline": {
                    key: left.get(key)
                    for key in (
                        "actualTerminalState",
                        "actualIntentType",
                        "actualTools",
                        "actionSequence",
                        "goalSuccess",
                        "stepCount",
                    )
                },
                "candidate": {
                    key: right.get(key)
                    for key in (
                        "actualTerminalState",
                        "actualIntentType",
                        "actualTools",
                        "actionSequence",
                        "goalSuccess",
                        "stepCount",
                    )
                },
            }
        )
    output = {
        "schemaVersion": 1,
        "suiteId": baseline_report["suiteId"],
        "generatedAt": datetime.now(UTC).isoformat(),
        "baseline": {
            "path": str(args.baseline),
            "mode": args.baseline_mode,
            "promptSha256": _prompt_sha(baseline_report, args.baseline_mode),
        },
        "candidate": {
            "path": str(args.candidate),
            "mode": args.candidate_mode,
            "promptSha256": _prompt_sha(candidate_report, args.candidate_mode),
        },
        "pairedGoalSuccess": _paired_interval(goal_differences),
        "improvements": improvements,
        "regressions": regressions,
        "noRegression": not regressions,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
