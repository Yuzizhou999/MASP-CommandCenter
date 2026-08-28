from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.preflight_agent_system import suite_quality, suite_sha256


INTENT_RETENTION_METRICS = (
    "requestSuccessRate",
    "rawJsonValidRate",
    "rawSchemaValidRate",
    "rawIntentMacroF1",
    "rawExactMatchRate",
    "rawSlotExactMatchRate",
    "rawMaspValidRate",
    "systemSafetyGateRecall",
    "clarificationAccuracy",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据冻结意图与轨迹门槛判定 Agent adapter 是否晋级"
    )
    parser.add_argument("--baseline-intent", type=Path, required=True)
    parser.add_argument("--candidate-intent", type=Path, required=True)
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--candidate-trajectory", type=Path, required=True)
    parser.add_argument(
        "--suite", type=Path, default=Path("evals/agent-trajectories-v1.json")
    )
    parser.add_argument("--baseline-mode", default="linear_v1")
    parser.add_argument("--candidate-mode", default="loop_local")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"报告必须是 JSON 对象：{path}")
    return value


def _trajectory_system(report: dict[str, Any], mode: str) -> dict[str, Any]:
    for system in report.get("systems") or []:
        if system.get("mode") == mode:
            if system.get("status") != "COMPLETED":
                raise ValueError(f"轨迹模式 {mode} 未完成：{system.get('status')}")
            return system
    raise ValueError(f"轨迹报告不包含模式 {mode}")


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    return float(value)


def _trajectory_prompt_sha(report: dict[str, Any], mode: str) -> str | None:
    by_mode = report.get("promptSha256ByMode") or {}
    value = by_mode.get(mode) or report.get("promptSha256")
    return str(value) if value else None


def _same_contract_value(
    baseline: dict[str, Any], candidate: dict[str, Any], key: str
) -> dict[str, Any]:
    baseline_value = baseline.get(key)
    candidate_value = candidate.get(key)
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "passed": bool(
            baseline_value
            and candidate_value
            and baseline_value == candidate_value
        ),
    }


def qualify(
    *,
    baseline_intent: dict[str, Any],
    candidate_intent: dict[str, Any],
    baseline_trajectory: dict[str, Any],
    candidate_trajectory: dict[str, Any],
    suite: dict[str, Any],
    baseline_mode: str = "linear_v1",
    candidate_mode: str = "loop_local",
) -> dict[str, Any]:
    if baseline_intent.get("suiteId") != candidate_intent.get("suiteId"):
        raise ValueError("意图无退化对照必须使用同一冻结 suite")
    if baseline_trajectory.get("suiteId") != candidate_trajectory.get("suiteId"):
        raise ValueError("轨迹对照必须使用同一冻结 suite")
    if candidate_trajectory.get("suiteId") != suite.get("suiteId"):
        raise ValueError("轨迹报告与门槛 suiteId 不一致")

    quality = suite_quality(suite)
    require_suite_hash = bool(
        (suite.get("evaluationDesign") or {}).get("requireSuiteHash")
    )
    expected_suite_sha = suite_sha256(suite)

    contract_checks = {
        "intentProtocol": _same_contract_value(
            baseline_intent, candidate_intent, "protocol"
        ),
        "intentEvaluationContractSha256": _same_contract_value(
            baseline_intent, candidate_intent, "evaluationContractSha256"
        ),
        "intentRequestPromptSetSha256": _same_contract_value(
            baseline_intent, candidate_intent, "requestPromptSetSha256"
        ),
        "trajectoryPromptSha256": {
            "baseline": _trajectory_prompt_sha(
                baseline_trajectory, baseline_mode
            ),
            "candidate": _trajectory_prompt_sha(
                candidate_trajectory, candidate_mode
            ),
        },
    }
    if require_suite_hash:
        contract_checks["trajectorySuiteSha256"] = {
            "expected": expected_suite_sha,
            "baseline": baseline_trajectory.get("suiteSha256"),
            "candidate": candidate_trajectory.get("suiteSha256"),
            "passed": (
                baseline_trajectory.get("suiteSha256") == expected_suite_sha
                and candidate_trajectory.get("suiteSha256") == expected_suite_sha
            ),
        }
    trajectory_prompt = contract_checks["trajectoryPromptSha256"]
    trajectory_prompt["passed"] = bool(
        trajectory_prompt["baseline"]
        and trajectory_prompt["candidate"]
        and trajectory_prompt["baseline"] == trajectory_prompt["candidate"]
    )
    contract_passed = all(row["passed"] for row in contract_checks.values())

    qualification = suite.get("qualification") or {}
    tolerance = float(qualification.get("intentNoRegressionTolerance", 0.01))
    baseline_intent_metrics = baseline_intent.get("metrics") or {}
    candidate_intent_metrics = candidate_intent.get("metrics") or {}
    retention_checks: dict[str, dict[str, Any]] = {}
    for name in INTENT_RETENTION_METRICS:
        baseline_value = _metric_value(baseline_intent_metrics, name)
        candidate_value = _metric_value(candidate_intent_metrics, name)
        passed = (
            baseline_value is not None
            and candidate_value is not None
            and candidate_value + tolerance >= baseline_value
        )
        retention_checks[name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "minimumAllowed": (
                round(baseline_value - tolerance, 6)
                if baseline_value is not None
                else None
            ),
            "passed": passed,
        }

    candidate_native_qualification = bool(
        (candidate_intent.get("qualification") or {}).get("passed")
    )
    retention_passed = all(row["passed"] for row in retention_checks.values())

    baseline_system = _trajectory_system(baseline_trajectory, baseline_mode)
    candidate_system = _trajectory_system(candidate_trajectory, candidate_mode)
    baseline_metrics = baseline_system.get("metrics") or {}
    candidate_metrics = candidate_system.get("metrics") or {}
    trajectory_checks: dict[str, dict[str, Any]] = {}
    for name, rule in (qualification.get("trajectoryThresholds") or {}).items():
        actual = _metric_value(candidate_metrics, name)
        target = float(rule["value"])
        operator = str(rule["operator"])
        if operator == "min":
            passed = actual is not None and actual >= target
        elif operator == "max":
            passed = actual is not None and actual <= target
        else:
            raise ValueError(f"不支持的门槛操作符：{operator}")
        trajectory_checks[name] = {
            "operator": operator,
            "target": target,
            "actual": actual,
            "baseline": _metric_value(baseline_metrics, name),
            "passed": passed,
        }
    trajectory_passed = bool(trajectory_checks) and all(
        row["passed"] for row in trajectory_checks.values()
    )
    passed = bool(
        quality["passed"]
        and
        contract_passed
        and candidate_native_qualification
        and retention_passed
        and trajectory_passed
    )
    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "suiteId": suite["suiteId"],
        "decision": "PROMOTE" if passed else "KEEP_V1",
        "passed": passed,
        "suiteQuality": quality,
        "evaluationContract": {
            "checks": contract_checks,
            "passed": contract_passed,
        },
        "intentRetention": {
            "tolerance": tolerance,
            "candidateThresholdQualificationPassed": candidate_native_qualification,
            "checks": retention_checks,
            "passed": candidate_native_qualification and retention_passed,
        },
        "trajectoryQualification": {
            "baselineMode": baseline_mode,
            "candidateMode": candidate_mode,
            "checks": trajectory_checks,
            "passed": trajectory_passed,
        },
    }


def main() -> None:
    args = _arguments()
    result = qualify(
        baseline_intent=_load(args.baseline_intent),
        candidate_intent=_load(args.candidate_intent),
        baseline_trajectory=_load(args.baseline_trajectory),
        candidate_trajectory=_load(args.candidate_trajectory),
        suite=_load(args.suite),
        baseline_mode=args.baseline_mode,
        candidate_mode=args.candidate_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
