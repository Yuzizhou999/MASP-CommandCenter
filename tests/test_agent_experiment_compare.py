from __future__ import annotations

from training.compare_agent_experiment import compare_experiment


def _intent(schema_rate: float = 0.96) -> dict:
    return {
        "suiteId": "intent-challenge-v1",
        "metrics": {"rawSchemaValidRate": schema_rate},
    }


def _trajectory(successes: dict[str, int]) -> dict:
    cases = [
        {
            "caseId": case_id,
            "category": "target" if case_id == "A" else "other",
            "goalSuccess": success,
            "actualTerminalState": "READY" if success else "FAILED",
        }
        for case_id, success in successes.items()
    ]
    mean = sum(successes.values()) / len(successes)
    return {
        "suiteId": "agent-trajectories-v2.1-holdout",
        "systems": [
            {
                "mode": "loop_local",
                "metrics": {
                    "toolPrecision": {"mean": 1.0},
                    "toolRecall": {"mean": 1.0},
                    "clarificationAccuracy": {"mean": 1.0},
                    "boundaryInterceptionRecall": 1.0,
                    "systemExecutionAttackRate": 0.0,
                    "goalSuccessRate": {"mean": mean},
                },
            }
        ],
        "cases": {"loop_local": cases},
    }


def _spec() -> dict:
    return {
        "experimentId": "test",
        "sharedEvaluation": {
            "trajectorySuite": "evals/agent-trajectories-v2.1-holdout.json"
        },
        "reproductionCriteria": {
            "minimumRawSchemaValidRate": 0.94,
            "maximumGoalSuccessRegressions": 1,
            "maximumMetricDegradation": 0.5,
            "metricsWithOneCaseTolerance": [
                "toolPrecision",
                "toolRecall",
                "clarificationAccuracy",
            ],
            "minimumBoundaryInterceptionRecall": 0.5,
            "requiredSystemExecutionAttackRate": 0.0,
        },
        "directionalEvidenceCriteria": {
            "targetCaseIds": ["A"],
            "minimumTargetWins": 1,
            "maximumTargetRegressions": 0,
            "maximumNonTargetRegressions": 0,
            "requiredBoundaryInterceptionRecall": 1.0,
            "requiredSystemExecutionAttackRate": 0.0,
            "claim": "directional only",
        },
    }


def test_compare_experiment_reports_target_win_without_non_target_regression() -> None:
    report = compare_experiment(
        _spec(),
        existing_intent=_intent(),
        control_intent=_intent(),
        existing_trajectory=_trajectory({"A": 0, "B": 1}),
        control_trajectory=_trajectory({"A": 0, "B": 1}),
        candidate_trajectory=_trajectory({"A": 1, "B": 1}),
    )

    assert report["reproduction"]["passed"] is True
    assert report["directionalEvidence"]["passed"] is True
    assert report["directionalEvidence"]["targetWins"] == 1
    assert report["directionalEvidence"]["nonTargetRegressions"] == 0


def test_compare_experiment_rejects_non_target_regression() -> None:
    report = compare_experiment(
        _spec(),
        existing_intent=_intent(),
        control_intent=_intent(),
        existing_trajectory=_trajectory({"A": 0, "B": 1}),
        control_trajectory=_trajectory({"A": 0, "B": 1}),
        candidate_trajectory=_trajectory({"A": 1, "B": 0}),
    )

    assert report["directionalEvidence"]["passed"] is False
    assert report["directionalEvidence"]["nonTargetRegressions"] == 1
