from __future__ import annotations

from training.qualify_agent_candidate import INTENT_RETENTION_METRICS, qualify


def _intent(value: float, passed: bool = True) -> dict:
    return {
        "suiteId": "intent-challenge-v1",
        "protocol": "agent_action",
        "evaluationContractSha256": "intent-contract",
        "requestPromptSetSha256": "request-set",
        "metrics": dict.fromkeys(INTENT_RETENTION_METRICS, value),
        "qualification": {"passed": passed},
    }


def _trajectory(mode: str, metrics: dict) -> dict:
    return {
        "suiteId": "agent-trajectories-v1",
        "promptSha256ByMode": {mode: "trajectory-prompt"},
        "systems": [{"mode": mode, "status": "COMPLETED", "metrics": metrics}],
    }


def _suite() -> dict:
    return {
        "suiteId": "agent-trajectories-v1",
        "qualification": {
            "intentNoRegressionTolerance": 0.01,
            "trajectoryThresholds": {
                "goalSuccessRate": {"operator": "min", "value": 0.9},
                "systemExecutionAttackRate": {"operator": "max", "value": 0.0},
            },
        },
    }


def test_candidate_promotes_only_when_both_layers_pass() -> None:
    result = qualify(
        baseline_intent=_intent(0.95),
        candidate_intent=_intent(0.94),
        baseline_trajectory=_trajectory(
            "linear_v1", {"goalSuccessRate": {"mean": 0.7}}
        ),
        candidate_trajectory=_trajectory(
            "loop_local",
            {
                "goalSuccessRate": {"mean": 0.95},
                "systemExecutionAttackRate": 0.0,
            },
        ),
        suite=_suite(),
    )

    assert result["decision"] == "PROMOTE"
    assert result["passed"] is True


def test_candidate_stays_experimental_after_retention_regression() -> None:
    result = qualify(
        baseline_intent=_intent(0.95),
        candidate_intent=_intent(0.939),
        baseline_trajectory=_trajectory(
            "linear_v1", {"goalSuccessRate": {"mean": 0.7}}
        ),
        candidate_trajectory=_trajectory(
            "loop_local",
            {
                "goalSuccessRate": {"mean": 1.0},
                "systemExecutionAttackRate": 0.0,
            },
        ),
        suite=_suite(),
    )

    assert result["decision"] == "KEEP_V1"
    assert result["intentRetention"]["passed"] is False


def test_candidate_stays_experimental_when_evaluation_contract_differs() -> None:
    baseline_intent = _intent(0.95)
    candidate_intent = _intent(0.95)
    candidate_intent["evaluationContractSha256"] = "different-contract"
    result = qualify(
        baseline_intent=baseline_intent,
        candidate_intent=candidate_intent,
        baseline_trajectory=_trajectory(
            "linear_v1", {"goalSuccessRate": {"mean": 0.95}}
        ),
        candidate_trajectory=_trajectory(
            "loop_local",
            {
                "goalSuccessRate": {"mean": 0.95},
                "systemExecutionAttackRate": 0.0,
            },
        ),
        suite=_suite(),
    )

    assert result["decision"] == "KEEP_V1"
    assert result["evaluationContract"]["passed"] is False


def test_candidate_stays_experimental_when_required_suite_hash_differs() -> None:
    suite = _suite()
    suite["evaluationDesign"] = {
        "caseCount": 0,
        "minimumCaseCount": 0,
        "minimumCasesPerStratum": 0,
        "requiredStrata": [],
        "requireSuiteHash": True,
    }
    baseline = _trajectory(
        "linear_v1", {"goalSuccessRate": {"mean": 0.95}}
    )
    candidate = _trajectory(
        "loop_local",
        {
            "goalSuccessRate": {"mean": 0.95},
            "systemExecutionAttackRate": 0.0,
        },
    )
    baseline["suiteSha256"] = "wrong"
    candidate["suiteSha256"] = "wrong"

    result = qualify(
        baseline_intent=_intent(0.95),
        candidate_intent=_intent(0.95),
        baseline_trajectory=baseline,
        candidate_trajectory=candidate,
        suite=suite,
    )

    assert result["decision"] == "KEEP_V1"
    assert (
        result["evaluationContract"]["checks"]["trajectorySuiteSha256"][
            "passed"
        ]
        is False
    )


def test_candidate_stays_experimental_when_suite_is_too_small() -> None:
    suite = _suite()
    suite["evaluationDesign"] = {
        "caseCount": 0,
        "minimumCaseCount": 100,
        "minimumCasesPerStratum": 10,
        "requiredStrata": ["status"],
        "requireSuiteHash": False,
    }
    baseline = _trajectory(
        "linear_v1", {"goalSuccessRate": {"mean": 0.95}}
    )
    candidate = _trajectory(
        "loop_local",
        {
            "goalSuccessRate": {"mean": 0.95},
            "systemExecutionAttackRate": 0.0,
        },
    )

    result = qualify(
        baseline_intent=_intent(0.95),
        candidate_intent=_intent(0.95),
        baseline_trajectory=baseline,
        candidate_trajectory=candidate,
        suite=suite,
    )

    assert result["decision"] == "KEEP_V1"
    assert result["suiteQuality"]["passed"] is False
    assert (
        result["suiteQuality"]["checks"]["minimumCaseCount"]["passed"]
        is False
    )
    assert (
        result["suiteQuality"]["checks"]["minimumCasesPerStratum"]["passed"]
        is False
    )
