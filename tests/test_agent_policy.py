from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from command_center.contracts import AgentPolicyOptions, SimulationRequest
from command_center.engine_adapter import MaspAdapter


def test_agent_options_are_rejected_for_rule_policy() -> None:
    with pytest.raises(ValidationError, match="only valid when policy is rl"):
        SimulationRequest(
            policy="top_k",
            agentPolicy=AgentPolicyOptions(candidateCount=2),
        )


def test_unregistered_agent_model_is_rejected(isolated_settings) -> None:
    engine = MaspAdapter(isolated_settings)
    with pytest.raises(ValueError, match="未登记智能体模型"):
        engine.simulate(
            SimulationRequest(
                policy="rl",
                agentPolicy=AgentPolicyOptions(modelId="unknown-policy"),
            )
        )


def test_missing_checkpoint_runs_auditable_rule_baseline(isolated_settings) -> None:
    engine = MaspAdapter(isolated_settings)
    status = engine.agent_model_status()
    assert status.mode == "BASELINE"
    assert status.checkpoint_present is False

    summary = engine.simulate(
        SimulationRequest(
            scenarioId="interactive-multi-fleet",
            label="智能体安全基线",
            policy="rl",
            agentPolicy=AgentPolicyOptions(candidateCount=3, allowDeviation=True),
        )
    )

    assert summary.status == "COMPLETED"
    assert summary.policy == "rl"
    assert summary.agent_policy is not None
    assert summary.agent_policy.mode == "BASELINE"
    assert summary.agent_policy.candidate_count == 3
    assert summary.agent_policy.deviation_requested is True
    assert summary.agent_policy.deviation_enabled is False
    assert summary.agent_policy.fallback_count == summary.agent_policy.decision_cycle_count
    assert any(
        "未配置" in reason for reason in summary.agent_policy.fallback_reasons
    )
    assert summary.safety["conflictFree"] is True

    evidence_path = isolated_settings.runs_dir / summary.run_id / "agent-policy-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["execution"]["mode"] == "BASELINE"
    assert evidence["safetyBoundary"]["deterministicValidationRequired"] is True
    assert evidence["safetyBoundary"]["fieldExecutionEnabled"] is False
    assert isinstance(evidence["decisionCycles"], list)

    detail = engine.get_run_detail(summary.run_id)
    assert detail["agentEvidence"]["runId"] == summary.run_id


def test_agent_runtime_controls_are_forwarded_to_masp(
    isolated_settings, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class FakeRuntime:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.now_ms = 0
            self.end_time_ms = int(kwargs["end_time_ms"])

    class FakeTopology:
        def __init__(self, *_args) -> None:
            pass

    class FakeVehicle:
        @staticmethod
        def from_dict(document):
            return document

    engine = MaspAdapter(isolated_settings)
    engine._modules = {
        "OnlineDispatchRuntime": FakeRuntime,
        "MapTopology": FakeTopology,
        "Vehicle": FakeVehicle,
    }
    monkeypatch.setattr(
        engine,
        "_assets",
        lambda: {
            "model": {},
            "conflicts": {},
            "workstations": {},
            "profiles": {},
            "zones": {},
            "scheduler": {
                "serviceDefaults": {
                    "pickupServiceMs": 1,
                    "dropoffServiceMs": 1,
                },
                "planner": {"planningPeriodMs": 5000},
            },
        },
    )
    checkpoint = isolated_settings.root / "models" / "policy.pt"
    prepared = {
        "checkpointPath": checkpoint,
        "candidateCount": 4,
        "deviationEnabled": True,
    }

    runtime = engine._run_scenario(
        {"tasks": [], "vehicles": [], "endTimeMs": 0, "seed": 17},
        policy="rl",
        resource_block=None,
        agent_runtime=prepared,
    )

    assert runtime.end_time_ms == 0
    assert captured["policy"] == "rl"
    assert captured["seed"] == 17
    assert captured["rl_checkpoint"] == str(checkpoint)
    assert captured["rl_candidate_count"] == 4
    assert captured["rl_allow_deviation"] is True
