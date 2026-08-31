from __future__ import annotations

from fastapi.testclient import TestClient

from command_center.api import app

client = TestClient(app)


def test_health_exposes_model_and_safety_boundary() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"]["commitMatches"] is True
    assert payload["safety"] == {
        "mode": "simulation-only",
        "fieldExecutionEnabled": False,
        "approvalBoundaryEnabled": True,
    }
    assert payload["agentPolicy"]["modelId"] == "masp-ppo-priority"
    assert payload["agentPolicy"]["safetyController"].startswith("MASP Top-K")
    assert payload["agentRuntime"]["mode"] in {"linear", "loop"}
    assert payload["agentRuntime"]["strategy"] in {
        "LINEAR_PIPELINE",
        "ACTION_PROTOCOL_LOOP",
    }
    assert payload["agentRuntime"]["budgets"]["maxToolCalls"] >= 1
    assert payload["agentRuntime"]["budgets"]["maxTotalTokens"] >= 128


def test_agent_policy_status_does_not_expose_server_path() -> None:
    response = client.get("/api/v1/agent-policy")
    assert response.status_code == 200
    payload = response.json()
    assert "checkpointPath" not in payload
    assert payload["mode"] in {"LEARNED", "BASELINE"}
    assert payload["device"] == "cpu"


def test_real_map_is_exposed() -> None:
    response = client.get("/api/v1/map")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["nodes"]) == 552
    assert len(payload["edges"]) == 1204
    assert payload["bounds"]["minX"] < payload["bounds"]["maxX"]
