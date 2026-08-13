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


def test_real_map_is_exposed() -> None:
    response = client.get("/api/v1/map")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["nodes"]) == 552
    assert len(payload["edges"]) == 1204
    assert payload["bounds"]["minX"] < payload["bounds"]["maxX"]
