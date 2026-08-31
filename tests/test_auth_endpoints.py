"""端到端验证 token 中间件真的挂在应用上。

单元测试只能证明 is_protected / token_matches 的判定正确，证明不了中间件被
正确注册。这里用 TestClient 走真实请求路径。

这些用例需要按 engine.lock.json 锁定的 MASP 检出才能导入 command_center.api，
因此归入 integration（见 tests/conftest.py 的 INTEGRATION_MODULES）。
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def token_client(monkeypatch) -> TestClient:
    """在配置了 token 的环境下重新加载 app。

    api.py 在模块导入时读取 Settings 并构造单例，所以必须先设环境变量再 reload。
    这本身就是 api.py 模块级副作用的代价，测试里只能这样处理。
    """
    monkeypatch.setenv("COMMAND_CENTER_API_TOKEN", "test-token")
    monkeypatch.setenv("COMMAND_CENTER_API_TOKEN_OPERATOR", "supervisor-under-test")
    import command_center.api as api_module
    import command_center.settings as settings_module

    importlib.reload(settings_module)
    reloaded = importlib.reload(api_module)
    try:
        with TestClient(reloaded.app) as client:
            yield client
    finally:
        # 还原全局模块状态，避免污染其他用例。
        monkeypatch.delenv("COMMAND_CENTER_API_TOKEN", raising=False)
        monkeypatch.delenv("COMMAND_CENTER_API_TOKEN_OPERATOR", raising=False)
        importlib.reload(settings_module)
        importlib.reload(api_module)


def test_health_reports_token_disabled_by_default() -> None:
    from command_center.api import app

    with TestClient(app) as client:
        payload = client.get("/api/health").json()

    assert payload["safety"]["apiTokenEnabled"] is False
    assert payload["safety"]["approverIdentityTrusted"] is False


def test_mutation_allowed_without_token_when_unconfigured() -> None:
    from command_center.api import app

    with TestClient(app) as client:
        # 未配置 token 时不应因为鉴权被拒；422/404 等业务错误都算通过。
        response = client.post("/api/v1/approvals/does-not-exist/decision", json={})

    assert response.status_code != 401


def test_health_stays_open_when_token_configured(token_client: TestClient) -> None:
    payload = token_client.get("/api/health").json()

    assert payload["safety"]["apiTokenEnabled"] is True
    assert payload["safety"]["approverIdentityTrusted"] is True


def test_read_requests_stay_open_when_token_configured(
    token_client: TestClient,
) -> None:
    assert token_client.get("/api/v1/approvals").status_code == 200


def test_mutation_without_token_is_rejected(token_client: TestClient) -> None:
    response = token_client.post(
        "/api/v1/approvals/any-id/decision",
        json={"approved": True, "decidedBy": "attacker", "reason": "伪造审批"},
    )

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_mutation_with_wrong_token_is_rejected(token_client: TestClient) -> None:
    response = token_client.post(
        "/api/v1/approvals/any-id/decision",
        json={"approved": True},
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401


def test_mutation_with_valid_token_passes_auth(token_client: TestClient) -> None:
    response = token_client.post(
        "/api/v1/approvals/does-not-exist/decision",
        json={"approved": True, "reason": "已核对仿真结果"},
        headers={"Authorization": "Bearer test-token"},
    )

    # 通过鉴权后应当落到业务逻辑（审批单不存在 → 404），而不是 401。
    assert response.status_code == 404
