from __future__ import annotations

from dataclasses import replace

import pytest

from command_center.auth import (
    is_protected,
    operator_dependency,
    token_matches,
)


def test_read_requests_are_never_protected() -> None:
    assert not is_protected("GET", "/api/v1/approvals")
    assert not is_protected("GET", "/api/v1/agent/runs/run-1")


def test_health_and_static_stay_open_for_mutations() -> None:
    assert not is_protected("POST", "/api/health")
    assert not is_protected("POST", "/assets/index.js")
    assert not is_protected("POST", "/openapi.json")


def test_mutating_api_requests_are_protected() -> None:
    assert is_protected("POST", "/api/v1/approvals/ap-1/decision")
    assert is_protected("POST", "/api/v1/agent/runs/run-1/resume")
    assert is_protected("PUT", "/api/v1/scenario-drafts/pkg-1")
    assert is_protected("DELETE", "/api/v1/intents/intent-1")


def test_token_check_passes_when_unconfigured(isolated_settings) -> None:
    assert token_matches(isolated_settings, None) is True


def test_token_check_requires_bearer_scheme(isolated_settings) -> None:
    settings = replace(isolated_settings, api_token="secret")

    assert token_matches(settings, None) is False
    assert token_matches(settings, "secret") is False
    assert token_matches(settings, "Basic secret") is False
    assert token_matches(settings, "Bearer ") is False
    assert token_matches(settings, "Bearer wrong") is False
    assert token_matches(settings, "Bearer secret") is True
    assert token_matches(settings, "bearer secret") is True


def test_unauthenticated_identity_keeps_client_name(isolated_settings) -> None:
    dependency = operator_dependency(isolated_settings)

    identity = dependency(authorization=None)

    assert identity.authenticated is False
    # 演示模式沿用客户端提交的名字，保持零配置体验不变。
    assert identity.resolve("alice") == "alice"
    assert identity.resolve(None) == "demo-operator"
    assert identity.resolve("   ") == "demo-operator"


def test_authenticated_identity_overrides_client_name(isolated_settings) -> None:
    settings = replace(
        isolated_settings,
        api_token="secret",
        api_token_operator="supervisor-a",
    )
    dependency = operator_dependency(settings)

    identity = dependency(authorization="Bearer secret")

    assert identity.authenticated is True
    # 已认证时客户端提交的审批人身份必须被忽略，不能伪造。
    assert identity.resolve("attacker") == "supervisor-a"
    assert identity.resolve(None) == "supervisor-a"


@pytest.mark.parametrize(
    "authorization",
    [None, "secret", "Bearer wrong", "Basic secret", "Bearer "],
)
def test_dependency_rejects_bad_credentials(isolated_settings, authorization) -> None:
    from fastapi import HTTPException

    settings = replace(isolated_settings, api_token="secret")
    dependency = operator_dependency(settings)

    with pytest.raises(HTTPException) as error:
        dependency(authorization=authorization)

    assert error.value.status_code == 401
    assert error.value.headers == {"WWW-Authenticate": "Bearer"}
