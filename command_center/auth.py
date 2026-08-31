"""操作者身份与可选的 Bearer token 鉴权。

当前版本是仿真验证系统，默认零配置启动，因此鉴权是**可选**的：未配置
``COMMAND_CENTER_API_TOKEN`` 时所有接口保持开放，方便本机演示。

但"开放"不等于"可以假装有身份"。审批人身份此前完全由客户端提交
（``ApprovalDecision.decidedBy`` 默认 ``demo-supervisor``），任何能访问端口的
人都能以任意身份批准 R3 高风险操作，而审计日志会把这个自称的身份记成
``actor``。这让人工审批这道安全门形同虚设。

因此这里区分两种身份：

* ``authenticated=False``：演示模式，沿用客户端提交的名字，但审计事件会带上
  ``authenticated: false``，事后可以分辨哪些决策没有经过鉴权。
* ``authenticated=True``：配置了 token，服务端用 token 绑定的身份**覆盖**客户端
  提交的名字，客户端无法伪造审批人。

token 只从服务端环境变量读取，不进入前端资源，也不写入审计 payload。
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from .settings import Settings


@dataclass(frozen=True)
class OperatorIdentity:
    """一次请求的操作者身份。"""

    name: str
    authenticated: bool

    def resolve(self, requested_by: str | None) -> str:
        """决定审计与记录里使用的身份。

        已认证时一律使用 token 绑定的身份，忽略客户端提交值；未认证时沿用
        客户端提交值，保持演示体验不变。
        """
        if self.authenticated:
            return self.name
        candidate = (requested_by or "").strip()
        return candidate or self.name


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def operator_dependency(settings: Settings):
    """构造读取 ``Authorization`` 头的 FastAPI 依赖。"""

    expected = (settings.api_token or "").strip()
    bound_name = settings.api_token_operator

    def current_operator(
        authorization: str | None = Header(default=None),
    ) -> OperatorIdentity:
        if not expected:
            # 未配置 token：保持开放，但身份标记为未认证。
            return OperatorIdentity(name="demo-operator", authenticated=False)
        if not authorization:
            raise _unauthorized("缺少 Authorization: Bearer <token>。")
        scheme, _, presented = authorization.partition(" ")
        if scheme.lower() != "bearer" or not presented.strip():
            raise _unauthorized("Authorization 必须是 Bearer 方案。")
        # 定长比较，避免通过响应时间区分前缀是否命中。
        if not hmac.compare_digest(presented.strip(), expected):
            raise _unauthorized("token 无效。")
        return OperatorIdentity(name=bound_name, authenticated=True)

    return current_operator


def operator_param(settings: Settings):
    """``Depends(...)`` 形式的便捷封装，供路由签名直接使用。"""
    return Depends(operator_dependency(settings))


# 配置 token 后仍然保持开放的路径：健康探针和前端静态资源。
# 健康探针要能在未持有 token 时判断服务是否存活；静态资源本身不含机密，
# 密钥只由后端读取，不进入前端产物。
_OPEN_PREFIXES = ("/api/health", "/assets", "/docs", "/openapi.json", "/redoc")

# 只有变更类方法需要 token。读接口在当前仿真系统里不含机密数据，
# 保持开放可以让演示和排查不必先配密钥。
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def is_protected(method: str, path: str) -> bool:
    """判断一个请求是否需要 token。"""
    if method.upper() not in _MUTATING_METHODS:
        return False
    if path.startswith(_OPEN_PREFIXES):
        return False
    return path.startswith("/api/")


def token_matches(settings: Settings, authorization: str | None) -> bool:
    """校验 ``Authorization`` 头是否匹配配置的 token。"""
    expected = (settings.api_token or "").strip()
    if not expected:
        return True
    if not authorization:
        return False
    scheme, _, presented = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return False
    presented = presented.strip()
    if not presented:
        return False
    return hmac.compare_digest(presented, expected)
