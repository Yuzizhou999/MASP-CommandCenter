from __future__ import annotations

from dataclasses import replace
from typing import Any

from .agent_protocol import AgentObservation
from .contracts import DiagnosisReport, IncidentRecord, PlanExplanationReport
from .diagnosis import deterministic_diagnosis
from .model_registry import model_registration
from .provider import (
    AgentDecisionResult,
    AgentToolPlan,
    DeepSeekProvider,
    PlannedToolCall,
)
from .settings import Settings


class OpenAICompatibleLocalProvider(DeepSeekProvider):
    """Use a domain-tuned local API for intent parsing or Agent actions."""

    def __init__(self, settings: Settings) -> None:
        adapted = replace(
            settings,
            deepseek_api_key=settings.local_llm_api_key or "local",
            deepseek_base_url=settings.local_llm_base_url,
            deepseek_model=settings.local_llm_model,
            deepseek_timeout_seconds=settings.local_llm_timeout_seconds,
            deepseek_input_cost_per_million=0,
            deepseek_output_cost_per_million=0,
        )
        super().__init__(adapted)
        self.runtime_settings = settings

    def _response_format(self, name: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": schema,
            },
        }

    def plan_context_tools(
        self,
        text: str,
        tool_definitions: list[dict[str, Any]],
        *,
        has_memory: bool = False,
    ) -> AgentToolPlan:
        del tool_definitions
        calls = [PlannedToolCall(name="get_world_snapshot", arguments={})]
        if has_memory:
            calls.append(
                PlannedToolCall(name="recall_conversation_memory", arguments={})
            )
        calls.append(
            PlannedToolCall(name="search_sop", arguments={"query": text, "limit": 2})
        )
        return AgentToolPlan(
            calls=tuple(calls),
            strategy="DETERMINISTIC_POLICY",
            model="deterministic-tool-policy",
        )

    def decide_agent_action(
        self,
        text: str,
        tool_definitions: list[dict[str, Any]],
        *,
        observations: list[AgentObservation],
        authoritative_parameters: dict[str, Any],
        action_history: list[dict[str, Any]] | None = None,
    ) -> AgentDecisionResult:
        return self._decide_agent_action(
            text,
            tool_definitions,
            observations=observations,
            authoritative_parameters=authoritative_parameters,
            action_history=action_history,
            native_tools=False,
        )

    def diagnose_incident(self, incident: IncidentRecord) -> DiagnosisReport:
        self._mark_fallback()
        return deterministic_diagnosis(
            incident, model="deterministic-local-intent-only"
        )

    def explain_plan(
        self, deterministic: PlanExplanationReport
    ) -> PlanExplanationReport:
        self._mark_fallback()
        return deterministic.model_copy(
            update={"model": "deterministic-local-intent-only"}
        )

    def status(self) -> dict[str, Any]:
        status = super().status()
        status.update(
            {
                "provider": "local-openai-compatible",
                "mode": "local-api" if self.configured else "deterministic-fallback",
                # configured 只说明 base_url 和 key 有值，而本地 provider 的 key
                # 默认是字面量 "local"，所以它恒为 True，说明不了模型服务是否真的
                # 在监听。observedMode 用实际请求结果回答这个问题，避免 health
                # 在模型不可达时仍然报告 local-api。
                "observedMode": self._observed_mode(),
                "capability": "dispatch-intent-and-agent-actions",
                "agentCapability": "single-action-protocol",
                "registration": model_registration(
                    self.runtime_settings.local_llm_model_card
                ),
            }
        )
        return status

    def _observed_mode(self) -> str:
        telemetry = self.telemetry()
        if not telemetry.get("requestCount"):
            return "unverified"
        if telemetry.get("successCount"):
            return "local-api"
        return "deterministic-fallback"


def create_llm_provider(settings: Settings) -> DeepSeekProvider:
    provider = settings.llm_provider
    use_local = provider == "local" or (
        provider == "auto" and settings.local_llm_enabled
    )
    if use_local:
        return OpenAICompatibleLocalProvider(settings)
    return DeepSeekProvider(settings)
