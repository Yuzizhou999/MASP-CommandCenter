from __future__ import annotations

from dataclasses import replace

from command_center.agent_protocol import AgentActionType, AgentObservation
from command_center.llm_provider import OpenAICompatibleLocalProvider
from command_center.provider import DeepSeekProvider


class _Response:
    def __init__(self, message: dict, usage: dict | None = None) -> None:
        self._message = message
        self._usage = usage or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": self._message}],
            "usage": self._usage,
        }


def _initial() -> list[AgentObservation]:
    return [
        AgentObservation(
            sequence=1,
            kind="INITIAL",
            code="request.received",
            summary="received",
            data={"hasMemory": False},
        )
    ]


def test_deterministic_driver_observes_before_next_action(isolated_settings) -> None:
    provider = DeepSeekProvider(isolated_settings)
    tools = [
        {"type": "function", "function": {"name": "get_world_snapshot"}},
        {"type": "function", "function": {"name": "search_sop"}},
    ]
    observations = _initial()

    first = provider.decide_agent_action(
        "查询状态",
        tools,
        observations=observations,
        authoritative_parameters={"task": None, "resourceBlock": None},
        action_history=[],
    )
    observations.append(
        AgentObservation(
            sequence=2,
            kind="TOOL_RESULT",
            code="tool.ok",
            summary="snapshot",
            toolName="get_world_snapshot",
            data={"value": {"worldRevision": 3}},
        )
    )
    second = provider.decide_agent_action(
        "查询状态",
        tools,
        observations=observations,
        authoritative_parameters={"task": None, "resourceBlock": None},
        action_history=[
            {"action": "CALL_TOOL", "tool": "get_world_snapshot", "arguments": {}}
        ],
    )

    assert first.action is not None and first.action.tool == "get_world_snapshot"
    assert second.action is not None and second.action.tool == "search_sop"


def test_deepseek_native_tool_call_is_normalized(isolated_settings, monkeypatch) -> None:
    provider = DeepSeekProvider(replace(isolated_settings, deepseek_api_key="key"))
    monkeypatch.setattr(
        provider,
        "_post",
        lambda **_: _Response(
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_sop",
                            "arguments": '{"query":"封路","limit":2}',
                        }
                    }
                ]
            },
            {"prompt_tokens": 10, "completion_tokens": 5},
        ),
    )

    decision = provider.decide_agent_action(
        "封路",
        [],
        observations=_initial(),
        authoritative_parameters={},
        action_history=[],
    )

    assert decision.action is not None
    assert decision.action.action is AgentActionType.CALL_TOOL
    assert decision.action.tool == "search_sop"
    assert decision.prompt_tokens + decision.completion_tokens == 15
    expected_cost = (
        10 * provider.settings.deepseek_input_cost_per_million
        + 5 * provider.settings.deepseek_output_cost_per_million
    ) / 1_000_000
    assert decision.estimated_cost_usd == expected_cost


def test_local_v1_intent_output_has_transitional_wrapper(
    isolated_settings, monkeypatch
) -> None:
    provider = OpenAICompatibleLocalProvider(
        replace(isolated_settings, local_llm_api_key="local")
    )
    monkeypatch.setattr(
        provider,
        "_post",
        lambda **_: _Response(
            {
                "content": '{"intentType":"QUERY_STATUS","reason":"查询","query":"查询"}'
            }
        ),
    )

    decision = provider.decide_agent_action(
        "查询状态",
        [],
        observations=_initial(),
        authoritative_parameters={},
        action_history=[],
    )

    assert decision.action is not None
    assert decision.action.action is AgentActionType.PROPOSE_INTENT


def test_multiple_native_tool_calls_are_rejected(isolated_settings, monkeypatch) -> None:
    provider = DeepSeekProvider(replace(isolated_settings, deepseek_api_key="key"))
    monkeypatch.setattr(
        provider,
        "_post",
        lambda **_: _Response(
            {
                "tool_calls": [
                    {"function": {"name": "a", "arguments": "{}"}},
                    {"function": {"name": "b", "arguments": "{}"}},
                ]
            }
        ),
    )

    decision = provider.decide_agent_action(
        "查询状态",
        [],
        observations=_initial(),
        authoritative_parameters={},
        action_history=[],
    )

    assert decision.action is None
    assert decision.error_code == "protocol.multiple_actions"
