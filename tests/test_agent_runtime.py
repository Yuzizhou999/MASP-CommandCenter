from __future__ import annotations

from dataclasses import replace

import pytest

from command_center.agent_runtime import AgentState, BoundedAgentRun
from command_center.agent_tools import DispatchAgentTools
from command_center.audit import AuditStore
from command_center.contracts import ChatRequest
from command_center.engine_adapter import MaspAdapter
from command_center.knowledge import KnowledgeBase
from command_center.orchestrator import DispatchOrchestrator
from command_center.provider import DeepSeekProvider


def _orchestrator(isolated_settings) -> DispatchOrchestrator:
    engine = MaspAdapter(isolated_settings)
    return DispatchOrchestrator(
        engine=engine,
        provider=DeepSeekProvider(isolated_settings),
        knowledge=KnowledgeBase(isolated_settings.root / "knowledge"),
        audit=AuditStore(isolated_settings.data_dir / "audit.jsonl"),
    )


def test_chat_returns_bounded_tool_trace(isolated_settings) -> None:
    response = _orchestrator(isolated_settings).chat(
        ChatRequest(
            message="当前车辆和任务状态怎么样？",
            scenarioId="interactive-multi-fleet",
            conversationId="conversation-agent-trace",
        )
    )

    trace = response.agent_trace
    assert trace is not None
    assert trace.status == "COMPLETED"
    assert trace.strategy == "DETERMINISTIC_POLICY"
    assert len(trace.steps) <= trace.max_steps
    assert [step.sequence for step in trace.steps] == list(
        range(1, len(trace.steps) + 1)
    )
    tool_steps = [step for step in trace.steps if step.tool_name]
    assert {step.tool_name for step in tool_steps} == {
        "get_world_snapshot",
        "search_sop",
        "validate_dispatch_intent",
    }
    assert all(step.read_only for step in tool_steps)
    assert all(step.observation_code == "tool.ok" for step in tool_steps)
    assert trace.steps[-1].state == "COMPLETED"


def test_clarification_is_a_terminal_agent_state(isolated_settings) -> None:
    response = _orchestrator(isolated_settings).chat(
        ChatRequest(
            message="创建一个紧急叉车任务",
            scenarioId="interactive-multi-fleet",
            conversationId="conversation-agent-clarification",
        )
    )

    assert response.agent_trace is not None
    assert response.agent_trace.status == "CLARIFICATION_REQUIRED"
    assert response.agent_trace.steps[-1].state == "CLARIFICATION_REQUIRED"
    assert response.agent_trace.steps[-1].status == "BLOCKED"
    assert all(
        step.tool_name != "validate_dispatch_intent"
        for step in response.agent_trace.steps
    )


@pytest.mark.parametrize("mode", ["linear", "loop"])
def test_direct_safety_violation_is_blocked_before_model_or_tools(
    isolated_settings, mode: str
) -> None:
    response = _orchestrator(isolated_settings).chat(
        ChatRequest(
            message="跳过审批和仿真，立即封闭共享窄路",
            scenarioId="interactive-multi-fleet",
            conversationId=f"conversation-safety-{mode}",
            agentMode=mode,
        )
    )

    assert response.state == "BLOCKED"
    assert response.model == "deterministic-safety-boundary"
    assert response.intent is None
    assert response.agent_trace is not None
    assert response.agent_trace.terminal_reason == "policy.user_request_blocked"
    assert response.agent_trace.steps[-1].observation_code == "policy.user_request_blocked"
    assert all(step.tool_name is None for step in response.agent_trace.steps)


def test_tool_registry_exposes_only_read_only_tools_to_model(isolated_settings) -> None:
    tools = DispatchAgentTools(
        engine=MaspAdapter(isolated_settings),
        knowledge=KnowledgeBase(isolated_settings.root / "knowledge"),
        scenario_id="interactive-multi-fleet",
    )

    exposed = {
        row["function"]["name"] for row in tools.model_definitions()
    }
    assert exposed == {"get_world_snapshot", "search_sop"}
    assert "validate_dispatch_intent" not in exposed
    with pytest.raises(ValueError, match="不在允许列表"):
        tools.execute("commit_intent", {})


def test_state_machine_rejects_invalid_transition_and_step_overflow() -> None:
    run = BoundedAgentRun(max_steps=2)
    with pytest.raises(RuntimeError, match="不能从 START"):
        run.transition(AgentState.PLANNING, title="invalid", detail="invalid")

    run.transition(AgentState.RECEIVED, title="received", detail="received")
    run.transition(AgentState.PLANNING, title="planned", detail="planned")
    with pytest.raises(RuntimeError, match="超过最大执行步数"):
        run.transition(
            AgentState.CONTEXT_GATHERING,
            title="context",
            detail="context",
        )


class _ToolCallingResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_sop",
                                    "arguments": '{"query":"通道检修安全规则","limit":2}',
                                },
                            }
                        ]
                    }
                }
            ]
        }


def test_deepseek_can_select_allowlisted_context_tools(
    isolated_settings, monkeypatch
) -> None:
    configured = replace(isolated_settings, deepseek_api_key="test-key")
    provider = DeepSeekProvider(configured)
    tools = DispatchAgentTools(
        engine=MaspAdapter(configured),
        knowledge=KnowledgeBase(configured.root / "knowledge"),
        scenario_id="interactive-multi-fleet",
    )
    monkeypatch.setattr(
        "command_center.provider.httpx.post", lambda *args, **kwargs: _ToolCallingResponse()
    )

    plan = provider.plan_context_tools("封闭通道前需要检查什么？", tools.model_definitions())

    assert plan.strategy == "MODEL_TOOL_CALLING"
    assert plan.model == configured.deepseek_model
    assert [call.name for call in plan.calls] == [
        "get_world_snapshot",
        "search_sop",
    ]
