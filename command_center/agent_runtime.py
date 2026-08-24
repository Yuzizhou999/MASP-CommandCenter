from __future__ import annotations

from enum import Enum
from time import perf_counter
from typing import Any

from .agent_tools import AgentToolResult, DispatchAgentTools
from .contracts import AgentExecutionTrace, AgentTraceStep


class AgentState(str, Enum):
    RECEIVED = "RECEIVED"
    PLANNING = "PLANNING"
    CONTEXT_GATHERING = "CONTEXT_GATHERING"
    PARAMETER_RESOLUTION = "PARAMETER_RESOLUTION"
    INTENT_DRAFTING = "INTENT_DRAFTING"
    SAFETY_VALIDATION = "SAFETY_VALIDATION"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    COMPLETED = "COMPLETED"


_TRANSITIONS: dict[AgentState | None, set[AgentState]] = {
    None: {AgentState.RECEIVED},
    AgentState.RECEIVED: {AgentState.PLANNING},
    AgentState.PLANNING: {AgentState.CONTEXT_GATHERING},
    AgentState.CONTEXT_GATHERING: {AgentState.PARAMETER_RESOLUTION},
    AgentState.PARAMETER_RESOLUTION: {
        AgentState.INTENT_DRAFTING,
        AgentState.CLARIFICATION_REQUIRED,
    },
    AgentState.INTENT_DRAFTING: {AgentState.SAFETY_VALIDATION},
    AgentState.SAFETY_VALIDATION: {AgentState.COMPLETED},
    AgentState.CLARIFICATION_REQUIRED: set(),
    AgentState.COMPLETED: set(),
}


class BoundedAgentRun:
    def __init__(self, *, max_steps: int = 16) -> None:
        self.max_steps = max_steps
        self.current_state: AgentState | None = None
        self.steps: list[AgentTraceStep] = []
        self.strategy = "DETERMINISTIC_POLICY"
        self.planner_model = "deterministic-tool-policy"
        self._started = perf_counter()

    def set_planner(self, *, strategy: str, model: str) -> None:
        if strategy not in {"MODEL_TOOL_CALLING", "DETERMINISTIC_POLICY"}:
            raise ValueError(f"未知 Agent 规划策略：{strategy}")
        self.strategy = strategy
        self.planner_model = model

    def transition(
        self,
        state: AgentState,
        *,
        title: str,
        detail: str,
        duration_ms: float = 0,
        status: str = "COMPLETED",
    ) -> None:
        allowed = _TRANSITIONS[self.current_state]
        if state not in allowed:
            current = self.current_state.value if self.current_state else "START"
            raise RuntimeError(f"Agent 状态不能从 {current} 转移到 {state.value}")
        self.current_state = state
        self._append(
            state=state,
            title=title,
            detail=detail,
            duration_ms=duration_ms,
            status=status,
        )

    def execute_tool(
        self,
        tools: DispatchAgentTools,
        name: str,
        arguments: dict[str, Any],
    ) -> AgentToolResult:
        if self.current_state not in {
            AgentState.CONTEXT_GATHERING,
            AgentState.SAFETY_VALIDATION,
        }:
            raise RuntimeError(f"当前状态不允许调用 Agent 工具：{self.current_state}")
        tool = tools.tool(name)
        started = perf_counter()
        try:
            result = tools.execute(name, arguments)
        except Exception as error:
            self._append(
                state=self.current_state,
                title=f"工具失败：{name}",
                detail=str(error),
                duration_ms=(perf_counter() - started) * 1000,
                status="FAILED",
                tool_name=name,
                read_only=tool.read_only,
            )
            raise
        self._append(
            state=self.current_state,
            title=f"调用工具：{name}",
            detail=result.summary,
            duration_ms=(perf_counter() - started) * 1000,
            tool_name=name,
            read_only=tool.read_only,
        )
        return result

    def build_trace(self) -> AgentExecutionTrace:
        if self.current_state is AgentState.COMPLETED:
            status = "COMPLETED"
        elif self.current_state is AgentState.CLARIFICATION_REQUIRED:
            status = "CLARIFICATION_REQUIRED"
        else:
            status = "FAILED"
        return AgentExecutionTrace(
            strategy=self.strategy,
            plannerModel=self.planner_model,
            status=status,
            maxSteps=self.max_steps,
            durationMs=round((perf_counter() - self._started) * 1000, 3),
            steps=self.steps,
        )

    def _append(
        self,
        *,
        state: AgentState,
        title: str,
        detail: str,
        duration_ms: float,
        status: str = "COMPLETED",
        tool_name: str | None = None,
        read_only: bool | None = None,
    ) -> None:
        if len(self.steps) >= self.max_steps:
            raise RuntimeError(f"Agent 超过最大执行步数 {self.max_steps}")
        self.steps.append(
            AgentTraceStep(
                sequence=len(self.steps) + 1,
                state=state.value,
                status=status,
                title=title,
                detail=detail,
                toolName=tool_name,
                readOnly=read_only,
                durationMs=round(max(0, duration_ms), 3),
            )
        )
