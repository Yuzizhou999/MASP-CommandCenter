from __future__ import annotations

from enum import Enum
from time import perf_counter
from typing import Any, Callable

from .agent_tools import AgentToolResult, DispatchAgentTools
from .contracts import AgentExecutionTrace, AgentTraceStep


class AgentState(str, Enum):
    RECEIVED = "RECEIVED"
    PLANNING = "PLANNING"
    DECIDING = "DECIDING"
    OBSERVING = "OBSERVING"
    CONTEXT_GATHERING = "CONTEXT_GATHERING"
    PARAMETER_RESOLUTION = "PARAMETER_RESOLUTION"
    INTENT_DRAFTING = "INTENT_DRAFTING"
    REPAIRING = "REPAIRING"
    SAFETY_VALIDATION = "SAFETY_VALIDATION"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    BLOCKED = "BLOCKED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    COMPLETED = "COMPLETED"


_TRANSITIONS: dict[AgentState | None, set[AgentState]] = {
    None: {AgentState.RECEIVED},
    AgentState.RECEIVED: {
        AgentState.PLANNING,
        AgentState.PARAMETER_RESOLUTION,
        AgentState.BLOCKED,
        AgentState.BUDGET_EXCEEDED,
    },
    AgentState.PLANNING: {
        AgentState.CONTEXT_GATHERING,
        AgentState.PARAMETER_RESOLUTION,
        AgentState.DECIDING,
        AgentState.BUDGET_EXCEEDED,
    },
    AgentState.DECIDING: {
        AgentState.CONTEXT_GATHERING,
        AgentState.INTENT_DRAFTING,
        AgentState.CLARIFICATION_REQUIRED,
        AgentState.BLOCKED,
        AgentState.BUDGET_EXCEEDED,
    },
    AgentState.OBSERVING: {AgentState.DECIDING, AgentState.BUDGET_EXCEEDED},
    AgentState.CONTEXT_GATHERING: {
        AgentState.CONTEXT_GATHERING,
        AgentState.OBSERVING,
        AgentState.PARAMETER_RESOLUTION,
        AgentState.DECIDING,
        AgentState.INTENT_DRAFTING,
        AgentState.BUDGET_EXCEEDED,
    },
    AgentState.PARAMETER_RESOLUTION: {
        AgentState.PLANNING,
        AgentState.DECIDING,
        AgentState.INTENT_DRAFTING,
        AgentState.CLARIFICATION_REQUIRED,
        AgentState.BUDGET_EXCEEDED,
    },
    AgentState.INTENT_DRAFTING: {
        AgentState.SAFETY_VALIDATION,
        AgentState.BUDGET_EXCEEDED,
    },
    AgentState.REPAIRING: {
        AgentState.DECIDING,
        AgentState.INTENT_DRAFTING,
        AgentState.BUDGET_EXCEEDED,
    },
    AgentState.SAFETY_VALIDATION: {
        AgentState.REPAIRING,
        AgentState.COMPLETED,
        AgentState.BLOCKED,
        AgentState.BUDGET_EXCEEDED,
    },
    AgentState.CLARIFICATION_REQUIRED: set(),
    AgentState.BLOCKED: set(),
    AgentState.BUDGET_EXCEEDED: set(),
    AgentState.COMPLETED: set(),
}


class AgentStepLimitExceeded(RuntimeError):
    code = "budget.steps"

    def __init__(self, max_steps: int) -> None:
        super().__init__(f"Agent 超过最大执行步数 {max_steps}")


class BoundedAgentRun:
    def __init__(
        self,
        *,
        max_steps: int = 16,
        reserve_terminal_step: bool = False,
        on_step: Callable[[AgentTraceStep], None] | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.reserve_terminal_step = reserve_terminal_step
        self.current_state: AgentState | None = None
        self.steps: list[AgentTraceStep] = []
        self.strategy = "DETERMINISTIC_POLICY"
        self.planner_model = "deterministic-tool-policy"
        self.budgets: dict[str, int | float] = {}
        self.usage: dict[str, int | float] = {}
        self.terminal_reason: str | None = None
        self._started = perf_counter()
        self._on_step = on_step

    def set_planner(self, *, strategy: str, model: str) -> None:
        if strategy not in {
            "MODEL_TOOL_CALLING",
            "DETERMINISTIC_POLICY",
            "ACTION_PROTOCOL_LOOP",
        }:
            raise ValueError(f"未知 Agent 规划策略：{strategy}")
        self.strategy = strategy
        self.planner_model = model

    def set_budget_summary(
        self,
        *,
        budgets: dict[str, int | float],
        usage: dict[str, int | float],
        terminal_reason: str | None = None,
    ) -> None:
        self.budgets = dict(budgets)
        self.usage = dict(usage)
        self.terminal_reason = terminal_reason

    def transition(
        self,
        state: AgentState,
        *,
        title: str,
        detail: str,
        duration_ms: float = 0,
        status: str = "COMPLETED",
        observation_code: str | None = None,
        attempt: int | None = None,
    ) -> None:
        allowed = _TRANSITIONS[self.current_state]
        if state not in allowed:
            current = self.current_state.value if self.current_state else "START"
            raise RuntimeError(f"Agent 状态不能从 {current} 转移到 {state.value}")
        self._append(
            state=state,
            title=title,
            detail=detail,
            duration_ms=duration_ms,
            status=status,
            observation_code=observation_code,
            attempt=attempt,
        )
        self.current_state = state

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
        elif self.current_state is AgentState.BLOCKED:
            status = "BLOCKED"
        elif self.current_state is AgentState.BUDGET_EXCEEDED:
            status = "BUDGET_EXCEEDED"
        else:
            status = "FAILED"
        return AgentExecutionTrace(
            strategy=self.strategy,
            plannerModel=self.planner_model,
            status=status,
            maxSteps=self.max_steps,
            durationMs=round((perf_counter() - self._started) * 1000, 3),
            budgets=self.budgets,
            usage=self.usage,
            terminalReason=self.terminal_reason,
            steps=self.steps,
        )

    def record(
        self,
        *,
        title: str,
        detail: str,
        state: AgentState | None = None,
        status: str = "COMPLETED",
        tool_name: str | None = None,
        read_only: bool | None = None,
        action: str | None = None,
        observation_code: str | None = None,
        attempt: int | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_ms: float = 0,
    ) -> None:
        target = state or self.current_state
        if target is None:
            raise RuntimeError("Agent 尚未进入任何状态")
        self._append(
            state=target,
            title=title,
            detail=detail,
            duration_ms=duration_ms,
            status=status,
            tool_name=tool_name,
            read_only=read_only,
            action=action,
            observation_code=observation_code,
            attempt=attempt,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
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
        action: str | None = None,
        observation_code: str | None = None,
        attempt: int | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        terminal_states = {
            AgentState.CLARIFICATION_REQUIRED,
            AgentState.BLOCKED,
            AgentState.BUDGET_EXCEEDED,
            AgentState.COMPLETED,
        }
        if len(self.steps) >= self.max_steps or (
            self.reserve_terminal_step
            and state not in terminal_states
            and len(self.steps) >= self.max_steps - 1
        ):
            raise AgentStepLimitExceeded(self.max_steps)
        step = AgentTraceStep(
            sequence=len(self.steps) + 1,
            state=state.value,
            status=status,
            title=title,
            detail=detail,
            toolName=tool_name,
            readOnly=read_only,
            action=action,
            observationCode=observation_code,
            attempt=attempt,
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            durationMs=round(max(0, duration_ms), 3),
        )
        self.steps.append(step)
        if self._on_step is not None:
            self._on_step(step)
