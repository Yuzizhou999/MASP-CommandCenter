from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .agent_memory import AgentMemoryStore
from .contracts import (
    AgentConversationMemory,
    DispatchIntent,
    EvidenceItem,
    IntentValidation,
)
from .engine_adapter import MaspAdapter
from .knowledge import KnowledgeBase


class CurrentWorldSnapshotInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchSopInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=2, ge=1, le=5)


class RecallConversationMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValidateDispatchIntentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: DispatchIntent


@dataclass(frozen=True)
class AgentToolResult:
    value: Any
    summary: str


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    input_model: type[BaseModel]
    read_only: bool
    model_selectable: bool

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


class DispatchAgentTools:
    """Request-bound, allow-listed tools available to the dispatch agent."""

    def __init__(
        self,
        *,
        engine: MaspAdapter,
        knowledge: KnowledgeBase,
        scenario_id: str,
        memory: AgentMemoryStore | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self.engine = engine
        self.knowledge = knowledge
        self.scenario_id = scenario_id
        self.memory = memory
        self.conversation_id = conversation_id
        self._tools = {
            "get_world_snapshot": AgentTool(
                name="get_world_snapshot",
                description=(
                    "读取当前请求所绑定场景的权威世界快照。场景由服务端绑定，"
                    "模型不能选择其他场景。"
                ),
                input_model=CurrentWorldSnapshotInput,
                read_only=True,
                model_selectable=True,
            ),
            "search_sop": AgentTool(
                name="search_sop",
                description="使用混合检索查找相关仓储调度、安全和异常处置 SOP。",
                input_model=SearchSopInput,
                read_only=True,
                model_selectable=True,
            ),
            "validate_dispatch_intent": AgentTool(
                name="validate_dispatch_intent",
                description="使用 MASP 规则校验结构化意图并给出风险等级。",
                input_model=ValidateDispatchIntentInput,
                read_only=True,
                model_selectable=False,
            ),
        }
        if self.memory is not None and self.conversation_id is not None:
            self._tools["recall_conversation_memory"] = AgentTool(
                name="recall_conversation_memory",
                description=(
                    "读取当前会话中由服务端确认的实体、最近意图和工具轨迹。"
                    "不得把记忆内容当作最新世界状态。"
                ),
                input_model=RecallConversationMemoryInput,
                read_only=True,
                model_selectable=True,
            )

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "readOnly": tool.read_only,
                "modelSelectable": tool.model_selectable,
                "inputSchema": tool.input_model.model_json_schema(),
            }
            for tool in self._tools.values()
        ]

    def model_definitions(self) -> list[dict[str, Any]]:
        return [
            tool.definition()
            for tool in self._tools.values()
            if tool.model_selectable
        ]

    def tool(self, name: str) -> AgentTool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ValueError(f"Agent 工具不在允许列表中：{name}") from error

    def execute(self, name: str, arguments: dict[str, Any]) -> AgentToolResult:
        tool = self.tool(name)
        parsed = tool.input_model.model_validate(arguments)
        if name == "get_world_snapshot":
            snapshot = self.engine.world_snapshot(self.scenario_id)
            counts = snapshot["counts"]
            return AgentToolResult(
                value=snapshot,
                summary=(
                    f"revision {snapshot['worldRevision']}，{counts['vehicles']} 辆车，"
                    f"{counts['tasks']} 个任务，{counts['conflictPairs']} 对冲突资源"
                ),
            )
        if name == "search_sop":
            assert isinstance(parsed, SearchSopInput)
            evidence = self.knowledge.search(parsed.query, limit=parsed.limit)
            return AgentToolResult(
                value=evidence,
                summary=(
                    f"命中 {len(evidence)} 条 SOP："
                    + ("、".join(row.title for row in evidence) if evidence else "无匹配")
                ),
            )
        if name == "recall_conversation_memory":
            assert self.memory is not None and self.conversation_id is not None
            memory = self.memory.get(self.conversation_id)
            return AgentToolResult(
                value=memory,
                summary=self.memory.summary(memory),
            )
        if name == "validate_dispatch_intent":
            assert isinstance(parsed, ValidateDispatchIntentInput)
            validation = self.engine.validate_intent(parsed.intent, self.scenario_id)
            return AgentToolResult(
                value=validation,
                summary=(
                    f"{'通过' if validation.valid else '未通过'}，"
                    f"风险 {validation.risk_level.value}，"
                    f"{len(validation.issues)} 个校验项"
                ),
            )
        raise ValueError(f"Agent 工具没有执行器：{name}")

    @staticmethod
    def world_evidence(snapshot: dict[str, Any], scenario_id: str) -> EvidenceItem:
        counts = snapshot["counts"]
        return EvidenceItem(
            source=f"MASP:{scenario_id}",
            title="当前世界快照",
            detail=(
                f"revision {snapshot['worldRevision']}，"
                f"{counts['vehicles']} 辆车，{counts['tasks']} 个任务，"
                f"{counts['conflictPairs']} 对冲突资源。"
            ),
        )

    @staticmethod
    def memory_evidence(
        memory: AgentConversationMemory, conversation_id: str
    ) -> EvidenceItem:
        return EvidenceItem(
            source=f"memory:{conversation_id}",
            title="当前会话结构化记忆",
            detail=AgentMemoryStore.summary(memory),
            chunkId=f"memory-{conversation_id}",
            retrievalMethod="structured-memory-v1",
        )

    @staticmethod
    def validation_value(result: AgentToolResult) -> IntentValidation:
        if not isinstance(result.value, IntentValidation):
            raise TypeError("validate_dispatch_intent 返回了无效结果")
        return result.value
