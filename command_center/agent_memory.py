from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from .contracts import (
    AgentConversationMemory,
    AgentExecutionTrace,
    AgentMemoryTurn,
    ClarificationRequest,
    DispatchIntent,
    IntentValidation,
    utc_now,
)


class AgentMemoryStore:
    """Persist structured, evidence-derived conversation memory only."""

    max_turns = 8

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def get(self, conversation_id: str) -> AgentConversationMemory | None:
        with self._lock:
            payload = self._load().get(conversation_id)
        if not isinstance(payload, dict):
            return None
        try:
            return AgentConversationMemory.model_validate(payload)
        except ValueError:
            return None

    def record(
        self,
        *,
        conversation_id: str,
        scenario_id: str,
        message: str,
        outcome: str,
        trace: AgentExecutionTrace,
        intent: DispatchIntent | None = None,
        validation: IntentValidation | None = None,
        clarification: ClarificationRequest | None = None,
    ) -> AgentConversationMemory:
        with self._lock:
            rows = self._load()
            previous_payload = rows.get(conversation_id)
            try:
                previous = (
                    AgentConversationMemory.model_validate(previous_payload)
                    if isinstance(previous_payload, dict)
                    else None
                )
            except ValueError:
                previous = None

            entities = {
                key: set(values)
                for key, values in (previous.confirmed_entities if previous else {}).items()
            }
            self._collect_entities(
                entities,
                intent=intent,
                clarification=clarification,
            )
            tool_names = list(
                dict.fromkeys(
                    step.tool_name for step in trace.steps if step.tool_name is not None
                )
            )
            turn = AgentMemoryTurn(
                message=message[:500],
                outcome=outcome,
                intentType=intent.intent_type.value if intent else None,
                riskLevel=validation.risk_level.value if validation else None,
                toolNames=tool_names,
            )
            turns = [*(previous.turns if previous else []), turn][-self.max_turns :]
            memory = AgentConversationMemory(
                conversationId=conversation_id,
                scenarioId=scenario_id,
                confirmedEntities={
                    key: sorted(values) for key, values in entities.items() if values
                },
                turns=turns,
                updatedAt=utc_now(),
            )
            rows[conversation_id] = memory.model_dump(by_alias=True, mode="json")
            self.path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return memory

    @staticmethod
    def _collect_entities(
        entities: dict[str, set[str]],
        *,
        intent: DispatchIntent | None,
        clarification: ClarificationRequest | None,
    ) -> None:
        def add(category: str, value: Any) -> None:
            if isinstance(value, str) and value:
                entities.setdefault(category, set()).add(value)

        if intent and intent.task:
            add("nodeIds", intent.task.pickup_node_id)
            add("nodeIds", intent.task.dropoff_node_id)
            add("robotGroups", intent.task.required_robot_group)
            add("taskIds", intent.task.task_id)
        if intent and intent.resource_block:
            for resource_id in intent.resource_block.resource_ids:
                add("resourceIds", resource_id)
        if clarification:
            for key, value in clarification.collected_parameters.items():
                category = {
                    "pickupNodeId": "nodeIds",
                    "dropoffNodeId": "nodeIds",
                    "requiredRobotGroup": "robotGroups",
                    "resourceIds": "resourceIds",
                }.get(key)
                if category is None:
                    continue
                if isinstance(value, list):
                    for item in value:
                        add(category, item)
                else:
                    add(category, value)

    @staticmethod
    def summary(memory: AgentConversationMemory | None) -> str:
        if memory is None:
            return "当前会话没有可召回的结构化记忆"
        entity_parts = [
            f"{key}={','.join(values)}"
            for key, values in sorted(memory.confirmed_entities.items())
            if values
        ]
        last_turn = memory.turns[-1] if memory.turns else None
        details = [f"已记录 {len(memory.turns)} 轮"]
        if entity_parts:
            details.append("确认实体 " + "；".join(entity_parts))
        if last_turn and last_turn.intent_type:
            details.append(f"最近意图 {last_turn.intent_type}")
        if last_turn and last_turn.tool_names:
            details.append("最近工具 " + "、".join(last_turn.tool_names))
        return "，".join(details)
