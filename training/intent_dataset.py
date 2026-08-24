from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from command_center.contracts import DispatchIntent, EvidenceItem, IntentType
from command_center.engine_adapter import MaspAdapter
from command_center.model_safety import enforce_intent_authority
from command_center.provider import intent_training_messages


@dataclass(frozen=True)
class IntentDatasetExample:
    messages: list[dict[str, str]]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"messages": self.messages, "metadata": self.metadata}


def stable_split(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "valid"
    return "train"


def canonical_assistant_payload(
    text: str,
    *,
    intent_type: IntentType,
    resolved_task: dict[str, Any] | None = None,
    resolved_resource_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "intentType": intent_type.value,
        "reason": text,
    }
    if intent_type is IntentType.CREATE_TASK:
        payload["task"] = {
            **dict(resolved_task or {}),
            "priorityClass": 3,
            "dueTimeMs": 300000,
        }
    elif intent_type is IntentType.BLOCK_RESOURCE:
        payload["resourceBlock"] = dict(resolved_resource_block or {})
    else:
        payload["query"] = text
    return payload


def build_example(
    *,
    example_id: str,
    text: str,
    scenario_id: str,
    world_revision: int,
    intent_type: IntentType,
    source: str,
    split_key: str,
    template_id: str,
    resolved_task: dict[str, Any] | None = None,
    resolved_resource_block: dict[str, Any] | None = None,
    evidence: list[EvidenceItem] | None = None,
) -> IntentDatasetExample:
    messages = intent_training_messages(
        text,
        world_revision=world_revision,
        requested_by="finetune-dataset",
        resolved_task=resolved_task,
        resolved_resource_block=resolved_resource_block,
        context_evidence=evidence,
    )
    assistant = canonical_assistant_payload(
        text,
        intent_type=intent_type,
        resolved_task=resolved_task,
        resolved_resource_block=resolved_resource_block,
    )
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(assistant, ensure_ascii=False, separators=(",", ":")),
        }
    )
    return IntentDatasetExample(
        messages=messages,
        metadata={
            "exampleId": example_id,
            "category": intent_type.value,
            "scenarioId": scenario_id,
            "source": source,
            "templateId": template_id,
            "split": stable_split(split_key),
            "authoritativeParameters": {
                "task": resolved_task,
                "resourceBlock": resolved_resource_block,
            },
        },
    )


def validate_example(
    example: dict[str, Any], engine: MaspAdapter | None = None
) -> dict[str, Any]:
    messages = example.get("messages")
    metadata = example.get("metadata")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("训练样本必须包含 system、user、assistant 三条消息")
    if [row.get("role") for row in messages] != ["system", "user", "assistant"]:
        raise ValueError("训练样本消息角色顺序无效")
    if not isinstance(metadata, dict):
        raise ValueError("训练样本缺少 metadata")

    request_payload = json.loads(messages[1]["content"])
    assistant_payload = json.loads(messages[2]["content"])
    authoritative = request_payload.get("authoritativeParameters") or {}
    resolved_task = authoritative.get("task")
    resolved_block = authoritative.get("resourceBlock")
    parsed = dict(assistant_payload)
    parsed["basedOnWorldRevision"] = int(request_payload["worldRevision"])
    parsed["requestedBy"] = str(request_payload["requestedBy"])
    parsed["environment"] = "simulation"
    if resolved_task is not None:
        parsed["intentType"] = IntentType.CREATE_TASK.value
        parsed["task"] = {**dict(parsed.get("task") or {}), **resolved_task}
    if resolved_block is not None:
        parsed["intentType"] = IntentType.BLOCK_RESOURCE.value
        parsed["resourceBlock"] = resolved_block
    intent = DispatchIntent.model_validate(parsed)
    enforce_intent_authority(
        intent,
        resolved_task=resolved_task,
        resolved_resource_block=resolved_block,
    )
    validation = None
    if engine is not None:
        validation = engine.validate_intent(intent, str(metadata["scenarioId"]))
        if not validation.valid:
            issues = "; ".join(row.message for row in validation.issues)
            raise ValueError(f"训练样本未通过 MASP 校验：{issues}")
    return {
        "intentType": intent.intent_type.value,
        "valid": validation.valid if validation is not None else True,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
