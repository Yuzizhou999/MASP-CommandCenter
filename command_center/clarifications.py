from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from .contracts import ClarificationRequest, IntentType, utc_now
from .engine_adapter import MaspAdapter

TASK_TERMS = (
    "创建",
    "新建",
    "新增",
    "插单",
    "安排",
    "送到",
    "送去",
    "送过去",
    "送往",
    "运到",
    "运至",
    "运输",
    "转运",
    "搬运",
    "移到",
    "挪到",
    "急货",
    "急活",
    "紧急任务",
)
BLOCK_TERMS = (
    "封闭",
    "封路",
    "检修",
    "停用",
    "禁行",
    "暂停",
    "暂时关闭",
    "临时关闭",
)
READ_ONLY_RESOURCE_QUERY_TERMS = (
    "哪些",
    "什么",
    "如何",
    "怎么",
    "为什么",
    "流程",
    "规范",
    "规则",
    "要求",
    "SOP",
    "sop",
)
EXPLICIT_BLOCK_COMMAND_TERMS = (
    "请封闭",
    "请封路",
    "请停用",
    "请禁行",
    "立即封闭",
    "立即封路",
    "立即停用",
    "执行封闭",
    "执行封路",
    "执行停用",
)
NEW_INTENT_TERMS = (
    "报告",
    "总结",
    "班次",
    "状态",
    "多少",
    "为什么",
    "原因",
    "解释",
)
NODE_PATTERN = re.compile(r"(?:(?:fork|jack):)?AP\d+", flags=re.IGNORECASE)
RESOURCE_PATTERN = re.compile(r"(?:zone|edge):[a-zA-Z0-9:_|.-]+")


@dataclass(frozen=True)
class ResolvedRequest:
    intent_type: IntentType | None
    message: str
    task: dict[str, Any] | None = None
    resource_block: dict[str, Any] | None = None
    clarification: ClarificationRequest | None = None


class ClarificationStore:
    """Persist only the minimum parameters needed to finish one intent."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._load().get(conversation_id)

    def put(self, conversation_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            rows = self._load()
            rows[conversation_id] = {**record, "updatedAt": utc_now().isoformat()}
            self.path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            rows = self._load()
            if rows.pop(conversation_id, None) is not None:
                self.path.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )


class ClarificationResolver:
    def __init__(self, store: ClarificationStore, engine: MaspAdapter) -> None:
        self.store = store
        self.engine = engine

    @staticmethod
    def _intent_type(message: str, pending: dict[str, Any] | None) -> IntentType | None:
        has_block_term = any(term in message for term in BLOCK_TERMS)
        is_explicit_block_command = any(
            term in message for term in EXPLICIT_BLOCK_COMMAND_TERMS
        ) or bool(re.search(r"(?:封闭|封路|停用|禁行).{0,8}\d+\s*(?:分钟|秒)", message))
        if (
            has_block_term
            and any(term in message for term in READ_ONLY_RESOURCE_QUERY_TERMS)
            and not is_explicit_block_command
        ):
            return None
        if has_block_term:
            return IntentType.BLOCK_RESOURCE
        if any(term in message for term in TASK_TERMS):
            return IntentType.CREATE_TASK
        if pending is not None and not any(term in message for term in NEW_INTENT_TERMS):
            return IntentType(pending["intentType"])
        return None

    @staticmethod
    def _combined(message: str, pending: dict[str, Any] | None) -> str:
        if pending is None:
            return message.strip()
        original = str(pending.get("combinedMessage", "")).strip()
        return f"{original}\n{message.strip()}" if original else message.strip()

    @staticmethod
    def _explicit_group(message: str) -> str | None:
        lowered = message.lower()
        fork = any(term in lowered for term in ("fork", "叉车", "托盘"))
        jack = any(term in lowered for term in ("jack", "搬运车", "顶升车", "料架"))
        if fork == jack:
            return None
        return "fork" if fork else "jack"

    @staticmethod
    def _field_reference(message: str, field: str) -> str | None:
        if field == "pickupNodeId":
            patterns = (
                rf"(?:从|把|将|起点(?:是|为)?|取货点(?:是|为)?)\s*({NODE_PATTERN.pattern})",
            )
        else:
            patterns = (
                rf"(?:搬运至|送到|运到|送往|运至|移到|挪到|终点(?:是|为)?|放货点(?:是|为)?|至|到)\s*({NODE_PATTERN.pattern})",
            )
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _task(self, conversation_id: str, combined: str) -> ResolvedRequest:
        explicit_group = self._explicit_group(combined)
        pickup_ref = self._field_reference(combined, "pickupNodeId")
        dropoff_ref = self._field_reference(combined, "dropoffNodeId")
        ordered_refs = list(dict.fromkeys(NODE_PATTERN.findall(combined)))
        if pickup_ref is None and dropoff_ref is None:
            pickup_ref = ordered_refs[0] if ordered_refs else None
            dropoff_ref = ordered_refs[1] if len(ordered_refs) > 1 else None
        elif pickup_ref is None:
            pickup_ref = next(
                (
                    value
                    for value in ordered_refs
                    if value.lower() != str(dropoff_ref).lower()
                ),
                None,
            )
        elif dropoff_ref is None:
            dropoff_ref = next(
                (
                    value
                    for value in ordered_refs
                    if value.lower() != str(pickup_ref).lower()
                ),
                None,
            )

        raw_refs = {
            "pickupNodeId": pickup_ref,
            "dropoffNodeId": dropoff_ref,
        }
        candidates = {
            field: self.engine.resolve_node_reference(value, explicit_group)
            if value
            else []
            for field, value in raw_refs.items()
        }
        candidate_groups = [
            {value.split(":", 1)[0] for value in values}
            for values in candidates.values()
            if values
        ]
        inferred_groups = set.intersection(*candidate_groups) if candidate_groups else set()
        group = explicit_group or (next(iter(inferred_groups)) if len(inferred_groups) == 1 else None)
        if group is not None:
            candidates = {
                field: self.engine.resolve_node_reference(value, group) if value else []
                for field, value in raw_refs.items()
            }

        missing: list[str] = []
        questions: list[str] = []
        code = "MISSING_REQUIRED_FIELDS"
        for field, label in (
            ("pickupNodeId", "取货站点"),
            ("dropoffNodeId", "放货站点"),
        ):
            raw = raw_refs[field]
            values = candidates[field]
            if raw is None:
                missing.append(field)
                questions.append(f"请提供{label}编号，例如 fork:AP1123。")
            elif not values:
                missing.append(field)
                questions.append(f"未找到与车型匹配的{label} {raw}，请核对编号或车型。")
                code = "AMBIGUOUS_ENTITY"
            elif len(values) > 1:
                missing.append(field)
                questions.append(f"{label} {raw} 对应多个车型，请明确使用叉车还是顶升车。")
                code = "AMBIGUOUS_ENTITY"
        if group is None:
            missing.append("requiredRobotGroup")
            questions.append("请明确执行车型：叉车（fork）或顶升车（jack）。")
            if candidate_groups:
                code = "AMBIGUOUS_ENTITY"

        collected = {
            key: values[0]
            for key, values in candidates.items()
            if len(values) == 1
        }
        if group:
            collected["requiredRobotGroup"] = group
        if missing:
            clarification = ClarificationRequest(
                code=code,
                missingFields=list(dict.fromkeys(missing)),
                questions=list(dict.fromkeys(questions)),
                collectedParameters=collected,
            )
            self.store.put(
                conversation_id,
                {
                    "intentType": IntentType.CREATE_TASK.value,
                    "combinedMessage": combined,
                    "collectedParameters": collected,
                },
            )
            return ResolvedRequest(
                intent_type=IntentType.CREATE_TASK,
                message=combined,
                clarification=clarification,
            )

        self.store.clear(conversation_id)
        return ResolvedRequest(
            intent_type=IntentType.CREATE_TASK,
            message=combined,
            task={
                "pickupNodeId": candidates["pickupNodeId"][0],
                "dropoffNodeId": candidates["dropoffNodeId"][0],
                "requiredRobotGroup": group,
                "payloadType": "shelf" if group == "jack" else "pallet",
            },
        )

    @staticmethod
    def _duration_ms(message: str) -> int:
        minute = re.search(r"(\d+)\s*分钟", message)
        if minute:
            return max(1, int(minute.group(1))) * 60000
        chinese_minute = re.search(r"([零〇一二两三四五六七八九十百]+)\s*分钟", message)
        if chinese_minute:
            return ClarificationResolver._chinese_number(
                chinese_minute.group(1)
            ) * 60000
        second = re.search(r"(\d+)\s*秒", message)
        if second:
            return max(1, int(second.group(1))) * 1000
        chinese_second = re.search(r"([零〇一二两三四五六七八九十百]+)\s*秒", message)
        if chinese_second:
            return ClarificationResolver._chinese_number(
                chinese_second.group(1)
            ) * 1000
        return 180000

    @staticmethod
    def _chinese_number(value: str) -> int:
        digits = {
            "零": 0,
            "〇": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if "百" in value:
            head, tail = value.split("百", 1)
            total = digits.get(head, 1) * 100
            return max(1, total + ClarificationResolver._chinese_number(tail))
        if "十" in value:
            head, tail = value.split("十", 1)
            total = digits.get(head, 1) * 10
            return max(1, total + digits.get(tail, 0))
        if not value:
            return 0
        return max(1, int("".join(str(digits[row]) for row in value)))

    def _block(self, conversation_id: str, combined: str) -> ResolvedRequest:
        resources = RESOURCE_PATTERN.findall(combined)
        if "共享窄路" in combined or "共享通道" in combined:
            resources.append("zone:zone-jack-pp363-pp365")
        resources = list(dict.fromkeys(resources))
        if not resources:
            clarification = ClarificationRequest(
                code="MISSING_REQUIRED_FIELDS",
                missingFields=["resourceIds"],
                questions=["请提供需要停用的通道、工位或资源编号，或在地图中选择目标。"],
                collectedParameters={"durationMs": self._duration_ms(combined)},
            )
            self.store.put(
                conversation_id,
                {
                    "intentType": IntentType.BLOCK_RESOURCE.value,
                    "combinedMessage": combined,
                    "collectedParameters": clarification.collected_parameters,
                },
            )
            return ResolvedRequest(
                intent_type=IntentType.BLOCK_RESOURCE,
                message=combined,
                clarification=clarification,
            )
        self.store.clear(conversation_id)
        return ResolvedRequest(
            intent_type=IntentType.BLOCK_RESOURCE,
            message=combined,
            resource_block={
                "resourceIds": resources,
                "startMs": 0,
                "endMs": self._duration_ms(combined),
                "reason": combined,
            },
        )

    def resolve(self, message: str, conversation_id: str) -> ResolvedRequest:
        pending = self.store.get(conversation_id)
        intent_type = self._intent_type(message, pending)
        combined = self._combined(message, pending if intent_type else None)
        if intent_type is IntentType.CREATE_TASK:
            return self._task(conversation_id, combined)
        if intent_type is IntentType.BLOCK_RESOURCE:
            return self._block(conversation_id, combined)
        if pending is not None:
            self.store.clear(conversation_id)
        return ResolvedRequest(intent_type=None, message=message.strip())
