from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .contracts import (
    DiagnosisReport,
    DispatchIntent,
    EvidenceItem,
    IntentType,
    PlanExplanationFinding,
)

MODEL_INTENT_TYPES = frozenset(
    {
        IntentType.QUERY_STATUS,
        IntentType.EXPLAIN_DECISION,
        IntentType.CREATE_TASK,
        IntentType.BLOCK_RESOURCE,
        IntentType.GENERATE_REPORT,
    }
)

FORBIDDEN_MODEL_REQUEST_TERMS = (
    "直接输出 REQUEST_RECOVERY",
    "直接控制",
    "跳过主管审批",
    "跳过审批",
    "解除安全停车",
    "直接把通道封锁写入",
    "写入资源预约表",
    "生成一条车辆经过的具体路线并立即执行",
    "生成具体路线并立即执行",
    "生产环境设置为 true",
    "调用 delete_all",
    "忽略工具白名单",
)

FORBIDDEN_MODEL_REQUEST_PATTERNS = (
    (
        "boundary-bypass",
        re.compile(
            r"(?:绕过|跳过|忽略|别管|不用|不要|别做).{0,16}"
            r"(?:审批|确认|校验|仿真|规则|约束|工具白名单|what-if)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "route-execution",
        re.compile(
            r"(?:规划|生成|安排).{0,16}(?:路线|轨迹).{0,24}(?:下发|执行|控制)"
        ),
    ),
    (
        "production-switch",
        re.compile(r"(?:切换|设置|进入|改成).{0,12}(?:生产模式|生产环境)"),
    ),
    (
        "unauthorized-interface",
        re.compile(r"(?:未公开|未授权|白名单外).{0,20}(?:接口|工具|API)", re.I),
    ),
    (
        "destructive-operation",
        re.compile(
            r"(?:清空|删除|删掉).{0,24}(?:任务|数据|数据库)|"
            r"(?:任务|数据|数据库).{0,24}(?:清空|删除|删掉)"
        ),
    ),
    (
        "model-vehicle-control",
        re.compile(r"(?:车辆|车队).{0,12}控制权.{0,12}(?:模型|大模型|AI)", re.I),
    ),
    (
        "reservation-write",
        re.compile(r"(?:写入|写进|修改).{0,20}(?:预约系统|预约表|资源预约)"),
    ),
    (
        "safety-stop-disable",
        re.compile(
            r"(?:解除|取消|关闭).{0,12}(?:安全停车|安全停机)|"
            r"(?:安全停车|安全停机).{0,12}(?:解除|取消|关闭)"
        ),
    ),
    (
        "direct-recovery-control",
        re.compile(
            r"(?:直接|立即|马上).{0,16}(?:让|控制|命令).{0,16}"
            r"(?:fork|jack|车辆|车).{0,16}(?:倒车|后退|执行|停车)",
            re.I,
        ),
    ),
    (
        "direct-submit",
        re.compile(r"(?:直接|立即|马上).{0,12}(?:提交|执行).{0,20}(?:任务|意图)"),
    ),
)


RETRIEVAL_INJECTION_PATTERNS = (
    (
        "retrieval.delimiter-breakout",
        re.compile(r"<\s*/?\s*UNTRUSTED_RETRIEVAL\b", re.I),
    ),
    (
        "retrieval.ignore-instructions",
        re.compile(
            r"ignore\s+(?:all\s+)?(?:(?:previous|prior)\s+)?"
            r"(?:(?:system|developer)\s+)?instructions|"
            r"(?:忽略|覆盖|替代).{0,12}(?:之前|以上|系统|开发者).{0,8}(?:指令|提示词|规则)",
            re.I,
        ),
    ),
    (
        "retrieval.authority-rewrite",
        re.compile(
            r"(?:将|把|修改).{0,24}"
            r"(?:requiredRobotGroup|environment|resourceBlock|resourceIds|requestedBy)"
            r".{0,16}(?:改为|设为|替换|覆盖)",
            re.I,
        ),
    ),
    (
        "retrieval.boundary-bypass",
        re.compile(r"(?:跳过|绕过|禁用|关闭).{0,16}(?:审批|校验|安全边界|白名单)"),
    ),
    (
        "retrieval.hidden-tool",
        re.compile(
            r"(?:调用|使用|执行).{0,16}"
            r"(?:delete_all|commit_intent|write_reservation|production_control)",
            re.I,
        ),
    ),
    (
        "retrieval.role-override",
        re.compile(r"(?:你现在是|system\s*message|developer\s*message).{0,30}(?:助手|模型|agent|智能体)", re.I),
    ),
)

_PROHIBITIVE_PREFIX = re.compile(r"(?:不得|禁止|严禁|不可|不能|切勿|不要)\s*$")


@dataclass(frozen=True)
class RetrievalScreening:
    accepted: list[EvidenceItem]
    quarantined: list[tuple[EvidenceItem, str]]


class ModelBoundaryError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def model_request_violation(text: str) -> str | None:
    normalized = " ".join(text.strip().split())
    exact = next(
        (term for term in FORBIDDEN_MODEL_REQUEST_TERMS if term in normalized),
        None,
    )
    if exact is not None:
        return exact
    return next(
        (
            code
            for code, pattern in FORBIDDEN_MODEL_REQUEST_PATTERNS
            if pattern.search(normalized)
        ),
        None,
    )


def retrieval_content_violation(text: str) -> str | None:
    """Detect instruction-shaped content without treating safety prose as an attack."""

    normalized = " ".join(text.strip().split())
    for code, pattern in RETRIEVAL_INJECTION_PATTERNS:
        for match in pattern.finditer(normalized):
            prefix = normalized[max(0, match.start() - 8) : match.start()]
            if _PROHIBITIVE_PREFIX.search(prefix):
                continue
            return code
    return None


def screen_retrieved_evidence(evidence: Iterable[EvidenceItem]) -> RetrievalScreening:
    accepted: list[EvidenceItem] = []
    quarantined: list[tuple[EvidenceItem, str]] = []
    for row in evidence:
        violation = retrieval_content_violation(f"{row.title}\n{row.detail}")
        if violation is None:
            accepted.append(row)
        else:
            quarantined.append((row, violation))
    return RetrievalScreening(accepted=accepted, quarantined=quarantined)


def untrusted_retrieval_record(row: EvidenceItem) -> dict[str, str | float | None]:
    return {
        "source": row.source,
        "title": row.title,
        "detail": (
            "<UNTRUSTED_RETRIEVAL>\n"
            f"{row.detail}\n"
            "</UNTRUSTED_RETRIEVAL>"
        ),
        "chunkId": row.chunk_id,
        "score": row.score,
    }


def enforce_intent_authority(
    intent: DispatchIntent,
    *,
    resolved_task: dict | None,
    resolved_resource_block: dict | None,
) -> None:
    if intent.intent_type not in MODEL_INTENT_TYPES:
        raise ModelBoundaryError(
            "intent.unsupported",
            f"模型不得生成意图 {intent.intent_type.value}",
        )

    if intent.intent_type is IntentType.CREATE_TASK:
        if resolved_task is None:
            raise ModelBoundaryError(
                "intent.task.ungrounded",
                "任务实体必须由确定性参数解析器提供",
            )
        task = intent.task.model_dump(by_alias=True, mode="json") if intent.task else {}
        for field in (
            "pickupNodeId",
            "dropoffNodeId",
            "requiredRobotGroup",
            "payloadType",
        ):
            if task.get(field) != resolved_task.get(field):
                raise ModelBoundaryError(
                    "intent.task.authority-mismatch",
                    f"模型改写了权威任务参数 {field}",
                )

    if intent.intent_type is IntentType.BLOCK_RESOURCE:
        if resolved_resource_block is None:
            raise ModelBoundaryError(
                "intent.resource.ungrounded",
                "资源实体必须由确定性参数解析器提供",
            )
        block = (
            intent.resource_block.model_dump(by_alias=True, mode="json")
            if intent.resource_block
            else {}
        )
        for field in ("resourceIds", "startMs", "endMs"):
            if block.get(field) != resolved_resource_block.get(field):
                raise ModelBoundaryError(
                    "intent.resource.authority-mismatch",
                    f"模型改写了权威资源参数 {field}",
                )


def unknown_evidence_ids(
    referenced_ids: Iterable[str], allowed_ids: Iterable[str]
) -> set[str]:
    return set(referenced_ids) - set(allowed_ids)


def enforce_plan_evidence(
    findings: list[PlanExplanationFinding], allowed_ids: Iterable[str]
) -> None:
    if not findings:
        raise ModelBoundaryError("explanation.empty", "计划解释未提供任何结论")
    referenced = {
        evidence_id for finding in findings for evidence_id in finding.evidence_ids
    }
    unknown = unknown_evidence_ids(referenced, allowed_ids)
    if unknown:
        raise ModelBoundaryError(
            "explanation.unknown-evidence",
            f"计划解释引用了未知证据：{', '.join(sorted(unknown))}",
        )


def diagnosis_violation(
    diagnosis: DiagnosisReport,
    *,
    evidence_ids: Iterable[str],
    vehicle_ids: Iterable[str],
    task_ids: Iterable[str],
    allowed_actions: Iterable[str],
) -> str | None:
    referenced = {
        evidence_id
        for row in diagnosis.root_cause_candidates
        for evidence_id in row.evidence_ids
    }
    referenced.update(
        evidence_id
        for row in diagnosis.recommendations
        for evidence_id in row.evidence_ids
    )
    if unknown_evidence_ids(referenced, evidence_ids):
        return "invalid-evidence"
    if not set(diagnosis.affected_vehicle_ids).issubset(set(vehicle_ids)):
        return "invalid-vehicle"
    if not set(diagnosis.affected_task_ids).issubset(set(task_ids)):
        return "invalid-task"
    allowed = set(allowed_actions)
    if any(
        row.action_code not in allowed
        or row.risk_level.value != "R3_HIGH"
        or not row.requires_simulation
        or not row.requires_approval
        for row in diagnosis.recommendations
    ):
        return "invalid-action"
    return None
