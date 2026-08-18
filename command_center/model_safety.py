from __future__ import annotations

from typing import Iterable

from .contracts import DiagnosisReport, DispatchIntent, IntentType, PlanExplanationFinding


MODEL_INTENT_TYPES = frozenset(
    {
        IntentType.QUERY_STATUS,
        IntentType.EXPLAIN_DECISION,
        IntentType.CREATE_TASK,
        IntentType.BLOCK_RESOURCE,
        IntentType.GENERATE_REPORT,
    }
)


class ModelBoundaryError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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
