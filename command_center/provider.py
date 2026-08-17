from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .contracts import (
    ClarificationRequest,
    DiagnosisReport,
    DispatchIntent,
    IncidentRecord,
    IntentType,
    PlanExplanationNarrative,
    PlanExplanationReport,
    ResourceBlockDraft,
    TaskDraft,
)
from .diagnosis import allowed_actions_for, deterministic_diagnosis
from .settings import Settings


SYSTEM_PROMPT = """你是保利智仓·灵枢的调度意图解析器。
你只能把用户请求转换为结构化调度意图，不得生成车辆轨迹、资源预约或解除安全停车。
可用意图为 QUERY_STATUS、EXPLAIN_DECISION、CREATE_TASK、BLOCK_RESOURCE、GENERATE_REPORT。
站点ID必须保留车型前缀。叉车使用 fork，搬运车使用 jack。
封闭共享窄路时使用资源 zone:zone-jack-pp363-pp365。
authoritativeParameters 中的实体已经由确定性目录解析，必须原样使用，不得替换或补充其他ID。
信息不足时不得自行补齐站点、车辆、工位或资源ID。
输出必须是单个JSON对象，不得包含Markdown。
"""


DIAGNOSIS_SYSTEM_PROMPT = """你是保利智仓·灵枢的异常诊断解释器。
你只能根据输入中的 Incident、Evidence 和 DeterministicFindings 解释车辆故障、工位停用或等待环，不得补充未提供的遥测、实体或事实。
每个根因候选和每条建议必须引用一个或多个输入中真实存在的 evidenceId。
只能从 allowedActions 中选择建议，所有建议均为 R3_HIGH，必须仿真并人工审批。
不得声称已经控制车辆、解除停车、重派任务或执行恢复；不得生成路线和资源预约。
事实和推断必须明确区分。证据不足时必须写入 uncertainties。
输出必须是符合所给 schema 的单个 JSON 对象，不得包含 Markdown。
"""


PLAN_EXPLANATION_SYSTEM_PROMPT = """你是仓储群车调度计划解释器。
你只能依据输入的 Evidence 和 DeterministicFindings 组织业务解释，不得自行计算路线、时间、指标或资源状态。
每条 finding 必须引用输入中真实存在的 evidenceId，并明确标记 FACT 或 INFERENCE。
不得声称选择了证据中未出现的车辆、任务、路线或资源。证据不足时写入 uncertainties。
输出必须是符合所给 schema 的单个 JSON 对象，不得包含 Markdown。
"""


@dataclass(frozen=True)
class ParseResult:
    intent: DispatchIntent | None
    model: str
    fallback_used: bool
    clarification: ClarificationRequest | None = None


class DeepSeekProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def configured(self) -> bool:
        return bool(self.settings.deepseek_api_key)

    def status(self) -> dict[str, Any]:
        return {
            "provider": "deepseek",
            "model": self.settings.deepseek_model,
            "configured": self.configured,
            "mode": "api" if self.configured else "deterministic-fallback",
            "baseUrl": self.settings.deepseek_base_url,
        }

    def parse_intent(
        self,
        text: str,
        *,
        world_revision: int,
        requested_by: str,
        resolved_task: dict[str, Any] | None = None,
        resolved_resource_block: dict[str, Any] | None = None,
    ) -> ParseResult:
        if not self.configured:
            return self._fallback_result(
                text,
                world_revision=world_revision,
                requested_by=requested_by,
                resolved_task=resolved_task,
                resolved_resource_block=resolved_resource_block,
                model="deterministic-fallback",
            )

        try:
            schema = DispatchIntent.model_json_schema(by_alias=True)
            response = httpx.post(
                f"{self.settings.deepseek_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.deepseek_model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "request": text,
                                    "worldRevision": world_revision,
                                    "requestedBy": requested_by,
                                    "authoritativeParameters": {
                                        "task": resolved_task,
                                        "resourceBlock": resolved_resource_block,
                                    },
                                    "schema": schema,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                },
                timeout=self.settings.deepseek_timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["basedOnWorldRevision"] = world_revision
            parsed["requestedBy"] = requested_by
            parsed["environment"] = "simulation"
            if resolved_task is not None:
                parsed["intentType"] = IntentType.CREATE_TASK.value
                parsed["task"] = {
                    **dict(parsed.get("task") or {}),
                    **resolved_task,
                }
            if resolved_resource_block is not None:
                parsed["intentType"] = IntentType.BLOCK_RESOURCE.value
                parsed["resourceBlock"] = resolved_resource_block
            return ParseResult(
                intent=DispatchIntent.model_validate(parsed),
                model=self.settings.deepseek_model,
                fallback_used=False,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return self._fallback_result(
                text,
                world_revision=world_revision,
                requested_by=requested_by,
                resolved_task=resolved_task,
                resolved_resource_block=resolved_resource_block,
                model=f"{self.settings.deepseek_model}:fallback",
            )

    def diagnose_incident(self, incident: IncidentRecord) -> DiagnosisReport:
        if not self.configured:
            return deterministic_diagnosis(incident)
        try:
            schema = DiagnosisReport.model_json_schema(by_alias=True)
            payload = {
                "incident": {
                    "incidentId": incident.incident_id,
                    "incidentType": incident.incident_type.value,
                    "severity": incident.severity.value,
                    "scenarioId": incident.scenario_id,
                    "runId": incident.run_id,
                    "vehicleIds": incident.vehicle_ids,
                    "taskIds": incident.task_ids,
                    "resourceIds": incident.resource_ids,
                    "faultCode": incident.fault_code,
                    "faultAtMs": incident.fault_at_ms,
                    "locationNodeId": incident.location_node_id,
                    "workstationId": incident.workstation_id,
                    "loadState": incident.load_state,
                    "eventAttributes": incident.event_attributes,
                },
                "evidence": [
                    row.model_dump(by_alias=True, mode="json")
                    for row in incident.evidence
                ],
                "deterministicFindings": [
                    row.model_dump(by_alias=True, mode="json")
                    for row in incident.deterministic_findings
                ],
                "allowedActions": sorted(allowed_actions_for(incident)),
                "schema": schema,
            }
            response = httpx.post(
                f"{self.settings.deepseek_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.deepseek_model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                },
                timeout=self.settings.deepseek_timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["model"] = self.settings.deepseek_model
            parsed["fallbackUsed"] = False
            return DiagnosisReport.model_validate(parsed)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return deterministic_diagnosis(
                incident,
                model=f"{self.settings.deepseek_model}:fallback",
            )

    def explain_plan(
        self, deterministic: PlanExplanationReport
    ) -> PlanExplanationReport:
        if not self.configured:
            return deterministic
        try:
            schema = PlanExplanationNarrative.model_json_schema(by_alias=True)
            response = httpx.post(
                f"{self.settings.deepseek_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.deepseek_model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": PLAN_EXPLANATION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "question": deterministic.question,
                                    "filters": {
                                        "vehicleId": deterministic.vehicle_id,
                                        "taskId": deterministic.task_id,
                                    },
                                    "evidence": [
                                        row.model_dump(by_alias=True, mode="json")
                                        for row in deterministic.evidence
                                    ],
                                    "deterministicFindings": [
                                        row.model_dump(by_alias=True, mode="json")
                                        for row in deterministic.findings
                                    ],
                                    "schema": schema,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                },
                timeout=self.settings.deepseek_timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            narrative = PlanExplanationNarrative.model_validate(json.loads(content))
            allowed_ids = {row.evidence_id for row in deterministic.evidence}
            if not narrative.findings or any(
                not set(row.evidence_ids).issubset(allowed_ids)
                for row in narrative.findings
            ):
                raise ValueError("plan explanation cited unknown evidence")
            return deterministic.model_copy(
                update={
                    "summary": narrative.summary,
                    "findings": narrative.findings,
                    "uncertainties": narrative.uncertainties,
                    "model": self.settings.deepseek_model,
                    "fallback_used": False,
                }
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return deterministic.model_copy(
                update={"model": f"{self.settings.deepseek_model}:fallback"}
            )

    def _fallback_result(
        self,
        text: str,
        *,
        world_revision: int,
        requested_by: str,
        resolved_task: dict[str, Any] | None,
        resolved_resource_block: dict[str, Any] | None,
        model: str,
    ) -> ParseResult:
        try:
            intent = self._fallback_intent(
                text,
                world_revision=world_revision,
                requested_by=requested_by,
                resolved_task=resolved_task,
                resolved_resource_block=resolved_resource_block,
            )
        except ValueError:
            task_like = any(
                term in text for term in ("创建", "插单", "送到", "运到", "紧急")
            )
            missing = (
                ["pickupNodeId", "dropoffNodeId", "requiredRobotGroup"]
                if task_like
                else ["resourceIds"]
            )
            questions = (
                ["请明确取货站点、放货站点和执行车型。"]
                if task_like
                else ["请明确需要封闭的通道、工位或资源编号。"]
            )
            return ParseResult(
                intent=None,
                model=model,
                fallback_used=True,
                clarification=ClarificationRequest(
                    code="MISSING_REQUIRED_FIELDS",
                    missingFields=missing,
                    questions=questions,
                ),
            )
        return ParseResult(intent=intent, model=model, fallback_used=True)

    def _fallback_intent(
        self,
        text: str,
        *,
        world_revision: int,
        requested_by: str,
        resolved_task: dict[str, Any] | None = None,
        resolved_resource_block: dict[str, Any] | None = None,
    ) -> DispatchIntent:
        normalized = text.strip()
        if any(term in normalized for term in ("封闭", "封路", "检修", "禁行")):
            if resolved_resource_block is None:
                resources = (
                    ["zone:zone-jack-pp363-pp365"]
                    if any(term in normalized for term in ("共享窄路", "共享通道"))
                    else []
                )
                if not resources:
                    raise ValueError("resourceIds are required before parsing BLOCK_RESOURCE")
                resolved_resource_block = {
                    "resourceIds": resources,
                    "startMs": 0,
                    "endMs": 180000,
                    "reason": normalized,
                }
            return DispatchIntent(
                intentType=IntentType.BLOCK_RESOURCE,
                requestedBy=requested_by,
                basedOnWorldRevision=world_revision,
                reason=normalized,
                resourceBlock=ResourceBlockDraft.model_validate(resolved_resource_block),
            )
        if any(term in normalized for term in ("报告", "总结", "班次")):
            return DispatchIntent(
                intentType=IntentType.GENERATE_REPORT,
                requestedBy=requested_by,
                basedOnWorldRevision=world_revision,
                reason=normalized,
                query=normalized,
            )
        if any(term in normalized for term in ("创建", "插单", "送到", "运到", "紧急")):
            if resolved_task is None:
                group = (
                    "jack"
                    if any(term in normalized.lower() for term in ("jack", "搬运车", "顶升车", "料架"))
                    else "fork"
                    if any(term in normalized.lower() for term in ("fork", "叉车", "托盘"))
                    else None
                )
                ap_ids = re.findall(
                    r"(?:fork:|jack:)?AP\d+", normalized, flags=re.IGNORECASE
                )
                if group is None or len(ap_ids) < 2:
                    raise ValueError("task endpoints and requiredRobotGroup must be explicit")
                prefix = f"{group}:"
                pickup = ap_ids[0]
                dropoff = ap_ids[1]
                if ":" not in pickup:
                    pickup = prefix + pickup.upper()
                if ":" not in dropoff:
                    dropoff = prefix + dropoff.upper()
                resolved_task = {
                    "pickupNodeId": pickup,
                    "dropoffNodeId": dropoff,
                    "requiredRobotGroup": group,
                    "payloadType": "shelf" if group == "jack" else "pallet",
                }
            return DispatchIntent(
                intentType=IntentType.CREATE_TASK,
                requestedBy=requested_by,
                basedOnWorldRevision=world_revision,
                reason=normalized,
                task=TaskDraft.model_validate(
                    {**resolved_task, "priorityClass": 3, "dueTimeMs": 300000}
                ),
            )
        return DispatchIntent(
            intentType=(
                IntentType.EXPLAIN_DECISION
                if any(term in normalized for term in ("为什么", "原因", "解释"))
                else IntentType.QUERY_STATUS
            ),
            requestedBy=requested_by,
            basedOnWorldRevision=world_revision,
            reason=normalized,
            query=normalized,
        )
