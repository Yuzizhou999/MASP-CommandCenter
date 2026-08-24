from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

import httpx

from .contracts import (
    ClarificationRequest,
    DiagnosisReport,
    DispatchIntent,
    EvidenceItem,
    IncidentRecord,
    IntentType,
    PlanExplanationNarrative,
    PlanExplanationReport,
    ResourceBlockDraft,
    TaskDraft,
)
from .diagnosis import allowed_actions_for, deterministic_diagnosis
from .model_safety import enforce_intent_authority, enforce_plan_evidence
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


@dataclass(frozen=True)
class PlannedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentToolPlan:
    calls: tuple[PlannedToolCall, ...]
    strategy: str
    model: str


def intent_request_payload(
    text: str,
    *,
    world_revision: int,
    requested_by: str,
    resolved_task: dict[str, Any] | None = None,
    resolved_resource_block: dict[str, Any] | None = None,
    context_evidence: list[EvidenceItem] | None = None,
) -> dict[str, Any]:
    return {
        "request": text,
        "worldRevision": world_revision,
        "requestedBy": requested_by,
        "authoritativeParameters": {
            "task": resolved_task,
            "resourceBlock": resolved_resource_block,
        },
        "retrievedContext": [
            {
                "source": row.source,
                "title": row.title,
                "detail": row.detail,
            }
            for row in (context_evidence or [])
        ],
        "schema": DispatchIntent.model_json_schema(by_alias=True),
    }


def intent_training_messages(
    text: str,
    *,
    world_revision: int,
    requested_by: str,
    resolved_task: dict[str, Any] | None = None,
    resolved_resource_block: dict[str, Any] | None = None,
    context_evidence: list[EvidenceItem] | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                intent_request_payload(
                    text,
                    world_revision=world_revision,
                    requested_by=requested_by,
                    resolved_task=resolved_task,
                    resolved_resource_block=resolved_resource_block,
                    context_evidence=context_evidence,
                ),
                ensure_ascii=False,
            ),
        },
    ]


class DeepSeekProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._telemetry_lock = RLock()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._run_context: ContextVar[str | None] = ContextVar(
            "deepseek_run_id", default=None
        )
        self._control_context: ContextVar[Callable[[], None] | None] = ContextVar(
            "deepseek_control", default=None
        )
        self._aggregate = self._empty_telemetry()
        self._runs: dict[str, dict[str, Any]] = {}

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
            "resilience": {
                "maxRetries": self.settings.deepseek_max_retries,
                "circuitFailureThreshold": self.settings.deepseek_circuit_failure_threshold,
                "circuitResetSeconds": self.settings.deepseek_circuit_reset_seconds,
            },
            "telemetry": self.telemetry(),
        }

    @staticmethod
    def _empty_telemetry() -> dict[str, Any]:
        return {
            "requestCount": 0,
            "attemptCount": 0,
            "successCount": 0,
            "failureCount": 0,
            "retryCount": 0,
            "fallbackCount": 0,
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
            "estimatedCostUsd": 0.0,
        }

    @contextmanager
    def telemetry_scope(
        self,
        run_id: str,
        *,
        control: Callable[[], None] | None = None,
    ):
        token = self._run_context.set(run_id)
        control_token = self._control_context.set(control)
        with self._telemetry_lock:
            self._runs[run_id] = self._empty_telemetry()
        try:
            yield
        finally:
            self._control_context.reset(control_token)
            self._run_context.reset(token)

    def telemetry(self, run_id: str | None = None) -> dict[str, Any]:
        with self._telemetry_lock:
            source = self._runs.get(run_id, {}) if run_id else self._aggregate
            result = dict(source)
            result["circuitOpen"] = time.monotonic() < self._circuit_open_until
            result["pricingUsdPerMillionTokens"] = {
                "input": self.settings.deepseek_input_cost_per_million,
                "output": self.settings.deepseek_output_cost_per_million,
            }
            return result

    def _increment(self, key: str, value: int | float = 1) -> None:
        with self._telemetry_lock:
            targets = [self._aggregate]
            run_id = self._run_context.get()
            if run_id:
                targets.append(self._runs.setdefault(run_id, self._empty_telemetry()))
            for target in targets:
                target[key] = target.get(key, 0) + value
                if key == "estimatedCostUsd":
                    target[key] = round(float(target[key]), 8)

    def _mark_fallback(self) -> None:
        self._increment("fallbackCount")

    def _post(self, *, payload: dict[str, Any]):
        with self._telemetry_lock:
            circuit_open = time.monotonic() < self._circuit_open_until
        if circuit_open:
            self._increment("failureCount")
            raise ValueError("DeepSeek circuit breaker is open")

        self._increment("requestCount")
        last_error: httpx.HTTPError | None = None
        for attempt in range(self.settings.deepseek_max_retries + 1):
            control = self._control_context.get()
            if control is not None:
                control()
            self._increment("attemptCount")
            if attempt:
                self._increment("retryCount")
                time.sleep(min(0.2 * (2 ** (attempt - 1)), 1.0))
            try:
                response = httpx.post(
                    f"{self.settings.deepseek_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.settings.deepseek_timeout_seconds,
                )
                response.raise_for_status()
                usage = response.json().get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = int(
                    usage.get("total_tokens") or prompt_tokens + completion_tokens
                )
                self._increment("promptTokens", prompt_tokens)
                self._increment("completionTokens", completion_tokens)
                self._increment("totalTokens", total_tokens)
                self._increment(
                    "estimatedCostUsd",
                    (
                        prompt_tokens * self.settings.deepseek_input_cost_per_million
                        + completion_tokens
                        * self.settings.deepseek_output_cost_per_million
                    )
                    / 1_000_000,
                )
                self._increment("successCount")
                with self._telemetry_lock:
                    self._consecutive_failures = 0
                return response
            except httpx.HTTPStatusError as error:
                last_error = error
                status_code = error.response.status_code
                if status_code not in {408, 429} and status_code < 500:
                    break
            except httpx.HTTPError as error:
                last_error = error

        self._increment("failureCount")
        with self._telemetry_lock:
            self._consecutive_failures += 1
            if (
                self._consecutive_failures
                >= self.settings.deepseek_circuit_failure_threshold
            ):
                self._circuit_open_until = (
                    time.monotonic() + self.settings.deepseek_circuit_reset_seconds
                )
        if last_error is not None:
            raise last_error
        raise ValueError("DeepSeek request failed")

    def plan_context_tools(
        self,
        text: str,
        tool_definitions: list[dict[str, Any]],
        *,
        has_memory: bool = False,
    ) -> AgentToolPlan:
        """Let the model choose read-only context tools, with a deterministic fallback."""
        fallback_calls = [
            PlannedToolCall(name="get_world_snapshot", arguments={})
        ]
        if has_memory:
            fallback_calls.append(
                PlannedToolCall(name="recall_conversation_memory", arguments={})
            )
        fallback_calls.append(
            PlannedToolCall(name="search_sop", arguments={"query": text, "limit": 2})
        )
        fallback = AgentToolPlan(
            calls=tuple(fallback_calls),
            strategy="DETERMINISTIC_POLICY",
            model="deterministic-tool-policy",
        )
        if not self.configured:
            self._mark_fallback()
            return fallback

        allowed_names = {
            str(item.get("function", {}).get("name"))
            for item in tool_definitions
            if item.get("type") == "function"
        }
        try:
            response = self._post(
                payload={
                    "model": self.settings.deepseek_model,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是仓储调度 Agent 的只读上下文规划器。"
                                "必须调用 get_world_snapshot；当请求涉及调度规则、"
                                "安全、检修、异常或操作流程时调用 search_sop。"
                                "当工具可用且请求引用之前、刚才或同一会话实体时，"
                                "调用 recall_conversation_memory。"
                                "只能使用提供的工具，不得请求写操作或车辆控制。"
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                    "tools": tool_definitions,
                    "tool_choice": "auto",
                }
            )
            message = response.json()["choices"][0]["message"]
            raw_calls = message.get("tool_calls") or []
            calls: list[PlannedToolCall] = []
            seen: set[tuple[str, str]] = set()
            for raw_call in raw_calls[:4]:
                function = raw_call.get("function") or {}
                name = str(function.get("name") or "")
                if name not in allowed_names:
                    continue
                raw_arguments = function.get("arguments") or "{}"
                arguments = (
                    raw_arguments
                    if isinstance(raw_arguments, dict)
                    else json.loads(raw_arguments)
                )
                if not isinstance(arguments, dict):
                    continue
                if name == "get_world_snapshot":
                    arguments = {}
                elif name == "recall_conversation_memory":
                    arguments = {}
                elif name == "search_sop":
                    query = arguments.get("query")
                    if not isinstance(query, str) or not query.strip():
                        continue
                    arguments = {
                        "query": query[:1000],
                        "limit": max(1, min(int(arguments.get("limit", 2)), 5)),
                    }
                signature = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
                if signature not in seen:
                    calls.append(PlannedToolCall(name=name, arguments=arguments))
                    seen.add(signature)
            if not calls:
                return fallback
            if not any(call.name == "get_world_snapshot" for call in calls):
                calls.insert(0, PlannedToolCall(name="get_world_snapshot", arguments={}))
            if has_memory and not any(
                call.name == "recall_conversation_memory" for call in calls
            ):
                calls.insert(
                    1,
                    PlannedToolCall(
                        name="recall_conversation_memory", arguments={}
                    ),
                )
            return AgentToolPlan(
                calls=tuple(calls),
                strategy="MODEL_TOOL_CALLING",
                model=self.settings.deepseek_model,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._mark_fallback()
            return fallback

    def parse_intent(
        self,
        text: str,
        *,
        world_revision: int,
        requested_by: str,
        resolved_task: dict[str, Any] | None = None,
        resolved_resource_block: dict[str, Any] | None = None,
        context_evidence: list[EvidenceItem] | None = None,
    ) -> ParseResult:
        if not self.configured:
            self._mark_fallback()
            return self._fallback_result(
                text,
                world_revision=world_revision,
                requested_by=requested_by,
                resolved_task=resolved_task,
                resolved_resource_block=resolved_resource_block,
                model="deterministic-fallback",
            )

        try:
            response = self._post(
                payload={
                    "model": self.settings.deepseek_model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": intent_training_messages(
                        text,
                        world_revision=world_revision,
                        requested_by=requested_by,
                        resolved_task=resolved_task,
                        resolved_resource_block=resolved_resource_block,
                        context_evidence=context_evidence,
                    ),
                }
            )
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
            intent = DispatchIntent.model_validate(parsed)
            enforce_intent_authority(
                intent,
                resolved_task=resolved_task,
                resolved_resource_block=resolved_resource_block,
            )
            return ParseResult(
                intent=intent,
                model=self.settings.deepseek_model,
                fallback_used=False,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._mark_fallback()
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
            self._mark_fallback()
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
            response = self._post(
                payload={
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
                }
            )
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["model"] = self.settings.deepseek_model
            parsed["fallbackUsed"] = False
            return DiagnosisReport.model_validate(parsed)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._mark_fallback()
            return deterministic_diagnosis(
                incident,
                model=f"{self.settings.deepseek_model}:fallback",
            )

    def explain_plan(
        self, deterministic: PlanExplanationReport
    ) -> PlanExplanationReport:
        if not self.configured:
            self._mark_fallback()
            return deterministic
        try:
            schema = PlanExplanationNarrative.model_json_schema(by_alias=True)
            response = self._post(
                payload={
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
                }
            )
            content = response.json()["choices"][0]["message"]["content"]
            narrative = PlanExplanationNarrative.model_validate(json.loads(content))
            enforce_plan_evidence(
                narrative.findings,
                (row.evidence_id for row in deterministic.evidence),
            )
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
            self._mark_fallback()
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
                term in text
                for term in (
                    "创建",
                    "新增",
                    "插单",
                    "安排",
                    "送到",
                    "送去",
                    "送过去",
                    "运到",
                    "运输",
                    "转运",
                    "搬运",
                    "急货",
                    "紧急任务",
                )
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
        if resolved_resource_block is not None or any(
            term in normalized for term in ("封闭", "封路", "检修", "停用", "禁行")
        ):
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
        if resolved_task is not None or any(
            term in normalized
            for term in (
                "创建",
                "新增",
                "插单",
                "安排",
                "送到",
                "送去",
                "送过去",
                "运到",
                "运输",
                "转运",
                "搬运",
                "急货",
                "紧急任务",
            )
        ):
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
