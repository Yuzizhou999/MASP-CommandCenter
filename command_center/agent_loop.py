from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .agent_memory import AgentMemoryStore
from .agent_observability import AgentObservabilityStore
from .agent_protocol import (
    AgentActionType,
    AgentBudgetExceeded,
    AgentBudgets,
    AgentBudgetTracker,
    AgentObservation,
    authoritative_intent,
    classify_validation,
)
from .agent_runtime import AgentState, AgentStepLimitExceeded, BoundedAgentRun
from .agent_tools import AgentToolResult, DispatchAgentTools
from .audit import AuditStore
from .clarifications import ClarificationResolver, ResolvedRequest
from .contracts import (
    AgentTraceStep,
    ChatRequest,
    ChatResponse,
    ClarificationRequest,
    DispatchIntent,
    EvidenceItem,
    IntentType,
    IntentValidation,
    new_id,
)
from .engine_adapter import MaspAdapter
from .knowledge import KnowledgeBase
from .model_safety import (
    ModelBoundaryError,
    enforce_intent_authority,
    model_request_violation,
    untrusted_retrieval_record,
)
from .provider import DeepSeekProvider


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True, mode="json")
    if isinstance(value, list):
        return [_json_value(row) for row in value]
    if isinstance(value, tuple):
        return [_json_value(row) for row in value]
    if isinstance(value, dict):
        return {str(key): _json_value(row) for key, row in value.items()}
    return value


def _proposal_authority_violation(
    proposal: dict[str, Any], resolved: ResolvedRequest
) -> ModelBoundaryError | None:
    if resolved.task is not None:
        proposed = proposal.get("task")
        if isinstance(proposed, dict):
            for field in (
                "pickupNodeId",
                "dropoffNodeId",
                "requiredRobotGroup",
                "payloadType",
            ):
                if field in proposed and proposed[field] != resolved.task.get(field):
                    return ModelBoundaryError(
                        "intent.task.authority-mismatch",
                        f"模型试图改写权威任务参数 {field}",
                    )
    if resolved.resource_block is not None:
        proposed = proposal.get("resourceBlock")
        if isinstance(proposed, dict):
            for field in ("resourceIds", "startMs", "endMs"):
                if (
                    field in proposed
                    and proposed[field] != resolved.resource_block.get(field)
                ):
                    return ModelBoundaryError(
                        "intent.resource.authority-mismatch",
                        f"模型试图改写权威资源参数 {field}",
                    )
    environment = proposal.get("environment")
    if environment not in {None, "simulation"}:
        return ModelBoundaryError(
            "intent.environment.forbidden",
            "模型不得切换出 simulation 环境",
        )
    return None


class AgentLoopExecutor:
    def __init__(
        self,
        *,
        engine: MaspAdapter,
        provider: DeepSeekProvider,
        knowledge: KnowledgeBase,
        audit: AuditStore,
        clarifications: ClarificationResolver,
        memory: AgentMemoryStore,
        observability: AgentObservabilityStore,
        budgets: AgentBudgets | None = None,
    ) -> None:
        self.engine = engine
        self.provider = provider
        self.knowledge = knowledge
        self.audit = audit
        self.clarifications = clarifications
        self.memory = memory
        self.observability = observability
        self.budgets = budgets or AgentBudgets()

    def chat(
        self,
        request: ChatRequest,
        *,
        on_step: Callable[[AgentTraceStep], None] | None = None,
        approval_gate: Callable[[DispatchIntent, IntentValidation], None] | None = None,
    ) -> ChatResponse:
        trace_id = new_id("trace")
        tracker = AgentBudgetTracker(self.budgets)
        run = BoundedAgentRun(
            max_steps=self.budgets.max_steps,
            reserve_terminal_step=True,
            on_step=on_step,
        )
        run.set_planner(strategy="ACTION_PROTOCOL_LOOP", model=self._driver_name())
        tools = DispatchAgentTools(
            engine=self.engine,
            knowledge=self.knowledge,
            scenario_id=request.scenario_id,
            memory=self.memory,
            conversation_id=request.conversation_id,
        )
        prior_memory = self.memory.get(request.conversation_id)
        evidence: list[EvidenceItem] = []
        observations: list[AgentObservation] = []
        action_history: list[dict[str, Any]] = []
        snapshot: dict[str, Any] | None = None
        intent: DispatchIntent | None = None
        validation: IntentValidation | None = None
        model = self._driver_name()
        fallback_used = False
        terminal_reason: str | None = None

        run.transition(
            AgentState.RECEIVED,
            title="接收调度目标",
            detail=f"绑定会话 {request.conversation_id} 和场景 {request.scenario_id}",
        )
        started = perf_counter()
        resolved = self.clarifications.resolve(
            request.message, request.conversation_id
        )
        run.transition(
            AgentState.PARAMETER_RESOLUTION,
            title="绑定权威业务参数",
            detail=(
                "硬性参数完整，允许策略模型开始决策"
                if resolved.clarification is None
                else "硬性参数缺失，由服务端强制澄清"
            ),
            duration_ms=(perf_counter() - started) * 1000,
        )
        if resolved.clarification is not None:
            return self._clarification_response(
                request=request,
                trace_id=trace_id,
                run=run,
                clarification=resolved.clarification,
                evidence=evidence,
                model="deterministic-parameter-resolver",
                fallback_used=False,
                tracker=tracker,
                reason="policy.hard_required_fields",
            )

        run.transition(
            AgentState.PLANNING,
            title="启动有界 observe-decide-act 循环",
            detail="每轮只允许一个动作，工具结果会进入下一轮决策",
        )
        observations.append(
            AgentObservation(
                sequence=1,
                kind="INITIAL",
                code="request.received",
                summary="用户目标和服务端权威参数已就绪",
                data={
                    "hasMemory": prior_memory is not None,
                    "scenarioId": request.scenario_id,
                },
            )
        )
        run.transition(
            AgentState.DECIDING,
            title="等待策略模型选择下一动作",
            detail="可调用一个只读工具、请求澄清或提出意图",
        )

        try:
            while True:
                tracker.check()
                started = perf_counter()
                decision = self.provider.decide_agent_action(
                    resolved.message,
                    tools.model_definitions(),
                    observations=observations,
                    authoritative_parameters={
                        "task": resolved.task,
                        "resourceBlock": resolved.resource_block,
                    },
                    action_history=action_history,
                )
                decision_tokens = decision.prompt_tokens + decision.completion_tokens
                tracker.consume_decision(
                    decision_tokens, decision.estimated_cost_usd
                )
                model = decision.model
                fallback_used = fallback_used or decision.fallback_used
                attempt = tracker.decisions

                if decision.action is None:
                    code = decision.error_code or "protocol.invalid_action"
                    detail = decision.error_message or "模型没有返回合法单动作"
                    run.record(
                        title="拒绝非法模型动作",
                        detail=detail,
                        status="REJECTED",
                        action="INVALID",
                        observation_code=code,
                        attempt=attempt,
                        prompt_tokens=decision.prompt_tokens,
                        completion_tokens=decision.completion_tokens,
                        duration_ms=(perf_counter() - started) * 1000,
                    )
                    observations.append(
                        AgentObservation(
                            sequence=len(observations) + 1,
                            kind=(
                                "SECURITY_BOUNDARY"
                                if code.startswith("policy.")
                                else "TOOL_REJECTION"
                            ),
                            code=code,
                            summary=detail,
                            data={"attempt": attempt},
                        )
                    )
                    action_history.append(
                        {
                            "action": "INVALID",
                            "errorCode": code,
                        }
                    )
                    if code.startswith("policy."):
                        terminal_reason = code
                        run.transition(
                            AgentState.BLOCKED,
                            title="确定性安全边界阻断请求",
                            detail=detail,
                            status="BLOCKED",
                        )
                        break
                    continue

                action = decision.action
                action_history.append(
                    action.model_dump(mode="json", exclude_none=True)
                )
                run.record(
                    title=f"策略动作：{action.action.value}",
                    detail=(
                        f"选择工具 {action.tool}"
                        if action.action is AgentActionType.CALL_TOOL
                        else "模型只提交决策，参数和终局由服务端控制"
                    ),
                    action=action.action.value,
                    attempt=attempt,
                    prompt_tokens=decision.prompt_tokens,
                    completion_tokens=decision.completion_tokens,
                    duration_ms=(perf_counter() - started) * 1000,
                )

                if action.action is AgentActionType.CALL_TOOL:
                    tracker.consume_tool_call()
                    try:
                        tool = tools.tool(str(action.tool))
                        if not tool.model_selectable:
                            raise ValueError(f"工具 {tool.name} 不允许由模型选择")
                        run.transition(
                            AgentState.CONTEXT_GATHERING,
                            title="执行单个只读工具动作",
                            detail=f"服务端校验工具 {tool.name} 的名称和参数",
                        )
                        result = run.execute_tool(
                            tools, tool.name, dict(action.arguments or {})
                        )
                    except (ValueError, TypeError, ValidationError) as error:
                        code = "tool.rejected"
                        run.record(
                            title="工具动作被拒绝",
                            detail=str(error),
                            status="REJECTED",
                            tool_name=str(action.tool),
                            read_only=None,
                            action=action.action.value,
                            observation_code=code,
                            attempt=attempt,
                        )
                        observations.append(
                            AgentObservation(
                                sequence=len(observations) + 1,
                                kind="TOOL_REJECTION",
                                code=code,
                                summary=str(error),
                                data={"arguments": dict(action.arguments or {})},
                                toolName=str(action.tool),
                            )
                        )
                        if run.current_state is AgentState.CONTEXT_GATHERING:
                            run.transition(
                                AgentState.OBSERVING,
                                title="观察工具拒绝结果",
                                detail="拒绝信息将返回策略模型",
                            )
                            run.transition(
                                AgentState.DECIDING,
                                title="基于拒绝结果重新决策",
                                detail="工具预算已计入本次无效调用",
                            )
                        continue

                    observation_data = self._tool_observation_data(tool.name, result)
                    observations.append(
                        AgentObservation(
                            sequence=len(observations) + 1,
                            kind="TOOL_RESULT",
                            code="tool.ok",
                            summary=result.summary,
                            data=observation_data,
                            toolName=tool.name,
                            trusted=tool.name != "search_sop",
                        )
                    )
                    if tool.name == "get_world_snapshot":
                        snapshot = result.value
                        evidence.append(tools.world_evidence(snapshot, request.scenario_id))
                    elif tool.name == "search_sop":
                        evidence.extend(result.value)
                        quarantined = (result.metadata or {}).get("quarantined") or []
                        for item in quarantined:
                            run.record(
                                title="隔离可疑检索内容",
                                detail=f"{item['source']}：{item['violation']}",
                                state=AgentState.OBSERVING,
                                status="BLOCKED",
                                observation_code=str(item["violation"]),
                                attempt=attempt,
                            )
                    elif tool.name == "recall_conversation_memory" and result.value:
                        evidence.append(
                            tools.memory_evidence(result.value, request.conversation_id)
                        )
                    run.transition(
                        AgentState.OBSERVING,
                        title="观察工具执行结果",
                        detail=result.summary,
                    )
                    run.transition(
                        AgentState.DECIDING,
                        title="基于新观察继续决策",
                        detail="策略模型现在可以看到上一工具结果",
                    )
                    continue

                if action.action is AgentActionType.REQUEST_CLARIFICATION:
                    clarification = self._soft_clarification(resolved)
                    return self._clarification_response(
                        request=request,
                        trace_id=trace_id,
                        run=run,
                        clarification=clarification,
                        evidence=evidence,
                        model=model,
                        fallback_used=fallback_used,
                        tracker=tracker,
                        reason="policy.model_requested_clarification",
                    )

                assert action.intent is not None
                if snapshot is None:
                    code = "policy.world_snapshot.required"
                    observations.append(
                        AgentObservation(
                            sequence=len(observations) + 1,
                            kind="POLICY_OVERRIDE",
                            code=code,
                            summary="提出意图前必须先读取权威世界快照",
                            data={},
                        )
                    )
                    run.record(
                        title="服务端覆盖过早终局",
                        detail="提出意图前必须先读取权威世界快照",
                        status="REJECTED",
                        action=action.action.value,
                        observation_code=code,
                        attempt=attempt,
                    )
                    continue

                authority_error = _proposal_authority_violation(action.intent, resolved)
                if authority_error is not None:
                    terminal_reason = authority_error.code
                    run.transition(
                        AgentState.BLOCKED,
                        title="权威实体边界阻断模型草案",
                        detail=str(authority_error),
                        status="BLOCKED",
                    )
                    break
                try:
                    intent = authoritative_intent(
                        action.intent,
                        world_revision=int(snapshot["worldRevision"]),
                        requested_by=request.requested_by,
                        resolved_task=resolved.task,
                        resolved_resource_block=resolved.resource_block,
                    )
                    enforce_intent_authority(
                        intent,
                        resolved_task=resolved.task,
                        resolved_resource_block=resolved.resource_block,
                    )
                except ModelBoundaryError as error:
                    terminal_reason = error.code
                    run.transition(
                        AgentState.BLOCKED,
                        title="权威实体边界阻断模型草案",
                        detail=str(error),
                        status="BLOCKED",
                    )
                    break
                except (ValidationError, ValueError) as error:
                    tracker.consume_repair()
                    code = getattr(error, "code", "intent.schema.invalid")
                    observations.append(
                        AgentObservation(
                            sequence=len(observations) + 1,
                            kind="VALIDATION_ISSUES",
                            code=code,
                            summary=str(error),
                            data={"fixable": True, "attempt": tracker.repair_attempts},
                        )
                    )
                    run.record(
                        title="草案 Schema 校验失败",
                        detail=str(error),
                        status="REJECTED",
                        action=action.action.value,
                        observation_code=code,
                        attempt=tracker.repair_attempts,
                    )
                    continue

                run.transition(
                    AgentState.INTENT_DRAFTING,
                    title="接受结构化意图草案",
                    detail=f"形成 {intent.intent_type.value}，权威字段已由服务端覆盖",
                )
                run.transition(
                    AgentState.SAFETY_VALIDATION,
                    title="执行确定性 MASP 校验",
                    detail="模型不能跳过或修改校验结果",
                )
                result = run.execute_tool(
                    tools,
                    "validate_dispatch_intent",
                    {"intent": intent.model_dump(by_alias=True, mode="json")},
                )
                validation = tools.validation_value(result)
                if validation.valid:
                    if validation.approval_required and approval_gate:
                        approval_gate(intent, validation)
                    run.transition(
                        AgentState.COMPLETED,
                        title="完成 Agent 决策",
                        detail="意图通过确定性校验，进入仿真或审批流程",
                    )
                    break

                disposition = classify_validation(validation)
                if disposition.can_repair:
                    tracker.consume_repair()
                    observations.append(
                        AgentObservation(
                            sequence=len(observations) + 1,
                            kind="VALIDATION_ISSUES",
                            code="validation.fixable",
                            summary="MASP 返回可修复问题",
                            data={
                                "attempt": tracker.repair_attempts,
                                "issues": [
                                    row.model_dump(mode="json")
                                    for row in disposition.fixable
                                ],
                            },
                        )
                    )
                    run.transition(
                        AgentState.REPAIRING,
                        title="进入有界意图修复",
                        detail=f"第 {tracker.repair_attempts} 次修复，只返回 fixable issues",
                        observation_code="validation.fixable",
                        attempt=tracker.repair_attempts,
                    )
                    run.transition(
                        AgentState.DECIDING,
                        title="根据校验问题重新提出意图",
                        detail="安全、权限和审批问题不会交给模型修复",
                    )
                    continue

                terminal_reason = (
                    disposition.blocking[0].code
                    if disposition.blocking
                    else "validation.blocked"
                )
                run.transition(
                    AgentState.BLOCKED,
                    title="确定性校验阻断草案",
                    detail="；".join(row.message for row in disposition.blocking)
                    or "意图未通过 MASP 校验",
                    status="BLOCKED",
                )
                break
        except (AgentBudgetExceeded, AgentStepLimitExceeded) as error:
            terminal_reason = error.code
            run.transition(
                AgentState.BUDGET_EXCEEDED,
                title="Agent 预算耗尽",
                detail=str(error),
                status="BLOCKED",
            )

        run.set_budget_summary(
            budgets=self.budgets.model_dump(by_alias=True),
            usage=tracker.snapshot(),
            terminal_reason=terminal_reason,
        )
        trace = run.build_trace()
        state = (
            "READY"
            if trace.status == "COMPLETED"
            else "BUDGET_EXCEEDED"
            if trace.status == "BUDGET_EXCEEDED"
            else "BLOCKED"
        )
        message, actions = self._intent_message(intent, validation, snapshot, state)
        if state == "READY":
            self.memory.record(
                conversation_id=request.conversation_id,
                scenario_id=request.scenario_id,
                message=request.message,
                outcome="READY",
                trace=trace,
                intent=intent,
                validation=validation,
            )
        self._record_observability(
            trace_id, request, trace, model, fallback_used, validation
        )
        response = ChatResponse(
            traceId=trace_id,
            conversationId=request.conversation_id,
            state=state,
            message=message,
            intent=intent,
            validation=validation,
            evidence=evidence,
            model=model,
            fallbackUsed=fallback_used,
            suggestedActions=actions,
            agentTrace=trace,
        )
        self.audit.append(
            trace_id=trace_id,
            event_type="AGENT_LOOP_TERMINAL",
            actor=request.requested_by,
            payload={
                "request": request.message,
                "scenarioId": request.scenario_id,
                "state": state,
                "terminalReason": terminal_reason,
                "intent": (
                    intent.model_dump(by_alias=True, mode="json") if intent else None
                ),
                "validation": (
                    validation.model_dump(by_alias=True, mode="json")
                    if validation
                    else None
                ),
                "agentTrace": trace.model_dump(by_alias=True, mode="json"),
            },
        )
        return response

    def _clarification_response(
        self,
        *,
        request: ChatRequest,
        trace_id: str,
        run: BoundedAgentRun,
        clarification: ClarificationRequest,
        evidence: list[EvidenceItem],
        model: str,
        fallback_used: bool,
        tracker: AgentBudgetTracker,
        reason: str,
    ) -> ChatResponse:
        run.transition(
            AgentState.CLARIFICATION_REQUIRED,
            title="等待用户补充信息",
            detail="；".join(clarification.questions),
            status="BLOCKED",
        )
        run.set_budget_summary(
            budgets=self.budgets.model_dump(by_alias=True),
            usage=tracker.snapshot(),
            terminal_reason=reason,
        )
        trace = run.build_trace()
        self.memory.record(
            conversation_id=request.conversation_id,
            scenario_id=request.scenario_id,
            message=request.message,
            outcome="CLARIFICATION_REQUIRED",
            trace=trace,
            clarification=clarification,
        )
        self._record_observability(
            trace_id, request, trace, model, fallback_used, None
        )
        response = ChatResponse(
            traceId=trace_id,
            conversationId=request.conversation_id,
            state="CLARIFICATION_REQUIRED",
            message="还不能形成可执行草案。" + " ".join(clarification.questions),
            clarification=clarification,
            evidence=evidence,
            model=model,
            fallbackUsed=fallback_used,
            suggestedActions=clarification.questions,
            agentTrace=trace,
        )
        self.audit.append(
            trace_id=trace_id,
            event_type="AGENT_CLARIFICATION_REQUESTED",
            actor=request.requested_by,
            payload={
                "request": request.message,
                "scenarioId": request.scenario_id,
                "reason": reason,
                "clarification": clarification.model_dump(
                    by_alias=True, mode="json"
                ),
                "agentTrace": trace.model_dump(by_alias=True, mode="json"),
            },
        )
        return response

    @staticmethod
    def _soft_clarification(resolved: ResolvedRequest) -> ClarificationRequest:
        return ClarificationRequest(
            code="AMBIGUOUS_ENTITY",
            missingFields=[],
            questions=["请明确希望系统处理的业务对象或目标。"],
            collectedParameters={
                "intentType": resolved.intent_type.value
                if resolved.intent_type is not None
                else None
            },
        )

    @staticmethod
    def _tool_observation_data(name: str, result: AgentToolResult) -> dict[str, Any]:
        if name == "get_world_snapshot":
            snapshot = result.value
            return {
                "value": {
                    "worldRevision": snapshot["worldRevision"],
                    "counts": snapshot["counts"],
                }
            }
        if name == "search_sop":
            return {
                "value": [untrusted_retrieval_record(row) for row in result.value],
                "quarantined": (result.metadata or {}).get("quarantined", []),
            }
        if name == "recall_conversation_memory":
            return {
                "value": _json_value(result.value) if result.value else None,
            }
        return {"value": _json_value(result.value)}

    @staticmethod
    def _intent_message(
        intent: DispatchIntent | None,
        validation: IntentValidation | None,
        snapshot: dict[str, Any] | None,
        state: str,
    ) -> tuple[str, list[str]]:
        if state == "BUDGET_EXCEEDED":
            return "Agent 已达到本次运行预算，未形成可执行草案。", ["查看执行轨迹"]
        if state == "BLOCKED":
            issues = "；".join(
                row.message for row in (validation.issues if validation else [])
            )
            return (
                f"草案被确定性安全边界阻断{f'：{issues}' if issues else '。'}",
                ["查看安全边界", "查看执行轨迹"],
            )
        if intent is None:
            return "Agent 未形成结构化意图。", ["查看执行轨迹"]
        if intent.intent_type is IntentType.CREATE_TASK and intent.task:
            task = intent.task
            return (
                f"已形成紧急运输任务草案：{task.pickup_node_id} 到 "
                f"{task.dropoff_node_id}，由 {task.required_robot_group} 车型执行，"
                f"优先级 {task.priority_class}。",
                ["运行数字孪生", "查看任务参数"],
            )
        if intent.intent_type is IntentType.BLOCK_RESOURCE and intent.resource_block:
            resources = "、".join(intent.resource_block.resource_ids)
            return (
                f"已识别临时封锁意图：{resources}。该操作必须先仿真并由主管审批。",
                ["运行封路推演", "查看安全规则"],
            )
        if intent.intent_type is IntentType.GENERATE_REPORT:
            return "可以根据仿真和审计记录生成班次运营报告。", ["生成运营报告"]
        counts = (snapshot or {}).get("counts") or {}
        return (
            f"当前场景共有 {counts.get('vehicles', 0)} 辆车、"
            f"{counts.get('tasks', 0)} 个任务。系统处于仿真模式。",
            ["注入紧急任务", "推演通道封闭"],
        )

    def _record_observability(
        self,
        trace_id: str,
        request: ChatRequest,
        trace,
        model: str,
        fallback_used: bool,
        validation: IntentValidation | None,
    ) -> None:
        self.observability.record(
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            scenario_id=request.scenario_id,
            trace=trace,
            model=model,
            fallback_used=fallback_used,
            validation=validation,
        )

    def _driver_name(self) -> str:
        status = self.provider.status()
        return str(status.get("model") or status.get("provider") or "agent-driver")
