from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from .agent_loop import AgentLoopExecutor
from .agent_memory import AgentMemoryStore
from .agent_observability import AgentObservabilityStore
from .agent_protocol import AgentBudgets
from .agent_runtime import AgentState, BoundedAgentRun
from .agent_tools import DispatchAgentTools
from .audit import AuditStore
from .clarifications import ClarificationResolver, ClarificationStore
from .contracts import (
    AgentTraceStep,
    ChatRequest,
    ChatResponse,
    DispatchIntent,
    IntentType,
    IntentValidation,
    new_id,
)
from .engine_adapter import MaspAdapter
from .knowledge import KnowledgeBase
from .model_safety import model_request_violation
from .provider import DeepSeekProvider


class DispatchOrchestrator:
    def __init__(
        self,
        *,
        engine: MaspAdapter,
        provider: DeepSeekProvider,
        knowledge: KnowledgeBase,
        audit: AuditStore,
        clarifications: ClarificationResolver | None = None,
        memory: AgentMemoryStore | None = None,
        observability: AgentObservabilityStore | None = None,
        runtime_mode: str = "linear",
        budgets: AgentBudgets | None = None,
    ) -> None:
        self.engine = engine
        self.provider = provider
        self.knowledge = knowledge
        self.audit = audit
        self.memory = memory or AgentMemoryStore(
            engine.settings.data_dir / "agent-memories.json"
        )
        self.observability = observability or AgentObservabilityStore(
            engine.settings.data_dir / "agent-metrics.jsonl"
        )
        self.clarifications = clarifications or ClarificationResolver(
            ClarificationStore(engine.settings.data_dir / "clarifications.json"), engine
        )
        if runtime_mode not in {"linear", "loop"}:
            raise ValueError("Agent runtime_mode 必须是 linear 或 loop")
        self.runtime_mode = runtime_mode
        self.budgets = budgets or AgentBudgets()
        self.loop = AgentLoopExecutor(
            engine=engine,
            provider=provider,
            knowledge=knowledge,
            audit=audit,
            clarifications=self.clarifications,
            memory=self.memory,
            observability=self.observability,
            budgets=self.budgets,
        )

    def chat(
        self,
        request: ChatRequest,
        *,
        on_step: Callable[[AgentTraceStep], None] | None = None,
        approval_gate: Callable[[DispatchIntent, IntentValidation], None] | None = None,
    ) -> ChatResponse:
        mode = request.agent_mode or self.runtime_mode
        if model_request_violation(request.message) is not None:
            return self._blocked_request_response(request, mode=mode, on_step=on_step)
        if mode == "loop":
            return self.loop.chat(
                request,
                on_step=on_step,
                approval_gate=approval_gate,
            )
        return self._chat_linear(
            request,
            on_step=on_step,
            approval_gate=approval_gate,
        )

    def _blocked_request_response(
        self,
        request: ChatRequest,
        *,
        mode: str,
        on_step: Callable[[AgentTraceStep], None] | None,
    ) -> ChatResponse:
        trace_id = new_id("trace")
        run = BoundedAgentRun(
            max_steps=self.budgets.max_steps if mode == "loop" else 16,
            reserve_terminal_step=True,
            on_step=on_step,
        )
        run.set_planner(
            strategy="ACTION_PROTOCOL_LOOP" if mode == "loop" else "DETERMINISTIC_POLICY",
            model="deterministic-safety-boundary",
        )
        run.transition(
            AgentState.RECEIVED,
            title="接收调度请求",
            detail=f"已绑定会话 {request.conversation_id} 和场景 {request.scenario_id}",
        )
        run.transition(
            AgentState.BLOCKED,
            title="确定性安全边界阻断请求",
            detail="请求试图绕过审批、仿真或 simulation 环境边界",
            status="BLOCKED",
            observation_code="policy.user_request_blocked",
        )
        run.set_budget_summary(
            budgets=(
                self.budgets.model_dump(by_alias=True, mode="json")
                if mode == "loop"
                else {}
            ),
            usage={},
            terminal_reason="policy.user_request_blocked",
        )
        trace = run.build_trace()
        self.observability.record(
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            scenario_id=request.scenario_id,
            trace=trace,
            model="deterministic-safety-boundary",
            fallback_used=False,
            validation=None,
        )
        response = ChatResponse(
            traceId=trace_id,
            conversationId=request.conversation_id,
            state="BLOCKED",
            message="请求被确定性安全边界阻断，不能绕过仿真、审批或环境限制。",
            model="deterministic-safety-boundary",
            fallbackUsed=False,
            suggestedActions=["查看安全边界", "修改请求"],
            agentTrace=trace,
        )
        self.audit.append(
            trace_id=trace_id,
            event_type="AGENT_REQUEST_BLOCKED",
            actor=request.requested_by,
            payload={
                "request": request.message,
                "scenarioId": request.scenario_id,
                "mode": mode,
                "reason": "policy.user_request_blocked",
                "agentTrace": trace.model_dump(by_alias=True, mode="json"),
            },
        )
        return response

    def _chat_linear(
        self,
        request: ChatRequest,
        *,
        on_step: Callable[[AgentTraceStep], None] | None = None,
        approval_gate: Callable[[DispatchIntent, IntentValidation], None] | None = None,
    ) -> ChatResponse:
        trace_id = new_id("trace")
        run = BoundedAgentRun(on_step=on_step)
        tools = DispatchAgentTools(
            engine=self.engine,
            knowledge=self.knowledge,
            scenario_id=request.scenario_id,
            memory=self.memory,
            conversation_id=request.conversation_id,
        )
        prior_memory = self.memory.get(request.conversation_id)
        run.transition(
            AgentState.RECEIVED,
            title="接收调度请求",
            detail=f"已绑定会话 {request.conversation_id} 和场景 {request.scenario_id}",
        )

        started = perf_counter()
        plan = self.provider.plan_context_tools(
            request.message,
            tools.model_definitions(),
            has_memory=prior_memory is not None,
        )
        run.set_planner(strategy=plan.strategy, model=plan.model)
        run.transition(
            AgentState.PLANNING,
            title="制定上下文工具计划",
            detail=(
                f"{plan.strategy} 选择 {len(plan.calls)} 个只读工具，"
                "写操作未开放给模型"
            ),
            duration_ms=(perf_counter() - started) * 1000,
        )
        run.transition(
            AgentState.CONTEXT_GATHERING,
            title="收集权威上下文",
            detail="按允许列表执行工具并记录调用结果",
        )
        snapshot = None
        evidence = []
        for call in plan.calls:
            result = run.execute_tool(tools, call.name, call.arguments)
            if call.name == "get_world_snapshot":
                snapshot = result.value
                evidence.append(
                    tools.world_evidence(result.value, request.scenario_id)
                )
            elif call.name == "search_sop":
                evidence.extend(result.value)
                for item in (result.metadata or {}).get("quarantined") or []:
                    run.record(
                        title="隔离可疑检索内容",
                        detail=f"{item['source']}：{item['violation']}",
                        status="BLOCKED",
                        observation_code=str(item["violation"]),
                    )
            elif call.name == "recall_conversation_memory" and result.value is not None:
                evidence.append(
                    tools.memory_evidence(result.value, request.conversation_id)
                )
        if snapshot is None:
            raise RuntimeError("Agent 工具计划缺少强制世界快照")

        started = perf_counter()
        resolved = self.clarifications.resolve(
            request.message, request.conversation_id
        )
        run.transition(
            AgentState.PARAMETER_RESOLUTION,
            title="解析并绑定业务实体",
            detail=(
                "请求参数完整，可以形成结构化意图"
                if resolved.clarification is None
                else "缺少必要参数，暂停执行并请求用户补充"
            ),
            duration_ms=(perf_counter() - started) * 1000,
        )
        if resolved.clarification is not None:
            run.transition(
                AgentState.CLARIFICATION_REQUIRED,
                title="等待补充信息",
                detail="；".join(resolved.clarification.questions),
                status="BLOCKED",
            )
            agent_trace = run.build_trace()
            self.memory.record(
                conversation_id=request.conversation_id,
                scenario_id=request.scenario_id,
                message=request.message,
                outcome="CLARIFICATION_REQUIRED",
                trace=agent_trace,
                clarification=resolved.clarification,
            )
            self.observability.record(
                trace_id=trace_id,
                conversation_id=request.conversation_id,
                scenario_id=request.scenario_id,
                trace=agent_trace,
                model="deterministic-parameter-resolver",
                fallback_used=False,
                validation=None,
            )
            message = "还不能形成可执行草案。" + " ".join(
                resolved.clarification.questions
            )
            response = ChatResponse(
                traceId=trace_id,
                conversationId=request.conversation_id,
                state="CLARIFICATION_REQUIRED",
                message=message,
                clarification=resolved.clarification,
                evidence=evidence,
                model="deterministic-parameter-resolver",
                fallbackUsed=False,
                suggestedActions=resolved.clarification.questions,
                agentTrace=agent_trace,
            )
            self.audit.append(
                trace_id=trace_id,
                event_type="AGENT_CLARIFICATION_REQUESTED",
                actor=request.requested_by,
                payload={
                    "conversationId": request.conversation_id,
                    "request": request.message,
                    "scenarioId": request.scenario_id,
                    "clarification": resolved.clarification.model_dump(
                        by_alias=True, mode="json"
                    ),
                    "agentTrace": agent_trace.model_dump(by_alias=True, mode="json"),
                },
            )
            return response

        started = perf_counter()
        parsed = self.provider.parse_intent(
            resolved.message,
            world_revision=int(snapshot["worldRevision"]),
            requested_by=request.requested_by,
            resolved_task=resolved.task,
            resolved_resource_block=resolved.resource_block,
            context_evidence=evidence,
        )
        if parsed.intent is None:
            raise ValueError("意图解析未生成结构化结果")
        run.transition(
            AgentState.INTENT_DRAFTING,
            title="生成结构化调度意图",
            detail=f"形成 {parsed.intent.intent_type.value} 草案，模型 {parsed.model}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        run.transition(
            AgentState.SAFETY_VALIDATION,
            title="进入确定性安全校验",
            detail="意图必须经过 MASP 规则和风险边界，模型不能跳过此步骤",
        )
        validation_result = run.execute_tool(
            tools,
            "validate_dispatch_intent",
            {"intent": parsed.intent.model_dump(by_alias=True, mode="json")},
        )
        validation = tools.validation_value(validation_result)
        if validation.valid and validation.approval_required and approval_gate:
            approval_gate(parsed.intent, validation)

        intent_type = parsed.intent.intent_type
        memory_reference = prior_memory is not None and any(
            term in request.message
            for term in ("刚才", "之前", "上次", "前面", "哪些实体", "哪些工具")
        )
        if memory_reference and intent_type in {
            IntentType.QUERY_STATUS,
            IntentType.EXPLAIN_DECISION,
        }:
            message = self.memory.summary(prior_memory)
            actions = ["查看 Agent 执行轨迹", "查看依据与引用"]
        elif intent_type is IntentType.CREATE_TASK:
            task = parsed.intent.task
            message = (
                f"已形成紧急运输任务草案：{task.pickup_node_id} 到 "
                f"{task.dropoff_node_id}，由 {task.required_robot_group} 车型执行，"
                f"优先级 {task.priority_class}。"
            )
            actions = ["运行数字孪生", "查看任务参数"]
        elif intent_type is IntentType.BLOCK_RESOURCE:
            resources = "、".join(parsed.intent.resource_block.resource_ids)
            message = (
                f"已识别临时封锁意图：{resources}。该操作属于高风险变更，"
                "必须先运行数字孪生并由调度主管审批。"
            )
            actions = ["运行封路推演", "查看安全规则"]
        elif intent_type is IntentType.GENERATE_REPORT:
            message = "可以根据已完成的仿真和审计记录生成班次运营报告。"
            actions = ["生成运营报告"]
        elif intent_type is IntentType.EXPLAIN_DECISION:
            message = (
                "当前解释必须以MASP规划记录和资源预约为依据。"
                "请先选择一个已完成的仿真方案。"
            )
            actions = ["查看最近方案", "比较仿真结果"]
        else:
            message = (
                f"当前场景共有 {snapshot['counts']['vehicles']} 辆车、"
                f"{snapshot['counts']['tasks']} 个任务和 "
                f"{snapshot['counts']['workstations']} 个工位。"
                "系统处于仿真模式，未连接真实车辆。"
            )
            actions = ["注入紧急任务", "推演通道封闭"]

        if not validation.valid:
            errors = "；".join(
                item.message for item in validation.issues if item.severity == "error"
            )
            message = f"意图未通过确定性校验：{errors}"
            actions = ["修改意图"]

        run.transition(
            AgentState.COMPLETED,
            title="完成 Agent 决策",
            detail=(
                "已生成可继续仿真的安全草案"
                if validation.valid
                else "草案已被确定性安全边界拦截"
            ),
        )
        agent_trace = run.build_trace()
        self.memory.record(
            conversation_id=request.conversation_id,
            scenario_id=request.scenario_id,
            message=request.message,
            outcome="READY",
            trace=agent_trace,
            intent=parsed.intent,
            validation=validation,
        )
        self.observability.record(
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            scenario_id=request.scenario_id,
            trace=agent_trace,
            model=parsed.model,
            fallback_used=parsed.fallback_used,
            validation=validation,
        )

        response = ChatResponse(
            traceId=trace_id,
            conversationId=request.conversation_id,
            message=message,
            intent=parsed.intent,
            validation=validation,
            evidence=evidence,
            model=parsed.model,
            fallbackUsed=parsed.fallback_used,
            suggestedActions=actions,
            agentTrace=agent_trace,
        )
        self.audit.append(
            trace_id=trace_id,
            event_type="AGENT_INTENT_PARSED",
            actor=request.requested_by,
            payload={
                "request": request.message,
                "scenarioId": request.scenario_id,
                "intent": parsed.intent.model_dump(by_alias=True, mode="json"),
                "validation": validation.model_dump(by_alias=True, mode="json"),
                "model": parsed.model,
                "fallbackUsed": parsed.fallback_used,
                "agentTrace": agent_trace.model_dump(by_alias=True, mode="json"),
            },
        )
        return response

    def tool_catalog(self, scenario_id: str = "interactive-multi-fleet") -> list[dict]:
        return DispatchAgentTools(
            engine=self.engine,
            knowledge=self.knowledge,
            scenario_id=scenario_id,
            memory=self.memory,
            conversation_id="catalog",
        ).catalog()

