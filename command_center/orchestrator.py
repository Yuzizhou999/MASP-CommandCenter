from __future__ import annotations

from .audit import AuditStore
from .clarifications import ClarificationResolver, ClarificationStore
from .contracts import (
    ChatRequest,
    ChatResponse,
    EvidenceItem,
    IntentType,
    new_id,
)
from .engine_adapter import MaspAdapter
from .knowledge import KnowledgeBase
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
    ) -> None:
        self.engine = engine
        self.provider = provider
        self.knowledge = knowledge
        self.audit = audit
        self.clarifications = clarifications or ClarificationResolver(
            ClarificationStore(engine.settings.data_dir / "clarifications.json"), engine
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        trace_id = new_id("trace")
        snapshot = self.engine.world_snapshot(request.scenario_id)
        resolved = self.clarifications.resolve(
            request.message, request.conversation_id
        )
        evidence = [
            EvidenceItem(
                source=f"MASP:{request.scenario_id}",
                title="当前世界快照",
                detail=(
                    f"revision {snapshot['worldRevision']}，"
                    f"{snapshot['counts']['vehicles']} 辆车，"
                    f"{snapshot['counts']['tasks']} 个任务，"
                    f"{snapshot['counts']['conflictPairs']} 对冲突资源。"
                ),
            )
        ]
        evidence.extend(self.knowledge.search(request.message, limit=2))
        if resolved.clarification is not None:
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
                },
            )
            return response
        parsed = self.provider.parse_intent(
            resolved.message,
            world_revision=int(snapshot["worldRevision"]),
            requested_by=request.requested_by,
            resolved_task=resolved.task,
            resolved_resource_block=resolved.resource_block,
        )
        if parsed.intent is None:
            raise ValueError("意图解析未生成结构化结果")
        validation = self.engine.validate_intent(parsed.intent, request.scenario_id)

        intent_type = parsed.intent.intent_type
        if intent_type is IntentType.CREATE_TASK:
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
            },
        )
        return response

