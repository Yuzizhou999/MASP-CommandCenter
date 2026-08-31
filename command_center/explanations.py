from __future__ import annotations

from typing import Any

from .audit import AuditStore
from .contracts import (
    PlanExplanationEvidence,
    PlanExplanationFinding,
    PlanExplanationReport,
    PlanExplanationRequest,
    new_id,
)
from .engine_adapter import MaspAdapter
from .provider import DeepSeekProvider


class PlanExplanationService:
    """Build explanations from persisted MASP planning evidence."""

    def __init__(
        self,
        *,
        engine: MaspAdapter,
        provider: DeepSeekProvider,
        audit: AuditStore,
    ) -> None:
        self.engine = engine
        self.provider = provider
        self.audit = audit

    @staticmethod
    def _matches(row: dict[str, Any], request: PlanExplanationRequest) -> bool:
        if request.vehicle_id and row.get("vehicleId") != request.vehicle_id:
            return False
        if request.task_id and row.get("taskId") != request.task_id:
            return False
        return True

    def _deterministic(
        self, run_id: str, request: PlanExplanationRequest
    ) -> PlanExplanationReport:
        detail = self.engine.get_run_detail(run_id)
        summary = detail["summary"]
        planning = detail["planning"]
        scenario = detail["scenario"]
        evidence: list[PlanExplanationEvidence] = []
        findings: list[PlanExplanationFinding] = []
        uncertainties: list[str] = []

        def add_evidence(
            category: str, fact: str, source: str, attributes: dict[str, Any]
        ) -> str:
            evidence_id = f"PE-{len(evidence) + 1:03d}"
            evidence.append(
                PlanExplanationEvidence(
                    evidenceId=evidence_id,
                    category=category,
                    fact=fact,
                    source=source,
                    attributes=attributes,
                )
            )
            return evidence_id

        run_evidence = add_evidence(
            "RUN",
            (
                f"运行 {run_id} 使用 {summary['policy']} 策略，状态为 "
                f"{summary['status']}。"
            ),
            "command-center-summary.json",
            {
                "policy": summary["policy"],
                "status": summary["status"],
                "seed": summary["seed"],
                "scenarioId": summary["scenarioId"],
            },
        )
        safety = summary.get("safety", {})
        safety_evidence = add_evidence(
            "SAFETY",
            (
                "MASP 安全结果记录资源预约冲突拒绝 "
                f"{safety.get('reservationConflictRejections', 0)} 次，"
                f"未规划任务 {safety.get('unplannedTaskCount', 0)} 个。"
            ),
            "command-center-summary.json#safety",
            dict(safety),
        )
        findings.append(
            PlanExplanationFinding(
                code="PLAN.SAFETY_BOUNDARY",
                title="计划经过确定性安全校验",
                explanation=(
                    "该结果来自 MASP 的连续时间规划、资源预约和计划校验，"
                    "不是由大模型生成路线。"
                ),
                classification="FACT",
                evidenceIds=[run_evidence, safety_evidence],
            )
        )

        vehicles = {row["vehicleId"]: row for row in scenario.get("vehicles", [])}
        tasks = {row["taskId"]: row for row in scenario.get("tasks", [])}
        assignments = [
            row
            for row in planning.get("assignments", [])
            if self._matches(row, request)
        ]
        plans = [
            row for row in scenario.get("plans", []) if self._matches(row, request)
        ]
        if request.vehicle_id and request.vehicle_id not in vehicles:
            raise ValueError(f"运行中不存在车辆 {request.vehicle_id}")
        if request.task_id and request.task_id not in tasks:
            raise ValueError(f"运行中不存在任务 {request.task_id}")

        for assignment in assignments[:12]:
            vehicle_id = assignment["vehicleId"]
            task_id = assignment["taskId"]
            vehicle_group = vehicles.get(vehicle_id, {}).get("robotGroup")
            task_group = tasks.get(task_id, {}).get("requiredRobotGroup")
            assignment_evidence = add_evidence(
                "ASSIGNMENT",
                (
                    f"{task_id} 在 {assignment['decisionTimeMs']} ms 分配给 {vehicle_id}，"
                    f"预计完成时刻 {assignment['completionTimeMs']} ms。"
                ),
                "planning-summary.json#assignments",
                dict(assignment),
            )
            explanation = (
                f"车辆和任务车型均为 {vehicle_group}，能力约束匹配；"
                f"本次候选的分配代价为 {assignment['assignmentCostMs']} ms。"
                if vehicle_group and vehicle_group == task_group
                else "规划记录确认了该任务与车辆的分配关系。"
            )
            findings.append(
                PlanExplanationFinding(
                    code="PLAN.ASSIGNMENT_SELECTED",
                    title=f"{task_id} 分配给 {vehicle_id}",
                    explanation=explanation,
                    classification="FACT",
                    evidenceIds=[assignment_evidence],
                )
            )
            wait_ms = int(assignment.get("insertedWaitMs") or 0)
            if wait_ms > 0:
                wait_evidence = add_evidence(
                    "WAIT",
                    f"该次分配插入了 {wait_ms} ms 等待。",
                    "planning-summary.json#assignments.insertedWaitMs",
                    {
                        "vehicleId": vehicle_id,
                        "taskId": task_id,
                        "insertedWaitMs": wait_ms,
                    },
                )
                findings.append(
                    PlanExplanationFinding(
                        code="PLAN.WAIT_INSERTED",
                        title=f"{vehicle_id} 的等待由时序规划插入",
                        explanation=(
                            "等待用于满足连续时间路径和资源预约的可行时窗；"
                            "当前汇总证据不能把全部等待归因到某一个资源。"
                        ),
                        classification="INFERENCE",
                        evidenceIds=[wait_evidence, safety_evidence],
                    )
                )

        for plan in plans[:12]:
            segments = plan.get("segments", [])
            traversals = [row for row in segments if row.get("kind") == "traverse"]
            waits = [row for row in segments if row.get("kind") == "wait"]
            resources = sorted(
                {
                    resource
                    for row in segments
                    for resource in row.get("resourceIds", [])
                }
            )
            route_evidence = add_evidence(
                "ROUTE",
                (
                    f"计划 {plan['id']} 包含 {len(traversals)} 个移动段、"
                    f"{len(waits)} 个等待段，并预约 {len(resources)} 个唯一资源。"
                ),
                "planned-scenario.json#plans",
                {
                    "planId": plan["id"],
                    "vehicleId": plan["vehicleId"],
                    "taskId": plan["taskId"],
                    "traverseSegmentCount": len(traversals),
                    "waitSegmentCount": len(waits),
                    "resourceCount": len(resources),
                    "startNodeId": traversals[0].get("startNodeId")
                    if traversals
                    else None,
                    "endNodeId": traversals[-1].get("endNodeId")
                    if traversals
                    else None,
                },
            )
            findings.append(
                PlanExplanationFinding(
                    code="PLAN.ROUTE_RESERVED",
                    title=f"{plan['vehicleId']} 的路线按时窗预约",
                    explanation=(
                        "路线中的节点、道路、冲突区和工位资源均随计划段记录，"
                        "车辆只能按通过校验的时序计划运行。"
                    ),
                    classification="FACT",
                    evidenceIds=[route_evidence, safety_evidence],
                )
            )

        tried = int(planning.get("routeCombinationsTried") or 0)
        pruned = int(planning.get("routeCombinationsPruned") or 0)
        if tried or pruned:
            route_search_evidence = add_evidence(
                "ROUTE",
                f"规划器尝试 {tried} 个路线组合，剪枝 {pruned} 个组合。",
                "planning-summary.json#route-search",
                {"routeCombinationsTried": tried, "routeCombinationsPruned": pruned},
            )
            findings.append(
                PlanExplanationFinding(
                    code="PLAN.ROUTE_ALTERNATIVES",
                    title="候选路线经过可行性筛选",
                    explanation=(
                        "候选组合会按时间窗、资源冲突和计划可行性筛选；"
                        "当前汇总记录不包含每条被排除路线的逐项原因。"
                    ),
                    classification="INFERENCE",
                    evidenceIds=[route_search_evidence],
                )
            )
            uncertainties.append("逐条候选路线的排除原因未写入当前规划汇总。")

        fallback_reasons = list(
            (summary.get("agentPolicy") or {}).get("fallbackReasons", [])
        )
        deadline_count = int(planning.get("planningDeadlineExhaustedCount") or 0)
        sipp_deadline_count = int(planning.get("sippDeadlineExhaustedCount") or 0)
        if fallback_reasons or deadline_count or sipp_deadline_count:
            fallback_evidence = add_evidence(
                "FALLBACK",
                (
                    f"策略接管原因 {fallback_reasons or ['无']}；规划周期耗尽 "
                    f"{deadline_count} 次，SIPP 时限耗尽 {sipp_deadline_count} 次。"
                ),
                "planning-summary.json#fallback",
                {
                    "fallbackReasons": fallback_reasons,
                    "planningDeadlineExhaustedCount": deadline_count,
                    "sippDeadlineExhaustedCount": sipp_deadline_count,
                },
            )
            findings.append(
                PlanExplanationFinding(
                    code="PLAN.FALLBACK_RECORDED",
                    title="计划回退与时限状态已留证",
                    explanation=(
                        "模型候选不可用或规划超出周期时，系统保留确定性规则候选，"
                        "并把原因写入运行证据。"
                    ),
                    classification="FACT",
                    evidenceIds=[fallback_evidence],
                )
            )

        if not assignments:
            uncertainties.append("当前筛选条件下没有任务分配记录。")
        if not plans:
            uncertainties.append("当前筛选条件下没有可解释的计划段。")
        return PlanExplanationReport(
            runId=run_id,
            question=request.question,
            vehicleId=request.vehicle_id,
            taskId=request.task_id,
            summary=(
                f"运行 {run_id} 的解释来自 {len(evidence)} 条 MASP 持久化证据，"
                "结论按事实与推断分开标注。"
            ),
            findings=findings,
            uncertainties=list(dict.fromkeys(uncertainties)),
            evidence=evidence,
            model="deterministic-evidence-explainer",
            fallbackUsed=True,
        )

    def explain(
        self, run_id: str, request: PlanExplanationRequest
    ) -> PlanExplanationReport:
        deterministic = self._deterministic(run_id, request)
        report = self.provider.explain_plan(deterministic)
        self.audit.append(
            trace_id=new_id("trace"),
            event_type="PLAN_EXPLAINED",
            actor=request.requested_by,
            payload={
                "runId": run_id,
                "vehicleId": request.vehicle_id,
                "taskId": request.task_id,
                "question": request.question,
                "evidenceIds": [row.evidence_id for row in report.evidence],
                "model": report.model,
                "fallbackUsed": report.fallback_used,
            },
        )
        return report
