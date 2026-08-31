from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


class IntentType(str, Enum):
    QUERY_STATUS = "QUERY_STATUS"
    EXPLAIN_DECISION = "EXPLAIN_DECISION"
    CREATE_TASK = "CREATE_TASK"
    CHANGE_TASK_PRIORITY = "CHANGE_TASK_PRIORITY"
    RUN_WHAT_IF = "RUN_WHAT_IF"
    REPORT_VEHICLE_FAULT = "REPORT_VEHICLE_FAULT"
    BLOCK_RESOURCE = "BLOCK_RESOURCE"
    CANCEL_TASK = "CANCEL_TASK"
    REQUEST_RECOVERY = "REQUEST_RECOVERY"
    GENERATE_REPORT = "GENERATE_REPORT"


class RiskLevel(str, Enum):
    R0_READ_ONLY = "R0_READ_ONLY"
    R1_LOW = "R1_LOW"
    R2_MEDIUM = "R2_MEDIUM"
    R3_HIGH = "R3_HIGH"
    R4_FORBIDDEN = "R4_FORBIDDEN"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class IncidentType(str, Enum):
    VEHICLE_FAULT = "VEHICLE_FAULT"
    WORKSTATION_DISABLED = "WORKSTATION_DISABLED"
    TASK_FAILED = "TASK_FAILED"
    PLANNING_FAILED = "PLANNING_FAILED"
    PLANNING_TIMEOUT = "PLANNING_TIMEOUT"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
    DEADLOCK_RISK = "DEADLOCK_RISK"
    SAFETY_STOP = "SAFETY_STOP"
    SIMULATION_ERROR = "SIMULATION_ERROR"


class IncidentSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, Enum):
    OPEN = "OPEN"
    DIAGNOSED = "DIAGNOSED"
    MITIGATING = "MITIGATING"
    RESOLVED = "RESOLVED"


class WhatIfMode(str, Enum):
    WAIT_RECOVERY = "WAIT_RECOVERY"
    ISOLATE_REASSIGN = "ISOLATE_REASSIGN"
    SUSPEND_AFFECTED_TASKS = "SUSPEND_AFFECTED_TASKS"
    CONTROLLED_REVERSE = "CONTROLLED_REVERSE"
    SAFETY_STOP = "SAFETY_STOP"


class TaskDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    task_id: str = Field(default_factory=lambda: new_id("urgent-task"), alias="taskId")
    release_time_ms: int = Field(default=0, ge=0, alias="releaseTimeMs")
    pickup_node_id: str = Field(alias="pickupNodeId")
    dropoff_node_id: str = Field(alias="dropoffNodeId")
    required_robot_group: Literal["fork", "jack"] = Field(alias="requiredRobotGroup")
    payload_type: str = Field(default="pallet", min_length=1, alias="payloadType")
    payload_id: str | None = Field(default=None, alias="payloadId")
    pickup_service_ms: int = Field(default=5000, gt=0, alias="pickupServiceMs")
    dropoff_service_ms: int = Field(default=5000, gt=0, alias="dropoffServiceMs")
    priority_class: int = Field(default=3, ge=0, le=9, alias="priorityClass")
    due_time_ms: int | None = Field(default=300000, ge=0, alias="dueTimeMs")

    @model_validator(mode="after")
    def validate_due_time(self) -> TaskDraft:
        if self.due_time_ms is not None and self.due_time_ms < self.release_time_ms:
            raise ValueError("dueTimeMs must not be earlier than releaseTimeMs")
        return self


class ResourceBlockDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    resource_ids: list[str] = Field(min_length=1, alias="resourceIds")
    start_ms: int = Field(default=0, ge=0, alias="startMs")
    end_ms: int = Field(default=180000, gt=0, alias="endMs")
    reason: str = Field(default="现场检修临时封闭", min_length=2)

    @model_validator(mode="after")
    def validate_window(self) -> ResourceBlockDraft:
        if self.end_ms <= self.start_ms:
            raise ValueError("endMs must be later than startMs")
        return self


class DispatchIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schemaVersion")
    intent_id: str = Field(default_factory=lambda: new_id("intent"), alias="intentId")
    intent_type: IntentType = Field(alias="intentType")
    requested_by: str = Field(default="demo-operator", alias="requestedBy")
    environment: Literal["simulation", "shadow", "production"] = "simulation"
    based_on_world_revision: int = Field(default=0, ge=0, alias="basedOnWorldRevision")
    reason: str = Field(default="用户调度请求", min_length=1)
    task: TaskDraft | None = None
    resource_block: ResourceBlockDraft | None = Field(
        default=None, alias="resourceBlock"
    )
    query: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> DispatchIntent:
        if self.intent_type is IntentType.CREATE_TASK and self.task is None:
            raise ValueError("CREATE_TASK requires task")
        if self.intent_type is IntentType.BLOCK_RESOURCE and self.resource_block is None:
            raise ValueError("BLOCK_RESOURCE requires resourceBlock")
        return self


class ValidationIssue(BaseModel):
    code: str
    message: str
    severity: Literal["error", "warning"]


class IntentValidation(BaseModel):
    intent_id: str = Field(alias="intentId")
    valid: bool
    risk_level: RiskLevel = Field(alias="riskLevel")
    approval_required: bool = Field(alias="approvalRequired")
    policy_code: str = Field(alias="policyCode")
    issues: list[ValidationIssue] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utc_now, alias="checkedAt")


class AgentPolicyOptions(BaseModel):
    """Run-time controls for the server-registered vehicle policy model."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model_id: str | None = Field(default=None, alias="modelId")
    candidate_count: int = Field(default=2, ge=1, le=8, alias="candidateCount")
    allow_deviation: bool = Field(default=True, alias="allowDeviation")


class AgentModelStatus(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    model_version: str = Field(alias="modelVersion")
    algorithm: str
    mode: Literal["LEARNED", "BASELINE"]
    configured: bool
    checkpoint_present: bool = Field(alias="checkpointPresent")
    checkpoint_name: str | None = Field(default=None, alias="checkpointName")
    checkpoint_sha256: str | None = Field(default=None, alias="checkpointSha256")
    device: str
    safety_controller: str = Field(alias="safetyController")
    notice: str


class AgentPolicyEvidence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    requested: bool = True
    mode: Literal["LEARNED", "BASELINE"]
    model_id: str = Field(alias="modelId")
    model_version: str = Field(alias="modelVersion")
    checkpoint_sha256: str | None = Field(default=None, alias="checkpointSha256")
    candidate_count: int = Field(alias="candidateCount")
    deviation_requested: bool = Field(alias="deviationRequested")
    deviation_enabled: bool = Field(alias="deviationEnabled")
    inference_count: int = Field(default=0, alias="inferenceCount")
    inference_ms: float = Field(default=0.0, alias="inferenceMs")
    fallback_count: int = Field(default=0, alias="fallbackCount")
    safety_fallback_count: int = Field(default=0, alias="safetyFallbackCount")
    guardian_candidate_count: int = Field(default=0, alias="guardianCandidateCount")
    guardian_override_count: int = Field(default=0, alias="guardianOverrideCount")
    agent_candidate_count: int = Field(default=0, alias="agentCandidateCount")
    selected_agent_candidate_count: int = Field(
        default=0, alias="selectedAgentCandidateCount"
    )
    decision_cycle_count: int = Field(default=0, alias="decisionCycleCount")
    fallback_reasons: list[str] = Field(default_factory=list, alias="fallbackReasons")
    notes: list[str] = Field(default_factory=list)
    evidence_path: str | None = Field(default=None, alias="evidencePath")


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    suite_name: str = Field(default="仓储群车高负载基准", min_length=2, alias="suiteName")
    base_scenario_id: str = Field(
        default="rhpp-long-distance-conflict", alias="baseScenarioId"
    )
    vehicle_counts: list[int] = Field(
        default_factory=lambda: [14], min_length=1, max_length=4, alias="vehicleCounts"
    )
    arrival_profiles: list[Literal["low", "medium", "high"]] = Field(
        default_factory=lambda: ["high"],
        min_length=1,
        max_length=3,
        alias="arrivalProfiles",
    )
    fleet_mixes: list[Literal["mixed", "fork", "jack"]] = Field(
        default_factory=lambda: ["mixed"],
        min_length=1,
        max_length=3,
        alias="fleetMixes",
    )
    policies: list[
        Literal[
            "top_k",
            "task_age",
            "shortest_remaining",
            "congestion",
            "previous_order",
            "random",
            "rl",
        ]
    ] = Field(default_factory=lambda: ["top_k", "congestion"], min_length=1)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1, max_length=10)
    horizon_ms: int = Field(default=900000, ge=60000, le=7200000, alias="horizonMs")
    agent_policy: AgentPolicyOptions | None = Field(default=None, alias="agentPolicy")
    requested_by: str = Field(default="evaluation-operator", alias="requestedBy")

    @model_validator(mode="after")
    def validate_matrix(self) -> BenchmarkRequest:
        if any(value < 1 or value > 100 for value in self.vehicle_counts):
            raise ValueError("vehicleCounts must be between 1 and 100")
        for name in (
            "vehicle_counts",
            "arrival_profiles",
            "fleet_mixes",
            "policies",
            "seeds",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        case_count = (
            len(self.vehicle_counts)
            * len(self.arrival_profiles)
            * len(self.fleet_mixes)
            * len(self.policies)
            * len(self.seeds)
        )
        if case_count > 2000:
            raise ValueError("benchmark matrix exceeds the 2000 case limit")
        if "rl" not in self.policies and self.agent_policy is not None:
            raise ValueError("agentPolicy requires the rl policy")
        return self

    @property
    def case_count(self) -> int:
        return (
            len(self.vehicle_counts)
            * len(self.arrival_profiles)
            * len(self.fleet_mixes)
            * len(self.policies)
            * len(self.seeds)
        )


class ModelEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    suite_name: str = Field(
        default="大模型调度安全回归", min_length=2, max_length=80, alias="suiteName"
    )
    requested_by: str = Field(default="model-evaluator", alias="requestedBy")


class DatasetExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(default="仓储调度评测数据", min_length=2)
    include_audit: bool = Field(default=True, alias="includeAudit")
    include_incidents: bool = Field(default=True, alias="includeIncidents")
    include_evidence_text: bool = Field(default=False, alias="includeEvidenceText")
    requested_by: str = Field(default="data-steward", alias="requestedBy")


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    scenario_id: str = Field(default="interactive-multi-fleet", alias="scenarioId")
    policy: Literal[
        "top_k",
        "task_age",
        "shortest_remaining",
        "congestion",
        "previous_order",
        "random",
        "rl",
    ] = "top_k"
    seed: int = 0
    intent: DispatchIntent | None = None
    label: str = "候选方案"
    agent_policy: AgentPolicyOptions | None = Field(default=None, alias="agentPolicy")

    @model_validator(mode="after")
    def validate_agent_policy(self) -> SimulationRequest:
        if self.policy != "rl" and self.agent_policy is not None:
            raise ValueError("agentPolicy is only valid when policy is rl")
        return self


class SimulationSummary(BaseModel):
    run_id: str = Field(alias="runId")
    scenario_id: str = Field(alias="scenarioId")
    label: str
    policy: str
    seed: int
    status: Literal["COMPLETED", "FAILED"]
    duration_ms: float = Field(alias="durationMs")
    metrics: dict[str, Any]
    planning: dict[str, Any]
    safety: dict[str, Any]
    agent_policy: AgentPolicyEvidence | None = Field(default=None, alias="agentPolicy")
    intent_id: str | None = Field(default=None, alias="intentId")
    manifest_path: str | None = Field(default=None, alias="manifestPath")
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")
    error: str | None = None


class ComparisonRequest(BaseModel):
    run_ids: list[str] = Field(min_length=2, max_length=4, alias="runIds")


class ComparisonResult(BaseModel):
    comparison_id: str = Field(default_factory=lambda: new_id("comparison"), alias="comparisonId")
    runs: list[SimulationSummary]
    recommended_run_id: str = Field(alias="recommendedRunId")
    rationale: list[str]
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")


class ApprovalRequest(BaseModel):
    approval_id: str = Field(default_factory=lambda: new_id("approval"), alias="approvalId")
    intent: DispatchIntent
    validation: IntentValidation
    simulation_run_ids: list[str] = Field(default_factory=list, alias="simulationRunIds")
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_by: str = Field(default="demo-operator", alias="requestedBy")
    decided_by: str | None = Field(default=None, alias="decidedBy")
    decision_reason: str | None = Field(default=None, alias="decisionReason")
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")
    decided_at: datetime | None = Field(default=None, alias="decidedAt")


class ApprovalDecision(BaseModel):
    approved: bool
    decided_by: str = Field(default="demo-supervisor", alias="decidedBy")
    reason: str = Field(default="已核对仿真结果和影响范围")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    scenario_id: str = Field(default="interactive-multi-fleet", alias="scenarioId")
    requested_by: str = Field(default="demo-operator", alias="requestedBy")
    conversation_id: str = Field(
        default_factory=lambda: new_id("conversation"), alias="conversationId"
    )
    agent_mode: Literal["linear", "loop"] | None = Field(
        default=None, alias="agentMode"
    )


class AgentRunCreateRequest(ChatRequest):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    timeout_seconds: int = Field(default=60, ge=5, le=300, alias="timeoutSeconds")
    execution_mode: Literal["ADVISORY", "GOAL_EXECUTION"] = Field(
        default="ADVISORY", alias="executionMode"
    )


class AgentRunResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    approved: bool
    decided_by: str = Field(default="demo-supervisor", alias="decidedBy")
    reason: str = Field(default="已核对 Agent 草案和风险边界", min_length=2)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: str
    title: str
    detail: str
    chunk_id: str | None = Field(default=None, alias="chunkId")
    score: float | None = Field(default=None, ge=0, le=1)
    retrieval_method: str | None = Field(default=None, alias="retrievalMethod")


class ClarificationRequest(BaseModel):
    code: Literal["MISSING_REQUIRED_FIELDS", "AMBIGUOUS_ENTITY"]
    missing_fields: list[str] = Field(default_factory=list, alias="missingFields")
    questions: list[str] = Field(default_factory=list)
    collected_parameters: dict[str, Any] = Field(
        default_factory=dict, alias="collectedParameters"
    )


class AgentTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    step_id: str = Field(default_factory=lambda: new_id("agent-step"), alias="stepId")
    sequence: int = Field(ge=1)
    state: Literal[
        "RECEIVED",
        "PLANNING",
        "DECIDING",
        "OBSERVING",
        "CONTEXT_GATHERING",
        "PARAMETER_RESOLUTION",
        "INTENT_DRAFTING",
        "REPAIRING",
        "SAFETY_VALIDATION",
        "CLARIFICATION_REQUIRED",
        "BLOCKED",
        "BUDGET_EXCEEDED",
        "COMPLETED",
    ]
    status: Literal["COMPLETED", "BLOCKED", "FAILED", "REJECTED"] = "COMPLETED"
    title: str
    detail: str
    tool_name: str | None = Field(default=None, alias="toolName")
    read_only: bool | None = Field(default=None, alias="readOnly")
    action: str | None = None
    observation_code: str | None = Field(default=None, alias="observationCode")
    attempt: int | None = Field(default=None, ge=1)
    prompt_tokens: int = Field(default=0, ge=0, alias="promptTokens")
    completion_tokens: int = Field(default=0, ge=0, alias="completionTokens")
    duration_ms: float = Field(default=0, ge=0, alias="durationMs")


class AgentRunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event_id: int = Field(ge=1, alias="eventId")
    event_type: str = Field(alias="eventType")
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")


class AgentRunEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    passed: bool
    score: float = Field(ge=0, le=1)
    checks: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class AgentWorkflowRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    decision: Literal["PROCEED", "BLOCK"]
    reasons: list[str] = Field(default_factory=list)
    safety_checks: dict[str, bool] = Field(
        default_factory=dict, alias="safetyChecks"
    )


class AgentWorkflowStep(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sequence: int = Field(ge=1)
    action: Literal["SIMULATE", "REQUEST_APPROVAL", "COMMIT"]
    status: Literal["RUNNING", "COMPLETED", "BLOCKED", "FAILED"]
    title: str
    detail: str
    output_ref: str | None = Field(default=None, alias="outputRef")
    duration_ms: float = Field(default=0, ge=0, alias="durationMs")


class AgentGoalWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    phase: Literal[
        "PENDING",
        "NOT_APPLICABLE",
        "SIMULATING",
        "WAITING_APPROVAL",
        "COMMITTING",
        "COMPLETED",
        "BLOCKED",
    ] = "PENDING"
    intent_id: str | None = Field(default=None, alias="intentId")
    simulation: dict[str, Any] | None = None
    recommendation: AgentWorkflowRecommendation | None = None
    approval_request: ApprovalRequest | None = Field(
        default=None, alias="approvalRequest"
    )
    commitment: dict[str, Any] | None = None
    steps: list[AgentWorkflowStep] = Field(default_factory=list)


class AgentRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    run_id: str = Field(alias="runId")
    status: Literal[
        "QUEUED",
        "RUNNING",
        "WAITING_APPROVAL",
        "COMPLETED",
        "REJECTED",
        "CANCELLED",
        "TIMED_OUT",
        "FAILED",
    ]
    request: AgentRunCreateRequest
    idempotency_key: str | None = Field(default=None, alias="idempotencyKey")
    attempt: int = Field(default=0, ge=0)
    recovered: bool = False
    cancel_requested: bool = Field(default=False, alias="cancelRequested")
    trace_steps: list[AgentTraceStep] = Field(default_factory=list, alias="traceSteps")
    response: ChatResponse | None = None
    approval: dict[str, Any] | None = None
    evaluation: AgentRunEvaluation | None = None
    workflow: AgentGoalWorkflow | None = None
    provider_usage: dict[str, Any] = Field(default_factory=dict, alias="providerUsage")
    error: str | None = None
    events: list[AgentRunEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")
    updated_at: datetime = Field(default_factory=utc_now, alias="updatedAt")
    deadline_at: datetime = Field(alias="deadlineAt")
    started_at: datetime | None = Field(default=None, alias="startedAt")
    completed_at: datetime | None = Field(default=None, alias="completedAt")


class AgentExecutionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    strategy: Literal[
        "MODEL_TOOL_CALLING",
        "DETERMINISTIC_POLICY",
        "ACTION_PROTOCOL_LOOP",
    ]
    planner_model: str = Field(alias="plannerModel")
    status: Literal[
        "COMPLETED",
        "CLARIFICATION_REQUIRED",
        "BLOCKED",
        "BUDGET_EXCEEDED",
        "FAILED",
    ]
    max_steps: int = Field(ge=1, alias="maxSteps")
    duration_ms: float = Field(ge=0, alias="durationMs")
    budgets: dict[str, int | float] = Field(default_factory=dict)
    usage: dict[str, int | float] = Field(default_factory=dict)
    terminal_reason: str | None = Field(default=None, alias="terminalReason")
    steps: list[AgentTraceStep] = Field(default_factory=list)


class AgentMemoryTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    message: str
    outcome: Literal["READY", "CLARIFICATION_REQUIRED"]
    intent_type: str | None = Field(default=None, alias="intentType")
    risk_level: str | None = Field(default=None, alias="riskLevel")
    tool_names: list[str] = Field(default_factory=list, alias="toolNames")
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")


class AgentConversationMemory(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    conversation_id: str = Field(alias="conversationId")
    scenario_id: str = Field(alias="scenarioId")
    confirmed_entities: dict[str, list[str]] = Field(
        default_factory=dict, alias="confirmedEntities"
    )
    turns: list[AgentMemoryTurn] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now, alias="updatedAt")


class ChatResponse(BaseModel):
    trace_id: str = Field(default_factory=lambda: new_id("trace"), alias="traceId")
    conversation_id: str = Field(alias="conversationId")
    state: Literal[
        "READY",
        "CLARIFICATION_REQUIRED",
        "BLOCKED",
        "BUDGET_EXCEEDED",
    ] = "READY"
    message: str
    intent: DispatchIntent | None = None
    validation: IntentValidation | None = None
    clarification: ClarificationRequest | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    model: str
    fallback_used: bool = Field(alias="fallbackUsed")
    suggested_actions: list[str] = Field(default_factory=list, alias="suggestedActions")
    agent_trace: AgentExecutionTrace | None = Field(default=None, alias="agentTrace")


AgentRunRecord.model_rebuild()


class PlanExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    question: str = Field(default="为什么这样调度？", min_length=1, max_length=1000)
    vehicle_id: str | None = Field(default=None, alias="vehicleId")
    task_id: str | None = Field(default=None, alias="taskId")
    requested_by: str = Field(default="demo-operator", alias="requestedBy")


class PlanExplanationEvidence(BaseModel):
    evidence_id: str = Field(alias="evidenceId")
    category: Literal["RUN", "ASSIGNMENT", "WAIT", "ROUTE", "SAFETY", "FALLBACK"]
    fact: str
    source: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class PlanExplanationFinding(BaseModel):
    code: str
    title: str
    explanation: str
    classification: Literal["FACT", "INFERENCE"]
    evidence_ids: list[str] = Field(min_length=1, alias="evidenceIds")


class PlanExplanationNarrative(BaseModel):
    summary: str
    findings: list[PlanExplanationFinding] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class PlanExplanationReport(PlanExplanationNarrative):
    schema_version: Literal[1] = Field(default=1, alias="schemaVersion")
    run_id: str = Field(alias="runId")
    question: str
    vehicle_id: str | None = Field(default=None, alias="vehicleId")
    task_id: str | None = Field(default=None, alias="taskId")
    evidence: list[PlanExplanationEvidence] = Field(default_factory=list)
    model: str
    fallback_used: bool = Field(alias="fallbackUsed")
    generated_at: datetime = Field(default_factory=utc_now, alias="generatedAt")


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("audit"), alias="eventId")
    trace_id: str = Field(alias="traceId")
    event_type: str = Field(alias="eventType")
    actor: str
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")


class FaultInjectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    run_id: str = Field(alias="runId")
    vehicle_id: str | None = Field(default=None, alias="vehicleId")
    fault_code: str = Field(
        default="DRIVE_MOTOR_OVERHEAT", min_length=2, max_length=80, alias="faultCode"
    )
    requested_at_ms: int | None = Field(default=None, ge=0, alias="requestedAtMs")
    recovery_duration_ms: int = Field(
        default=120000, ge=10000, le=900000, alias="recoveryDurationMs"
    )
    requested_by: str = Field(default="demo-operator", alias="requestedBy")


class WorkstationInjectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    run_id: str = Field(alias="runId")
    workstation_node_id: str | None = Field(default=None, alias="workstationNodeId")
    requested_at_ms: int | None = Field(default=None, ge=0, alias="requestedAtMs")
    recovery_duration_ms: int = Field(
        default=180000, ge=10000, le=900000, alias="recoveryDurationMs"
    )
    requested_by: str = Field(default="demo-operator", alias="requestedBy")


class DeadlockInjectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    run_id: str = Field(alias="runId")
    deadlock_case: Literal["RECOVERABLE", "UNRECOVERABLE"] = Field(
        default="RECOVERABLE", alias="deadlockCase"
    )
    requested_by: str = Field(default="demo-operator", alias="requestedBy")


class IncidentEvidence(BaseModel):
    evidence_id: str = Field(alias="evidenceId")
    evidence_type: str = Field(alias="evidenceType")
    fact: str
    source: str
    observed_at_ms: int | None = Field(default=None, alias="observedAtMs")
    attributes: dict[str, Any] = Field(default_factory=dict)


class DeterministicFinding(BaseModel):
    code: str
    title: str
    detail: str
    certainty: Literal["CONFIRMED", "INFERRED"]
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")


class RootCauseCandidate(BaseModel):
    code: str
    title: str
    explanation: str
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1, alias="evidenceIds")
    classification: Literal["FACT", "INFERENCE"] = "INFERENCE"


class IncidentRecommendation(BaseModel):
    action_code: str = Field(alias="actionCode")
    action: str
    rationale: str
    risk_level: RiskLevel = Field(alias="riskLevel")
    requires_simulation: bool = Field(default=True, alias="requiresSimulation")
    requires_approval: bool = Field(default=True, alias="requiresApproval")
    evidence_ids: list[str] = Field(min_length=1, alias="evidenceIds")


class DiagnosisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    summary: str
    confirmed_facts: list[str] = Field(alias="confirmedFacts")
    root_cause_candidates: list[RootCauseCandidate] = Field(
        min_length=1, alias="rootCauseCandidates"
    )
    affected_vehicle_ids: list[str] = Field(default_factory=list, alias="affectedVehicleIds")
    affected_task_ids: list[str] = Field(default_factory=list, alias="affectedTaskIds")
    recommendations: list[IncidentRecommendation] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    model: str = "deterministic-fallback"
    fallback_used: bool = Field(default=True, alias="fallbackUsed")
    generated_at: datetime = Field(default_factory=utc_now, alias="generatedAt")


class IncidentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    incident_id: str = Field(default_factory=lambda: new_id("incident"), alias="incidentId")
    incident_type: IncidentType = Field(alias="incidentType")
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.OPEN
    scenario_id: str = Field(alias="scenarioId")
    run_id: str = Field(alias="runId")
    vehicle_ids: list[str] = Field(default_factory=list, alias="vehicleIds")
    task_ids: list[str] = Field(default_factory=list, alias="taskIds")
    resource_ids: list[str] = Field(default_factory=list, alias="resourceIds")
    fault_code: str | None = Field(default=None, alias="faultCode")
    fault_at_ms: int = Field(ge=0, alias="faultAtMs")
    recovery_duration_ms: int = Field(ge=0, alias="recoveryDurationMs")
    location_node_id: str | None = Field(default=None, alias="locationNodeId")
    location_edge_id: str | None = Field(default=None, alias="locationEdgeId")
    workstation_id: str | None = Field(default=None, alias="workstationId")
    load_state: str | None = Field(default=None, alias="loadState")
    event_attributes: dict[str, Any] = Field(default_factory=dict, alias="eventAttributes")
    evidence: list[IncidentEvidence] = Field(default_factory=list)
    deterministic_findings: list[DeterministicFinding] = Field(
        default_factory=list, alias="deterministicFindings"
    )
    diagnosis: DiagnosisReport | None = None
    what_if_run_ids: dict[str, str] = Field(default_factory=dict, alias="whatIfRunIds")
    approval_ids: dict[str, str] = Field(default_factory=dict, alias="approvalIds")
    created_by: str = Field(default="demo-operator", alias="createdBy")
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")
    updated_at: datetime = Field(default_factory=utc_now, alias="updatedAt")


class IncidentWhatIfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: WhatIfMode
    requested_by: str = Field(default="demo-operator", alias="requestedBy")


class IncidentApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: WhatIfMode
    requested_by: str = Field(default="demo-operator", alias="requestedBy")
