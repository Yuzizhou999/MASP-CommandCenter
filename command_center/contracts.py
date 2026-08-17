from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    def validate_due_time(self) -> "TaskDraft":
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
    def validate_window(self) -> "ResourceBlockDraft":
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
    def validate_payload(self) -> "DispatchIntent":
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
    def validate_agent_policy(self) -> "SimulationRequest":
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


class EvidenceItem(BaseModel):
    source: str
    title: str
    detail: str


class ChatResponse(BaseModel):
    trace_id: str = Field(default_factory=lambda: new_id("trace"), alias="traceId")
    message: str
    intent: DispatchIntent | None = None
    validation: IntentValidation | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    model: str
    fallback_used: bool = Field(alias="fallbackUsed")
    suggested_actions: list[str] = Field(default_factory=list, alias="suggestedActions")


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
    load_state: str | None = Field(default=None, alias="loadState")
    evidence: list[IncidentEvidence] = Field(default_factory=list)
    deterministic_findings: list[DeterministicFinding] = Field(
        default_factory=list, alias="deterministicFindings"
    )
    diagnosis: DiagnosisReport | None = None
    what_if_run_ids: dict[str, str] = Field(default_factory=dict, alias="whatIfRunIds")
    created_by: str = Field(default="demo-operator", alias="createdBy")
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")
    updated_at: datetime = Field(default_factory=utc_now, alias="updatedAt")


class IncidentWhatIfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: WhatIfMode
    requested_by: str = Field(default="demo-operator", alias="requestedBy")
