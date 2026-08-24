export interface EngineStatus {
  expectedCommit: string;
  currentCommit: string;
  commitMatches: boolean;
  dirty: boolean;
  dirtyFileCount: number;
  allowed: boolean;
  warning?: string | null;
}

export interface AgentModelStatus {
  modelId: string;
  modelVersion: string;
  algorithm: string;
  mode: "LEARNED" | "BASELINE";
  configured: boolean;
  checkpointPresent: boolean;
  checkpointName?: string | null;
  checkpointSha256?: string | null;
  device: string;
  safetyController: string;
  notice: string;
}

export interface Health {
  status: string;
  environment: string;
  engine: EngineStatus;
  model: {
    provider: string;
    model: string;
    configured: boolean;
    mode: string;
  };
  agentPolicy: AgentModelStatus;
  safety: {
    mode: string;
    fieldExecutionEnabled: boolean;
    approvalBoundaryEnabled: boolean;
  };
}

export interface ScenarioMeta {
  scenarioId: string;
  file: string;
  vehicleCount: number;
  taskCount: number;
  endTimeMs: number;
}

export interface Vehicle {
  vehicleId: string;
  robotGroup: "fork" | "jack";
  currentNodeId: string;
  state: string;
  loadState: string;
}

export interface Task {
  taskId: string;
  pickupNodeId: string;
  dropoffNodeId: string;
  requiredRobotGroup: "fork" | "jack";
  priorityClass: number;
  releaseTimeMs: number;
  dueTimeMs?: number | null;
  state: string;
}

export interface Snapshot {
  scenarioId: string;
  worldRevision: number;
  mode: string;
  endTimeMs: number;
  counts: Record<string, number>;
  groups: Record<string, number>;
  vehicles: Vehicle[];
  tasks: Task[];
  zones: Array<{ id: string; memberEdgeIds: string[] }>;
  engine: EngineStatus;
}

export interface MapNode {
  id: string;
  type: string;
  x: number;
  y: number;
  groups: string[];
}

export interface MapEdge {
  id: string;
  group: string;
  start: string;
  end: string;
  p0: [number, number];
  p1: [number, number];
  p2: [number, number];
  p3: [number, number];
  length?: number;
  motionDirection?: number;
  shared: boolean;
}

export interface MapModel {
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
  stats: Record<string, number>;
  nodes: MapNode[];
  edges: MapEdge[];
  sharedOverlays: Array<MapEdge & { forkEdge?: string; jackEdge?: string }>;
}

export interface TaskDraft {
  taskId: string;
  pickupNodeId: string;
  dropoffNodeId: string;
  requiredRobotGroup: "fork" | "jack";
  payloadType: string;
  priorityClass: number;
  dueTimeMs?: number | null;
}

export interface ResourceBlock {
  resourceIds: string[];
  startMs: number;
  endMs: number;
  reason: string;
}

export interface DispatchIntent {
  schemaVersion: number;
  intentId: string;
  intentType: string;
  requestedBy: string;
  environment: string;
  basedOnWorldRevision: number;
  reason: string;
  task?: TaskDraft | null;
  resourceBlock?: ResourceBlock | null;
  query?: string | null;
}

export interface Validation {
  intentId: string;
  valid: boolean;
  riskLevel: string;
  approvalRequired: boolean;
  policyCode: string;
  issues: Array<{ code: string; message: string; severity: string }>;
}

export interface Evidence {
  source: string;
  title: string;
  detail: string;
  chunkId?: string | null;
  score?: number | null;
  retrievalMethod?: string | null;
}

export interface AgentTraceStep {
  stepId: string;
  sequence: number;
  state:
    | "RECEIVED"
    | "PLANNING"
    | "CONTEXT_GATHERING"
    | "PARAMETER_RESOLUTION"
    | "INTENT_DRAFTING"
    | "SAFETY_VALIDATION"
    | "CLARIFICATION_REQUIRED"
    | "COMPLETED";
  status: "COMPLETED" | "BLOCKED" | "FAILED";
  title: string;
  detail: string;
  toolName?: string | null;
  readOnly?: boolean | null;
  durationMs: number;
}

export interface AgentExecutionTrace {
  strategy: "MODEL_TOOL_CALLING" | "DETERMINISTIC_POLICY";
  plannerModel: string;
  status: "COMPLETED" | "CLARIFICATION_REQUIRED" | "FAILED";
  maxSteps: number;
  durationMs: number;
  steps: AgentTraceStep[];
}

export interface AgentMetricEvent {
  traceId: string;
  conversationId: string;
  scenarioId: string;
  createdAt: string;
  status: "COMPLETED" | "CLARIFICATION_REQUIRED" | "FAILED";
  strategy: "MODEL_TOOL_CALLING" | "DETERMINISTIC_POLICY";
  plannerModel: string;
  intentModel: string;
  fallbackUsed: boolean;
  durationMs: number;
  stepCount: number;
  toolNames: string[];
  validationPassed?: boolean | null;
  riskLevel?: string | null;
}

export interface AgentMetricsSummary {
  generatedAt: string;
  requestCount: number;
  completedCount: number;
  clarificationCount: number;
  taskCompletionRate: number;
  modelToolPlanningRate: number;
  fallbackRate: number;
  safetyBlockRate: number;
  averageDurationMs: number;
  p95DurationMs: number;
  averageStepCount: number;
  toolCallCounts: Record<string, number>;
  recent: AgentMetricEvent[];
}

export interface ChatResponse {
  traceId: string;
  conversationId: string;
  state: "READY" | "CLARIFICATION_REQUIRED";
  message: string;
  intent?: DispatchIntent | null;
  validation?: Validation | null;
  clarification?: {
    code: "MISSING_REQUIRED_FIELDS" | "AMBIGUOUS_ENTITY";
    missingFields: string[];
    questions: string[];
    collectedParameters: Record<string, unknown>;
  } | null;
  evidence: Evidence[];
  model: string;
  fallbackUsed: boolean;
  suggestedActions: string[];
  agentTrace?: AgentExecutionTrace | null;
}

export type AgentRunStatus =
  | "QUEUED"
  | "RUNNING"
  | "WAITING_APPROVAL"
  | "COMPLETED"
  | "REJECTED"
  | "CANCELLED"
  | "TIMED_OUT"
  | "FAILED";

export interface AgentRunEvent {
  eventId: number;
  eventType: string;
  payload: Record<string, unknown>;
  createdAt: string;
}

export interface AgentRunEvaluation {
  passed: boolean;
  score: number;
  checks: Record<string, boolean>;
  notes: string[];
}

export interface AgentRunRecord {
  runId: string;
  status: AgentRunStatus;
  request: {
    message: string;
    scenarioId: string;
    requestedBy: string;
    conversationId: string;
    timeoutSeconds: number;
  };
  idempotencyKey?: string | null;
  attempt: number;
  recovered: boolean;
  cancelRequested: boolean;
  traceSteps: AgentTraceStep[];
  response?: ChatResponse | null;
  approval?: {
    intent: DispatchIntent;
    validation: Validation;
    requestedAt: string;
    decision?: {
      approved: boolean;
      decidedBy: string;
      reason: string;
      decidedAt: string;
    } | null;
  } | null;
  evaluation?: AgentRunEvaluation | null;
  providerUsage: Record<string, number | boolean | Record<string, number>>;
  error?: string | null;
  events: AgentRunEvent[];
  createdAt: string;
  updatedAt: string;
  deadlineAt: string;
  startedAt?: string | null;
  completedAt?: string | null;
}

export interface AgentPolicyOptions {
  modelId?: string | null;
  candidateCount: number;
  allowDeviation: boolean;
}

export interface AgentPolicyEvidence {
  requested: boolean;
  mode: "LEARNED" | "BASELINE";
  modelId: string;
  modelVersion: string;
  checkpointSha256?: string | null;
  candidateCount: number;
  deviationRequested: boolean;
  deviationEnabled: boolean;
  inferenceCount: number;
  inferenceMs: number;
  fallbackCount: number;
  safetyFallbackCount: number;
  guardianCandidateCount: number;
  guardianOverrideCount: number;
  agentCandidateCount: number;
  selectedAgentCandidateCount: number;
  decisionCycleCount: number;
  fallbackReasons: string[];
  notes: string[];
  evidencePath?: string | null;
}

export interface AgentDecisionCandidate {
  candidateId: string;
  strategy: string;
  feasible: boolean;
  plannedTaskCount: number;
  order: Array<{ vehicleId: string; taskId: string }>;
  failureCode?: string;
}

export interface AgentDecisionCycle {
  cycleIndex: number;
  decisionTimeMs: number;
  candidateCount: number;
  feasibleCandidateCount: number;
  selectedCandidateIds: string[];
  candidates: AgentDecisionCandidate[];
}

export interface AgentPolicyArtifact {
  schemaVersion: number;
  runId: string;
  model: Record<string, unknown>;
  execution: AgentPolicyEvidence;
  safetyBoundary: Record<string, unknown>;
  decisionCycles: AgentDecisionCycle[];
}

export interface SimulationSummary {
  runId: string;
  scenarioId: string;
  label: string;
  policy: string;
  seed: number;
  status: string;
  durationMs: number;
  metrics: Record<string, number | null | Record<string, number>>;
  planning: Record<string, unknown>;
  safety: Record<string, unknown>;
  agentPolicy?: AgentPolicyEvidence | null;
  intentId?: string | null;
  createdAt: string;
  error?: string | null;
}

export interface ReplayMotion {
  startRotationMs: number;
  linearMs: number;
  endRotationMs: number;
  startHeadingRad: number;
  travelStartHeadingRad: number;
  travelEndHeadingRad: number;
  endHeadingRad: number;
}

export interface ReplaySegment {
  id: string;
  kind: string;
  startMs: number;
  endMs: number;
  startNodeId?: string | null;
  endNodeId?: string | null;
  edgeId?: string | null;
  expectedLoadState?: string;
  resourceIds?: string[];
  commandPayload?: {
    startHeadingRad?: number;
    endHeadingRad?: number;
    [key: string]: unknown;
  };
  motion?: ReplayMotion;
}

export interface ReplayPlan {
  id: string;
  vehicleId: string;
  taskId?: string | null;
  createdAtMs: number;
  committedUntilMs: number;
  segments: ReplaySegment[];
}

export interface ReplayVehicle {
  vehicleId: string;
  robotGroup: "fork" | "jack";
  initialNodeId: string;
  initialHeadingRad: number;
  state: string;
  loadState: string;
  activeTaskId?: string | null;
  availableAtMs?: number | null;
}

export interface ReplayTask {
  taskId: string;
  releaseTimeMs: number;
  pickupNodeId: string;
  dropoffNodeId: string;
  requiredRobotGroup: "fork" | "jack";
  state: string;
  assignedVehicleId?: string | null;
  assignedAtMs?: number | null;
  pickedAtMs?: number | null;
  completedAtMs?: number | null;
  dueTimeMs?: number | null;
  initialGlobalRouteMs?: number | null;
}

export interface DispatchReplay {
  scenarioId: string;
  seed?: number;
  endTimeMs: number;
  replayMode: "online" | "offline";
  sweepModel: {
    sampleSpacing: number;
    footprintMargin: number;
    baseGeometryOnly: boolean;
  };
  vehicleProfiles: Record<string, { length: number; width: number }>;
  vehicles: ReplayVehicle[];
  tasks: ReplayTask[];
  plans: ReplayPlan[];
  events: Array<{
    id?: string;
    type: string;
    timeMs: number;
    payload?: Record<string, unknown>;
  }>;
  metrics: Record<string, number | string | Record<string, number> | null>;
  planning: Record<string, unknown>;
  baselinePlanning: Record<string, unknown>;
  manifest: Record<string, unknown>;
}

export interface RunDetail {
  summary: SimulationSummary;
  scenario: {
    endTimeMs: number;
    vehicles: Array<{
      vehicleId: string;
      robotGroup: "fork" | "jack";
      initialNodeId: string;
      initialHeadingRad?: number;
    }>;
    plans: Array<{
      id: string;
      vehicleId: string;
      taskId: string;
      createdAtMs?: number;
      committedUntilMs?: number;
      segments: Array<{
        id: string;
        kind: string;
        startMs: number;
        endMs: number;
        startNodeId?: string;
        endNodeId?: string;
        edgeId?: string;
        expectedLoadState?: string;
        resourceIds?: string[];
        commandPayload?: Record<string, unknown>;
      }>;
    }>;
  };
  result: {
    eventLog: Array<Record<string, unknown>>;
  };
  planning: Record<string, unknown>;
  replay: DispatchReplay;
  agentEvidence?: AgentPolicyArtifact;
}

export interface PlanExplanationEvidence {
  evidenceId: string;
  category: "RUN" | "ASSIGNMENT" | "WAIT" | "ROUTE" | "SAFETY" | "FALLBACK";
  fact: string;
  source: string;
  attributes: Record<string, unknown>;
}

export interface PlanExplanationFinding {
  code: string;
  title: string;
  explanation: string;
  classification: "FACT" | "INFERENCE";
  evidenceIds: string[];
}

export interface PlanExplanationReport {
  schemaVersion: 1;
  runId: string;
  question: string;
  vehicleId?: string | null;
  taskId?: string | null;
  summary: string;
  findings: PlanExplanationFinding[];
  uncertainties: string[];
  evidence: PlanExplanationEvidence[];
  model: string;
  fallbackUsed: boolean;
  generatedAt: string;
}

export type WhatIfMode =
  | "WAIT_RECOVERY"
  | "ISOLATE_REASSIGN"
  | "SUSPEND_AFFECTED_TASKS"
  | "CONTROLLED_REVERSE"
  | "SAFETY_STOP";

export interface IncidentEvidence {
  evidenceId: string;
  evidenceType: string;
  fact: string;
  source: string;
  observedAtMs?: number | null;
  attributes: Record<string, unknown>;
}

export interface DeterministicFinding {
  code: string;
  title: string;
  detail: string;
  certainty: "CONFIRMED" | "INFERRED";
  evidenceIds: string[];
}

export interface RootCauseCandidate {
  code: string;
  title: string;
  explanation: string;
  confidence: number;
  evidenceIds: string[];
  classification: "FACT" | "INFERENCE";
}

export interface IncidentRecommendation {
  actionCode: WhatIfMode;
  action: string;
  rationale: string;
  riskLevel: string;
  requiresSimulation: boolean;
  requiresApproval: boolean;
  evidenceIds: string[];
}

export interface DiagnosisReport {
  summary: string;
  confirmedFacts: string[];
  rootCauseCandidates: RootCauseCandidate[];
  affectedVehicleIds: string[];
  affectedTaskIds: string[];
  recommendations: IncidentRecommendation[];
  uncertainties: string[];
  model: string;
  fallbackUsed: boolean;
  generatedAt: string;
}

export interface Incident {
  incidentId: string;
  incidentType: string;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  status: "OPEN" | "DIAGNOSED" | "MITIGATING" | "RESOLVED";
  scenarioId: string;
  runId: string;
  vehicleIds: string[];
  taskIds: string[];
  resourceIds: string[];
  faultCode?: string | null;
  faultAtMs: number;
  recoveryDurationMs: number;
  locationNodeId?: string | null;
  locationEdgeId?: string | null;
  workstationId?: string | null;
  loadState?: string | null;
  eventAttributes: Record<string, any>;
  evidence: IncidentEvidence[];
  deterministicFindings: DeterministicFinding[];
  diagnosis?: DiagnosisReport | null;
  whatIfRunIds: Partial<Record<WhatIfMode, string>>;
  approvalIds: Partial<Record<WhatIfMode, string>>;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
}

export interface IncidentReport {
  title: string;
  mode: string;
  incident: Incident;
  whatIfRuns: Array<{ mode: WhatIfMode; run: SimulationSummary }>;
  safetyNotice: string;
}

export interface Approval {
  approvalId: string;
  intent: DispatchIntent;
  validation: Validation;
  simulationRunIds: string[];
  status: string;
  requestedBy: string;
  decidedBy?: string | null;
  decisionReason?: string | null;
  createdAt: string;
  decidedAt?: string | null;
}

export interface Comparison {
  comparisonId: string;
  runs: SimulationSummary[];
  recommendedRunId: string;
  rationale: string[];
  createdAt: string;
}

export interface AuditEvent {
  eventId: string;
  traceId: string;
  eventType: string;
  actor: string;
  payload: Record<string, unknown>;
  createdAt: string;
}

export interface ShiftReport {
  title: string;
  mode: string;
  runCount: number;
  successfulRunCount: number;
  approvalCount: number;
  pendingApprovalCount: number;
  latestRun?: SimulationSummary | null;
  notice: string;
}

export interface ScenarioDraftSummary {
  packageId: string;
  version: string;
  status: "draft" | "published" | "archived";
  revision: number;
  sceneId: string;
  streamId: string;
  taskCount: number;
  updatedAt?: string | null;
  build?: Record<string, unknown> | null;
}

export interface ScenarioPackageDocument {
  schemaVersion: 1;
  packageId: string;
  version: string;
  status: "draft" | "published" | "archived";
  metadata?: Record<string, unknown>;
  warehouseScene: {
    sceneId: string;
    name: string;
    bounds: { minX: number; maxX: number; minY: number; maxY: number };
    robotProfiles: Record<string, Record<string, unknown>>;
    nodes: Array<Record<string, any>>;
    edges: Array<Record<string, any>>;
    workstations: Array<Record<string, any>>;
    vehicles: Array<Record<string, any>>;
    recoveryNodes: Array<Record<string, any>>;
    trafficZones: Array<Record<string, any>>;
    safety: Record<string, any>;
  };
  taskStream: {
    streamId: string;
    seed: number;
    endTimeMs: number;
    tasks: Array<Record<string, any>>;
    events: Array<Record<string, any>>;
  };
}

export interface ScenarioValidationReport {
  valid: boolean;
  issues: Array<{ severity: string; code: string; path: string; message: string }>;
  stats: Record<string, number>;
}

export interface BenchmarkRequest {
  suiteName: string;
  baseScenarioId: string;
  vehicleCounts: number[];
  arrivalProfiles: Array<"low" | "medium" | "high">;
  fleetMixes: Array<"mixed" | "fork" | "jack">;
  policies: Array<"top_k" | "task_age" | "shortest_remaining" | "congestion" | "previous_order" | "random" | "rl">;
  seeds: number[];
  horizonMs: number;
  requestedBy: string;
  agentPolicy?: AgentPolicyOptions;
}

export interface BenchmarkCoverage {
  baseScenarioId: string;
  vehicleCounts: number[];
  arrivalProfiles: string[];
  fleetMixes: string[];
  policies: string[];
  seeds: number[];
  horizonMs: number;
}

export interface BenchmarkSafetyGate {
  passed: boolean;
  conflictCaseCount: number;
  planningTimeoutCaseCount: number;
  failedCaseCount: number;
  fieldExecutionEnabled: boolean;
}

export interface BenchmarkSummary {
  benchmarkId: string;
  suiteName: string;
  status: string;
  createdAt: string;
  durationMs: number;
  caseCount: number;
  completedCaseCount: number;
  coverage: BenchmarkCoverage;
  safetyGate: BenchmarkSafetyGate;
}

export interface BenchmarkStatistic {
  count: number;
  mean?: number | null;
  stddev?: number | null;
  ci95Low?: number | null;
  ci95High?: number | null;
}

export interface BenchmarkAggregate {
  vehicleCount: number;
  arrivalProfile: string;
  fleetMix: string;
  policy: string;
  caseCount: number;
  successfulCaseCount: number;
  failedCaseCount: number;
  metrics: Record<string, BenchmarkStatistic>;
}

export interface BenchmarkReport extends BenchmarkSummary {
  aggregates: BenchmarkAggregate[];
  failureCases: Array<{ caseId: string; error?: string | null }>;
  cases: Array<Record<string, unknown>>;
  artifacts: Record<string, string>;
}

export interface ModelSafetyCaseResult {
  caseId: string;
  category: string;
  title: string;
  severity: "NORMAL" | "CRITICAL";
  requestedExecution: string;
  executionMode: string;
  passed: boolean;
  latencyMs: number;
  expected: Record<string, unknown>;
  observed: Record<string, unknown>;
  error?: string | null;
}

export interface ModelSafetyEvaluationSummary {
  evaluationId: string;
  suiteName: string;
  status: "PASSED" | "FAILED";
  createdAt: string;
  durationMs: number;
  passedCaseCount: number;
  failedCaseCount: number;
  fallbackCaseCount: number;
  liveProviderCaseCount: number;
  liveProviderEvaluated: boolean;
  safetyGate: {
    passed: boolean;
    criticalFailureCount: number;
    fieldExecutionEnabled: boolean;
  };
  coverage: {
    caseCount: number;
    categories: string[];
    providerCaseCount: number;
    deterministicBoundaryCaseCount: number;
  };
}

export interface ModelSafetyEvaluationReport extends ModelSafetyEvaluationSummary {
  schemaVersion: number;
  suiteId: string;
  suiteVersion: number;
  suiteSha256: string;
  description: string;
  createdBy: string;
  provider: {
    provider: string;
    model: string;
    configured: boolean;
    mode: string;
    baseUrl: string;
  };
  cases: ModelSafetyCaseResult[];
  notes: string[];
  artifacts: Record<string, string>;
}

export interface DatasetExportRequest {
  name: string;
  includeAudit: boolean;
  includeIncidents: boolean;
  includeEvidenceText: boolean;
  requestedBy: string;
}

export interface DatasetQualityReport {
  passed: boolean;
  recordCount: number;
  recordTypeCounts: Record<string, number>;
  splitCounts: Record<string, number>;
  duplicateRecordIds: string[];
  missingRequiredRecordIndexes: number[];
  sensitiveFieldFindings: Array<Record<string, unknown>>;
  checks: Record<string, boolean>;
}

export interface DatasetExportManifest {
  schemaVersion: number;
  exportId: string;
  name: string;
  createdAt: string;
  createdBy: string;
  classification: string;
  simulationOnly: boolean;
  includeEvidenceText: boolean;
  recordCount: number;
  quality: DatasetQualityReport;
  splits: Record<string, string>;
  usageRestrictions: string[];
  artifacts: Record<string, string>;
}
