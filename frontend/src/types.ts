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

export interface RunDetail {
  summary: SimulationSummary;
  scenario: {
    endTimeMs: number;
    vehicles: Array<{
      vehicleId: string;
      robotGroup: "fork" | "jack";
      initialNodeId: string;
    }>;
    plans: Array<{
      id: string;
      vehicleId: string;
      taskId: string;
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
      }>;
    }>;
  };
  result: {
    eventLog: Array<Record<string, unknown>>;
  };
  planning: Record<string, unknown>;
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

export type WhatIfMode = "WAIT_RECOVERY" | "ISOLATE_REASSIGN" | "SAFETY_STOP";

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
  loadState?: string | null;
  evidence: IncidentEvidence[];
  deterministicFindings: DeterministicFinding[];
  diagnosis?: DiagnosisReport | null;
  whatIfRunIds: Partial<Record<WhatIfMode, string>>;
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
