export interface EngineStatus {
  expectedCommit: string;
  currentCommit: string;
  commitMatches: boolean;
  dirty: boolean;
  dirtyFileCount: number;
  allowed: boolean;
  warning?: string | null;
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
  message: string;
  intent?: DispatchIntent | null;
  validation?: Validation | null;
  evidence: Evidence[];
  model: string;
  fallbackUsed: boolean;
  suggestedActions: string[];
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
