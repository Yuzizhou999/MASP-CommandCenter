import type {
  AgentModelStatus,
  AgentPolicyOptions,
  Approval,
  AuditEvent,
  BenchmarkReport,
  BenchmarkRequest,
  BenchmarkSummary,
  ChatResponse,
  Comparison,
  DispatchIntent,
  DatasetExportManifest,
  DatasetExportRequest,
  Health,
  Incident,
  IncidentReport,
  MapModel,
  RunDetail,
  ScenarioMeta,
  SimulationSummary,
  Snapshot,
  ShiftReport,
  WhatIfMode,
  ScenarioDraftSummary,
  ScenarioPackageDocument,
  ScenarioValidationReport,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || response.statusText);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  agentPolicy: () => request<AgentModelStatus>("/api/v1/agent-policy"),
  scenarios: () => request<ScenarioMeta[]>("/api/v1/scenarios"),
  snapshot: (scenarioId: string) =>
    request<Snapshot>(`/api/v1/world/snapshot?scenarioId=${encodeURIComponent(scenarioId)}`),
  map: () => request<MapModel>("/api/v1/map"),
  chat: (message: string, scenarioId: string) =>
    request<ChatResponse>("/api/v1/agent/chat", {
      method: "POST",
      body: JSON.stringify({ message, scenarioId, requestedBy: "demo-operator" }),
    }),
  simulate: (
    scenarioId: string,
    label: string,
    intent?: DispatchIntent | null,
    policy = "top_k",
    agentPolicy?: AgentPolicyOptions,
  ) =>
    request<SimulationSummary>("/api/v1/simulations", {
      method: "POST",
      body: JSON.stringify({
        scenarioId,
        label,
        policy,
        seed: 0,
        intent: intent || null,
        agentPolicy,
      }),
    }),
  simulations: () => request<SimulationSummary[]>("/api/v1/simulations"),
  runDetail: (runId: string) => request<RunDetail>(`/api/v1/simulations/${runId}`),
  compare: (runIds: string[]) =>
    request<Comparison>("/api/v1/simulations/compare", {
      method: "POST",
      body: JSON.stringify({ runIds }),
    }),
  approvals: () => request<Approval[]>("/api/v1/approvals"),
  createApproval: (scenarioId: string, intent: DispatchIntent, runIds: string[]) => {
    const query = runIds.map((id) => `runId=${encodeURIComponent(id)}`).join("&");
    return request<Approval>(
      `/api/v1/approvals?scenarioId=${encodeURIComponent(scenarioId)}&${query}`,
      { method: "POST", body: JSON.stringify(intent) },
    );
  },
  decideApproval: (approvalId: string, approved: boolean) =>
    request<Approval>(`/api/v1/approvals/${approvalId}/decision`, {
      method: "POST",
      body: JSON.stringify({
        approved,
        decidedBy: "demo-supervisor",
        reason: approved ? "已核对仿真结果和安全影响" : "当前影响范围不可接受",
      }),
    }),
  commitIntent: (
    scenarioId: string,
    intent: DispatchIntent,
    approvalId?: string | null,
  ) => {
    const approval = approvalId ? `&approvalId=${encodeURIComponent(approvalId)}` : "";
    return request<Record<string, unknown>>(
      `/api/v1/intents/${intent.intentId}/commit?scenarioId=${encodeURIComponent(scenarioId)}${approval}`,
      { method: "POST", body: JSON.stringify(intent) },
    );
  },
  audit: () => request<AuditEvent[]>("/api/v1/audit?limit=80"),
  report: () => request<ShiftReport>("/api/v1/reports/shift"),
  incidents: () => request<Incident[]>("/api/v1/incidents"),
  injectVehicleFault: (
    runId: string,
    options: {
      vehicleId?: string;
      faultCode: string;
      requestedAtMs?: number;
      recoveryDurationMs: number;
    },
  ) => request<Incident>("/api/v1/incidents/inject", {
    method: "POST",
    body: JSON.stringify({ runId, requestedBy: "demo-operator", ...options }),
  }),
  diagnoseIncident: (incidentId: string) =>
    request<Incident>(
      `/api/v1/incidents/${encodeURIComponent(incidentId)}/diagnose?requestedBy=demo-operator`,
      { method: "POST" },
    ),
  runIncidentWhatIf: (incidentId: string, mode: WhatIfMode) =>
    request<Incident>(`/api/v1/incidents/${encodeURIComponent(incidentId)}/what-if`, {
      method: "POST",
      body: JSON.stringify({ mode, requestedBy: "demo-operator" }),
    }),
  incidentReport: (incidentId: string) =>
    request<IncidentReport>(`/api/v1/incidents/${encodeURIComponent(incidentId)}/report`),
  scenarioDrafts: () => request<ScenarioDraftSummary[]>('/api/v1/scenario-drafts'),
  scenarioDraft: (packageId: string) =>
    request<ScenarioPackageDocument>(`/api/v1/scenario-drafts/${encodeURIComponent(packageId)}`),
  createScenarioDraftFromRuntime: (scenarioId: string, packageId: string) =>
    request<ScenarioDraftSummary>(`/api/v1/scenario-drafts/from-runtime?scenarioId=${encodeURIComponent(scenarioId)}&packageId=${encodeURIComponent(packageId)}`, { method: 'POST' }),
  updateScenarioDraft: (packageId: string, document: ScenarioPackageDocument, revision: number) =>
    request<ScenarioDraftSummary>(`/api/v1/scenario-drafts/${encodeURIComponent(packageId)}?expectedRevision=${revision}&requestedBy=demo-operator`, { method: 'PUT', body: JSON.stringify(document) }),
  validateScenarioDraft: (packageId: string) =>
    request<ScenarioValidationReport>(`/api/v1/scenario-drafts/${encodeURIComponent(packageId)}/validate?requestedBy=demo-operator`, { method: 'POST' }),
  generateScenarioTasks: (packageId: string, generation: Record<string, unknown>, revision: number) =>
    request<ScenarioDraftSummary>(`/api/v1/scenario-drafts/${encodeURIComponent(packageId)}/generate-tasks?expectedRevision=${revision}&requestedBy=demo-operator`, { method: 'POST', body: JSON.stringify(generation) }),
  compileScenarioDraft: (packageId: string) =>
    request<Record<string, unknown>>(`/api/v1/scenario-drafts/${encodeURIComponent(packageId)}/compile?requestedBy=demo-operator`, { method: 'POST' }),
  publishScenarioDraft: (packageId: string) =>
    request<ScenarioDraftSummary>(`/api/v1/scenario-drafts/${encodeURIComponent(packageId)}/publish?requestedBy=demo-supervisor`, { method: 'POST' }),
  benchmarks: () => request<BenchmarkSummary[]>("/api/v1/evaluations/benchmarks"),
  runBenchmark: (options: BenchmarkRequest) =>
    request<BenchmarkReport>("/api/v1/evaluations/benchmarks", {
      method: "POST",
      body: JSON.stringify(options),
    }),
  benchmarkDetail: (benchmarkId: string) =>
    request<BenchmarkReport>(`/api/v1/evaluations/benchmarks/${encodeURIComponent(benchmarkId)}`),
  datasetExports: () => request<DatasetExportManifest[]>("/api/v1/dataset-exports"),
  createDatasetExport: (options: DatasetExportRequest) =>
    request<DatasetExportManifest>("/api/v1/dataset-exports", {
      method: "POST",
      body: JSON.stringify(options),
    }),
  datasetExportDetail: (exportId: string) =>
    request<DatasetExportManifest>(`/api/v1/dataset-exports/${encodeURIComponent(exportId)}`),
  datasetExportDownloadUrl: (exportId: string) =>
    `/api/v1/dataset-exports/${encodeURIComponent(exportId)}/download`,
};
