import { useMemo, useState } from "react";
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Badge,
  Button,
  Field,
  Input,
  Select,
  Tab,
  TabList,
  Tooltip,
} from "@fluentui/react-components";
import {
  ArrowDownload20Regular,
  BrainCircuit20Regular,
  DocumentSearch20Regular,
  Play20Regular,
  ShieldError20Regular,
  VehicleTruckProfile20Regular,
} from "@fluentui/react-icons";
import type {
  Incident,
  IncidentEvidence,
  SimulationSummary,
  WhatIfMode,
} from "../types";

interface IncidentWorkbenchProps {
  incidents: Incident[];
  selected?: Incident | null;
  runs: SimulationSummary[];
  busy?: string | null;
  onSelect: (incident: Incident) => void;
  onInject: (options: {
    runId: string;
    vehicleId?: string;
    faultCode: string;
    requestedAtMs?: number;
    recoveryDurationMs: number;
  }) => Promise<void>;
  onDiagnose: (incidentId: string) => Promise<void>;
  onWhatIf: (incidentId: string, mode: WhatIfMode) => Promise<void>;
  onCompare: (runIds: string[]) => void;
  onDownloadReport: (incidentId: string) => Promise<void>;
}

const modeLabel: Record<WhatIfMode, string> = {
  WAIT_RECOVERY: "等待恢复推演",
  ISOLATE_REASSIGN: "隔离重派推演",
  SAFETY_STOP: "安全停车推演",
};

const evidenceTypeLabel: Record<string, string> = {
  FAULT_SIGNAL: "故障信号",
  VEHICLE_POSITION: "车辆位置",
  ACTIVE_TASK: "关联任务",
  RESOURCE_OCCUPANCY: "资源占用",
  PLANNING_METRICS: "基线指标",
  RECENT_EVENT: "运行事件",
  SOP: "处置规范",
};

function EvidenceLink({
  evidenceId,
  evidenceById,
}: {
  evidenceId: string;
  evidenceById: Map<string, IncidentEvidence>;
}) {
  const row = evidenceById.get(evidenceId);
  return (
    <Tooltip
      content={row ? row.fact : "证据不存在"}
      relationship="description"
    >
      <button className="evidence-link mono" type="button">
        {evidenceId}
      </button>
    </Tooltip>
  );
}

export function IncidentWorkbench({
  incidents,
  selected,
  runs,
  busy,
  onSelect,
  onInject,
  onDiagnose,
  onWhatIf,
  onCompare,
  onDownloadReport,
}: IncidentWorkbenchProps) {
  const completedRuns = runs.filter((run) => run.status === "COMPLETED");
  const [runId, setRunId] = useState("");
  const [vehicleId, setVehicleId] = useState("");
  const [faultCode, setFaultCode] = useState("DRIVE_MOTOR_OVERHEAT");
  const [requestedAtSeconds, setRequestedAtSeconds] = useState("");
  const [recoverySeconds, setRecoverySeconds] = useState("120");
  const [section, setSection] = useState<"analysis" | "evidence">("analysis");
  const effectiveRunId = runId || completedRuns[0]?.runId || "";
  const evidenceById = useMemo(
    () => new Map(selected?.evidence.map((row) => [row.evidenceId, row]) || []),
    [selected],
  );
  const branchRunIds = selected ? Object.values(selected.whatIfRunIds) : [];

  const submitInjection = async () => {
    if (!effectiveRunId || !faultCode.trim()) return;
    const requestedAt = requestedAtSeconds.trim()
      ? Math.round(Number(requestedAtSeconds) * 1000)
      : undefined;
    await onInject({
      runId: effectiveRunId,
      vehicleId: vehicleId.trim() || undefined,
      faultCode: faultCode.trim(),
      requestedAtMs: requestedAt,
      recoveryDurationMs: Math.max(10, Number(recoverySeconds) || 120) * 1000,
    });
  };

  return (
    <section className="incident-workbench" aria-label="AI 异常诊断工作台">
      <div className="incident-control-pane">
        <div className="panel-heading">
          <h2>故障演示控制</h2>
          <p>从已完成运行选择安全节点建立故障分支</p>
        </div>
        <div className="incident-form">
          <Field label="基线运行" required>
            <Select
              value={effectiveRunId}
              onChange={(_, data) => setRunId(data.value)}
              aria-label="选择故障注入基线"
            >
              {completedRuns.map((run) => (
                <option key={run.runId} value={run.runId}>
                  {run.label} | {run.runId}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="车辆编号" hint="留空时选择计划最丰富的车辆">
            <Input
              value={vehicleId}
              placeholder="例如 fork-001"
              onChange={(_, data) => setVehicleId(data.value)}
            />
          </Field>
          <Field label="故障码" required>
            <Input value={faultCode} onChange={(_, data) => setFaultCode(data.value)} />
          </Field>
          <div className="incident-form-row">
            <Field label="目标时刻（秒）" hint="留空时自动选择">
              <Input
                type="number"
                min={0}
                value={requestedAtSeconds}
                onChange={(_, data) => setRequestedAtSeconds(data.value)}
              />
            </Field>
            <Field label="恢复窗口（秒）">
              <Input
                type="number"
                min={10}
                max={900}
                value={recoverySeconds}
                onChange={(_, data) => setRecoverySeconds(data.value)}
              />
            </Field>
          </div>
          <Button
            appearance="primary"
            icon={<ShieldError20Regular />}
            disabled={!effectiveRunId || Boolean(busy)}
            onClick={() => void submitInjection()}
          >
            {busy === "incident-inject" ? "正在建立分支" : "注入车辆故障"}
          </Button>
          <p className="incident-form-notice">
            仅创建仿真事故记录，不向 MASP 或真实车辆发送故障指令。
          </p>
        </div>

        <div className="incident-list" aria-label="异常记录">
          <h3>异常记录</h3>
          {incidents.map((incident) => (
            <button
              type="button"
              className={selected?.incidentId === incident.incidentId ? "incident-row selected" : "incident-row"}
              key={incident.incidentId}
              onClick={() => onSelect(incident)}
            >
              <span>
                <strong>{incident.faultCode || incident.incidentType}</strong>
                <small className="mono">{incident.vehicleIds.join(", ")}</small>
              </span>
              <Badge appearance="tint" color={incident.severity === "CRITICAL" ? "danger" : "warning"}>
                {incident.status}
              </Badge>
            </button>
          ))}
          {incidents.length === 0 && <div className="empty-state">尚无异常记录。</div>}
        </div>
      </div>

      <div className="incident-analysis-pane">
        {!selected && (
          <div className="incident-empty">
            <VehicleTruckProfile20Regular />
            <strong>选择已完成运行并注入故障</strong>
            <p>系统将在完整移动段结束节点建立可审计的故障分支。</p>
          </div>
        )}
        {selected && (
          <>
            <div className="incident-heading">
              <div>
                <span className="mono">{selected.incidentId}</span>
                <h2>{selected.faultCode || "车辆故障"}</h2>
                <p>
                  {selected.vehicleIds.join(", ")} | {selected.locationNodeId} | {selected.faultAtMs / 1000}s
                </p>
              </div>
              <div className="incident-heading-actions">
                <Badge appearance="filled" color={selected.status === "OPEN" ? "warning" : "informative"}>
                  {selected.status}
                </Badge>
                <Button
                  appearance="primary"
                  icon={<BrainCircuit20Regular />}
                  disabled={Boolean(busy)}
                  onClick={() => void onDiagnose(selected.incidentId)}
                >
                  {busy === "incident-diagnose" ? "诊断中" : selected.diagnosis ? "重新分析" : "AI 分析原因"}
                </Button>
                <Tooltip content="导出带证据和仿真结果的 JSON 报告" relationship="description">
                  <Button
                    appearance="subtle"
                    icon={<ArrowDownload20Regular />}
                    aria-label="导出异常报告"
                    onClick={() => void onDownloadReport(selected.incidentId)}
                  />
                </Tooltip>
              </div>
            </div>

            <div className="incident-fact-strip">
              <div><span>故障车辆</span><strong className="mono">{selected.vehicleIds[0]}</strong></div>
              <div><span>故障节点</span><strong className="mono">{selected.locationNodeId || "-"}</strong></div>
              <div><span>载荷状态</span><strong>{selected.loadState || "unknown"}</strong></div>
              <div><span>关联任务</span><strong className="mono">{selected.taskIds[0] || "-"}</strong></div>
            </div>

            <TabList
              selectedValue={section}
              onTabSelect={(_, data) => setSection(data.value as "analysis" | "evidence")}
              aria-label="诊断详情"
            >
              <Tab value="analysis" icon={<BrainCircuit20Regular />}>原因与建议</Tab>
              <Tab value="evidence" icon={<DocumentSearch20Regular />}>证据链 ({selected.evidence.length})</Tab>
            </TabList>

            {section === "analysis" && (
              <div className="diagnosis-body">
                {!selected.diagnosis && (
                  <div className="diagnosis-pending">
                    <strong>确定性事实已经提取</strong>
                    <p>点击“AI 分析原因”后，DeepSeek 将仅基于下列证据和规则生成结构化解释。</p>
                  </div>
                )}
                {selected.diagnosis && (
                  <div className="diagnosis-summary">
                    <div>
                      <Badge appearance="tint" color={selected.diagnosis.fallbackUsed ? "warning" : "success"}>
                        {selected.diagnosis.fallbackUsed ? "规则降级诊断" : "DeepSeek 证据诊断"}
                      </Badge>
                      <span className="mono">{selected.diagnosis.model}</span>
                    </div>
                    <p>{selected.diagnosis.summary}</p>
                  </div>
                )}

                <div className="diagnosis-columns">
                  <div className="diagnosis-column">
                    <h3>确认事实</h3>
                    {selected.deterministicFindings
                      .filter((row) => row.certainty === "CONFIRMED")
                      .map((row) => (
                        <article key={row.code}>
                          <strong>{row.title}</strong>
                          <p>{row.detail}</p>
                          <div>{row.evidenceIds.map((id) => <EvidenceLink key={id} evidenceId={id} evidenceById={evidenceById} />)}</div>
                        </article>
                      ))}
                  </div>
                  <div className="diagnosis-column">
                    <h3>原因推断</h3>
                    {(selected.diagnosis?.rootCauseCandidates || selected.deterministicFindings
                      .filter((row) => row.certainty === "INFERRED")
                      .map((row) => ({ ...row, explanation: row.detail, confidence: 0, classification: "INFERENCE" as const })))
                      .map((row) => (
                        <article key={row.code}>
                          <strong>{row.title}</strong>
                          <p>{row.explanation}</p>
                          <div>{row.evidenceIds.map((id) => <EvidenceLink key={id} evidenceId={id} evidenceById={evidenceById} />)}</div>
                        </article>
                      ))}
                  </div>
                  <div className="diagnosis-column unknown-column">
                    <h3>未知信息</h3>
                    {(selected.diagnosis?.uncertainties || ["尚未调用诊断模型，物理故障根因未知。"])
                      .map((row) => <p key={row}>{row}</p>)}
                  </div>
                </div>

                <div className="what-if-section">
                  <div>
                    <h3>处置方案推演</h3>
                    <p>每个分支由 MASP 重新计算，结果仍需主管审批。</p>
                  </div>
                  <div className="what-if-actions">
                    {(Object.keys(modeLabel) as WhatIfMode[]).map((mode) => {
                      const finished = selected.whatIfRunIds[mode];
                      return (
                        <Button
                          key={mode}
                          appearance={finished ? "secondary" : "primary"}
                          icon={<Play20Regular />}
                          disabled={Boolean(busy)}
                          onClick={() => void onWhatIf(selected.incidentId, mode)}
                        >
                          {busy === `incident-${mode}` ? "推演中" : finished ? `${modeLabel[mode]}（已完成）` : modeLabel[mode]}
                        </Button>
                      );
                    })}
                    <Button
                      appearance="secondary"
                      disabled={branchRunIds.length < 2}
                      onClick={() => onCompare(branchRunIds)}
                    >
                      比较处置方案 ({branchRunIds.length})
                    </Button>
                  </div>
                </div>
              </div>
            )}

            {section === "evidence" && (
              <div className="incident-evidence-list">
                {selected.evidence.map((row) => (
                  <Accordion key={row.evidenceId} collapsible>
                    <AccordionItem value={row.evidenceId}>
                      <AccordionHeader>
                        <span className="incident-evidence-heading">
                          <strong className="mono">{row.evidenceId}</strong>
                          <span>{evidenceTypeLabel[row.evidenceType] || row.evidenceType}</span>
                          {row.observedAtMs != null && <time className="mono">{row.observedAtMs} ms</time>}
                        </span>
                      </AccordionHeader>
                      <AccordionPanel>
                        <p>{row.fact}</p>
                        <span className="evidence-source mono">{row.source}</span>
                        <pre>{JSON.stringify(row.attributes, null, 2)}</pre>
                      </AccordionPanel>
                    </AccordionItem>
                  </Accordion>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
