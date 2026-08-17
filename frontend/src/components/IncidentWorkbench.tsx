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
  Alert20Regular,
  ArrowDownload20Regular,
  BrainCircuit20Regular,
  DocumentSearch20Regular,
  Play20Regular,
  ShieldError20Regular,
  VehicleTruckProfile20Regular,
  Wrench20Regular,
} from "@fluentui/react-icons";
import type {
  Incident,
  IncidentEvidence,
  SimulationSummary,
  WhatIfMode,
} from "../types";

type InjectionKind =
  | "VEHICLE_FAULT"
  | "WORKSTATION_DISABLED"
  | "DEADLOCK_RECOVERABLE"
  | "DEADLOCK_UNRECOVERABLE";

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
  onInjectWorkstation: (options: {
    runId: string;
    workstationNodeId?: string;
    requestedAtMs?: number;
    recoveryDurationMs: number;
  }) => Promise<void>;
  onInjectDeadlock: (
    runId: string,
    deadlockCase: "RECOVERABLE" | "UNRECOVERABLE",
  ) => Promise<void>;
  onDiagnose: (incidentId: string) => Promise<void>;
  onWhatIf: (incidentId: string, mode: WhatIfMode) => Promise<void>;
  onRequestApproval: (incidentId: string, mode: WhatIfMode) => Promise<void>;
  onDemoTask: () => Promise<void>;
  onDemoRoadblock: () => Promise<void>;
  onCompare: (runIds: string[]) => void;
  onDownloadReport: (incidentId: string) => Promise<void>;
}

const modeLabel: Record<WhatIfMode, string> = {
  WAIT_RECOVERY: "等待恢复",
  ISOLATE_REASSIGN: "隔离重派",
  SUSPEND_AFFECTED_TASKS: "暂停关联任务",
  CONTROLLED_REVERSE: "受控倒退",
  SAFETY_STOP: "安全停车",
};

const incidentTypeLabel: Record<string, string> = {
  VEHICLE_FAULT: "车辆故障",
  WORKSTATION_DISABLED: "工位停用",
  DEADLOCK_RISK: "等待环 / 死锁",
};

const evidenceTypeLabel: Record<string, string> = {
  FAULT_SIGNAL: "故障信号",
  VEHICLE_POSITION: "车辆位置",
  ACTIVE_TASK: "关联任务",
  RESOURCE_OCCUPANCY: "资源占用",
  PLANNING_METRICS: "基线指标",
  RECENT_EVENT: "运行事件",
  WORKSTATION_OUTAGE: "工位停用",
  WORKSTATION_DEFINITION: "工位定义",
  AFFECTED_TASKS: "受影响任务",
  WAIT_GRAPH_CYCLE: "等待环",
  WAIT_DEPENDENCIES: "依赖边",
  RECOVERY_DECISION: "恢复决策",
  RECOVERY_PLAN: "倒退计划",
  SAFETY_STOP: "安全停车",
  RECOVERY_ACCEPTANCE: "恢复验收",
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
    <Tooltip content={row ? row.fact : "证据不存在"} relationship="description">
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
  onInjectWorkstation,
  onInjectDeadlock,
  onDiagnose,
  onWhatIf,
  onRequestApproval,
  onDemoTask,
  onDemoRoadblock,
  onCompare,
  onDownloadReport,
}: IncidentWorkbenchProps) {
  const completedRuns = runs.filter((run) => run.status === "COMPLETED");
  const [runId, setRunId] = useState("");
  const [kind, setKind] = useState<InjectionKind>("VEHICLE_FAULT");
  const [subjectId, setSubjectId] = useState("");
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
  const availableModes = useMemo<WhatIfMode[]>(() => {
    const configured = selected?.eventAttributes.availableWhatIfModes;
    if (Array.isArray(configured)) return configured as WhatIfMode[];
    return ["WAIT_RECOVERY", "ISOLATE_REASSIGN", "SAFETY_STOP"];
  }, [selected]);

  const eventAtMs = requestedAtSeconds.trim()
    ? Math.round(Number(requestedAtSeconds) * 1000)
    : undefined;
  const recoveryDurationMs = Math.max(10, Number(recoverySeconds) || 120) * 1000;

  const submitInjection = async () => {
    if (!effectiveRunId) return;
    if (kind === "VEHICLE_FAULT") {
      await onInject({
        runId: effectiveRunId,
        vehicleId: subjectId.trim() || undefined,
        faultCode: faultCode.trim() || "DRIVE_MOTOR_OVERHEAT",
        requestedAtMs: eventAtMs,
        recoveryDurationMs,
      });
      return;
    }
    if (kind === "WORKSTATION_DISABLED") {
      await onInjectWorkstation({
        runId: effectiveRunId,
        workstationNodeId: subjectId.trim() || undefined,
        requestedAtMs: eventAtMs,
        recoveryDurationMs,
      });
      return;
    }
    await onInjectDeadlock(
      effectiveRunId,
      kind === "DEADLOCK_RECOVERABLE" ? "RECOVERABLE" : "UNRECOVERABLE",
    );
  };

  const oneClickFault = () => onInject({
    runId: effectiveRunId,
    faultCode: "DRIVE_MOTOR_OVERHEAT",
    recoveryDurationMs: 120000,
  });
  const oneClickWorkstation = () => onInjectWorkstation({
    runId: effectiveRunId,
    recoveryDurationMs: 180000,
  });

  return (
    <section className="incident-workbench" aria-label="AI 异常诊断工作台">
      <div className="incident-control-pane">
        <div className="panel-heading">
          <h2>事件演示控制</h2>
          <p>在已完成运行上创建可重放的隔离分支</p>
        </div>

        <div className="demo-event-strip" aria-label="一键演示事件">
          <strong>一键演示</strong>
          <div>
            <Button icon={<Play20Regular />} disabled={Boolean(busy)} onClick={() => void onDemoTask()}>
              紧急插单
            </Button>
            <Button icon={<VehicleTruckProfile20Regular />} disabled={!effectiveRunId || Boolean(busy)} onClick={() => void oneClickFault()}>
              车辆故障
            </Button>
            <Button icon={<Wrench20Regular />} disabled={!effectiveRunId || Boolean(busy)} onClick={() => void oneClickWorkstation()}>
              工位停用
            </Button>
            <Button icon={<Alert20Regular />} disabled={Boolean(busy)} onClick={() => void onDemoRoadblock()}>
              通道封闭
            </Button>
            <Button icon={<ShieldError20Regular />} disabled={!effectiveRunId || Boolean(busy)} onClick={() => void onInjectDeadlock(effectiveRunId, "RECOVERABLE")}>
              等待环
            </Button>
          </div>
        </div>

        <div className="incident-form">
          <Field label="事件类型" required>
            <Select value={kind} onChange={(_, data) => setKind(data.value as InjectionKind)}>
              <option value="VEHICLE_FAULT">车辆故障</option>
              <option value="WORKSTATION_DISABLED">工位停用</option>
              <option value="DEADLOCK_RECOVERABLE">可恢复等待环</option>
              <option value="DEADLOCK_UNRECOVERABLE">不可恢复死锁</option>
            </Select>
          </Field>
          <Field label="基线运行" required>
            <Select
              value={effectiveRunId}
              onChange={(_, data) => setRunId(data.value)}
              aria-label="选择事件注入基线"
            >
              {completedRuns.map((run) => (
                <option key={run.runId} value={run.runId}>
                  {run.label} | {run.runId}
                </option>
              ))}
            </Select>
          </Field>
          {kind === "VEHICLE_FAULT" && (
            <>
              <Field label="车辆编号" hint="留空时选择计划最丰富的车辆">
                <Input value={subjectId} placeholder="例如 fork-001" onChange={(_, data) => setSubjectId(data.value)} />
              </Field>
              <Field label="故障码" required>
                <Input value={faultCode} onChange={(_, data) => setFaultCode(data.value)} />
              </Field>
            </>
          )}
          {kind === "WORKSTATION_DISABLED" && (
            <Field label="工位节点" hint="留空时选择影响任务最多的工位">
              <Input value={subjectId} placeholder="例如 fork:AP1123" onChange={(_, data) => setSubjectId(data.value)} />
            </Field>
          )}
          {!kind.startsWith("DEADLOCK_") && (
            <div className="incident-form-row">
              <Field label="目标时刻（秒）" hint="留空时自动选择">
                <Input type="number" min={0} value={requestedAtSeconds} onChange={(_, data) => setRequestedAtSeconds(data.value)} />
              </Field>
              <Field label="恢复窗口（秒）">
                <Input type="number" min={10} max={900} value={recoverySeconds} onChange={(_, data) => setRecoverySeconds(data.value)} />
              </Field>
            </div>
          )}
          <Button
            appearance="primary"
            icon={<ShieldError20Regular />}
            disabled={!effectiveRunId || Boolean(busy)}
            onClick={() => void submitInjection()}
          >
            {busy?.startsWith("incident-inject") ? "正在建立分支" : "注入所选事件"}
          </Button>
          <p className="incident-form-notice">
            事件只写入数字孪生分支，不向真实车辆、工位或控制系统发送指令。
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
                <strong>{incidentTypeLabel[incident.incidentType] || incident.incidentType}</strong>
                <small className="mono">
                  {incident.workstationId || incident.vehicleIds.join(", ") || incident.locationNodeId || "-"}
                </small>
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
            <ShieldError20Regular />
            <strong>选择或注入一个演示事件</strong>
            <p>系统将提取确定性证据，再由 AI 解释原因和处置边界。</p>
          </div>
        )}
        {selected && (
          <>
            <div className="incident-heading">
              <div>
                <span className="mono">{selected.incidentId}</span>
                <h2>{incidentTypeLabel[selected.incidentType] || selected.faultCode || "仓储异常"}</h2>
                <p>
                  {selected.faultCode || "-"} | {selected.locationNodeId || "多资源"} | {selected.faultAtMs / 1000}s
                </p>
              </div>
              <div className="incident-heading-actions">
                <Badge appearance="filled" color={selected.status === "OPEN" ? "warning" : "informative"}>
                  {selected.status}
                </Badge>
                <Button appearance="primary" icon={<BrainCircuit20Regular />} disabled={Boolean(busy)} onClick={() => void onDiagnose(selected.incidentId)}>
                  {busy === "incident-diagnose" ? "诊断中" : selected.diagnosis ? "重新分析" : "AI 分析原因"}
                </Button>
                <Tooltip content="导出带证据、推演和审批编号的 JSON 报告" relationship="description">
                  <Button appearance="subtle" icon={<ArrowDownload20Regular />} aria-label="导出异常报告" onClick={() => void onDownloadReport(selected.incidentId)} />
                </Tooltip>
              </div>
            </div>

            <div className="incident-fact-strip">
              <div><span>事件类型</span><strong>{incidentTypeLabel[selected.incidentType] || selected.incidentType}</strong></div>
              <div><span>影响位置</span><strong className="mono">{selected.workstationId || selected.locationNodeId || "多资源"}</strong></div>
              <div><span>影响车辆</span><strong className="mono">{selected.vehicleIds.length}</strong></div>
              <div><span>关联任务</span><strong className="mono">{selected.taskIds.length}</strong></div>
            </div>

            <TabList selectedValue={section} onTabSelect={(_, data) => setSection(data.value as "analysis" | "evidence")} aria-label="诊断详情">
              <Tab value="analysis" icon={<BrainCircuit20Regular />}>原因与建议</Tab>
              <Tab value="evidence" icon={<DocumentSearch20Regular />}>证据链 ({selected.evidence.length})</Tab>
            </TabList>

            {section === "analysis" && (
              <div className="diagnosis-body">
                {!selected.diagnosis && (
                  <div className="diagnosis-pending">
                    <strong>确定性事实已经提取</strong>
                    <p>AI 只基于已列出的运行证据和受控 SOP 生成结构化解释。</p>
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
                    {selected.deterministicFindings.filter((row) => row.certainty === "CONFIRMED").map((row) => (
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
                    {(selected.diagnosis?.uncertainties || ["尚未调用诊断模型，现场物理状态未知。"])
                      .map((row) => <p key={row}>{row}</p>)}
                  </div>
                </div>

                <div className="what-if-section">
                  <div>
                    <h3>处置方案推演与审批</h3>
                    <p>候选由 MASP 重新计算；推演成功后才能送交主管审批。</p>
                  </div>
                  <div className="what-if-actions">
                    {availableModes.map((mode) => {
                      const finished = selected.whatIfRunIds[mode];
                      const approvalId = selected.approvalIds[mode];
                      return (
                        <div className="what-if-option" key={mode}>
                          <Button
                            appearance={finished ? "secondary" : "primary"}
                            icon={<Play20Regular />}
                            disabled={Boolean(busy)}
                            onClick={() => void onWhatIf(selected.incidentId, mode)}
                          >
                            {busy === `incident-${mode}` ? "推演中" : finished ? `${modeLabel[mode]}（已完成）` : `${modeLabel[mode]}推演`}
                          </Button>
                          <Button
                            appearance="subtle"
                            disabled={!finished || Boolean(approvalId) || Boolean(busy)}
                            onClick={() => void onRequestApproval(selected.incidentId, mode)}
                          >
                            {approvalId ? "已送审" : "提交审批"}
                          </Button>
                        </div>
                      );
                    })}
                    <Button appearance="secondary" disabled={branchRunIds.length < 2} onClick={() => onCompare(branchRunIds)}>
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
