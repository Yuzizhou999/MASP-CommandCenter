import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactElement } from "react";
import {
  Badge,
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  DialogTrigger,
  Select,
  Skeleton,
  SkeletonItem,
  Tooltip,
} from "@fluentui/react-components";
import {
  Alert20Regular,
  ArrowClockwise20Regular,
  Bot20Regular,
  BrainCircuit24Regular,
  ClipboardTaskListLtr20Regular,
  DataTrending20Regular,
  DocumentArrowDown20Regular,
  DocumentText20Regular,
  Home20Regular,
  Play20Regular,
  ShieldLock20Regular,
  ShieldError20Regular,
  SignOut20Regular,
  VehicleTruck20Regular,
  Wrench20Regular,
} from "@fluentui/react-icons";
import { api } from "./api";
import { ApprovalsPanel } from "./components/ApprovalsPanel";
import { AgentPolicyPanel } from "./components/AgentPolicyPanel";
import { AssistantPanel } from "./components/AssistantPanel";
import { DispatchReplayPanel } from "./components/DispatchReplayPanel";
import { IncidentWorkbench } from "./components/IncidentWorkbench";
import { EvaluationCenter } from "./components/EvaluationCenter";
import { PlanExplanationPanel } from "./components/PlanExplanationPanel";
import { OperationsPanel } from "./components/OperationsPanel";
import { SimulationTable } from "./components/SimulationTable";
import { ScenarioDesigner } from "./components/ScenarioDesigner";
import { WarehouseMap } from "./components/WarehouseMap";
import type {
  Approval,
  AgentRunRecord,
  AuditEvent,
  ChatResponse,
  Comparison,
  DispatchIntent,
  Health,
  Incident,
  MapModel,
  PlanExplanationReport,
  RunDetail,
  ScenarioMeta,
  ShiftReport,
  SimulationSummary,
  Snapshot,
  WhatIfMode,
} from "./types";

type View = "command" | "designer" | "simulations" | "incidents" | "evaluation" | "approvals" | "operations";

const viewItems: Array<{ id: View; label: string; icon: ReactElement }> = [
  { id: "command", label: "调度总览", icon: <Home20Regular /> },
  { id: "designer", label: "场景设计", icon: <Wrench20Regular /> },
  { id: "simulations", label: "方案仿真", icon: <Play20Regular /> },
  { id: "incidents", label: "异常诊断", icon: <ShieldError20Regular /> },
  { id: "evaluation", label: "评测中心", icon: <DataTrending20Regular /> },
  { id: "approvals", label: "风险审批", icon: <ShieldLock20Regular /> },
  { id: "operations", label: "运营审计", icon: <ClipboardTaskListLtr20Regular /> },
];

const formatCount = (value?: number) => (typeof value === "number" ? value.toLocaleString("zh-CN") : "-");
const terminalAgentRunStatuses = new Set(["COMPLETED", "REJECTED", "CANCELLED", "TIMED_OUT", "FAILED"]);

export default function App() {
  const [view, setView] = useState<View>("command");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioMeta[]>([]);
  const [selectedScenario, setSelectedScenario] = useState("interactive-multi-fleet");
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [map, setMap] = useState<MapModel | null>(null);
  const [runs, setRuns] = useState<SimulationSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [checkedRunIds, setCheckedRunIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null);
  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null);
  const [agentRun, setAgentRun] = useState<AgentRunRecord | null>(null);
  const stopAgentWatch = useRef<(() => void) | null>(null);
  const [conversationId] = useState(() => `conversation-${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`);
  const [planExplanation, setPlanExplanation] = useState<PlanExplanationReport | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [busyApprovalId, setBusyApprovalId] = useState<string | null>(null);
  const [playbackMs, setPlaybackMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(5);
  const [selectedVehicleIds, setSelectedVehicleIds] = useState<string[]>([]);
  const [report, setReport] = useState<ShiftReport | null>(null);
  const [reportOpen, setReportOpen] = useState(false);

  const handleDesignerNotice = (message: string) => setNotice(message);
  const handleDesignerError = (message: string) => setError(message);

  const loadRun = useCallback(async (runId: string) => {
    setSelectedRunId(runId);
    setPlaying(false);
    setPlaybackMs(0);
    setSelectedVehicleIds([]);
    try {
      const detail = await api.runDetail(runId);
      setRunDetail(detail);
      setSelectedScenario(detail.summary.scenarioId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法加载仿真回放");
    }
  }, []);

  const toggleReplayVehicle = useCallback((vehicleId: string) => {
    setSelectedVehicleIds((current) =>
      current.includes(vehicleId)
        ? current.filter((item) => item !== vehicleId)
        : [...current, vehicleId],
    );
  }, []);

  const loadInitial = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextHealth, nextScenarios, nextMap, nextRuns, nextApprovals, nextAudit, nextIncidents] =
        await Promise.all([
          api.health(),
          api.scenarios(),
          api.map(),
          api.simulations(),
          api.approvals(),
          api.audit(),
          api.incidents(),
        ]);
      setHealth(nextHealth);
      setScenarios(nextScenarios);
      setMap(nextMap);
      setRuns(nextRuns);
      setApprovals(nextApprovals);
      setAudit(nextAudit);
      setIncidents(nextIncidents);
      setSelectedIncidentId((current) => current || nextIncidents[0]?.incidentId || null);
      const initialScenario = nextScenarios.some((item) => item.scenarioId === selectedScenario)
        ? selectedScenario
        : nextScenarios[0]?.scenarioId;
      if (initialScenario) {
        setSelectedScenario(initialScenario);
        setSnapshot(await api.snapshot(initialScenario));
      }
      if (nextRuns[0]) {
        await loadRun(nextRuns[0].runId);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "控制台初始化失败");
    } finally {
      setLoading(false);
    }
  }, [loadRun, selectedScenario]);

  useEffect(() => {
    void loadInitial();
    // Initial data is loaded once; scenario changes are handled by the next effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => stopAgentWatch.current?.(), []);

  useEffect(() => {
    if (loading || !selectedScenario) return;
    let active = true;
    api.snapshot(selectedScenario)
      .then((value) => {
        if (active) setSnapshot(value);
      })
      .catch((reason) => {
        if (active) setError(reason instanceof Error ? reason.message : "场景快照加载失败");
      });
    return () => { active = false; };
  }, [loading, selectedScenario]);

  useEffect(() => {
    if (!playing || !runDetail) return;
    const timer = window.setInterval(() => {
      setPlaybackMs((current) => {
        const next = current + 100 * speed;
        if (next >= runDetail.scenario.endTimeMs) {
          setPlaying(false);
          return runDetail.scenario.endTimeMs;
        }
        return next;
      });
    }, 100);
    return () => window.clearInterval(timer);
  }, [playing, runDetail, speed]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 4200);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const refresh = async () => {
    setRefreshing(true);
    setError(null);
    try {
      const [nextHealth, nextSnapshot, nextRuns, nextApprovals, nextAudit, nextIncidents] = await Promise.all([
        api.health(),
        api.snapshot(selectedScenario),
        api.simulations(),
        api.approvals(),
        api.audit(),
        api.incidents(),
      ]);
      setHealth(nextHealth);
      setSnapshot(nextSnapshot);
      setRuns(nextRuns);
      setApprovals(nextApprovals);
      setAudit(nextAudit);
      setIncidents(nextIncidents);
      setNotice("世界状态、仿真与审计记录已刷新");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "刷新失败");
    } finally {
      setRefreshing(false);
    }
  };

  const syncAgentRun = useCallback(async (runId: string) => {
    const current = await api.agentRun(runId);
    setAgentRun(current);
    const workflowSimulation = current.workflow?.simulation;
    if (workflowSimulation) {
      setRuns((rows) => [
        workflowSimulation,
        ...rows.filter((row) => row.runId !== workflowSimulation.runId),
      ]);
    }
    const workflowApproval = current.workflow?.approvalRequest;
    if (workflowApproval) {
      setApprovals((rows) => [
        workflowApproval,
        ...rows.filter((row) => row.approvalId !== workflowApproval.approvalId),
      ]);
    }
    if (current.status === "WAITING_APPROVAL") {
      setBusy(null);
      setNotice(
        current.approval?.stage === "POST_SIMULATION"
          ? "数字孪生已通过推进门槛，等待主管确认"
          : "高风险草案已暂停，等待主管确认",
      );
    } else if (terminalAgentRunStatuses.has(current.status)) {
      stopAgentWatch.current?.();
      stopAgentWatch.current = null;
      setBusy(null);
      if (current.response) {
        setChatResponse(current.response);
        setNotice(
          current.response.state === "CLARIFICATION_REQUIRED"
            ? "参数尚不完整，请在对话中补充缺失信息"
            : current.response.fallbackUsed
              ? "DeepSeek 不可用，已使用确定性本地解析"
              : "Agent run 已完成并通过轨迹评测",
        );
        setAudit(await api.audit());
      } else if (current.error) {
        setError(current.error);
      }
    } else {
      setBusy("chat");
    }
    return current;
  }, []);

  const watchAgentRun = useCallback((runId: string) => {
    stopAgentWatch.current?.();
    stopAgentWatch.current = api.watchAgentRun(
      runId,
      () => { void syncAgentRun(runId).catch(() => undefined); },
      () => { void syncAgentRun(runId).catch(() => undefined); },
    );
  }, [syncAgentRun]);

  const handleChat = async (message: string) => {
    setBusy("chat");
    setError(null);
    setChatResponse(null);
    try {
      const idempotencyKey = `ui-${crypto.randomUUID()}`;
      const created = await api.createAgentRun(
        message,
        selectedScenario,
        conversationId,
        idempotencyKey,
      );
      setAgentRun(created);
      watchAgentRun(created.runId);
      await syncAgentRun(created.runId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "调度请求解析失败");
      setBusy(null);
    }
  };

  const handleAgentApproval = async (approved: boolean) => {
    if (!agentRun) return;
    setBusy("agent-approval");
    setError(null);
    try {
      const resumed = await api.resumeAgentRun(agentRun.runId, approved);
      setAgentRun(resumed);
      if (approved) {
        setNotice("主管已批准，Agent 从检查点继续执行");
        watchAgentRun(agentRun.runId);
      } else {
        setBusy(null);
        setNotice("主管已拒绝高风险草案");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Agent 审批失败");
      setBusy(null);
    }
  };

  const handleAgentCancel = async () => {
    if (!agentRun) return;
    setError(null);
    try {
      const cancelled = await api.cancelAgentRun(agentRun.runId);
      setAgentRun(cancelled);
      setBusy(null);
      setNotice("Agent run 已取消");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "取消 Agent run 失败");
    }
  };

  const handleSimulate = async (intent: DispatchIntent) => {
    setBusy("simulate");
    setError(null);
    try {
      const label = intent.intentType === "BLOCK_RESOURCE" ? "共享通道封锁方案" : "紧急任务插单方案";
      const result = await api.simulate(selectedScenario, label, intent);
      setRuns((current) => [result, ...current.filter((row) => row.runId !== result.runId)]);
      await loadRun(result.runId);
      setCheckedRunIds((current) => [...new Set([result.runId, ...current])].slice(0, 4));
      setNotice(`仿真完成：${result.metrics.completedTaskCount ?? 0} 个任务，资源冲突 ${result.metrics.reservationConflictRejections ?? 0}`);
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "数字孪生仿真失败");
    } finally {
      setBusy(null);
    }
  };

  const runBaseline = async () => {
    const scenarioId = selectedScenario;
    setSelectedScenario(scenarioId);
    setBusy("baseline");
    setError(null);
    try {
      const result = await api.simulate(scenarioId, "当前场景规则基线");
      setRuns((current) => [result, ...current.filter((row) => row.runId !== result.runId)]);
      await loadRun(result.runId);
      setCheckedRunIds([result.runId]);
      setNotice("当前场景规则基线仿真已完成");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "基线仿真失败");
    } finally {
      setBusy(null);
    }
  };

  const handleDemoIntent = async (message: string) => {
    setBusy("demo-intent");
    setError(null);
    try {
      const response = await api.chat(
        message,
        selectedScenario,
        `demo-${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`,
      );
      setChatResponse(response);
      if (response.state !== "READY" || !response.intent) {
        throw new Error("演示意图缺少必要参数，未启动仿真。")
      }
      const label = response.intent.intentType === "BLOCK_RESOURCE"
        ? "一键演示 | 通道封闭"
        : "一键演示 | 紧急插单";
      const run = await api.simulate(selectedScenario, label, response.intent);
      setRuns((current) => [run, ...current.filter((row) => row.runId !== run.runId)]);
      await loadRun(run.runId);
      if (response.intent.intentType === "BLOCK_RESOURCE") {
        const approval = await api.createApproval(selectedScenario, response.intent, [run.runId]);
        setApprovals((current) => [approval, ...current.filter((row) => row.approvalId !== approval.approvalId)]);
        setNotice("通道封闭推演已完成并提交 R3 审批");
      } else {
        setNotice("紧急插单演示已完成 MASP 安全仿真");
      }
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "一键调度演示失败");
    } finally {
      setBusy(null);
    }
  };

  const handleExplainPlan = async (options: { question: string; vehicleId?: string; taskId?: string }) => {
    if (!runDetail) return;
    setBusy("plan-explain");
    setError(null);
    try {
      const explanation = await api.explainPlan(runDetail.summary.runId, options);
      setPlanExplanation(explanation);
      setNotice(explanation.fallbackUsed ? "已根据 MASP 证据生成确定性规划解释" : "DeepSeek 已根据 MASP 证据生成规划解释");
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "规划解释生成失败");
    } finally {
      setBusy(null);
    }
  };

  const handlePolicyRun = async (policy: "top_k" | "rl", candidateCount: number) => {
    setBusy("policy-run");
    setError(null);
    try {
      const label = policy === "rl" ? "群车智能体协同" : "Top-K 规则基线";
      const result = await api.simulate(
        selectedScenario,
        label,
        null,
        policy,
        policy === "rl"
          ? {
              modelId: health?.agentPolicy.modelId,
              candidateCount,
              allowDeviation: true,
            }
          : undefined,
      );
      setRuns((current) => [result, ...current.filter((row) => row.runId !== result.runId)]);
      await loadRun(result.runId);
      setCheckedRunIds((current) => [...new Set([result.runId, ...current])].slice(0, 4));
      if (result.agentPolicy?.mode === "BASELINE") {
        setNotice("智能体权重未启用，本次已由 MASP 规则策略安全接管");
      } else if (result.agentPolicy) {
        setNotice(`智能体推理完成：采用 ${result.agentPolicy.selectedAgentCandidateCount} 个安全候选`);
      } else {
        setNotice("Top-K 规则基线仿真已完成");
      }
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "策略仿真失败");
    } finally {
      setBusy(null);
    }
  };

  const handleCreateApproval = async (intent: DispatchIntent, runId: string) => {
    setBusy("approval");
    setError(null);
    try {
      const approval = await api.createApproval(selectedScenario, intent, [runId]);
      setApprovals((current) => [approval, ...current]);
      setNotice("高风险意图已提交主管审批");
      setView("approvals");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "创建审批单失败");
    } finally {
      setBusy(null);
    }
  };

  const handleDecision = async (approvalId: string, approved: boolean) => {
    setBusyApprovalId(approvalId);
    setError(null);
    try {
      const updated = await api.decideApproval(approvalId, approved);
      setApprovals((current) => current.map((item) => item.approvalId === approvalId ? updated : item));
      setNotice(approved ? "审批已批准，可提交到仿真环境" : "审批已驳回");
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "审批决策失败");
    } finally {
      setBusyApprovalId(null);
    }
  };

  const handleCommit = async (intent: DispatchIntent, approvalId?: string) => {
    setBusy("commit");
    setError(null);
    try {
      const record = await api.commitIntent(selectedScenario, intent, approvalId);
      setNotice(String(record.notice || "意图已提交到仿真环境"));
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "仿真态提交失败");
    } finally {
      setBusy(null);
    }
  };

  const handleCompare = async () => {
    setBusy("compare");
    setError(null);
    try {
      setComparison(await api.compare(checkedRunIds));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "方案比较失败");
    } finally {
      setBusy(null);
    }
  };

  const handleReport = async () => {
    setBusy("report");
    setError(null);
    try {
      setReport(await api.report());
      setReportOpen(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "运营报告生成失败");
    } finally {
      setBusy(null);
    }
  };

  const handleInjectIncident = async (options: {
    runId: string;
    vehicleId?: string;
    faultCode: string;
    requestedAtMs?: number;
    recoveryDurationMs: number;
  }) => {
    setBusy("incident-inject");
    setError(null);
    try {
      const incident = await api.injectVehicleFault(options.runId, options);
      setIncidents((current) => [incident, ...current.filter((row) => row.incidentId !== incident.incidentId)]);
      setSelectedIncidentId(incident.incidentId);
      await loadRun(incident.runId);
      setPlaybackMs(incident.faultAtMs);
      setNotice(`故障已在安全节点 ${incident.locationNodeId || "-"} 建立仿真分支`);
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "故障注入失败");
    } finally {
      setBusy(null);
    }
  };

  const handleInjectWorkstation = async (options: {
    runId: string;
    workstationNodeId?: string;
    requestedAtMs?: number;
    recoveryDurationMs: number;
  }) => {
    setBusy("incident-inject-workstation");
    setError(null);
    try {
      const incident = await api.injectWorkstationOutage(options.runId, options);
      setIncidents((current) => [incident, ...current.filter((row) => row.incidentId !== incident.incidentId)]);
      setSelectedIncidentId(incident.incidentId);
      await loadRun(incident.runId);
      setPlaybackMs(incident.faultAtMs);
      setNotice(`工位 ${incident.workstationId || incident.locationNodeId || "-"} 已建立停用分支`);
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "工位停用注入失败");
    } finally {
      setBusy(null);
    }
  };

  const handleInjectDeadlock = async (
    runId: string,
    deadlockCase: "RECOVERABLE" | "UNRECOVERABLE",
  ) => {
    setBusy("incident-inject-deadlock");
    setError(null);
    try {
      const incident = await api.injectDeadlock(runId, deadlockCase);
      setIncidents((current) => [incident, ...current.filter((row) => row.incidentId !== incident.incidentId)]);
      setSelectedIncidentId(incident.incidentId);
      await loadRun(incident.runId);
      setPlaybackMs(incident.faultAtMs);
      setNotice(
        incident.eventAttributes.recoveryAvailable
          ? "MASP 已检测等待环并生成受控倒退候选"
          : "MASP 已检测不可恢复死锁并保持安全停车",
      );
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "等待环注入失败");
    } finally {
      setBusy(null);
    }
  };

  const handleDiagnoseIncident = async (incidentId: string) => {
    setBusy("incident-diagnose");
    setError(null);
    try {
      const incident = await api.diagnoseIncident(incidentId);
      setIncidents((current) => current.map((row) => row.incidentId === incidentId ? incident : row));
      setNotice(incident.diagnosis?.fallbackUsed ? "DeepSeek 不可用或证据校验未通过，已生成规则诊断" : "DeepSeek 已完成证据约束诊断");
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "异常诊断失败");
    } finally {
      setBusy(null);
    }
  };

  const handleIncidentWhatIf = async (incidentId: string, mode: WhatIfMode) => {
    setBusy(`incident-${mode}`);
    setError(null);
    try {
      const incident = await api.runIncidentWhatIf(incidentId, mode);
      const nextRuns = await api.simulations();
      setIncidents((current) => current.map((row) => row.incidentId === incidentId ? incident : row));
      setRuns(nextRuns);
      setNotice(`${mode} 处置分支已完成 MASP 推演，尚未执行任何恢复动作`);
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "处置方案推演失败");
    } finally {
      setBusy(null);
    }
  };

  const handleIncidentApproval = async (incidentId: string, mode: WhatIfMode) => {
    setBusy(`incident-approval-${mode}`);
    setError(null);
    try {
      const approval = await api.createIncidentApproval(incidentId, mode);
      const nextIncidents = await api.incidents();
      setApprovals((current) => [approval, ...current.filter((row) => row.approvalId !== approval.approvalId)]);
      setIncidents(nextIncidents);
      setNotice("处置方案已绑定仿真证据并提交主管审批");
      setAudit(await api.audit());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "处置方案送审失败");
    } finally {
      setBusy(null);
    }
  };

  const handleIncidentCompare = async (runIds: string[]) => {
    setCheckedRunIds(runIds.slice(0, 4));
    setView("simulations");
    setBusy("compare");
    setError(null);
    try {
      setComparison(await api.compare(runIds.slice(0, 4)));
      if (runIds[0]) await loadRun(runIds[0]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "处置方案比较失败");
    } finally {
      setBusy(null);
    }
  };

  const downloadIncidentReport = async (incidentId: string) => {
    setError(null);
    try {
      const nextReport = await api.incidentReport(incidentId);
      const blob = new Blob([JSON.stringify(nextReport, null, 2)], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `lingshu-incident-${incidentId}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "异常报告导出失败");
    }
  };

  const downloadReport = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `lingshu-shift-report-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const currentIntentRun = useMemo(
    () => runs.find((run) => run.intentId && run.intentId === chatResponse?.intent?.intentId) || null,
    [chatResponse?.intent?.intentId, runs],
  );
  const currentApproval = useMemo(
    () => approvals.find((approval) => approval.intent.intentId === chatResponse?.intent?.intentId) || null,
    [approvals, chatResponse?.intent?.intentId],
  );
  const selectedIncident = useMemo(
    () => incidents.find((incident) => incident.incidentId === selectedIncidentId) || null,
    [incidents, selectedIncidentId],
  );
  const pendingApprovals = approvals.filter((approval) => approval.status === "PENDING").length;
  const scenarioMeta = scenarios.find((item) => item.scenarioId === selectedScenario);

  if (loading) {
    return (
      <div className="loading-shell" aria-label="正在加载控制台">
        <Skeleton><SkeletonItem className="loading-header" /></Skeleton>
        <div className="loading-grid">
          <Skeleton><SkeletonItem className="loading-nav" /></Skeleton>
          <Skeleton><SkeletonItem className="loading-main" /></Skeleton>
          <Skeleton><SkeletonItem className="loading-side" /></Skeleton>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark"><BrainCircuit24Regular /></span>
          <div>
            <strong>保利智仓·灵枢</strong>
            <span>AI 调度指挥中心</span>
          </div>
        </div>
        <div className="topbar-context">
          <Badge className="environment-badge" appearance="filled" color="informative">仿真环境</Badge>
          <Select
            aria-label="选择演示场景"
            value={selectedScenario}
            onChange={(_, data) => {
              setSelectedScenario(data.value);
              setRunDetail(null);
              setSelectedRunId(null);
              setPlaying(false);
              setPlaybackMs(0);
            }}
          >
            {scenarios.map((scenario) => (
              <option key={scenario.scenarioId} value={scenario.scenarioId}>
                {scenario.scenarioId} ({scenario.vehicleCount}车/{scenario.taskCount}任务)
              </option>
            ))}
          </Select>
          <Tooltip content="刷新状态" relationship="label">
            <Button
              appearance="subtle"
              icon={<ArrowClockwise20Regular />}
              aria-label="刷新状态"
              disabled={refreshing}
              onClick={() => void refresh()}
            />
          </Tooltip>
          <Tooltip content="演示账号，无生产权限" relationship="label">
            <Button appearance="subtle" icon={<SignOut20Regular />}>调度演示员</Button>
          </Tooltip>
        </div>
      </header>

      <aside className="side-nav">
        <nav aria-label="主导航">
          {viewItems.map((item) => (
            <Button
              key={item.id}
              appearance={view === item.id ? "primary" : "subtle"}
              icon={item.icon}
              onClick={() => setView(item.id)}
            >
              {item.label}
              {item.id === "approvals" && pendingApprovals > 0 && (
                <span className="nav-count">{pendingApprovals}</span>
              )}
            </Button>
          ))}
        </nav>
        <div className="engine-summary">
          <span>MASP 引擎</span>
          <strong className="mono">{health?.engine.currentCommit.slice(0, 8)}</strong>
          <Badge appearance="tint" color={health?.engine.allowed ? "success" : "danger"}>
            {health?.engine.allowed ? "版本已锁定" : "版本异常"}
          </Badge>
          <p>{health?.engine.dirty ? `${health.engine.dirtyFileCount} 个开发中修改` : "工作区干净"}</p>
        </div>
      </aside>

      <main className="main-content">
        <div className="page-heading">
          <div>
            <h1>{viewItems.find((item) => item.id === view)?.label}</h1>
            <p>场景 {selectedScenario} | 世界版本 <span className="mono">{snapshot?.worldRevision ?? "-"}</span></p>
          </div>
          <div className="demo-actions">
            <Button
              appearance="secondary"
              icon={<VehicleTruck20Regular />}
              disabled={Boolean(busy)}
              onClick={() => void runBaseline()}
            >
              {busy === "baseline" ? "基线运行中" : "运行当前场景基线"}
            </Button>
            <Button
              appearance="secondary"
              icon={<Bot20Regular />}
              disabled={Boolean(busy)}
              onClick={() => void handleChat("创建紧急叉车任务，从 AP1123 运到 AP2121，优先级设为最高")}
            >
              注入紧急任务
            </Button>
            <Button
              appearance="secondary"
              icon={<Alert20Regular />}
              disabled={Boolean(busy)}
              onClick={() => void handleChat("共享窄路需要检修，请封闭三分钟并评估任务影响")}
            >
              推演通道封闭
            </Button>
            <Button
              appearance="secondary"
              icon={<DocumentText20Regular />}
              disabled={Boolean(busy)}
              onClick={() => void handleReport()}
            >
              运营报告
            </Button>
          </div>
        </div>

        {error && (
          <div className="status-banner error-banner" role="alert">
            <Alert20Regular />
            <span>{error}</span>
            <Button appearance="subtle" onClick={() => setError(null)}>关闭</Button>
          </div>
        )}
        {notice && <div className="status-banner notice-banner" role="status">{notice}</div>}
        {health?.engine.warning && (
          <div className="status-banner warning-banner"><Alert20Regular />{health.engine.warning}</div>
        )}

        {view === "command" && (
          <section className="metric-strip" aria-label="关键运营指标">
            <div><span>车辆</span><strong className="mono">{formatCount(snapshot?.counts.vehicles)}</strong><small>fork {snapshot?.groups.fork || 0} / jack {snapshot?.groups.jack || 0}</small></div>
            <div><span>待调度任务</span><strong className="mono">{formatCount(snapshot?.counts.tasks)}</strong><small>{scenarioMeta ? `${Math.round(scenarioMeta.endTimeMs / 60000)} 分钟仿真窗` : "-"}</small></div>
            <div><span>路网节点</span><strong className="mono">{formatCount(snapshot?.counts.nodes)}</strong><small>{formatCount(snapshot?.counts.edges)} 条有向边</small></div>
            <div><span>冲突资源对</span><strong className="mono">{formatCount(snapshot?.counts.conflictPairs)}</strong><small>确定性安全校验</small></div>
            <div><span>模型状态</span><strong>{health?.model.configured ? (health.model.provider === "deepseek" ? "DeepSeek API" : "本地微调模型") : "本地降级"}</strong><small>{health?.model.model}</small></div>
          </section>
        )}

        {snapshot && map && view === "command" && (
          <div className="command-grid">
            <WarehouseMap
              map={map}
              snapshot={snapshot}
              run={runDetail}
              intent={chatResponse?.intent}
              playbackMs={playbackMs}
              playing={playing}
              speed={speed}
              onTogglePlaying={() => setPlaying((value) => !value)}
              onPlaybackChange={(value) => { setPlaybackMs(value); setPlaying(false); }}
              onSpeedChange={setSpeed}
              selectedVehicleIds={selectedVehicleIds}
              onToggleVehicle={toggleReplayVehicle}
            />
            <AssistantPanel
              response={chatResponse}
              agentRun={agentRun}
              run={currentIntentRun}
              approval={currentApproval}
              busy={busy}
              onSend={handleChat}
              onAgentApproval={handleAgentApproval}
              onAgentCancel={handleAgentCancel}
              onSimulate={handleSimulate}
              onCreateApproval={handleCreateApproval}
              onCommit={handleCommit}
            />
            <DispatchReplayPanel
              run={runDetail}
              playbackMs={playbackMs}
              selectedVehicleIds={selectedVehicleIds}
              onToggleVehicle={toggleReplayVehicle}
            />
          </div>
        )}

        {snapshot && map && view === "simulations" && (
          <div className="simulation-workspace">
            {health?.agentPolicy && (
              <AgentPolicyPanel
                status={health.agentPolicy}
                run={runDetail}
                busy={busy === "policy-run"}
                onRun={(policy, candidateCount) => void handlePolicyRun(policy, candidateCount)}
              />
            )}
            <PlanExplanationPanel
              run={runDetail}
              report={planExplanation?.runId === runDetail?.summary.runId ? planExplanation : null}
              busy={busy === "plan-explain"}
              onExplain={(options) => void handleExplainPlan(options)}
            />
            <WarehouseMap
              map={map}
              snapshot={snapshot}
              run={runDetail}
              intent={chatResponse?.intent}
              playbackMs={playbackMs}
              playing={playing}
              speed={speed}
              onTogglePlaying={() => setPlaying((value) => !value)}
              onPlaybackChange={(value) => { setPlaybackMs(value); setPlaying(false); }}
              onSpeedChange={setSpeed}
              selectedVehicleIds={selectedVehicleIds}
              onToggleVehicle={toggleReplayVehicle}
            />
            <DispatchReplayPanel
              run={runDetail}
              playbackMs={playbackMs}
              selectedVehicleIds={selectedVehicleIds}
              onToggleVehicle={toggleReplayVehicle}
            />
            <SimulationTable
              runs={runs}
              selectedRunId={selectedRunId}
              checkedRunIds={checkedRunIds}
              comparison={comparison}
              busy={busy === "compare"}
              onSelect={(runId) => void loadRun(runId)}
              onToggleChecked={(runId, checked) => setCheckedRunIds((current) =>
                checked ? [...new Set([...current, runId])].slice(0, 4) : current.filter((id) => id !== runId),
              )}
              onCompare={() => void handleCompare()}
            />
          </div>
        )}

        {snapshot && map && view === "incidents" && (
          <div className="incident-page-layout">
            <IncidentWorkbench
              incidents={incidents}
              selected={selectedIncident}
              runs={runs}
              busy={busy}
              onSelect={(incident) => {
                setSelectedIncidentId(incident.incidentId);
                void loadRun(incident.runId).then(() => setPlaybackMs(incident.faultAtMs));
              }}
              onInject={handleInjectIncident}
              onInjectWorkstation={handleInjectWorkstation}
              onInjectDeadlock={handleInjectDeadlock}
              onDiagnose={handleDiagnoseIncident}
              onWhatIf={handleIncidentWhatIf}
              onRequestApproval={handleIncidentApproval}
              onDemoTask={() => handleDemoIntent("创建紧急叉车任务，从 AP1123 运到 AP2121")}
              onDemoRoadblock={() => handleDemoIntent("共享窄路需要检修，请封闭三分钟并评估任务影响")}
              onCompare={(runIds) => void handleIncidentCompare(runIds)}
              onDownloadReport={downloadIncidentReport}
            />
            <WarehouseMap
              map={map}
              snapshot={snapshot}
              run={runDetail}
              incident={selectedIncident}
              playbackMs={selectedIncident?.faultAtMs ?? playbackMs}
              playing={false}
              speed={speed}
              onTogglePlaying={() => undefined}
              onPlaybackChange={setPlaybackMs}
              onSpeedChange={setSpeed}
              selectedVehicleIds={selectedVehicleIds}
              onToggleVehicle={toggleReplayVehicle}
            />
          </div>
        )}

        {view === "designer" && (
          <ScenarioDesigner
            scenarios={scenarios}
            initialScenarioId={selectedScenario}
            onNotice={handleDesignerNotice}
            onError={handleDesignerError}
          />
        )}

        {view === "evaluation" && (
          <EvaluationCenter onNotice={setNotice} onError={setError} />
        )}

        {view === "approvals" && (
          <ApprovalsPanel approvals={approvals} busyId={busyApprovalId} onDecision={handleDecision} />
        )}

        {snapshot && view === "operations" && <OperationsPanel snapshot={snapshot} audit={audit} />}
      </main>

      <Dialog open={reportOpen} onOpenChange={(_, data) => setReportOpen(data.open)}>
        <DialogSurface>
          <DialogBody>
            <DialogTitle>{report?.title || "运营报告"}</DialogTitle>
            <DialogContent>
              {report && (
                <div className="report-content">
                  <div className="report-metrics">
                    <div><span>仿真总数</span><strong className="mono">{report.runCount}</strong></div>
                    <div><span>成功仿真</span><strong className="mono">{report.successfulRunCount}</strong></div>
                    <div><span>审批单</span><strong className="mono">{report.approvalCount}</strong></div>
                    <div><span>待审批</span><strong className="mono">{report.pendingApprovalCount}</strong></div>
                  </div>
                  {report.latestRun && (
                    <div className="report-latest">
                      <h3>最近成功方案</h3>
                      <p>{report.latestRun.label}</p>
                      <dl>
                        <div><dt>完成任务</dt><dd>{String(report.latestRun.metrics.completedTaskCount ?? "-")}</dd></div>
                        <div><dt>资源冲突</dt><dd>{String(report.latestRun.metrics.reservationConflictRejections ?? "-")}</dd></div>
                        <div><dt>平均周期</dt><dd>{typeof report.latestRun.metrics.meanTaskCycleTimeMs === "number" ? `${(report.latestRun.metrics.meanTaskCycleTimeMs / 1000).toFixed(1)} s` : "-"}</dd></div>
                      </dl>
                    </div>
                  )}
                  <p className="report-notice">{report.notice}</p>
                </div>
              )}
            </DialogContent>
            <DialogActions>
              <DialogTrigger disableButtonEnhancement>
                <Button appearance="secondary">关闭</Button>
              </DialogTrigger>
              <Button appearance="primary" icon={<DocumentArrowDown20Regular />} onClick={downloadReport}>
                导出 JSON
              </Button>
            </DialogActions>
          </DialogBody>
        </DialogSurface>
      </Dialog>
    </div>
  );
}
