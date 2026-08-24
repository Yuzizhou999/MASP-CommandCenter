import { useEffect, useMemo, useState } from "react";
import {
  Badge,
  Button,
  Checkbox,
  Field,
  Input,
  Select,
  Spinner,
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableHeaderCell,
  TableRow,
} from "@fluentui/react-components";
import {
  ArrowDownload20Regular,
  ArrowSync20Regular,
  Database20Regular,
  BrainCircuit20Regular,
  Play20Regular,
  ShieldCheckmark20Regular,
} from "@fluentui/react-icons";
import { api } from "../api";
import type {
  AgentMetricsSummary,
  BenchmarkReport,
  BenchmarkRequest,
  BenchmarkSummary,
  DatasetExportManifest,
  ModelSafetyEvaluationReport,
  ModelSafetyEvaluationSummary,
} from "../types";

interface EvaluationCenterProps {
  onNotice: (message: string) => void;
  onError: (message: string) => void;
}

const vehicleOptions = [14, 30, 50, 100] as const;
const arrivalOptions = [
  { value: "low", label: "低负载" },
  { value: "medium", label: "中负载" },
  { value: "high", label: "高负载" },
] as const;
const fleetOptions = [
  { value: "mixed", label: "混合车型" },
  { value: "fork", label: "仅叉车" },
  { value: "jack", label: "仅顶升车" },
] as const;
const policyOptions = [
  { value: "top_k", label: "Top-K 基线" },
  { value: "task_age", label: "任务等待" },
  { value: "shortest_remaining", label: "最短剩余" },
  { value: "congestion", label: "拥堵感知" },
  { value: "previous_order", label: "历史顺序" },
  { value: "random", label: "随机策略" },
  { value: "rl", label: "智能体策略" },
] as const;

const arrivalLabels: Record<string, string> = { low: "低", medium: "中", high: "高" };
const fleetLabels: Record<string, string> = { mixed: "混合", fork: "叉车", jack: "顶升车" };
const policyLabels = Object.fromEntries(policyOptions.map((item) => [item.value, item.label]));
const executionLabels: Record<string, string> = {
  DEEPSEEK_API: "DeepSeek API",
  DETERMINISTIC_FALLBACK: "确定性降级",
  BOUNDARY_CHECK: "边界检查",
  KNOWLEDGE_RETRIEVAL: "知识检索",
  EVALUATOR_ERROR: "评测异常",
};

const formatNumber = (value?: number | null, digits = 1) =>
  typeof value === "number" ? value.toLocaleString("zh-CN", { maximumFractionDigits: digits }) : "-";

const formatSeconds = (value?: number | null) =>
  typeof value === "number" ? `${formatNumber(value / 1000)} s` : "-";

const formatPercent = (value?: number | null) =>
  typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "-";

const formatInterval = (metric?: { mean?: number | null; ci95Low?: number | null; ci95High?: number | null }) => {
  if (typeof metric?.mean !== "number") return "-";
  return `${formatNumber(metric.mean)} [${formatNumber(metric.ci95Low)}, ${formatNumber(metric.ci95High)}]`;
};

function toggleValue<T>(current: T[], value: T, checked: boolean): T[] {
  return checked ? [...new Set([...current, value])] : current.filter((item) => item !== value);
}

export function EvaluationCenter({ onNotice, onError }: EvaluationCenterProps) {
  const [benchmarks, setBenchmarks] = useState<BenchmarkSummary[]>([]);
  const [selectedReport, setSelectedReport] = useState<BenchmarkReport | null>(null);
  const [modelEvaluations, setModelEvaluations] = useState<ModelSafetyEvaluationSummary[]>([]);
  const [selectedModelReport, setSelectedModelReport] = useState<ModelSafetyEvaluationReport | null>(null);
  const [exports, setExports] = useState<DatasetExportManifest[]>([]);
  const [selectedExport, setSelectedExport] = useState<DatasetExportManifest | null>(null);
  const [agentMetrics, setAgentMetrics] = useState<AgentMetricsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<"benchmark" | "model" | "model-report" | "dataset" | "report" | null>(null);
  const [suiteName, setSuiteName] = useState("仓储群车高负载基准");
  const [vehicleCounts, setVehicleCounts] = useState<number[]>([14]);
  const [arrivalProfiles, setArrivalProfiles] = useState<string[]>(["medium", "high"]);
  const [fleetMixes, setFleetMixes] = useState<string[]>(["mixed"]);
  const [policies, setPolicies] = useState<string[]>(["top_k", "rl"]);
  const [seedText, setSeedText] = useState("0,1,2");
  const [horizonMs, setHorizonMs] = useState(900000);
  const [modelSuiteName, setModelSuiteName] = useState("大模型调度安全回归");
  const [datasetName, setDatasetName] = useState("仓储调度评测数据");
  const [includeAudit, setIncludeAudit] = useState(true);
  const [includeIncidents, setIncludeIncidents] = useState(true);
  const [includeEvidenceText, setIncludeEvidenceText] = useState(false);

  const seeds = useMemo(
    () => [...new Set(seedText.split(/[,，\s]+/).filter(Boolean).map(Number).filter(Number.isInteger))],
    [seedText],
  );
  const caseCount = vehicleCounts.length * arrivalProfiles.length * fleetMixes.length * policies.length * seeds.length;
  const configValid = Boolean(suiteName.trim()) && caseCount > 0 && caseCount <= 2000 && seeds.length <= 10;

  const loadLists = async () => {
    const [nextBenchmarks, nextModelEvaluations, nextExports, nextAgentMetrics] = await Promise.all([
      api.benchmarks(),
      api.modelSafetyEvaluations(),
      api.datasetExports(),
      api.agentMetrics(),
    ]);
    setBenchmarks(nextBenchmarks);
    setModelEvaluations(nextModelEvaluations);
    setExports(nextExports);
    setAgentMetrics(nextAgentMetrics);
    setSelectedExport((current) => current || nextExports[0] || null);
    return { nextBenchmarks, nextModelEvaluations };
  };

  useEffect(() => {
    let active = true;
    loadLists()
      .then(async ({ nextBenchmarks, nextModelEvaluations }) => {
        if (!active) return;
        const [report, modelReport] = await Promise.all([
          nextBenchmarks[0] ? api.benchmarkDetail(nextBenchmarks[0].benchmarkId) : null,
          nextModelEvaluations[0] ? api.modelSafetyEvaluationDetail(nextModelEvaluations[0].evaluationId) : null,
        ]);
        if (active) {
          setSelectedReport(report);
          setSelectedModelReport(modelReport);
        }
      })
      .catch((reason) => onError(reason instanceof Error ? reason.message : "评测资产加载失败"))
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
    // Parent callbacks are stable for the lifetime of this view.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runBenchmark = async () => {
    if (!configValid) return;
    setBusy("benchmark");
    try {
      const request: BenchmarkRequest = {
        suiteName: suiteName.trim(),
        baseScenarioId: "rhpp-long-distance-conflict",
        vehicleCounts,
        arrivalProfiles: arrivalProfiles as BenchmarkRequest["arrivalProfiles"],
        fleetMixes: fleetMixes as BenchmarkRequest["fleetMixes"],
        policies: policies as BenchmarkRequest["policies"],
        seeds,
        horizonMs,
        requestedBy: "evaluation-operator",
        ...(policies.includes("rl") ? { agentPolicy: { candidateCount: 3, allowDeviation: true } } : {}),
      };
      const report = await api.runBenchmark(request);
      setSelectedReport(report);
      setBenchmarks(await api.benchmarks());
      onNotice(`评测完成：${report.completedCaseCount}/${report.caseCount} 个用例完成，安全门槛${report.safetyGate.passed ? "通过" : "未通过"}`);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : "基准评测失败");
    } finally {
      setBusy(null);
    }
  };

  const openReport = async (benchmarkId: string) => {
    setBusy("report");
    try {
      setSelectedReport(await api.benchmarkDetail(benchmarkId));
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : "评测报告加载失败");
    } finally {
      setBusy(null);
    }
  };

  const runModelEvaluation = async () => {
    if (!modelSuiteName.trim()) return;
    setBusy("model");
    try {
      const report = await api.runModelSafetyEvaluation(modelSuiteName.trim());
      setSelectedModelReport(report);
      setModelEvaluations(await api.modelSafetyEvaluations());
      onNotice(
        `模型安全评测完成：${report.passedCaseCount}/${report.coverage.caseCount} 通过，` +
        `${report.liveProviderEvaluated ? "包含 DeepSeek 实测" : "本次为离线降级与边界验证"}`,
      );
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : "模型安全评测失败");
    } finally {
      setBusy(null);
    }
  };

  const openModelReport = async (evaluationId: string) => {
    setBusy("model-report");
    try {
      setSelectedModelReport(await api.modelSafetyEvaluationDetail(evaluationId));
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : "模型安全报告加载失败");
    } finally {
      setBusy(null);
    }
  };

  const createDataset = async () => {
    if (!datasetName.trim()) return;
    setBusy("dataset");
    try {
      const manifest = await api.createDatasetExport({
        name: datasetName.trim(),
        includeAudit,
        includeIncidents,
        includeEvidenceText,
        requestedBy: "data-steward",
      });
      setSelectedExport(manifest);
      setExports(await api.datasetExports());
      onNotice(`脱敏数据集已生成：${manifest.recordCount} 条记录，质量检查${manifest.quality.passed ? "通过" : "未通过"}`);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : "数据集导出失败");
    } finally {
      setBusy(null);
    }
  };

  const downloadDataset = (exportId: string) => {
    const anchor = document.createElement("a");
    anchor.href = api.datasetExportDownloadUrl(exportId);
    anchor.download = `${exportId}.zip`;
    anchor.click();
  };

  if (loading) {
    return <div className="evaluation-loading"><Spinner label="正在加载评测资产" /></div>;
  }

  return (
    <div className="evaluation-workspace">
      <section className="data-panel benchmark-config">
        <div className="panel-heading panel-heading-actions">
          <div><h2>基准矩阵</h2><p>固定输入、引擎版本与随机种子，结果可重复核验</p></div>
          <Badge appearance="tint" color={caseCount > 0 && caseCount <= 2000 ? "informative" : "danger"}>{caseCount} 个用例</Badge>
        </div>
        <div className="evaluation-form">
          <Field label="评测名称"><Input value={suiteName} onChange={(_, data) => setSuiteName(data.value)} /></Field>
          <div className="evaluation-option-group"><span>车辆规模</span><div>{vehicleOptions.map((value) => <Checkbox key={value} label={`${value} 车`} checked={vehicleCounts.includes(value)} onChange={(_, data) => setVehicleCounts(toggleValue(vehicleCounts, value, data.checked === true))} />)}</div></div>
          <div className="evaluation-option-group"><span>任务到达强度</span><div>{arrivalOptions.map((item) => <Checkbox key={item.value} label={item.label} checked={arrivalProfiles.includes(item.value)} onChange={(_, data) => setArrivalProfiles(toggleValue(arrivalProfiles, item.value, data.checked === true))} />)}</div></div>
          <div className="evaluation-option-group"><span>车型组合</span><div>{fleetOptions.map((item) => <Checkbox key={item.value} label={item.label} checked={fleetMixes.includes(item.value)} onChange={(_, data) => setFleetMixes(toggleValue(fleetMixes, item.value, data.checked === true))} />)}</div></div>
          <div className="evaluation-option-group policy-options"><span>对照策略</span><div>{policyOptions.map((item) => <Checkbox key={item.value} label={item.label} checked={policies.includes(item.value)} onChange={(_, data) => setPolicies(toggleValue(policies, item.value, data.checked === true))} />)}</div></div>
          <div className="evaluation-inline-fields">
            <Field label="随机种子" hint="逗号分隔，最多 10 个"><Input value={seedText} onChange={(_, data) => setSeedText(data.value)} /></Field>
            <Field label="仿真窗口"><Select value={String(horizonMs)} onChange={(_, data) => setHorizonMs(Number(data.value))}><option value="300000">5 分钟</option><option value="900000">15 分钟</option><option value="1800000">30 分钟</option><option value="3600000">60 分钟</option></Select></Field>
          </div>
          {!configValid && <p className="field-error">至少选择每类一项；随机种子不超过 10 个，总用例不超过 2000。</p>}
          <Button appearance="primary" icon={<Play20Regular />} disabled={!configValid || Boolean(busy)} onClick={() => void runBenchmark()}>{busy === "benchmark" ? "正在逐项评测" : "运行评测矩阵"}</Button>
        </div>
      </section>

      <section className="data-panel benchmark-history">
        <div className="panel-heading panel-heading-actions"><div><h2>评测记录</h2><p>{benchmarks.length} 份可追溯报告</p></div><Button appearance="subtle" icon={<ArrowSync20Regular />} aria-label="刷新评测记录" onClick={() => void loadLists().catch((reason) => onError(reason instanceof Error ? reason.message : "评测记录刷新失败"))} /></div>
        <div className="evaluation-list">
          {benchmarks.map((row) => <button type="button" key={row.benchmarkId} className={selectedReport?.benchmarkId === row.benchmarkId ? "evaluation-list-row selected-row" : "evaluation-list-row"} onClick={() => void openReport(row.benchmarkId)}><span><strong>{row.suiteName}</strong><small>{new Date(row.createdAt).toLocaleString("zh-CN", { hour12: false })}</small></span><span><Badge appearance="tint" color={row.safetyGate.passed ? "success" : "danger"}>{row.safetyGate.passed ? "安全通过" : "存在失败"}</Badge><small className="mono">{row.completedCaseCount}/{row.caseCount}</small></span></button>)}
          {benchmarks.length === 0 && <div className="evaluation-empty">尚无评测记录</div>}
        </div>
      </section>

      <section className="data-panel benchmark-report">
        <div className="panel-heading panel-heading-actions"><div><h2>{selectedReport?.suiteName || "评测报告"}</h2><p>{selectedReport ? `${selectedReport.benchmarkId} | ${selectedReport.caseCount} 个固定用例` : "运行矩阵后生成统计报告"}</p></div>{selectedReport && <Badge appearance="tint" color={selectedReport.safetyGate.passed ? "success" : "danger"} icon={<ShieldCheckmark20Regular />}>{selectedReport.safetyGate.passed ? "安全门槛通过" : "安全门槛未通过"}</Badge>}</div>
        {selectedReport ? <>
          <div className="evaluation-metrics">
            <div><span>完成用例</span><strong className="mono">{selectedReport.completedCaseCount}/{selectedReport.caseCount}</strong></div>
            <div><span>资源冲突失败</span><strong className="mono">{selectedReport.safetyGate.conflictCaseCount}</strong></div>
            <div><span>规划超时用例</span><strong className="mono">{selectedReport.safetyGate.planningTimeoutCaseCount}</strong></div>
            <div><span>运行耗时</span><strong className="mono">{formatNumber(selectedReport.durationMs / 1000)} s</strong></div>
          </div>
          <div className="coverage-line"><span>覆盖范围</span><strong>{selectedReport.coverage.vehicleCounts.join("/")} 车 · {selectedReport.coverage.arrivalProfiles.map((item) => arrivalLabels[item] || item).join("/")}负载 · {selectedReport.coverage.fleetMixes.map((item) => fleetLabels[item] || item).join("/")}车型 · {selectedReport.coverage.seeds.length} 个种子</strong></div>
          <div className="table-scroll evaluation-table"><Table size="small" aria-label="评测聚合统计"><TableHeader><TableRow><TableHeaderCell>车辆/负载/车型</TableHeaderCell><TableHeaderCell>策略</TableHeaderCell><TableHeaderCell>成功</TableHeaderCell><TableHeaderCell>完成任务 均值[95%CI]</TableHeaderCell><TableHeaderCell>吞吐 均值[95%CI]</TableHeaderCell><TableHeaderCell>平均周期</TableHeaderCell><TableHeaderCell>冲突拒绝</TableHeaderCell></TableRow></TableHeader><TableBody>{selectedReport.aggregates.map((row) => <TableRow key={`${row.vehicleCount}-${row.arrivalProfile}-${row.fleetMix}-${row.policy}`}><TableCell>{row.vehicleCount} / {arrivalLabels[row.arrivalProfile] || row.arrivalProfile} / {fleetLabels[row.fleetMix] || row.fleetMix}</TableCell><TableCell>{policyLabels[row.policy] || row.policy}</TableCell><TableCell className="mono">{row.successfulCaseCount}/{row.caseCount}</TableCell><TableCell className="mono">{formatInterval(row.metrics.completedTaskCount)}</TableCell><TableCell className="mono">{formatInterval(row.metrics.completedDropoffsPerHour)}</TableCell><TableCell className="mono">{formatSeconds(row.metrics.meanTaskCycleTimeMs?.mean)}</TableCell><TableCell className="mono">{formatNumber(row.metrics.reservationConflictRejections?.mean)}</TableCell></TableRow>)}</TableBody></Table></div>
          {selectedReport.failureCases.length > 0 && <div className="failure-summary"><strong>失败案例</strong>{selectedReport.failureCases.slice(0, 8).map((row) => <span key={row.caseId}><code>{row.caseId}</code>{row.error || "未返回错误说明"}</span>)}</div>}
        </> : <div className="evaluation-empty">选择历史评测或运行新的矩阵。</div>}
      </section>

      <section className="data-panel agent-observability">
        <div className="panel-heading panel-heading-actions">
          <div><h2>Agent 运行观测</h2><p>工具轨迹、完成率、模型降级与安全拦截</p></div>
          <Button
            appearance="subtle"
            icon={<ArrowSync20Regular />}
            aria-label="刷新 Agent 运行指标"
            onClick={() => void api.agentMetrics().then(setAgentMetrics).catch((reason) => onError(reason instanceof Error ? reason.message : "Agent 指标刷新失败"))}
          />
        </div>
        {agentMetrics ? <>
          <div className="agent-observability-metrics">
            <div><span>请求总数</span><strong className="mono">{agentMetrics.requestCount}</strong></div>
            <div><span>任务完成率</span><strong className="mono">{formatPercent(agentMetrics.taskCompletionRate)}</strong></div>
            <div><span>模型工具规划</span><strong className="mono">{formatPercent(agentMetrics.modelToolPlanningRate)}</strong></div>
            <div><span>模型降级率</span><strong className="mono">{formatPercent(agentMetrics.fallbackRate)}</strong></div>
            <div><span>P95 延迟</span><strong className="mono">{formatNumber(agentMetrics.p95DurationMs)} ms</strong></div>
            <div><span>平均执行步数</span><strong className="mono">{formatNumber(agentMetrics.averageStepCount)}</strong></div>
          </div>
          <div className="agent-tool-distribution">
            <span>工具调用</span>
            {Object.entries(agentMetrics.toolCallCounts).map(([name, count]) => (
              <code key={name}>{name} · {count}</code>
            ))}
            {Object.keys(agentMetrics.toolCallCounts).length === 0 && <strong>暂无轨迹</strong>}
          </div>
        </> : <div className="evaluation-empty">尚无 Agent 运行指标</div>}
      </section>

      <section className="data-panel model-evaluation">
        <div className="panel-heading panel-heading-actions">
          <div><h2>大模型安全评测</h2><p>意图、参数、知识、提示注入、证据、动作授权和降级韧性</p></div>
          {selectedModelReport && <Badge appearance="tint" color={selectedModelReport.safetyGate.passed ? "success" : "danger"} icon={<ShieldCheckmark20Regular />}>{selectedModelReport.safetyGate.passed ? "关键用例通过" : "关键用例失败"}</Badge>}
        </div>
        <div className="model-evaluation-layout">
          <div className="model-evaluation-controls">
            <Field label="评测名称"><Input value={modelSuiteName} onChange={(_, data) => setModelSuiteName(data.value)} /></Field>
            <p className="model-evaluation-note">运行仓库内固定测试集。配置 DeepSeek 时执行真实调用；未配置或调用失败时记录为确定性降级，边界恶意输出始终由后端固定向量验证。</p>
            <Button appearance="primary" icon={<BrainCircuit20Regular />} disabled={!modelSuiteName.trim() || Boolean(busy)} onClick={() => void runModelEvaluation()}>{busy === "model" ? "正在核对模型边界" : "运行模型安全评测"}</Button>
            <div className="model-evaluation-history">
              <strong>历史报告</strong>
              {modelEvaluations.map((row) => <button type="button" key={row.evaluationId} className={selectedModelReport?.evaluationId === row.evaluationId ? "evaluation-list-row selected-row" : "evaluation-list-row"} onClick={() => void openModelReport(row.evaluationId)}><span><strong>{row.suiteName}</strong><small>{new Date(row.createdAt).toLocaleString("zh-CN", { hour12: false })}</small></span><span><Badge appearance="tint" color={row.safetyGate.passed ? "success" : "danger"}>{row.passedCaseCount}/{row.coverage.caseCount}</Badge><small>{row.liveProviderEvaluated ? "DeepSeek 实测" : "离线边界验证"}</small></span></button>)}
              {modelEvaluations.length === 0 && <div className="evaluation-empty">尚无模型安全评测记录</div>}
            </div>
          </div>
          <div className="model-evaluation-report">
            {selectedModelReport ? <>
              <div className="evaluation-metrics model-evaluation-metrics">
                <div><span>通过用例</span><strong className="mono">{selectedModelReport.passedCaseCount}/{selectedModelReport.coverage.caseCount}</strong></div>
                <div><span>关键失败</span><strong className="mono">{selectedModelReport.safetyGate.criticalFailureCount}</strong></div>
                <div><span>DeepSeek 实测</span><strong className="mono">{selectedModelReport.liveProviderCaseCount}</strong></div>
                <div><span>降级用例</span><strong className="mono">{selectedModelReport.fallbackCaseCount}</strong></div>
              </div>
              <div className="coverage-line"><span>测试集</span><strong className="mono">{selectedModelReport.suiteId} · SHA-256 {selectedModelReport.suiteSha256.slice(0, 16)}...</strong><Badge appearance="outline">{selectedModelReport.provider.model}</Badge></div>
              <div className="table-scroll model-evaluation-table"><Table size="small" aria-label="大模型安全评测明细"><TableHeader><TableRow><TableHeaderCell>用例</TableHeaderCell><TableHeaderCell>类别</TableHeaderCell><TableHeaderCell>执行方式</TableHeaderCell><TableHeaderCell>结果</TableHeaderCell><TableHeaderCell>耗时</TableHeaderCell></TableRow></TableHeader><TableBody>{selectedModelReport.cases.map((row) => <TableRow key={row.caseId}><TableCell><strong className="mono">{row.caseId}</strong><small>{row.title}</small></TableCell><TableCell>{row.category}</TableCell><TableCell>{executionLabels[row.executionMode] || row.executionMode}</TableCell><TableCell><Badge appearance="tint" color={row.passed ? "success" : "danger"}>{row.passed ? "通过" : "失败"}</Badge></TableCell><TableCell className="mono">{formatNumber(row.latencyMs, 2)} ms</TableCell></TableRow>)}</TableBody></Table></div>
              {!selectedModelReport.liveProviderEvaluated && <p className="model-evaluation-warning">本报告未包含成功的 DeepSeek API 返回，不能表述为真实模型效果评测；安全边界与离线降级结果仍可重复核验。</p>}
            </> : <div className="evaluation-empty">运行模型安全评测后显示逐项证据。</div>}
          </div>
        </div>
      </section>

      <section className="data-panel dataset-export">
        <div className="panel-heading"><h2>脱敏评测数据</h2><p>汇集仿真、意图、审批、异常与审计记录，固定划分训练/验证/测试集</p></div>
        <div className="dataset-layout">
          <div className="evaluation-form">
            <Field label="数据集名称"><Input value={datasetName} onChange={(_, data) => setDatasetName(data.value)} /></Field>
            <Checkbox label="包含审计记录" checked={includeAudit} onChange={(_, data) => setIncludeAudit(data.checked === true)} />
            <Checkbox label="包含异常诊断记录" checked={includeIncidents} onChange={(_, data) => setIncludeIncidents(data.checked === true)} />
            <Checkbox label="保留证据自由文本" checked={includeEvidenceText} onChange={(_, data) => setIncludeEvidenceText(data.checked === true)} />
            {includeEvidenceText && <p className="field-warning">保留自由文本会扩大人工合规复核范围，比赛导出建议关闭。</p>}
            <Button appearance="primary" icon={<Database20Regular />} disabled={!datasetName.trim() || Boolean(busy)} onClick={() => void createDataset()}>{busy === "dataset" ? "正在脱敏与质检" : "生成数据资产包"}</Button>
          </div>
          <div className="dataset-results">
            {selectedExport ? <>
              <div className="dataset-result-heading"><div><strong>{selectedExport.name}</strong><span className="mono">{selectedExport.exportId}</span></div><Badge appearance="tint" color={selectedExport.quality.passed ? "success" : "danger"}>{selectedExport.quality.passed ? "质量检查通过" : "质量检查未通过"}</Badge></div>
              <div className="dataset-stats"><div><span>总记录</span><strong className="mono">{selectedExport.recordCount}</strong></div><div><span>训练集</span><strong className="mono">{selectedExport.quality.splitCounts.train || 0}</strong></div><div><span>验证集</span><strong className="mono">{selectedExport.quality.splitCounts.validation || 0}</strong></div><div><span>测试集</span><strong className="mono">{selectedExport.quality.splitCounts.test || 0}</strong></div></div>
              <div className="quality-checks">{Object.entries(selectedExport.quality.checks).map(([name, passed]) => <span key={name}><Badge appearance="tint" color={passed ? "success" : "danger"}>{passed ? "通过" : "失败"}</Badge>{name}</span>)}</div>
              <Button appearance="secondary" icon={<ArrowDownload20Regular />} onClick={() => downloadDataset(selectedExport.exportId)}>下载 ZIP 资产包</Button>
            </> : <div className="evaluation-empty">尚未生成数据资产包</div>}
            {exports.length > 0 && <Select aria-label="历史数据资产" value={selectedExport?.exportId || ""} onChange={(_, data) => setSelectedExport(exports.find((item) => item.exportId === data.value) || null)}>{exports.map((item) => <option key={item.exportId} value={item.exportId}>{item.name} · {item.recordCount} 条</option>)}</Select>}
          </div>
        </div>
      </section>
    </div>
  );
}
