import { useEffect, useMemo, useState } from "react";
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Badge,
  Button,
  Field,
  Select,
  Textarea,
} from "@fluentui/react-components";
import {
  BrainCircuit20Regular,
  DocumentSearch20Regular,
} from "@fluentui/react-icons";
import type { PlanExplanationReport, RunDetail } from "../types";

interface PlanExplanationPanelProps {
  run?: RunDetail | null;
  report?: PlanExplanationReport | null;
  busy?: boolean;
  onExplain: (options: { question: string; vehicleId?: string; taskId?: string }) => void;
}

const categoryLabels: Record<string, string> = {
  RUN: "运行",
  ASSIGNMENT: "分配",
  WAIT: "等待",
  ROUTE: "路线",
  SAFETY: "安全",
  FALLBACK: "回退",
};

export function PlanExplanationPanel({ run, report, busy, onExplain }: PlanExplanationPanelProps) {
  const [question, setQuestion] = useState("为什么这样分配车辆，等待和绕行是如何产生的？");
  const [vehicleId, setVehicleId] = useState("");
  const [taskId, setTaskId] = useState("");
  const vehicles = useMemo(() => run?.scenario.vehicles.map((row) => row.vehicleId) || [], [run]);
  const tasks = useMemo(() => [...new Set(run?.scenario.plans.map((row) => row.taskId) || [])], [run]);
  const evidenceById = useMemo(
    () => new Map(report?.evidence.map((row) => [row.evidenceId, row]) || []),
    [report],
  );

  useEffect(() => {
    setVehicleId("");
    setTaskId("");
  }, [run?.summary.runId]);

  return (
    <section className="data-panel plan-explanation-panel">
      <div className="panel-heading panel-heading-actions">
        <div><h2>规划证据解释</h2><p>{run ? `${run.summary.label} · ${run.summary.runId}` : "选择一个已完成运行"}</p></div>
        {report && <Badge appearance="tint" color={report.fallbackUsed ? "informative" : "success"}>{report.fallbackUsed ? "确定性解释" : "DeepSeek 证据解释"}</Badge>}
      </div>
      <div className="plan-explanation-controls">
        <Field label="车辆"><Select value={vehicleId} disabled={!run} onChange={(_, data) => setVehicleId(data.value)}><option value="">全部车辆</option>{vehicles.map((id) => <option key={id} value={id}>{id}</option>)}</Select></Field>
        <Field label="任务"><Select value={taskId} disabled={!run} onChange={(_, data) => setTaskId(data.value)}><option value="">全部任务</option>{tasks.map((id) => <option key={id} value={id}>{id}</option>)}</Select></Field>
        <Field label="解释问题"><Textarea resize="vertical" value={question} disabled={!run} onChange={(_, data) => setQuestion(data.value)} /></Field>
        <Button appearance="primary" icon={<BrainCircuit20Regular />} disabled={!run || !question.trim() || busy} onClick={() => onExplain({ question: question.trim(), vehicleId: vehicleId || undefined, taskId: taskId || undefined })}>{busy ? "正在核对证据" : "解释当前计划"}</Button>
      </div>
      {report ? <div className="plan-explanation-result">
        <p className="plan-explanation-summary">{report.summary}</p>
        <div className="plan-findings">{report.findings.map((finding) => <article key={`${finding.code}-${finding.title}`}><div><Badge appearance="tint" color={finding.classification === "FACT" ? "success" : "warning"}>{finding.classification === "FACT" ? "事实" : "推断"}</Badge><strong>{finding.title}</strong></div><p>{finding.explanation}</p><div className="evidence-chip-row">{finding.evidenceIds.map((id) => <span key={id} title={evidenceById.get(id)?.fact}>{id}</span>)}</div></article>)}</div>
        {report.uncertainties.length > 0 && <div className="plan-uncertainties"><strong>证据边界</strong>{report.uncertainties.map((item) => <p key={item}>{item}</p>)}</div>}
        <Accordion collapsible><AccordionItem value="plan-evidence"><AccordionHeader icon={<DocumentSearch20Regular />}>查看 {report.evidence.length} 条原始证据</AccordionHeader><AccordionPanel><div className="plan-evidence-list">{report.evidence.map((item) => <article key={item.evidenceId}><div><code>{item.evidenceId}</code><Badge appearance="outline">{categoryLabels[item.category] || item.category}</Badge></div><p>{item.fact}</p><small className="mono">{item.source}</small></article>)}</div></AccordionPanel></AccordionItem></Accordion>
      </div> : <div className="plan-explanation-empty">解释只读取当前运行的规划摘要、计划段和安全结果，不修改调度状态。</div>}
    </section>
  );
}
