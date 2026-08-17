import { useState } from "react";
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Badge,
  Button,
  Select,
  Tab,
  TabList,
} from "@fluentui/react-components";
import {
  Bot20Regular,
  Play20Regular,
  ShieldLock20Regular,
} from "@fluentui/react-icons";
import type { AgentModelStatus, RunDetail } from "../types";

type PolicyMode = "top_k" | "rl";

interface AgentPolicyPanelProps {
  status: AgentModelStatus;
  run?: RunDetail | null;
  busy?: boolean;
  onRun: (policy: PolicyMode, candidateCount: number) => void;
}

const strategyLabel: Record<string, string> = {
  rl: "PPO 智能体",
  congestion_guardian: "拥堵 guardian",
  congestion_fallback: "规则接管",
  congestion: "拥堵优先",
  task_age: "任务等待",
  shortest_remaining: "最短剩余",
  previous_order: "沿用顺序",
  random: "确定性扰动",
};

export function AgentPolicyPanel({ status, run, busy, onRun }: AgentPolicyPanelProps) {
  const [mode, setMode] = useState<PolicyMode>("top_k");
  const [candidateCount, setCandidateCount] = useState(2);
  const execution = run?.summary.agentPolicy;
  const cycles = run?.agentEvidence?.decisionCycles.filter(
    (cycle) => cycle.candidateCount > 0,
  ) || [];

  return (
    <section className="data-panel agent-policy-panel">
      <div className="panel-heading agent-policy-heading">
        <div>
          <h2>群车智能体策略</h2>
          <p className="mono">{status.modelId} / {status.modelVersion}</p>
        </div>
        <Badge
          appearance="tint"
          color={status.checkpointPresent ? "success" : "warning"}
        >
          {status.checkpointPresent ? "权重已登记" : "规则基线"}
        </Badge>
      </div>

      <div className="agent-policy-body">
        <div className="agent-policy-controls">
          <TabList
            selectedValue={mode}
            onTabSelect={(_, data) => setMode(data.value as PolicyMode)}
            aria-label="选择群车调度策略"
          >
            <Tab value="top_k" icon={<ShieldLock20Regular />}>Top-K 规则</Tab>
            <Tab value="rl" icon={<Bot20Regular />}>智能体协同</Tab>
          </TabList>
          <div className="agent-policy-runbar">
            <label>
              <span>候选数</span>
              <Select
                aria-label="智能体候选数"
                value={String(candidateCount)}
                disabled={mode !== "rl" || busy}
                onChange={(_, data) => setCandidateCount(Number(data.value))}
              >
                {[1, 2, 3, 4].map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </Select>
            </label>
            <Button
              appearance="primary"
              icon={<Play20Regular />}
              disabled={busy}
              onClick={() => onRun(mode, candidateCount)}
            >
              {busy ? "运行中" : "运行策略"}
            </Button>
          </div>
          <p className="agent-model-notice">{status.notice}</p>
          <dl className="agent-model-facts">
            <div><dt>推理设备</dt><dd className="mono">{status.device}</dd></div>
            <div><dt>安全控制</dt><dd>{status.safetyController}</dd></div>
          </dl>
        </div>

        <div className="agent-policy-evidence">
          <div className="agent-evidence-title">
            <div>
              <h3>选中方案证据</h3>
              <p>{execution ? run?.summary.label : "当前方案未使用智能体策略"}</p>
            </div>
            {execution && (
              <Badge appearance="tint" color={execution.mode === "LEARNED" ? "success" : "warning"}>
                {execution.mode === "LEARNED" ? "模型推理" : "安全降级"}
              </Badge>
            )}
          </div>

          {execution ? (
            <>
              <div className="agent-evidence-metrics">
                <div><span>推理</span><strong className="mono">{execution.inferenceCount}</strong></div>
                <div><span>智能体候选</span><strong className="mono">{execution.agentCandidateCount}</strong></div>
                <div><span>采用</span><strong className="mono">{execution.selectedAgentCandidateCount}</strong></div>
                <div><span>规则接管</span><strong className="mono">{execution.fallbackCount + execution.safetyFallbackCount + execution.guardianOverrideCount}</strong></div>
              </div>
              {[...execution.fallbackReasons, ...execution.notes].length > 0 && (
                <div className="agent-evidence-notes">
                  {[...execution.fallbackReasons, ...execution.notes].map((note) => (
                    <p key={note}>{note}</p>
                  ))}
                </div>
              )}
              {cycles.length > 0 && (
                <Accordion collapsible className="agent-cycle-list">
                  {cycles.slice(0, 10).map((cycle) => (
                    <AccordionItem key={cycle.cycleIndex} value={String(cycle.cycleIndex)}>
                      <AccordionHeader>
                        <span className="agent-cycle-heading">
                          <strong className="mono">T+{(cycle.decisionTimeMs / 1000).toFixed(1)}s</strong>
                          <span>{cycle.feasibleCandidateCount}/{cycle.candidateCount} 可行</span>
                        </span>
                      </AccordionHeader>
                      <AccordionPanel>
                        <div className="agent-candidate-list">
                          {cycle.candidates.map((candidate) => (
                            <div key={candidate.candidateId}>
                              <Badge
                                appearance="tint"
                                color={cycle.selectedCandidateIds.includes(candidate.candidateId) ? "success" : candidate.feasible ? "informative" : "danger"}
                              >
                                {cycle.selectedCandidateIds.includes(candidate.candidateId) ? "已采用" : candidate.feasible ? "可行" : "拒绝"}
                              </Badge>
                              <strong>{strategyLabel[candidate.strategy] || candidate.strategy}</strong>
                              <span className="mono">{candidate.plannedTaskCount} 任务</span>
                              {candidate.failureCode && <small className="mono">{candidate.failureCode}</small>}
                            </div>
                          ))}
                        </div>
                      </AccordionPanel>
                    </AccordionItem>
                  ))}
                </Accordion>
              )}
            </>
          ) : (
            <div className="empty-state">选择一条智能体协同运行后显示模型与候选证据。</div>
          )}
        </div>
      </div>
    </section>
  );
}
