import { useState } from "react";
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Badge,
  Button,
  Textarea,
} from "@fluentui/react-components";
import {
  Bot24Regular,
  CheckmarkCircle20Regular,
  DismissCircle20Regular,
  DocumentSearch20Regular,
  Play20Regular,
  Send20Regular,
  ShieldLock20Regular,
} from "@fluentui/react-icons";
import type {
  AgentGoalWorkflow,
  AgentRunRecord,
  Approval,
  ChatResponse,
  DispatchIntent,
  SimulationSummary,
} from "../types";

interface AssistantPanelProps {
  response?: ChatResponse | null;
  agentRun?: AgentRunRecord | null;
  run?: SimulationSummary | null;
  approval?: Approval | null;
  busy?: string | null;
  onSend: (message: string) => Promise<void>;
  onAgentApproval: (approved: boolean) => Promise<void>;
  onAgentCancel: () => Promise<void>;
  onSimulate: (intent: DispatchIntent) => Promise<void>;
  onCreateApproval: (intent: DispatchIntent, runId: string) => Promise<void>;
  onCommit: (intent: DispatchIntent, approvalId?: string) => Promise<void>;
}

const examples = [
  "创建紧急叉车任务，从 AP1123 运到 AP2121",
  "共享窄路需要检修，请封闭三分钟并评估影响",
  "当前车辆和任务状态怎么样？",
];

const riskLabel: Record<string, string> = {
  R0_READ_ONLY: "只读",
  R1_LOW: "低风险",
  R2_MEDIUM: "中风险",
  R3_HIGH: "高风险",
  R4_FORBIDDEN: "禁止",
};

const runStatusLabel: Record<string, string> = {
  QUEUED: "等待执行",
  RUNNING: "实时执行",
  WAITING_APPROVAL: "等待主管确认",
  COMPLETED: "运行完成",
  REJECTED: "主管拒绝",
  CANCELLED: "已取消",
  TIMED_OUT: "执行超时",
  FAILED: "运行失败",
};

const workflowPhaseLabel: Record<string, string> = {
  PENDING: "准备执行",
  NOT_APPLICABLE: "无需执行",
  SIMULATING: "数字孪生运行中",
  WAITING_APPROVAL: "等待审批",
  COMMITTING: "提交仿真环境",
  COMPLETED: "目标已完成",
  BLOCKED: "安全门槛阻断",
};

function WorkflowProgress({ workflow }: { workflow: AgentGoalWorkflow }) {
  const simulation = workflow.simulation;
  const recommendation = workflow.recommendation;
  return (
    <section className="agent-workflow" aria-label="Agent 目标执行">
      <div className="agent-workflow-heading">
        <strong>目标执行</strong>
        <Badge
          appearance="outline"
          color={
            workflow.phase === "COMPLETED"
              ? "success"
              : workflow.phase === "BLOCKED"
                ? "danger"
                : workflow.phase === "WAITING_APPROVAL"
                  ? "warning"
                  : "informative"
          }
        >
          {workflowPhaseLabel[workflow.phase]}
        </Badge>
      </div>

      {workflow.steps.length > 0 && (
        <ol className="agent-workflow-steps">
          {workflow.steps.map((step) => (
            <li key={`${step.sequence}-${step.action}`}>
              <span className={`workflow-step-mark workflow-step-${step.status.toLowerCase()}`}>
                {step.sequence}
              </span>
              <div>
                <strong>{step.title}</strong>
                <p>{step.detail}</p>
                <small className="mono">
                  {step.action} · {step.status} · {step.durationMs.toFixed(1)} ms
                </small>
              </div>
            </li>
          ))}
        </ol>
      )}

      {simulation && (
        <div className="workflow-simulation-summary">
          <div><span>仿真</span><strong className="mono">{simulation.runId}</strong></div>
          <div><span>完成任务</span><strong>{String(simulation.metrics.completedTaskCount ?? 0)}</strong></div>
          <div><span>资源冲突</span><strong>{String(simulation.metrics.reservationConflictRejections ?? 0)}</strong></div>
        </div>
      )}

      {recommendation && (
        <div className={`workflow-recommendation workflow-${recommendation.decision.toLowerCase()}`}>
          <strong>{recommendation.decision === "PROCEED" ? "建议推进" : "禁止推进"}</strong>
          <span>{recommendation.reasons.join("；")}</span>
        </div>
      )}

      {workflow.commitment && (
        <div className="workflow-commitment">
          <CheckmarkCircle20Regular />
          <span>已提交到仿真环境</span>
          <code>{String(workflow.commitment.commitId || "-")}</code>
        </div>
      )}
    </section>
  );
}

export function AssistantPanel({
  response,
  agentRun,
  run,
  approval,
  busy,
  onSend,
  onAgentApproval,
  onAgentCancel,
  onSimulate,
  onCreateApproval,
  onCommit,
}: AssistantPanelProps) {
  const [message, setMessage] = useState("");
  const agentActive = Boolean(
    agentRun && ["QUEUED", "RUNNING", "WAITING_APPROVAL"].includes(agentRun.status),
  );

  const submit = async (value = message) => {
    const normalized = value.trim();
    if (!normalized || busy || agentActive) return;
    setMessage("");
    await onSend(normalized);
  };

  const intent = response?.intent;
  const validation = response?.validation;
  const actionable = intent && ["CREATE_TASK", "BLOCK_RESOURCE"].includes(intent.intentType);
  const approved = approval?.status === "APPROVED";
  const goalManaged = agentRun?.request.executionMode === "GOAL_EXECUTION";

  return (
    <aside className="assistant-panel" aria-label="AI 调度助手">
      <div className="assistant-title">
        <span className="assistant-icon"><Bot24Regular /></span>
        <div>
          <h2>灵枢 AI 调度助手</h2>
          <p>
            {response
              ? `${response.model}${response.fallbackUsed ? "，本地降级" : ""}`
              : agentRun
                ? `${runStatusLabel[agentRun.status]} · SSE 实时轨迹`
                : "等待调度指令"}
          </p>
        </div>
      </div>

      <div className="assistant-conversation">
        {!response && !agentRun && (
          <div className="assistant-empty">
            <p>用自然语言描述任务或异常，系统会先生成结构化意图，再交给 MASP 校验与仿真。</p>
            <div className="example-list">
              {examples.map((example) => (
                <Button key={example} appearance="subtle" onClick={() => void submit(example)}>
                  {example}
                </Button>
              ))}
            </div>
          </div>
        )}

        {!response && agentRun && (
          <div className="assistant-response agent-runtime-live">
            <div className="agent-runtime-bar">
              <Badge
                appearance="filled"
                color={
                  agentRun.status === "WAITING_APPROVAL"
                    ? "warning"
                    : ["FAILED", "TIMED_OUT", "REJECTED"].includes(agentRun.status)
                      ? "danger"
                      : agentRun.status === "COMPLETED"
                        ? "success"
                        : "informative"
                }
              >
                {runStatusLabel[agentRun.status]}
              </Badge>
              <code>{agentRun.runId}</code>
              <span>第 {agentRun.attempt} 次执行</span>
            </div>
            <p className="response-copy">
              {agentRun.status === "WAITING_APPROVAL"
                ? agentRun.approval?.stage === "POST_SIMULATION"
                  ? "数字孪生和确定性推进门槛均已完成。高风险方案在仿真后暂停，批准后只提交到仿真环境。"
                  : "确定性安全校验已完成。高风险草案在检查点暂停，批准后才会完成本轮 Agent 决策。"
                : agentRun.error || "Agent 正在按有界状态机执行，步骤会实时写入并推送到这里。"}
            </p>

            {agentRun.workflow && <WorkflowProgress workflow={agentRun.workflow} />}

            {agentRun.traceSteps.length > 0 && (
              <ol className="agent-trace-list agent-trace-live-list" aria-label="实时 Agent 轨迹">
                {agentRun.traceSteps.map((step) => (
                  <li key={step.stepId} className={`agent-trace-step agent-trace-${step.status.toLowerCase()}`}>
                    <span className="agent-step-index mono">{step.sequence}</span>
                    <div>
                      <div className="agent-step-heading">
                        <strong>{step.title}</strong>
                        {step.toolName && <code>{step.toolName}</code>}
                        {step.readOnly && <span>只读</span>}
                      </div>
                      <p>{step.detail}</p>
                      <small className="mono">{step.state} · {step.durationMs.toFixed(1)} ms</small>
                    </div>
                  </li>
                ))}
              </ol>
            )}

            {agentRun.approval && agentRun.status === "WAITING_APPROVAL" && (
              <div className="agent-approval-gate">
                <div>
                  <strong>人工审批检查点</strong>
                  <span>
                    {riskLabel[agentRun.approval.validation.riskLevel]} · {agentRun.approval.intent.intentType}
                    {agentRun.approval.stage === "POST_SIMULATION" ? " · 仿真后" : ""}
                  </span>
                </div>
                <div className="assistant-actions">
                  <Button
                    appearance="primary"
                    icon={<CheckmarkCircle20Regular />}
                    disabled={Boolean(busy)}
                    onClick={() => void onAgentApproval(true)}
                  >
                    批准并继续
                  </Button>
                  <Button
                    appearance="secondary"
                    icon={<DismissCircle20Regular />}
                    disabled={Boolean(busy)}
                    onClick={() => void onAgentApproval(false)}
                  >
                    拒绝草案
                  </Button>
                </div>
              </div>
            )}

            {agentRun.status === "RUNNING" && (
              <div className="assistant-actions">
                <Button
                  appearance="subtle"
                  icon={<DismissCircle20Regular />}
                  onClick={() => void onAgentCancel()}
                >
                  取消运行
                </Button>
              </div>
            )}
          </div>
        )}

        {response && (
          <div className="assistant-response">
            <p className="response-copy">{response.message}</p>
            {agentRun?.evaluation && (
              <div className="agent-runtime-bar">
                <Badge
                  appearance="outline"
                  color={agentRun.evaluation.passed ? "success" : "danger"}
                >
                  轨迹评测 {Math.round(agentRun.evaluation.score * 100)}%
                </Badge>
                <span className="mono">
                  Token {Number(agentRun.providerUsage.totalTokens || 0).toLocaleString("zh-CN")}
                </span>
                <span className="mono">
                  ${Number(agentRun.providerUsage.estimatedCostUsd || 0).toFixed(6)}
                </span>
              </div>
            )}
            {agentRun?.workflow && <WorkflowProgress workflow={agentRun.workflow} />}
            {response.clarification && (
              <div className="clarification-box" role="status">
                <strong>需要补充信息</strong>
                <span>已保留：{Object.entries(response.clarification.collectedParameters).map(([key, value]) => `${key}=${String(value)}`).join("，") || "暂无"}</span>
                {response.clarification.questions.map((question) => <p key={question}>{question}</p>)}
              </div>
            )}
            {validation && (
              <div className="validation-row">
                <Badge
                  appearance="filled"
                  color={validation.valid ? "success" : "danger"}
                  icon={<CheckmarkCircle20Regular />}
                >
                  {validation.valid ? "确定性校验通过" : "校验未通过"}
                </Badge>
                <Badge appearance="outline" color={validation.approvalRequired ? "danger" : "informative"}>
                  {riskLabel[validation.riskLevel] || validation.riskLevel}
                </Badge>
              </div>
            )}

            {intent && (
              <div className="intent-sheet">
                <div><span>意图</span><strong className="mono">{intent.intentType}</strong></div>
                {intent.task && (
                  <>
                    <div><span>起点</span><strong className="mono">{intent.task.pickupNodeId}</strong></div>
                    <div><span>终点</span><strong className="mono">{intent.task.dropoffNodeId}</strong></div>
                    <div><span>车型 / 优先级</span><strong>{intent.task.requiredRobotGroup} / P{intent.task.priorityClass}</strong></div>
                  </>
                )}
                {intent.resourceBlock && (
                  <>
                    <div><span>资源</span><strong className="mono">{intent.resourceBlock.resourceIds.join(", ")}</strong></div>
                    <div><span>时窗</span><strong>{intent.resourceBlock.startMs / 1000}s - {intent.resourceBlock.endMs / 1000}s</strong></div>
                  </>
                )}
              </div>
            )}

            {response.agentTrace && (
              <Accordion collapsible className="agent-trace-accordion">
                <AccordionItem value="agent-trace">
                  <AccordionHeader icon={<Bot24Regular />}>
                    Agent 执行轨迹 · {response.agentTrace.steps.length} 步
                  </AccordionHeader>
                  <AccordionPanel>
                    <div className="agent-trace-summary">
                      <Badge
                        appearance="outline"
                        color={response.agentTrace.strategy === "ACTION_PROTOCOL_LOOP" ? "success" : response.agentTrace.strategy === "MODEL_TOOL_CALLING" ? "informative" : "warning"}
                      >
                        {response.agentTrace.strategy === "ACTION_PROTOCOL_LOOP" ? "单动作闭环" : response.agentTrace.strategy === "MODEL_TOOL_CALLING" ? "模型工具规划" : "确定性工具策略"}
                      </Badge>
                      <span className="mono">{response.agentTrace.plannerModel}</span>
                      <span>{response.agentTrace.durationMs.toFixed(1)} ms</span>
                      {typeof response.agentTrace.usage.decisions === "number" && <span>决策 {response.agentTrace.usage.decisions}</span>}
                      {typeof response.agentTrace.usage.toolCalls === "number" && <span>工具 {response.agentTrace.usage.toolCalls}</span>}
                      {typeof response.agentTrace.usage.totalTokens === "number" && <span className="mono">Token {response.agentTrace.usage.totalTokens.toLocaleString("zh-CN")}</span>}
                      {typeof response.agentTrace.usage.estimatedCostUsd === "number" && response.agentTrace.usage.estimatedCostUsd > 0 && <span className="mono">${response.agentTrace.usage.estimatedCostUsd.toFixed(4)}</span>}
                      {typeof response.agentTrace.usage.repairAttempts === "number" && response.agentTrace.usage.repairAttempts > 0 && <span>修复 {response.agentTrace.usage.repairAttempts}</span>}
                      {response.agentTrace.terminalReason && <code>{response.agentTrace.terminalReason}</code>}
                    </div>
                    <ol className="agent-trace-list">
                      {response.agentTrace.steps.map((step) => (
                        <li key={step.stepId} className={`agent-trace-step agent-trace-${step.status.toLowerCase()}`}>
                          <span className="agent-step-index mono">{step.sequence}</span>
                          <div>
                            <div className="agent-step-heading">
                              <strong>{step.title}</strong>
                              {step.toolName && <code>{step.toolName}</code>}
                              {step.action && <code>{step.action}</code>}
                              {step.readOnly && <span>只读</span>}
                            </div>
                            <p>{step.detail}</p>
                            <small className="mono">{step.state}{step.observationCode ? ` · ${step.observationCode}` : ""} · {step.durationMs.toFixed(1)} ms</small>
                          </div>
                        </li>
                      ))}
                    </ol>
                  </AccordionPanel>
                </AccordionItem>
              </Accordion>
            )}

            {response.evidence.length > 0 && (
              <Accordion collapsible className="evidence-accordion">
                <AccordionItem value="evidence">
                  <AccordionHeader icon={<DocumentSearch20Regular />}>查看依据与引用</AccordionHeader>
                  <AccordionPanel>
                    {response.evidence.map((item) => (
                      <div className="evidence-item" key={`${item.source}-${item.title}`}>
                        <strong>{item.title}</strong>
                        <div className="evidence-meta">
                          <span className="mono">{item.source}</span>
                          {item.chunkId && <span className="mono">{item.chunkId}</span>}
                          {typeof item.score === "number" && <span>相关度 {(item.score * 100).toFixed(1)}%</span>}
                          {item.retrievalMethod && <span className="mono">{item.retrievalMethod}</span>}
                        </div>
                        <p>{item.detail}</p>
                      </div>
                    ))}
                  </AccordionPanel>
                </AccordionItem>
              </Accordion>
            )}

            {actionable && validation?.valid && !goalManaged && (
              <div className="assistant-actions">
                <Button
                  appearance="primary"
                  icon={<Play20Regular />}
                  disabled={Boolean(busy)}
                  onClick={() => void onSimulate(intent)}
                >
                  {busy === "simulate" ? "仿真运行中" : run ? "重新运行仿真" : "运行数字孪生"}
                </Button>
                {run && validation.approvalRequired && !approval && (
                  <Button
                    appearance="secondary"
                    icon={<ShieldLock20Regular />}
                    disabled={Boolean(busy)}
                    onClick={() => void onCreateApproval(intent, run.runId)}
                  >
                    提交主管审批
                  </Button>
                )}
                {run && (!validation.approvalRequired || approved) && (
                  <Button
                    appearance="secondary"
                    icon={<CheckmarkCircle20Regular />}
                    disabled={Boolean(busy)}
                    onClick={() => void onCommit(intent, approval?.approvalId)}
                  >
                    提交仿真环境
                  </Button>
                )}
                {approval && validation.approvalRequired && !approved && (
                  <span className="action-note">审批状态：{approval.status}</span>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="assistant-composer">
        <Textarea
          resize="vertical"
          value={message}
          placeholder="描述任务、异常或需要评估的调度方案"
          aria-label="调度请求"
          onChange={(_, data) => setMessage(data.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void submit();
            }
          }}
        />
        <Button
          appearance="primary"
          icon={<Send20Regular />}
          aria-label="发送调度请求"
          disabled={!message.trim() || Boolean(busy) || agentActive}
          onClick={() => void submit()}
        />
      </div>
      <div className="safety-notice"><ShieldLock20Regular />大模型无权生成路径、预约资源或控制真实车辆</div>
    </aside>
  );
}
