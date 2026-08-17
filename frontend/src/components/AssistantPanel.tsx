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
  DocumentSearch20Regular,
  Play20Regular,
  Send20Regular,
  ShieldLock20Regular,
} from "@fluentui/react-icons";
import type { Approval, ChatResponse, DispatchIntent, SimulationSummary } from "../types";

interface AssistantPanelProps {
  response?: ChatResponse | null;
  run?: SimulationSummary | null;
  approval?: Approval | null;
  busy?: string | null;
  onSend: (message: string) => Promise<void>;
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

export function AssistantPanel({
  response,
  run,
  approval,
  busy,
  onSend,
  onSimulate,
  onCreateApproval,
  onCommit,
}: AssistantPanelProps) {
  const [message, setMessage] = useState("");

  const submit = async (value = message) => {
    const normalized = value.trim();
    if (!normalized || busy) return;
    setMessage("");
    await onSend(normalized);
  };

  const intent = response?.intent;
  const validation = response?.validation;
  const actionable = intent && ["CREATE_TASK", "BLOCK_RESOURCE"].includes(intent.intentType);
  const approved = approval?.status === "APPROVED";

  return (
    <aside className="assistant-panel" aria-label="AI 调度助手">
      <div className="assistant-title">
        <span className="assistant-icon"><Bot24Regular /></span>
        <div>
          <h2>灵枢 AI 调度助手</h2>
          <p>{response ? `${response.model}${response.fallbackUsed ? "，本地降级" : ""}` : "等待调度指令"}</p>
        </div>
      </div>

      <div className="assistant-conversation">
        {!response && (
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

        {response && (
          <div className="assistant-response">
            <p className="response-copy">{response.message}</p>
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

            {response.evidence.length > 0 && (
              <Accordion collapsible className="evidence-accordion">
                <AccordionItem value="evidence">
                  <AccordionHeader icon={<DocumentSearch20Regular />}>查看依据与引用</AccordionHeader>
                  <AccordionPanel>
                    {response.evidence.map((item) => (
                      <div className="evidence-item" key={`${item.source}-${item.title}`}>
                        <strong>{item.title}</strong>
                        <span className="mono">{item.source}</span>
                        <p>{item.detail}</p>
                      </div>
                    ))}
                  </AccordionPanel>
                </AccordionItem>
              </Accordion>
            )}

            {actionable && validation?.valid && (
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
          disabled={!message.trim() || Boolean(busy)}
          onClick={() => void submit()}
        />
      </div>
      <div className="safety-notice"><ShieldLock20Regular />大模型无权生成路径、预约资源或控制真实车辆</div>
    </aside>
  );
}
