import { Badge, Button, Card, CardHeader } from "@fluentui/react-components";
import {
  Checkmark20Regular,
  Dismiss20Regular,
  ShieldLock24Regular,
} from "@fluentui/react-icons";
import type { Approval } from "../types";

interface ApprovalsPanelProps {
  approvals: Approval[];
  busyId?: string | null;
  onDecision: (approvalId: string, approved: boolean) => Promise<void>;
}

const statusText: Record<string, string> = {
  PENDING: "待审批",
  APPROVED: "已批准",
  REJECTED: "已驳回",
  EXPIRED: "已过期",
};

export function ApprovalsPanel({ approvals, busyId, onDecision }: ApprovalsPanelProps) {
  return (
    <section className="data-panel approvals-panel">
      <div className="panel-heading">
        <h2>高风险审批</h2>
        <p>R3 操作必须基于有效世界版本和已完成仿真，由主管人工决策</p>
      </div>
      <div className="approval-list">
        {approvals.map((approval) => (
          <Card key={approval.approvalId} className="approval-card">
            <CardHeader
              image={<span className="approval-icon"><ShieldLock24Regular /></span>}
              header={
                <div className="approval-card-title">
                  <strong>{approval.intent.intentType === "BLOCK_RESOURCE" ? "共享资源封锁" : approval.intent.intentType}</strong>
                  <Badge
                    appearance="tint"
                    color={
                      approval.status === "APPROVED"
                        ? "success"
                        : approval.status === "PENDING"
                          ? "warning"
                          : "danger"
                    }
                  >
                    {statusText[approval.status] || approval.status}
                  </Badge>
                </div>
              }
              description={`${approval.requestedBy} | ${new Date(approval.createdAt).toLocaleString("zh-CN", { hour12: false })}`}
            />
            <div className="approval-details">
              <div><span>风险策略</span><strong className="mono">{approval.validation.policyCode}</strong></div>
              <div><span>意图编号</span><strong className="mono">{approval.intent.intentId}</strong></div>
              <div><span>关联仿真</span><strong className="mono">{approval.simulationRunIds.join(", ") || "未关联"}</strong></div>
              <div><span>影响资源</span><strong className="mono">{approval.intent.resourceBlock?.resourceIds.join(", ") || "-"}</strong></div>
              <div><span>有效时窗</span><strong>{approval.intent.resourceBlock ? `${approval.intent.resourceBlock.startMs / 1000}s - ${approval.intent.resourceBlock.endMs / 1000}s` : "-"}</strong></div>
              <div><span>申请理由</span><strong>{approval.intent.reason}</strong></div>
            </div>
            {approval.decisionReason && (
              <div className="decision-note">决策说明：{approval.decisionReason}</div>
            )}
            {approval.status === "PENDING" && (
              <div className="approval-actions">
                <Button
                  appearance="primary"
                  icon={<Checkmark20Regular />}
                  disabled={Boolean(busyId)}
                  onClick={() => void onDecision(approval.approvalId, true)}
                >
                  {busyId === approval.approvalId ? "处理中" : "批准"}
                </Button>
                <Button
                  appearance="secondary"
                  icon={<Dismiss20Regular />}
                  disabled={Boolean(busyId)}
                  onClick={() => void onDecision(approval.approvalId, false)}
                >
                  驳回
                </Button>
              </div>
            )}
          </Card>
        ))}
        {approvals.length === 0 && (
          <div className="empty-state">当前没有审批单。通道封锁等高风险意图完成仿真后会进入这里。</div>
        )}
      </div>
    </section>
  );
}
