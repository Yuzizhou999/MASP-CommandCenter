import {
  Badge,
  Button,
  Checkbox,
  Table,
  TableBody,
  TableCell,
  TableCellLayout,
  TableHeader,
  TableHeaderCell,
  TableRow,
} from "@fluentui/react-components";
import {
  BranchCompare20Regular,
  PlayCircle20Regular,
  Star20Filled,
} from "@fluentui/react-icons";
import type { Comparison, SimulationSummary } from "../types";

interface SimulationTableProps {
  runs: SimulationSummary[];
  selectedRunId?: string | null;
  checkedRunIds: string[];
  comparison?: Comparison | null;
  busy?: boolean;
  onSelect: (runId: string) => void;
  onToggleChecked: (runId: string, checked: boolean) => void;
  onCompare: () => void;
}

const metric = (run: SimulationSummary, key: string) => {
  const value = run.metrics[key];
  return typeof value === "number" ? value : null;
};

const formatSeconds = (value: number | null) =>
  value === null ? "-" : `${(value / 1000).toFixed(1)} s`;

export function SimulationTable({
  runs,
  selectedRunId,
  checkedRunIds,
  comparison,
  busy,
  onSelect,
  onToggleChecked,
  onCompare,
}: SimulationTableProps) {
  return (
    <section className="data-panel simulation-panel">
      <div className="panel-heading panel-heading-actions">
        <div>
          <h2>候选方案与指标</h2>
          <p>选择 2-4 个成功方案进行同口径比较</p>
        </div>
        <Button
          appearance="primary"
          icon={<BranchCompare20Regular />}
          disabled={checkedRunIds.length < 2 || checkedRunIds.length > 4 || busy}
          onClick={onCompare}
        >
          {busy ? "比较中" : `比较方案${checkedRunIds.length ? ` (${checkedRunIds.length})` : ""}`}
        </Button>
      </div>

      {comparison && (
        <div className="recommendation-banner">
          <Star20Filled />
          <div>
            <strong>推荐 {comparison.runs.find((run) => run.runId === comparison.recommendedRunId)?.label}</strong>
            <span>{comparison.rationale.join(" ")}</span>
          </div>
        </div>
      )}

      <div className="table-scroll">
        <Table aria-label="仿真方案指标表" size="small">
          <TableHeader>
            <TableRow>
              <TableHeaderCell className="check-column" />
              <TableHeaderCell>方案</TableHeaderCell>
              <TableHeaderCell>任务完成</TableHeaderCell>
              <TableHeaderCell>平均周期</TableHeaderCell>
              <TableHeaderCell>平均排队</TableHeaderCell>
              <TableHeaderCell>资源冲突</TableHeaderCell>
              <TableHeaderCell>状态</TableHeaderCell>
            </TableRow>
          </TableHeader>
          <TableBody>
            {runs.map((run) => {
              const recommended = comparison?.recommendedRunId === run.runId;
              const selected = selectedRunId === run.runId;
              return (
                <TableRow
                  key={run.runId}
                  className={selected ? "selected-row" : undefined}
                  onClick={() => onSelect(run.runId)}
                >
                  <TableCell className="check-column" onClick={(event) => event.stopPropagation()}>
                    <Checkbox
                      aria-label={`选择 ${run.label}`}
                      checked={checkedRunIds.includes(run.runId)}
                      disabled={run.status !== "COMPLETED"}
                      onChange={(_, data) => onToggleChecked(run.runId, data.checked === true)}
                    />
                  </TableCell>
                  <TableCell>
                    <TableCellLayout
                      media={recommended ? <Star20Filled className="recommendation-icon" /> : <PlayCircle20Regular />}
                      description={`${run.policy} | ${new Date(run.createdAt).toLocaleString("zh-CN", { hour12: false })}`}
                    >
                      {run.label}
                    </TableCellLayout>
                  </TableCell>
                  <TableCell className="mono">{metric(run, "completedTaskCount") ?? "-"}</TableCell>
                  <TableCell className="mono">{formatSeconds(metric(run, "meanTaskCycleTimeMs"))}</TableCell>
                  <TableCell className="mono">{formatSeconds(metric(run, "meanTaskQueueTimeMs"))}</TableCell>
                  <TableCell className="mono">{metric(run, "reservationConflictRejections") ?? "-"}</TableCell>
                  <TableCell>
                    <Badge
                      appearance="tint"
                      color={run.status === "COMPLETED" ? "success" : "danger"}
                    >
                      {run.status === "COMPLETED" ? "已完成" : "失败"}
                    </Badge>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
        {runs.length === 0 && (
          <div className="empty-state">尚无仿真结果。请从调度助手运行基线或候选方案。</div>
        )}
      </div>
    </section>
  );
}
