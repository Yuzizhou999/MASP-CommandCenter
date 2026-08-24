import { useMemo } from "react";
import type { ReplayPlan, ReplaySegment, RunDetail } from "../types";

interface DispatchReplayPanelProps {
  run?: RunDetail | null;
  playbackMs: number;
  selectedVehicleIds: string[];
  onToggleVehicle: (vehicleId: string) => void;
}

const formatDuration = (value: number) => {
  const seconds = Math.max(0, value) / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(1)}s`;
};

const formatClock = (value: number) => {
  const seconds = Math.floor(Math.max(0, value) / 1000);
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
};

const eventLabels: Record<string, string> = {
  TASK_RELEASED: "任务释放",
  TASK_ASSIGNED: "任务分配",
  PICKUP_STARTED: "开始取货",
  PICKUP_COMPLETED: "取货完成",
  DROPOFF_STARTED: "开始放货",
  DROPOFF_COMPLETED: "放货完成",
  VEHICLE_ENTER_EDGE: "车辆进入路段",
  VEHICLE_EXIT_EDGE: "车辆离开路段",
  VEHICLE_WAIT_STARTED: "车辆开始等待",
  VEHICLE_WAIT_ENDED: "车辆结束等待",
  PLAN_COMPUTED: "计划已计算",
  PLAN_COMMITTED: "计划已承诺",
};

const activeSegment = (segments: ReplaySegment[], timeMs: number) =>
  segments.find((segment) => timeMs >= segment.startMs && timeMs <= segment.endMs);

export function DispatchReplayPanel({
  run,
  playbackMs,
  selectedVehicleIds,
  onToggleVehicle,
}: DispatchReplayPanelProps) {
  const replay = run?.replay;
  const selected = useMemo(() => new Set(selectedVehicleIds), [selectedVehicleIds]);
  const plansByVehicle = useMemo(() => {
    const rows = new Map<string, ReplayPlan[]>();
    replay?.plans.forEach((plan) => rows.set(plan.vehicleId, [...(rows.get(plan.vehicleId) || []), plan]));
    return rows;
  }, [replay]);

  if (!run || !replay) return null;

  const vehicleRows = replay.vehicles.map((vehicle) => {
    const plans = plansByVehicle.get(vehicle.vehicleId) || [];
    const segments = plans.flatMap((plan) => plan.segments).sort((left, right) => left.startMs - right.startMs);
    const segment = activeSegment(segments, playbackMs);
    const plan = plans.filter((row) => row.createdAtMs <= playbackMs && playbackMs <= row.committedUntilMs).at(-1);
    let status = playbackMs < (plans[0]?.createdAtMs || 0) ? "待命" : "空闲";
    if (segment?.kind === "pickup") status = "取货服务";
    if (segment?.kind === "dropoff") status = "放货服务";
    if (segment?.kind === "wait") status = "等待";
    if (segment?.kind === "rotate") status = "原地转向";
    if (segment?.kind === "traverse") status = "行驶中";
    return { vehicle, segment, plan, status };
  });

  const taskRows = replay.tasks.map((task) => {
    const status = task.completedAtMs != null && playbackMs >= task.completedAtMs
      ? "已完成"
      : task.pickedAtMs != null && playbackMs >= task.pickedAtMs
        ? "运输中"
        : playbackMs < task.releaseTimeMs
          ? "未释放"
          : task.assignedAtMs != null && playbackMs >= task.assignedAtMs
            ? "已分配"
            : "排队中";
    return { task, status };
  });

  const comparable = replay.tasks.filter((task) =>
    task.completedAtMs != null
    && task.completedAtMs <= playbackMs
    && task.assignedAtMs != null
    && Number(task.initialGlobalRouteMs) > 0,
  );
  const baselineMs = comparable.reduce((sum, task) => sum + Number(task.initialGlobalRouteMs), 0);
  const actualMs = comparable.reduce((sum, task) => sum + Math.max(0, Number(task.completedAtMs) - Number(task.assignedAtMs)), 0);
  const ratio = baselineMs > 0 ? actualMs / baselineMs : null;
  const deltaMs = actualMs - baselineMs;
  const visibleEvents = replay.events.filter((event) => event.timeMs <= playbackMs).slice(-18).reverse();
  const planning = replay.planning;
  const latency = (planning.planningLatencyMs || {}) as Record<string, number>;
  const policy = String(planning.policy || replay.manifest.policy || run.summary.policy || "unknown");
  const completed = taskRows.filter((row) => row.status === "已完成").length;
  const moving = vehicleRows.filter((row) => row.status === "行驶中").length;
  const waiting = vehicleRows.filter((row) => row.status === "等待").length;

  return (
    <section className="dispatch-replay" aria-label="调度回放详情">
      <div className="replay-kpis">
        <div><strong>{completed}/{replay.tasks.length}</strong><span>当前完成任务</span></div>
        <div><strong>{moving}</strong><span>行驶车辆</span></div>
        <div><strong>{waiting}</strong><span>等待车辆</span></div>
        <div><strong>{Number(replay.metrics.completedDropoffsPerHour || 0).toFixed(1)}</strong><span>仿真吞吐 / 小时</span></div>
        <div><strong>{Number(replay.metrics.reservationConflictRejections || 0)}</strong><span>资源冲突拒绝</span></div>
        <div><strong>{replay.replayMode === "online" ? "在线" : "离线"}</strong><span>{policy} 调度策略</span></div>
      </div>

      <section className="replay-section route-time-section">
        <div className="replay-section-heading"><h2>路线执行耗时</h2><span>{comparable.length}/{replay.tasks.length} 个已完成任务</span></div>
        <div className="replay-stat-grid route-time-grid">
          <div><strong>{ratio == null ? "--" : `${(ratio * 100).toFixed(1)}%`}</strong><span>实际 / 初始全局路线</span><small>{ratio == null ? "等待任务完成" : `${ratio >= 1 ? "增加" : "减少"} ${Math.abs((ratio - 1) * 100).toFixed(1)}%`}</small></div>
          <div><strong>{formatDuration(actualMs)}</strong><span>实际累计耗时</span><small>任务分配到实际放货完成</small></div>
          <div><strong>{formatDuration(baselineMs)}</strong><span>初始全局路线累计耗时</span><small>连续航向无冲突最短路与服务</small></div>
          <div><strong>{deltaMs < 0 ? "-" : "+"}{formatDuration(Math.abs(deltaMs))}</strong><span>相对基线增量</span><small>{comparable.length} 个可比较任务</small></div>
        </div>
      </section>

      <details className="replay-section planning-section">
        <summary className="replay-section-heading"><h2>规划与 RL 性能</h2><span>{policy === "rl" ? "RL 优先级" : `策略 ${policy}`}</span></summary>
        <div className="replay-stat-grid planning-stat-grid">
          <div><strong>{Number(latency.p95 || 0).toFixed(1)} ms</strong><span>规划周期 P95</span></div>
          <div><strong>{Number(latency.max || 0).toFixed(1)} ms</strong><span>最慢规划周期</span></div>
          <div><strong>{Number(planning.planningTimeoutCount || 0)}</strong><span>规划超时周期</span></div>
          <div><strong>{Number(planning.routeCombinationsTried || 0)}</strong><span>路线组合尝试</span></div>
          <div><strong>{Number(planning.scheduleAttempts || 0)}</strong><span>SIPP 调度尝试</span></div>
          <div><strong>{policy === "rl" ? `${Number(planning.rlInferenceCount || 0)} 次` : "未启用"}</strong><span>RL 优先级参与</span></div>
        </div>
      </details>

      <div className="replay-list-grid">
        <section className="replay-section replay-list-section">
          <div className="replay-section-heading"><h2>车辆状态</h2><span>{replay.vehicles.length} 辆</span></div>
          <div className="replay-list">
            {vehicleRows.map(({ vehicle, segment, plan, status }) => <button key={vehicle.vehicleId} type="button" className={`replay-list-row ${selected.has(vehicle.vehicleId) ? "selected" : ""}`} aria-pressed={selected.has(vehicle.vehicleId)} onClick={() => onToggleVehicle(vehicle.vehicleId)}>
              <i className={`replay-dot ${vehicle.robotGroup}`} />
              <span className="replay-row-main"><strong>{vehicle.vehicleId}</strong><small>{segment?.edgeId || segment?.startNodeId || vehicle.initialNodeId}{plan ? ` · 承诺至 ${formatClock(plan.committedUntilMs)}` : ""}</small></span>
              <span className={`replay-status ${status === "空闲" ? "done" : status === "等待" || status === "待命" ? "wait" : "active"}`}>{status}</span>
            </button>)}
          </div>
        </section>

        <section className="replay-section replay-list-section">
          <div className="replay-section-heading"><h2>任务状态</h2><span>{replay.tasks.length} 个</span></div>
          <div className="replay-list">
            {taskRows.map(({ task, status }) => <button key={task.taskId} type="button" className="replay-list-row" disabled={!task.assignedVehicleId || Number(task.assignedAtMs) > playbackMs} onClick={() => task.assignedVehicleId && onToggleVehicle(task.assignedVehicleId)}>
              <i className={`replay-dot ${task.requiredRobotGroup}`} />
              <span className="replay-row-main"><strong>{task.taskId}</strong><small>{task.pickupNodeId} → {task.dropoffNodeId}</small></span>
              <span className={`replay-status ${status === "已完成" ? "done" : status === "运输中" || status === "已分配" ? "active" : "wait"}`}>{status}</span>
            </button>)}
          </div>
        </section>
      </div>

      <section className="replay-section timeline-section">
        <div className="replay-section-heading"><h2>事件时间线</h2><span>{visibleEvents.length}/{replay.events.length}</span></div>
        <div className="replay-timeline">
          {visibleEvents.length === 0 && <div className="replay-empty">时间线上还没有发生事件</div>}
          {visibleEvents.map((event, index) => {
            const payload = event.payload || {};
            const detail = payload.vehicleId || payload.taskId || payload.planId || payload.edgeId || "";
            return <div className="replay-event" key={event.id || `${event.timeMs}-${event.type}-${index}`}><time>{formatClock(event.timeMs)}</time><div><strong>{eventLabels[event.type] || event.type.replaceAll("_", " ")}</strong><span>{String(detail)}</span></div></div>;
          })}
        </div>
      </section>
    </section>
  );
}
