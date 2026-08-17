import { useMemo, useState } from "react";
import {
  Button,
  Select,
  Slider,
  Tooltip,
} from "@fluentui/react-components";
import {
  ArrowReset20Regular,
  Pause20Filled,
  Play20Filled,
  Subtract20Regular,
  Add20Regular,
} from "@fluentui/react-icons";
import type {
  DispatchIntent,
  Incident,
  MapEdge,
  MapModel,
  RunDetail,
  Snapshot,
} from "../types";

interface WarehouseMapProps {
  map: MapModel;
  snapshot: Snapshot;
  run?: RunDetail | null;
  intent?: DispatchIntent | null;
  playbackMs: number;
  playing: boolean;
  speed: number;
  onTogglePlaying: () => void;
  onPlaybackChange: (value: number) => void;
  onSpeedChange: (value: number) => void;
  incident?: Incident | null;
}

interface VehiclePosition {
  id: string;
  group: "fork" | "jack";
  x: number;
  y: number;
  state: string;
  activeEdgeId?: string;
}

const pathForEdge = (edge: MapEdge) =>
  `M ${edge.p0[0]} ${edge.p0[1]} C ${edge.p1[0]} ${edge.p1[1]}, ${edge.p2[0]} ${edge.p2[1]}, ${edge.p3[0]} ${edge.p3[1]}`;

const cubicPoint = (edge: MapEdge, t: number) => {
  const mt = 1 - t;
  const x =
    mt ** 3 * edge.p0[0] +
    3 * mt ** 2 * t * edge.p1[0] +
    3 * mt * t ** 2 * edge.p2[0] +
    t ** 3 * edge.p3[0];
  const y =
    mt ** 3 * edge.p0[1] +
    3 * mt ** 2 * t * edge.p1[1] +
    3 * mt * t ** 2 * edge.p2[1] +
    t ** 3 * edge.p3[1];
  return { x, y };
};

const formatTime = (value: number) => {
  const totalSeconds = Math.floor(value / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
};

export function WarehouseMap({
  map,
  snapshot,
  run,
  intent,
  playbackMs,
  playing,
  speed,
  onTogglePlaying,
  onPlaybackChange,
  onSpeedChange,
  incident,
}: WarehouseMapProps) {
  const [zoom, setZoom] = useState(1);
  const nodeById = useMemo(
    () => new Map(map.nodes.map((node) => [node.id, node])),
    [map.nodes],
  );
  const edgeById = useMemo(
    () => new Map(map.edges.map((edge) => [edge.id, edge])),
    [map.edges],
  );
  const routeEdges = useMemo(
    () =>
      new Set(
        run?.scenario.plans.flatMap((plan) =>
          plan.segments.flatMap((segment) =>
            segment.edgeId ? [segment.edgeId] : [],
          ),
        ) || [],
      ),
    [run],
  );
  const blockedEdges = useMemo(() => {
    const resources = new Set(intent?.resourceBlock?.resourceIds || []);
    const members = snapshot.zones
      .filter((zone) => resources.has(`zone:${zone.id}`))
      .flatMap((zone) => zone.memberEdgeIds);
    return new Set(members);
  }, [intent, snapshot.zones]);
  const incidentEdges = useMemo(() => {
    const resources = new Set(incident?.resourceIds || []);
    const edges = new Set(
      [...resources]
        .filter((row) => row.startsWith("edge:"))
        .map((row) => row.slice(5)),
    );
    if (incident?.locationEdgeId) edges.add(incident.locationEdgeId);
    return edges;
  }, [incident]);

  const positions = useMemo<VehiclePosition[]>(() => {
    if (!run) {
      return snapshot.vehicles.flatMap((vehicle) => {
        const node = nodeById.get(vehicle.currentNodeId);
        return node
          ? [{
              id: vehicle.vehicleId,
              group: vehicle.robotGroup,
              x: node.x,
              y: node.y,
              state: vehicle.state,
            }]
          : [];
      });
    }

    return run.scenario.vehicles.flatMap((vehicle) => {
      const allSegments = run.scenario.plans
        .filter((plan) => plan.vehicleId === vehicle.vehicleId)
        .flatMap((plan) => plan.segments)
        .sort((left, right) => left.startMs - right.startMs);
      const active = allSegments.find(
        (segment) => playbackMs >= segment.startMs && playbackMs < segment.endMs,
      );
      const completed = allSegments.filter((segment) => segment.endMs <= playbackMs).at(-1);
      const segment = active || completed;
      const initial = nodeById.get(vehicle.initialNodeId);
      if (!segment) {
        return initial
          ? [{
              id: vehicle.vehicleId,
              group: vehicle.robotGroup,
              x: initial.x,
              y: initial.y,
              state: "待命",
            }]
          : [];
      }
      if (active?.kind === "traverse" && active.edgeId) {
        const edge = edgeById.get(active.edgeId);
        if (edge) {
          const progress = Math.max(
            0,
            Math.min(1, (playbackMs - active.startMs) / (active.endMs - active.startMs)),
          );
          const point = cubicPoint(edge, progress);
          return [{
            id: vehicle.vehicleId,
            group: vehicle.robotGroup,
            ...point,
            state: active.expectedLoadState === "loaded" ? "载货行驶" : "空载行驶",
            activeEdgeId: edge.id,
          }];
        }
      }
      const node = nodeById.get(segment.endNodeId || segment.startNodeId || "") || initial;
      return node
        ? [{
            id: vehicle.vehicleId,
            group: vehicle.robotGroup,
            x: node.x,
            y: node.y,
            state:
              active?.kind === "pickup"
                ? "取货作业"
                : active?.kind === "dropoff"
                  ? "卸货作业"
                  : active?.kind === "wait"
                    ? "安全等待"
                    : "待命",
          }]
        : [];
    });
  }, [edgeById, nodeById, playbackMs, run, snapshot.vehicles]);

  const activeEdges = new Set(positions.flatMap((item) => item.activeEdgeId ? [item.activeEdgeId] : []));
  const width = map.bounds.maxX - map.bounds.minX;
  const height = map.bounds.maxY - map.bounds.minY;
  const viewWidth = width / zoom;
  const viewHeight = height / zoom;
  const viewX = map.bounds.minX + (width - viewWidth) / 2;
  const viewY = map.bounds.minY + (height - viewHeight) / 2;
  const endTimeMs = run?.scenario.endTimeMs || snapshot.endTimeMs;

  return (
    <section className="map-panel" aria-label="MASP 仓库数字孪生地图">
      <div className="panel-heading map-heading">
        <div>
          <h2>仓库数字孪生</h2>
          <p>{run ? `正在回放 ${run.summary.label}` : "当前场景初始状态"}</p>
        </div>
        <div className="map-tools">
          <Tooltip content="缩小地图" relationship="label">
            <Button
              appearance="subtle"
              icon={<Subtract20Regular />}
              aria-label="缩小地图"
              onClick={() => setZoom((value) => Math.max(1, value - 0.5))}
            />
          </Tooltip>
          <span className="zoom-value">{Math.round(zoom * 100)}%</span>
          <Tooltip content="放大地图" relationship="label">
            <Button
              appearance="subtle"
              icon={<Add20Regular />}
              aria-label="放大地图"
              onClick={() => setZoom((value) => Math.min(3, value + 0.5))}
            />
          </Tooltip>
          <Tooltip content="重置地图" relationship="label">
            <Button
              appearance="subtle"
              icon={<ArrowReset20Regular />}
              aria-label="重置地图"
              onClick={() => setZoom(1)}
            />
          </Tooltip>
        </div>
      </div>

      <div className="map-stage">
        <svg
          className="warehouse-svg"
          viewBox={`${viewX} ${viewY} ${viewWidth} ${viewHeight}`}
          role="img"
          aria-label={`仓库地图，${map.nodes.length} 个节点，${positions.length} 辆车`}
          preserveAspectRatio="xMidYMid meet"
        >
          <rect
            x={map.bounds.minX}
            y={map.bounds.minY}
            width={width}
            height={height}
            className="map-floor"
          />
          <g className="map-network">
            {map.edges.map((edge) => {
              const classes = [
                "map-edge",
                edge.group === "fork" ? "map-edge-fork" : "map-edge-jack",
                edge.shared ? "map-edge-shared" : "",
                routeEdges.has(edge.id) ? "map-edge-route" : "",
                blockedEdges.has(edge.id) ? "map-edge-blocked" : "",
                incidentEdges.has(edge.id) ? "map-edge-incident" : "",
                activeEdges.has(edge.id) ? "map-edge-active" : "",
              ].filter(Boolean).join(" ");
              return <path key={edge.id} d={pathForEdge(edge)} className={classes} />;
            })}
          </g>
          <g className="map-nodes">
            {map.nodes.filter((node) => node.type === "AP").map((node) => (
              <circle key={node.id} cx={node.x} cy={node.y} r={0.42} className="map-node-ap">
                <title>{node.id}</title>
              </circle>
            ))}
          </g>
          {incident?.locationNodeId && nodeById.get(incident.locationNodeId) && (
            <g className="incident-node-marker">
              <circle
                cx={nodeById.get(incident.locationNodeId)?.x}
                cy={nodeById.get(incident.locationNodeId)?.y}
                r={2.4}
              />
              <circle
                cx={nodeById.get(incident.locationNodeId)?.x}
                cy={nodeById.get(incident.locationNodeId)?.y}
                r={0.75}
                className="incident-node-core"
              />
              <title>{`异常位置 ${incident.locationNodeId}`}</title>
            </g>
          )}
          <g className="map-vehicles">
            {positions.map((vehicle) => (
              <g
                key={vehicle.id}
                transform={`translate(${vehicle.x} ${vehicle.y})`}
                className={incident?.vehicleIds.includes(vehicle.id) ? "incident-vehicle" : undefined}
              >
                <circle r={1.35} className={`vehicle-halo vehicle-halo-${vehicle.group}`} />
                <rect
                  x={-0.72}
                  y={-0.72}
                  width={1.44}
                  height={1.44}
                  rx={0.25}
                  className={`vehicle-marker vehicle-marker-${vehicle.group}`}
                />
                <title>{`${vehicle.id} | ${vehicle.state}`}</title>
              </g>
            ))}
          </g>
        </svg>
        <div className="map-legend" aria-label="地图图例">
          <span><i className="legend-line legend-fork" />叉车路网</span>
          <span><i className="legend-line legend-jack" />搬运车路网</span>
          <span><i className="legend-line legend-route" />仿真路径</span>
          {blockedEdges.size > 0 && <span><i className="legend-line legend-blocked" />封锁资源</span>}
          {incident && <span><i className="legend-line legend-incident" />异常影响</span>}
        </div>
      </div>

      <div className="playback-controls">
        <Tooltip content={playing ? "暂停回放" : "开始回放"} relationship="label">
          <Button
            appearance="primary"
            icon={playing ? <Pause20Filled /> : <Play20Filled />}
            aria-label={playing ? "暂停回放" : "开始回放"}
            onClick={onTogglePlaying}
            disabled={!run}
          />
        </Tooltip>
        <span className="playback-time mono">{formatTime(playbackMs)}</span>
        <Slider
          min={0}
          max={endTimeMs}
          step={100}
          value={Math.min(playbackMs, endTimeMs)}
          onChange={(_, data) => onPlaybackChange(data.value)}
          aria-label="仿真时间轴"
          disabled={!run}
        />
        <span className="playback-time mono">{formatTime(endTimeMs)}</span>
        <Select
          aria-label="回放速度"
          value={String(speed)}
          onChange={(_, data) => onSpeedChange(Number(data.value))}
          disabled={!run}
        >
          <option value="1">1x</option>
          <option value="2">2x</option>
          <option value="5">5x</option>
          <option value="10">10x</option>
        </Select>
      </div>
    </section>
  );
}
