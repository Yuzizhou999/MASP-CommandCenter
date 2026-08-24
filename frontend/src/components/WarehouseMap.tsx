import { useEffect, useMemo, useRef, useState } from "react";
import { Button, Select, Slider, Switch, Tooltip } from "@fluentui/react-components";
import {
  Add20Regular,
  ArrowReset20Regular,
  Pause20Filled,
  Play20Filled,
  Subtract20Regular,
  Target20Regular,
} from "@fluentui/react-icons";
import type {
  DispatchIntent,
  Incident,
  MapEdge,
  MapModel,
  ReplayPlan,
  ReplaySegment,
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
  selectedVehicleIds?: string[];
  onToggleVehicle?: (vehicleId: string) => void;
}

interface VehiclePosition {
  id: string;
  group: "fork" | "jack";
  x: number;
  y: number;
  heading: number;
  state: string;
  segment?: ReplaySegment;
  activeEdgeId?: string;
}

interface MapView {
  x: number;
  y: number;
  width: number;
  height: number;
}

const pathForEdge = (edge: MapEdge) =>
  `M ${edge.p0[0]} ${edge.p0[1]} C ${edge.p1[0]} ${edge.p1[1]}, ${edge.p2[0]} ${edge.p2[1]}, ${edge.p3[0]} ${edge.p3[1]}`;

const cubicPoint = (edge: MapEdge, t: number) => {
  const mt = 1 - t;
  return {
    x: mt ** 3 * edge.p0[0] + 3 * mt ** 2 * t * edge.p1[0] + 3 * mt * t ** 2 * edge.p2[0] + t ** 3 * edge.p3[0],
    y: mt ** 3 * edge.p0[1] + 3 * mt ** 2 * t * edge.p1[1] + 3 * mt * t ** 2 * edge.p2[1] + t ** 3 * edge.p3[1],
  };
};

const cubicDerivative = (edge: MapEdge, t: number) => {
  const mt = 1 - t;
  return {
    x: 3 * mt ** 2 * (edge.p1[0] - edge.p0[0]) + 6 * mt * t * (edge.p2[0] - edge.p1[0]) + 3 * t ** 2 * (edge.p3[0] - edge.p2[0]),
    y: 3 * mt ** 2 * (edge.p1[1] - edge.p0[1]) + 6 * mt * t * (edge.p2[1] - edge.p1[1]) + 3 * t ** 2 * (edge.p3[1] - edge.p2[1]),
  };
};

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

const shortestAngleDelta = (from: number, to: number) => {
  const turn = (to - from + Math.PI) % (2 * Math.PI);
  return (turn < 0 ? turn + 2 * Math.PI : turn) - Math.PI;
};

const interpolateHeading = (from: number, to: number, progress: number) =>
  from + shortestAngleDelta(from, to) * clamp01(progress);

const edgeHeading = (edge: MapEdge, t: number) => {
  const derivative = cubicDerivative(edge, t);
  let heading = Math.atan2(derivative.y, derivative.x);
  if (edge.motionDirection === 1) heading += Math.PI;
  return heading;
};

const geometryHeading = (edge: MapEdge, t: number) => {
  const derivative = cubicDerivative(edge, t);
  if (Math.abs(derivative.x) + Math.abs(derivative.y) < 1e-9) {
    return Math.atan2(edge.p3[1] - edge.p0[1], edge.p3[0] - edge.p0[0]);
  }
  return Math.atan2(derivative.y, derivative.x);
};

const traversalState = (segment: ReplaySegment, edge: MapEdge, timeMs: number) => {
  const duration = Math.max(0, segment.endMs - segment.startMs);
  const elapsed = Math.max(0, Math.min(duration, timeMs - segment.startMs));
  const motion = segment.motion;
  if (!motion) {
    const t = duration ? clamp01(elapsed / duration) : 1;
    return { t, heading: edgeHeading(edge, t), phase: "linear" };
  }
  const startRotationMs = Math.max(0, motion.startRotationMs || 0);
  const linearMs = Math.max(0, motion.linearMs || 0);
  const endRotationMs = Math.max(0, motion.endRotationMs || 0);
  if (startRotationMs > 0 && elapsed <= startRotationMs) {
    return {
      t: 0,
      heading: interpolateHeading(motion.startHeadingRad, motion.travelStartHeadingRad, elapsed / startRotationMs),
      phase: "start-rotation",
    };
  }
  const linearElapsed = elapsed - startRotationMs;
  if (linearMs > 0 && linearElapsed <= linearMs) {
    const t = clamp01(linearElapsed / linearMs);
    return { t, heading: edgeHeading(edge, t), phase: "linear" };
  }
  const progress = endRotationMs ? Math.max(0, linearElapsed - linearMs) / endRotationMs : 1;
  return {
    t: 1,
    heading: interpolateHeading(motion.travelEndHeadingRad, motion.endHeadingRad, progress),
    phase: "end-rotation",
  };
};

const partialPath = (edge: MapEdge, startT = 0, endT = 1) => {
  const from = clamp01(startT);
  const to = Math.max(from, clamp01(endT));
  const steps = Math.max(2, Math.ceil((to - from) * Math.max(4, (edge.length || 0) * 3)));
  return Array.from({ length: steps + 1 }, (_, index) => {
    const point = cubicPoint(edge, from + ((to - from) * index) / steps);
    return `${index ? "L" : "M"} ${point.x} ${point.y}`;
  }).join(" ");
};

const sampledFootprintPath = (
  edge: MapEdge,
  length: number,
  width: number,
  spacing: number,
  margin: number,
  startT = 0,
  endT = 1,
) => {
  const from = clamp01(startT);
  const to = Math.max(from, clamp01(endT));
  const samples = Math.max(3, Math.ceil(((edge.length || 0) * (to - from)) / Math.max(0.001, spacing)) + 1);
  const halfLength = length / 2 + margin;
  const halfWidth = width / 2 + margin;
  const corners = [[-halfLength, -halfWidth], [halfLength, -halfWidth], [halfLength, halfWidth], [-halfLength, halfWidth]];
  return Array.from({ length: samples }, (_, index) => {
    const t = from + ((to - from) * index) / (samples - 1);
    const center = cubicPoint(edge, t);
    const heading = geometryHeading(edge, t);
    const cosine = Math.cos(heading);
    const sine = Math.sin(heading);
    return corners.map(([x, y], cornerIndex) =>
      `${cornerIndex ? "L" : "M"} ${center.x + x * cosine - y * sine} ${center.y + x * sine + y * cosine}`,
    ).join(" ") + " Z";
  }).join(" ");
};

const formatTime = (value: number) => {
  const totalSeconds = Math.floor(value / 1000);
  return `${String(Math.floor(totalSeconds / 60)).padStart(2, "0")}:${String(totalSeconds % 60).padStart(2, "0")}`;
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
  selectedVehicleIds = [],
  onToggleVehicle,
}: WarehouseMapProps) {
  const fullView = useMemo<MapView>(() => ({
    x: map.bounds.minX,
    y: map.bounds.minY,
    width: map.bounds.maxX - map.bounds.minX,
    height: map.bounds.maxY - map.bounds.minY,
  }), [map.bounds]);
  const [view, setView] = useState(fullView);
  const [showStations, setShowStations] = useState(false);
  const [showVehicleLabels, setShowVehicleLabels] = useState(false);
  const panRef = useRef<{ pointerId: number; x: number; y: number; view: MapView } | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => setView(fullView), [fullView]);

  const nodeById = useMemo(() => new Map(map.nodes.map((node) => [node.id, node])), [map.nodes]);
  const edgeById = useMemo(() => new Map(map.edges.map((edge) => [edge.id, edge])), [map.edges]);
  const replay = run?.replay;
  const replayPlans = replay?.plans || run?.scenario.plans.map((plan) => ({
    ...plan,
    createdAtMs: plan.createdAtMs || 0,
    committedUntilMs: plan.committedUntilMs || run.scenario.endTimeMs,
  } as ReplayPlan)) || [];
  const replayVehicles = replay?.vehicles || run?.scenario.vehicles.map((vehicle) => ({
    ...vehicle,
    initialHeadingRad: vehicle.initialHeadingRad || 0,
    state: "UNKNOWN",
    loadState: "empty",
  })) || [];

  const plansByVehicle = useMemo(() => {
    const rows = new Map<string, ReplayPlan[]>();
    replayPlans.forEach((plan) => rows.set(plan.vehicleId, [...(rows.get(plan.vehicleId) || []), plan]));
    rows.forEach((plans) => plans.sort((left, right) => left.createdAtMs - right.createdAtMs));
    return rows;
  }, [replayPlans]);

  const routeEdges = useMemo(() => new Set(replayPlans.flatMap((plan) =>
    plan.segments.flatMap((segment) => segment.edgeId ? [segment.edgeId] : []),
  )), [replayPlans]);

  const blockedEdges = useMemo(() => {
    const resources = new Set(intent?.resourceBlock?.resourceIds || []);
    return new Set(snapshot.zones.filter((zone) => resources.has(`zone:${zone.id}`)).flatMap((zone) => zone.memberEdgeIds));
  }, [intent, snapshot.zones]);

  const incidentEdges = useMemo(() => {
    const resources = new Set(incident?.resourceIds || []);
    const edges = new Set([...resources].filter((row) => row.startsWith("edge:")).map((row) => row.slice(5)));
    if (incident?.locationEdgeId) edges.add(incident.locationEdgeId);
    return edges;
  }, [incident]);

  const positions = useMemo<VehiclePosition[]>(() => {
    if (!run) {
      return snapshot.vehicles.flatMap((vehicle) => {
        const node = nodeById.get(vehicle.currentNodeId);
        return node ? [{ id: vehicle.vehicleId, group: vehicle.robotGroup, x: node.x, y: node.y, heading: 0, state: vehicle.state }] : [];
      });
    }
    return replayVehicles.flatMap((vehicle) => {
      const segments = (plansByVehicle.get(vehicle.vehicleId) || []).flatMap((plan) => plan.segments).sort((left, right) => left.startMs - right.startMs);
      const initial = nodeById.get(vehicle.initialNodeId);
      let previous: ReplaySegment | undefined;
      for (const segment of segments) {
        if (playbackMs < segment.startMs) break;
        if (playbackMs <= segment.endMs) {
          if (segment.kind === "rotate") {
            const node = nodeById.get(segment.startNodeId || "") || nodeById.get(segment.endNodeId || "") || initial;
            if (!node) return [];
            const duration = Math.max(1, segment.endMs - segment.startMs);
            const start = Number(segment.commandPayload?.startHeadingRad || vehicle.initialHeadingRad || 0);
            const end = Number(segment.commandPayload?.endHeadingRad || start);
            return [{ id: vehicle.vehicleId, group: vehicle.robotGroup, x: node.x, y: node.y, heading: interpolateHeading(start, end, (playbackMs - segment.startMs) / duration), state: "原地转向", segment }];
          }
          const edge = segment.edgeId ? edgeById.get(segment.edgeId) : undefined;
          if (edge && segment.endMs > segment.startMs) {
            const motion = traversalState(segment, edge, playbackMs);
            const point = cubicPoint(edge, motion.t);
            return [{ id: vehicle.vehicleId, group: vehicle.robotGroup, ...point, heading: motion.heading, state: motion.phase === "linear" ? (segment.expectedLoadState === "loaded" ? "载货行驶" : "空载行驶") : "原地转向", segment, activeEdgeId: edge.id }];
          }
          const node = nodeById.get(segment.startNodeId || "") || nodeById.get(segment.endNodeId || "") || initial;
          if (!node) return [];
          const priorEdge = previous?.edgeId ? edgeById.get(previous.edgeId) : undefined;
          const heading = previous?.kind === "rotate"
            ? Number(previous.commandPayload?.endHeadingRad || vehicle.initialHeadingRad || 0)
            : previous?.motion?.endHeadingRad ?? (priorEdge ? edgeHeading(priorEdge, 1) : vehicle.initialHeadingRad || 0);
          const state = segment.kind === "pickup" ? "取货作业" : segment.kind === "dropoff" ? "卸货作业" : segment.kind === "wait" ? "安全等待" : "待命";
          return [{ id: vehicle.vehicleId, group: vehicle.robotGroup, x: node.x, y: node.y, heading, state, segment }];
        }
        previous = segment;
      }
      const node = nodeById.get(previous?.endNodeId || vehicle.initialNodeId) || initial;
      if (!node) return [];
      const priorEdge = previous?.edgeId ? edgeById.get(previous.edgeId) : undefined;
      const heading = previous?.kind === "rotate"
        ? Number(previous.commandPayload?.endHeadingRad || vehicle.initialHeadingRad || 0)
        : previous?.motion?.endHeadingRad ?? (priorEdge ? edgeHeading(priorEdge, 1) : vehicle.initialHeadingRad || 0);
      return [{ id: vehicle.vehicleId, group: vehicle.robotGroup, x: node.x, y: node.y, heading, state: previous ? "空闲" : "待命", segment: previous }];
    });
  }, [edgeById, nodeById, plansByVehicle, playbackMs, replayVehicles, run, snapshot.vehicles]);

  const selected = useMemo(() => new Set(selectedVehicleIds), [selectedVehicleIds]);
  const commitments = useMemo(() => {
    if (!replay) return [];
    const executionHorizonMs = Number(replay.planning.executionHorizonMs || 0);
    return selectedVehicleIds.flatMap((vehicleId) => {
      const plans = plansByVehicle.get(vehicleId) || [];
      const plan = plans.filter((row) => row.createdAtMs <= playbackMs && playbackMs <= row.committedUntilMs).at(-1);
      if (!plan) return [];
      const windowEnd = Math.min(plan.committedUntilMs, playbackMs + executionHorizonMs);
      const traversals = plan.segments.flatMap((segment) => {
        if (!segment.edgeId || segment.endMs <= playbackMs || segment.startMs >= windowEnd) return [];
        const edge = edgeById.get(segment.edgeId);
        if (!edge) return [];
        const startT = traversalState(segment, edge, Math.max(playbackMs, segment.startMs)).t;
        const endT = Math.max(startT, traversalState(segment, edge, Math.min(windowEnd, segment.endMs)).t);
        return [{ edge, startT, endT }];
      });
      const vehicle = replay.vehicles.find((row) => row.vehicleId === vehicleId);
      const profile = replay.vehicleProfiles[vehicle?.robotGroup || ""] || { length: 1, width: 0.5 };
      const spacing = replay.sweepModel.sampleSpacing || 0.25;
      const margin = replay.sweepModel.footprintMargin || 0;
      return [{
        vehicleId,
        group: vehicle?.robotGroup || "jack",
        route: traversals.map(({ edge, startT, endT }) => partialPath(edge, startT, endT)).join(" "),
        locked: [...new Map(traversals.map((item) => [item.edge.id, item.edge])).values()].map((edge) => sampledFootprintPath(edge, profile.length, profile.width, spacing, margin)).join(" "),
        swept: traversals.map(({ edge, startT, endT }) => sampledFootprintPath(edge, profile.length, profile.width, spacing, margin, startT, endT)).join(" "),
        end: traversals.length ? cubicPoint(traversals.at(-1)!.edge, traversals.at(-1)!.endT) : null,
      }];
    });
  }, [edgeById, plansByVehicle, playbackMs, replay, selectedVehicleIds]);

  const activeEdges = new Set(positions.flatMap((item) => item.activeEdgeId ? [item.activeEdgeId] : []));
  const endTimeMs = replay?.endTimeMs || run?.scenario.endTimeMs || snapshot.endTimeMs;
  const zoom = fullView.width / view.width;

  const constrainView = (next: MapView): MapView => {
    const width = Math.max(fullView.width / 6, Math.min(fullView.width, next.width));
    const height = Math.max(fullView.height / 6, Math.min(fullView.height, next.height));
    return {
      x: Math.max(fullView.x, Math.min(fullView.x + fullView.width - width, next.x)),
      y: Math.max(fullView.y, Math.min(fullView.y + fullView.height - height, next.y)),
      width,
      height,
    };
  };

  const zoomMap = (factor: number, focusX = view.x + view.width / 2, focusY = view.y + view.height / 2) => {
    setView((current) => {
      const nextWidth = current.width * factor;
      const nextHeight = current.height * factor;
      const xRatio = (focusX - current.x) / current.width;
      const yRatio = (focusY - current.y) / current.height;
      return constrainView({ x: focusX - nextWidth * xRatio, y: focusY - nextHeight * yRatio, width: nextWidth, height: nextHeight });
    });
  };

  const focusShared = () => {
    const shared = map.edges.filter((edge) => edge.shared);
    if (!shared.length) return;
    const xs = shared.flatMap((edge) => [edge.p0[0], edge.p1[0], edge.p2[0], edge.p3[0]]);
    const ys = shared.flatMap((edge) => [edge.p0[1], edge.p1[1], edge.p2[1], edge.p3[1]]);
    const padding = 5;
    setView(constrainView({ x: Math.min(...xs) - padding, y: Math.min(...ys) - padding, width: Math.max(...xs) - Math.min(...xs) + padding * 2, height: Math.max(...ys) - Math.min(...ys) + padding * 2 }));
  };

  return (
    <section className="map-panel" aria-label="MASP 仓库数字孪生地图">
      <div className="panel-heading map-heading">
        <div><h2>仓库数字孪生</h2><p>{run ? `正在回放 ${run.summary.label}` : "当前场景初始状态"}</p></div>
        <div className="map-tools">
          <Switch checked={showStations} onChange={(_, data) => setShowStations(data.checked)} label="工位" />
          <Switch checked={showVehicleLabels} onChange={(_, data) => setShowVehicleLabels(data.checked)} label="车号" />
          <Tooltip content="聚焦共享区" relationship="label"><Button appearance="subtle" icon={<Target20Regular />} aria-label="聚焦共享区" onClick={focusShared} /></Tooltip>
          <Tooltip content="缩小地图" relationship="label"><Button appearance="subtle" icon={<Subtract20Regular />} aria-label="缩小地图" onClick={() => zoomMap(1.25)} /></Tooltip>
          <span className="zoom-value">{Math.round(zoom * 100)}%</span>
          <Tooltip content="放大地图" relationship="label"><Button appearance="subtle" icon={<Add20Regular />} aria-label="放大地图" onClick={() => zoomMap(0.8)} /></Tooltip>
          <Tooltip content="重置地图" relationship="label"><Button appearance="subtle" icon={<ArrowReset20Regular />} aria-label="重置地图" onClick={() => setView(fullView)} /></Tooltip>
        </div>
      </div>

      <div className="map-stage">
        <svg
          ref={svgRef}
          className="warehouse-svg"
          viewBox={`${view.x} ${view.y} ${view.width} ${view.height}`}
          role="img"
          aria-label={`仓库地图，${map.nodes.length} 个节点，${positions.length} 辆车`}
          preserveAspectRatio="xMidYMid meet"
          onWheel={(event) => {
            event.preventDefault();
            const bounds = event.currentTarget.getBoundingClientRect();
            zoomMap(event.deltaY > 0 ? 1.12 : 0.88, view.x + ((event.clientX - bounds.left) / bounds.width) * view.width, view.y + ((event.clientY - bounds.top) / bounds.height) * view.height);
          }}
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            panRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, view };
          }}
          onPointerMove={(event) => {
            const pan = panRef.current;
            const element = svgRef.current;
            if (!pan || pan.pointerId !== event.pointerId || !element) return;
            const bounds = element.getBoundingClientRect();
            setView(constrainView({ ...pan.view, x: pan.view.x - ((event.clientX - pan.x) / bounds.width) * pan.view.width, y: pan.view.y - ((event.clientY - pan.y) / bounds.height) * pan.view.height }));
          }}
          onPointerUp={(event) => { if (panRef.current?.pointerId === event.pointerId) panRef.current = null; }}
          onPointerCancel={() => { panRef.current = null; }}
        >
          <defs>
            <pattern id="commitment-lock-fork" width="0.9" height="0.9" patternUnits="userSpaceOnUse"><path d="M 0 .9 L .9 0" className="commitment-hatch-fork" /></pattern>
            <pattern id="commitment-lock-jack" width="0.9" height="0.9" patternUnits="userSpaceOnUse"><path d="M 0 0 L .9 .9" className="commitment-hatch-jack" /></pattern>
          </defs>
          <rect x={fullView.x} y={fullView.y} width={fullView.width} height={fullView.height} className="map-floor" />
          <g className="map-network">
            {map.edges.map((edge) => {
              const classes = ["map-edge", edge.group === "fork" ? "map-edge-fork" : "map-edge-jack", edge.shared ? "map-edge-shared" : "", routeEdges.has(edge.id) ? "map-edge-route" : "", blockedEdges.has(edge.id) ? "map-edge-blocked" : "", incidentEdges.has(edge.id) ? "map-edge-incident" : "", activeEdges.has(edge.id) ? "map-edge-active" : ""].filter(Boolean).join(" ");
              return <path key={edge.id} d={pathForEdge(edge)} className={classes}><title>{edge.id}</title></path>;
            })}
          </g>
          <g className="map-nodes">
            {map.nodes.filter((node) => node.type === "AP").map((node) => (
              <g key={node.id}><circle cx={node.x} cy={node.y} r={0.42} className="map-node-ap"><title>{node.id}</title></circle>{showStations && <text x={node.x + 0.65} y={node.y - 0.65} className="station-label">{node.id.split(":").at(-1)}</text>}</g>
            ))}
          </g>
          <g className="commitment-layer">
            {commitments.map((item) => <path key={`${item.vehicleId}-locked`} d={item.locked} className={`commitment-lock commitment-lock-${item.group}`} />)}
            {commitments.map((item) => <path key={`${item.vehicleId}-swept`} d={item.swept} className={`commitment-sweep commitment-sweep-${item.group}`} />)}
            {commitments.map((item) => <path key={`${item.vehicleId}-route`} d={item.route} className={`commitment-route commitment-route-${item.group}`} />)}
            {commitments.map((item) => item.end && <circle key={`${item.vehicleId}-end`} cx={item.end.x} cy={item.end.y} r={0.32} className={`commitment-end commitment-end-${item.group}`} />)}
          </g>
          {incident?.locationNodeId && nodeById.get(incident.locationNodeId) && <g className="incident-node-marker"><circle cx={nodeById.get(incident.locationNodeId)?.x} cy={nodeById.get(incident.locationNodeId)?.y} r={2.4} /><circle cx={nodeById.get(incident.locationNodeId)?.x} cy={nodeById.get(incident.locationNodeId)?.y} r={0.75} className="incident-node-core" /><title>{`异常位置 ${incident.locationNodeId}`}</title></g>}
          <g className="map-vehicles">
            {positions.map((vehicle) => {
              const profile = replay?.vehicleProfiles[vehicle.group] || (vehicle.group === "fork" ? { length: 2, width: 1 } : { length: 1, width: 0.5 });
              const angle = vehicle.heading * 180 / Math.PI;
              const isSelected = selected.has(vehicle.id);
              return <g key={vehicle.id} data-vehicle={vehicle.id} transform={`translate(${vehicle.x} ${vehicle.y}) rotate(${angle})`} className={["vehicle-group", isSelected ? "selected" : "", incident?.vehicleIds.includes(vehicle.id) ? "incident-vehicle" : ""].filter(Boolean).join(" ")} onClick={(event) => { event.stopPropagation(); onToggleVehicle?.(vehicle.id); }} role="button" aria-label={`${vehicle.id}，${vehicle.state}`}>
                <rect x={-profile.length / 2 - 0.5} y={-profile.width / 2 - 0.5} width={profile.length + 1} height={profile.width + 1} rx={0.25} className="vehicle-hitbox" />
                <rect x={-profile.length / 2} y={-profile.width / 2} width={profile.length} height={profile.width} rx={0.12} className={`vehicle-body vehicle-body-${vehicle.group}`} />
                <path d={`M ${profile.length / 2 - 0.08} 0 L ${profile.length / 2 - 0.4} ${-Math.min(profile.width * 0.28, 0.18)} L ${profile.length / 2 - 0.4} ${Math.min(profile.width * 0.28, 0.18)} Z`} className="vehicle-front" />
                {showVehicleLabels && <text x={profile.length / 2 + 0.35} y={-profile.width / 2 - 0.35} transform={`rotate(${-angle} ${profile.length / 2 + 0.35} ${-profile.width / 2 - 0.35})`} className="vehicle-label">{vehicle.id}</text>}
                <title>{`${vehicle.id} | ${vehicle.state}`}</title>
              </g>;
            })}
          </g>
        </svg>
        <div className="map-legend" aria-label="地图图例">
          <span><i className="legend-line legend-fork" />叉车 2.0×1.0 m</span>
          <span><i className="legend-line legend-jack" />搬运车 1.0×0.5 m</span>
          <span><i className="legend-line legend-route" />全局路线</span>
          {selected.size > 0 && <span><i className="legend-line legend-commitment" />窗口承诺</span>}
          {blockedEdges.size > 0 && <span><i className="legend-line legend-blocked" />封锁资源</span>}
          {incident && <span><i className="legend-line legend-incident" />异常影响</span>}
        </div>
        {run && <div className="map-live-status">{positions.filter((item) => item.state.includes("行驶")).length} 行驶 · {positions.filter((item) => item.state.includes("等待")).length} 等待 · {selected.size ? `已选 ${selected.size} 辆，显示滚动承诺窗口` : "点击车辆查看承诺路线"}</div>}
      </div>

      <div className="playback-controls">
        <Tooltip content={playing ? "暂停回放" : "开始回放"} relationship="label"><Button appearance="primary" icon={playing ? <Pause20Filled /> : <Play20Filled />} aria-label={playing ? "暂停回放" : "开始回放"} onClick={onTogglePlaying} disabled={!run} /></Tooltip>
        <span className="playback-time mono">{formatTime(playbackMs)}</span>
        <Slider min={0} max={endTimeMs} step={100} value={Math.min(playbackMs, endTimeMs)} onChange={(_, data) => onPlaybackChange(data.value)} aria-label="仿真时间轴" disabled={!run} />
        <span className="playback-time mono">{formatTime(endTimeMs)}</span>
        <Select aria-label="回放速度" value={String(speed)} onChange={(_, data) => onSpeedChange(Number(data.value))} disabled={!run}><option value="1">1x</option><option value="2">2x</option><option value="5">5x</option><option value="10">10x</option></Select>
      </div>
    </section>
  );
}
