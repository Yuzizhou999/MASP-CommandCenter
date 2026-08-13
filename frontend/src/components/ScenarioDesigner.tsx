import { useEffect, useMemo, useRef, useState } from "react";
import {
  Badge,
  Button,
  Card,
  Field,
  Input,
  Select,
  Textarea,
  Tooltip,
} from "@fluentui/react-components";
import {
  Add20Regular,
  CheckmarkCircle20Regular,
  DocumentAdd20Regular,
  Play20Regular,
  Save20Regular,
  Wrench20Regular,
} from "@fluentui/react-icons";
import { api } from "../api";
import type {
  ScenarioDraftSummary,
  ScenarioPackageDocument,
  ScenarioValidationReport,
  ScenarioMeta,
} from "../types";

interface ScenarioDesignerProps {
  scenarios: ScenarioMeta[];
  initialScenarioId: string;
  onNotice: (message: string) => void;
  onError: (message: string) => void;
}

type DraftNode = Record<string, any> & { id: string; x: number; y: number; type: string };
type DraftEdge = Record<string, any> & { id: string; startNodeId: string; endNodeId: string; robotGroup: string };

const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

const defaultGeneration = (scene: ScenarioPackageDocument["warehouseScene"]) => {
  const stations = scene.workstations.filter((item) => item.allowedRobotGroups?.length);
  const pairs = stations.slice(0, 2).map((item, index) => {
    const group = item.allowedRobotGroups?.[0] || "fork";
    const target = stations.find((candidate) => candidate.nodeId !== item.nodeId && candidate.allowedRobotGroups?.includes(group));
    return {
      pickupNodeId: item.nodeId,
      dropoffNodeId: target?.nodeId || item.nodeId,
      requiredRobotGroup: group,
      payloadType: group === "jack" ? "shelf" : "pallet",
      weight: index === 0 ? 2 : 1,
    };
  }).filter((item) => item.pickupNodeId !== item.dropoffNodeId);
  return {
    streamId: "designer-generated",
    seed: 2026,
    endTimeMs: 180000,
    maxTasks: 12,
    arrival: { mode: "time_windows", windows: [
      { startTimeMs: 0, endTimeMs: 60000, mode: "fixed_interval", intervalMs: 15000 },
      { startTimeMs: 60000, endTimeMs: 180000, mode: "poisson", meanIntervalMs: 20000 },
    ] },
    odPairs: pairs,
    priorityDistribution: [{ priorityClass: 0, weight: 3 }, { priorityClass: 1, weight: 1 }],
    serviceTimePolicy: { mode: "workstation_defaults" },
    dueTimePolicy: { mode: "relative", offsetMs: 120000 },
  };
};

export function ScenarioDesigner({ scenarios, initialScenarioId, onNotice, onError }: ScenarioDesignerProps) {
  const [drafts, setDrafts] = useState<ScenarioDraftSummary[]>([]);
  const [selectedPackageId, setSelectedPackageId] = useState<string>("");
  const [document, setDocument] = useState<ScenarioPackageDocument | null>(null);
  const [validation, setValidation] = useState<ScenarioValidationReport | null>(null);
  const [generationText, setGenerationText] = useState("{}");
  const [scenarioId, setScenarioId] = useState(initialScenarioId);
  const [packageId, setPackageId] = useState(`${initialScenarioId}-designer`);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedTaskIndex, setSelectedTaskIndex] = useState(0);
  const [edgeStartId, setEdgeStartId] = useState("");
  const [edgeEndId, setEdgeEndId] = useState("");
  const [edgeGroup, setEdgeGroup] = useState<"fork" | "jack">("fork");
  const [busy, setBusy] = useState<string | null>(null);
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const selectedNode = useMemo(
    () => document?.warehouseScene.nodes.find((node) => node.id === selectedNodeId) as DraftNode | undefined,
    [document, selectedNodeId],
  );
  const nodes = (document?.warehouseScene.nodes || []) as DraftNode[];
  const edges = (document?.warehouseScene.edges || []) as DraftEdge[];

  const reloadDrafts = async () => {
    const rows = await api.scenarioDrafts();
    setDrafts(rows);
    if (!selectedPackageId && rows[0]) setSelectedPackageId(rows[0].packageId);
  };

  useEffect(() => {
    void reloadDrafts().catch((reason) => onError(reason instanceof Error ? reason.message : "场景草稿加载失败"));
  }, []);

  useEffect(() => {
    if (!selectedPackageId) return;
    setBusy("load");
    api.scenarioDraft(selectedPackageId)
      .then((value) => {
        setDocument(value);
        setValidation(null);
        setGenerationText(JSON.stringify(defaultGeneration(value.warehouseScene), null, 2));
        setSelectedNodeId(value.warehouseScene.nodes.find((node) => node.type === "AP")?.id || value.warehouseScene.nodes[0]?.id || null);
        setSelectedTaskIndex(0);
      })
      .catch((reason) => onError(reason instanceof Error ? reason.message : "场景草稿读取失败"))
      .finally(() => setBusy(null));
  }, [selectedPackageId]);

  const updateDocument = (next: ScenarioPackageDocument) => {
    setDocument(next);
    setValidation(null);
  };

  const updateNode = (nodeId: string, patch: Record<string, unknown>) => {
    if (!document) return;
    const next = clone(document);
    const node = next.warehouseScene.nodes.find((item) => item.id === nodeId);
    if (node) Object.assign(node, patch);
    updateDocument(next);
  };

  const svgPoint = (event: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg || !document) return null;
    const rect = svg.getBoundingClientRect();
    const bounds = document.warehouseScene.bounds;
    return {
      x: bounds.minX + ((event.clientX - rect.left) / rect.width) * (bounds.maxX - bounds.minX),
      y: bounds.minY + ((event.clientY - rect.top) / rect.height) * (bounds.maxY - bounds.minY),
    };
  };

  const handlePointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!draggingNodeId) return;
    const point = svgPoint(event);
    if (point) updateNode(draggingNodeId, point);
  };

  const addNode = () => {
    if (!document) return;
    const next = clone(document);
    const id = `${edgeGroup}:CP-design-${String(next.warehouseScene.nodes.length + 1).padStart(3, "0")}`;
    const center = {
      x: (next.warehouseScene.bounds.minX + next.warehouseScene.bounds.maxX) / 2,
      y: (next.warehouseScene.bounds.minY + next.warehouseScene.bounds.maxY) / 2,
    };
    next.warehouseScene.nodes.push({
      id, type: "CP", ...center,
      allowedRobotGroups: [edgeGroup],
      positionsByGroup: {}, headings: {}, propertiesByGroup: {},
      waitAllowedByGroup: { [edgeGroup]: true }, capacity: 1,
    });
    setSelectedNodeId(id);
    updateDocument(next);
  };

  const addEdge = () => {
    if (!document || !edgeStartId || !edgeEndId || edgeStartId === edgeEndId) return;
    const start = nodes.find((node) => node.id === edgeStartId);
    const end = nodes.find((node) => node.id === edgeEndId);
    if (!start || !end) return;
    const next = clone(document);
    const id = `${edgeGroup}:edge-design-${String(next.warehouseScene.edges.length + 1).padStart(4, "0")}`;
    next.warehouseScene.edges.push({
      id, name: "手工配置边", startNodeId: start.id, endNodeId: end.id,
      controlPoints: [[start.x, start.y], [start.x, start.y], [end.x, end.y], [end.x, end.y]],
      lengthM: Math.max(0.1, Math.hypot(end.x - start.x, end.y - start.y)),
      motionDirection: 0, moveStyle: 0, maxSpeedMps: null, loadedMaxSpeedMps: null,
      robotGroup: edgeGroup,
    });
    updateDocument(next);
    setEdgeStartId("");
    setEdgeEndId("");
  };

  const addTask = () => {
    if (!document) return;
    const stations = document.warehouseScene.workstations.filter((item) => item.allowedRobotGroups?.length);
    const first = stations[0];
    const second = stations.find((item) => item.nodeId !== first?.nodeId && item.allowedRobotGroups?.some((group: string) => first?.allowedRobotGroups?.includes(group)));
    if (!first || !second) return;
    const group = first.allowedRobotGroups[0] || "fork";
    const next = clone(document);
    const index = next.taskStream.tasks.length + 1;
    next.taskStream.tasks.push({
      taskId: `${next.taskStream.streamId}-manual-${String(index).padStart(4, "0")}`,
      releaseTimeMs: 0, pickupNodeId: first.nodeId, dropoffNodeId: second.nodeId,
      requiredRobotGroup: group, payloadType: group === "jack" ? "shelf" : "pallet",
      payloadId: `${next.taskStream.streamId}-payload-${String(index).padStart(4, "0")}`,
      pickupServiceMs: first.pickupServiceMs, dropoffServiceMs: second.dropoffServiceMs,
      priorityClass: 0, dueTimeMs: null,
    });
    setSelectedTaskIndex(next.taskStream.tasks.length - 1);
    updateDocument(next);
  };

  const updateTask = (index: number, patch: Record<string, unknown>) => {
    if (!document) return;
    const next = clone(document);
    Object.assign(next.taskStream.tasks[index], patch);
    updateDocument(next);
  };

  const createDraft = async () => {
    setBusy("create");
    try {
      const created = await api.createScenarioDraftFromRuntime(scenarioId, packageId);
      await reloadDrafts();
      setSelectedPackageId(created.packageId);
      onNotice(`已从 ${scenarioId} 创建场景草稿`);
    } catch (reason) { onError(reason instanceof Error ? reason.message : "场景草稿创建失败"); }
    finally { setBusy(null); }
  };

  const save = async () => {
    if (!document) return;
    setBusy("save");
    try {
      const summary = await api.updateScenarioDraft(document.packageId, document, Number(document.metadata?.revision || 1));
      updateDocument({ ...document, metadata: { ...(document.metadata || {}), revision: summary.revision } });
      await reloadDrafts();
      onNotice(`草稿已保存，revision ${summary.revision}`);
    } catch (reason) { onError(reason instanceof Error ? reason.message : "场景草稿保存失败"); }
    finally { setBusy(null); }
  };

  const validate = async () => {
    if (!document) return;
    setBusy("validate");
    try { setValidation(await api.validateScenarioDraft(document.packageId)); onNotice("场景包校验已完成"); }
    catch (reason) { onError(reason instanceof Error ? reason.message : "场景校验失败"); }
    finally { setBusy(null); }
  };

  const generateTasks = async () => {
    if (!document) return;
    setBusy("generate");
    try {
      const generation = JSON.parse(generationText) as Record<string, unknown>;
      const summary = await api.generateScenarioTasks(document.packageId, generation, Number(document.metadata?.revision || 1));
      const next = await api.scenarioDraft(document.packageId);
      updateDocument(next);
      await reloadDrafts();
      onNotice(`已生成 ${summary.taskCount} 个任务`);
    } catch (reason) { onError(reason instanceof Error ? reason.message : "任务流生成失败，请检查JSON参数"); }
    finally { setBusy(null); }
  };

  const compile = async () => {
    if (!document) return;
    setBusy("compile");
    try { const result = await api.compileScenarioDraft(document.packageId); onNotice(`场景已编译，生成 ${String((result.paths as Record<string, unknown> | undefined) ? Object.keys(result.paths as Record<string, unknown>).length : 0)} 个仿真资产`); await reloadDrafts(); }
    catch (reason) { onError(reason instanceof Error ? reason.message : "场景编译失败"); }
    finally { setBusy(null); }
  };

  const publish = async () => {
    if (!document) return;
    setBusy("publish");
    try { await api.publishScenarioDraft(document.packageId); setDocument(await api.scenarioDraft(document.packageId)); await reloadDrafts(); onNotice("场景包已发布为不可变仿真版本"); }
    catch (reason) { onError(reason instanceof Error ? reason.message : "场景发布失败"); }
    finally { setBusy(null); }
  };

  const pathForEdge = (edge: DraftEdge) => {
    const points = edge.controlPoints || [];
    if (points.length === 4) return `M ${points[0][0]} ${points[0][1]} C ${points[1][0]} ${points[1][1]}, ${points[2][0]} ${points[2][1]}, ${points[3][0]} ${points[3][1]}`;
    return "";
  };
  const bounds = document?.warehouseScene.bounds || { minX: 0, maxX: 100, minY: 0, maxY: 100 };

  return (
    <div className="designer-layout">
      <aside className="designer-sidebar data-panel">
        <div className="panel-heading"><h2>场景草稿</h2><p>地图、车辆与任务流的可编辑版本</p></div>
        <div className="designer-create">
          <Field label="来源运行场景">
            <Select value={scenarioId} onChange={(_, data) => { setScenarioId(data.value); setPackageId(`${data.value}-designer`); }}>
              {scenarios.map((item) => <option key={item.scenarioId} value={item.scenarioId}>{item.scenarioId}</option>)}
            </Select>
          </Field>
          <Field label="草稿ID"><Input value={packageId} onChange={(_, data) => setPackageId(data.value)} /></Field>
          <Button appearance="primary" icon={<DocumentAdd20Regular />} onClick={() => void createDraft()} disabled={Boolean(busy)}>{busy === "create" ? "创建中" : "从运行场景创建"}</Button>
        </div>
        <div className="designer-draft-list">
          {drafts.map((item) => <button key={item.packageId} className={item.packageId === selectedPackageId ? "draft-row selected-row" : "draft-row"} onClick={() => setSelectedPackageId(item.packageId)}><span>{item.packageId}</span><small>{item.status} · rev {item.revision} · {item.taskCount}任务</small></button>)}
          {!drafts.length && <p className="empty-copy">还没有草稿，从左上方选择运行场景开始。</p>}
        </div>
      </aside>

      <section className="designer-main">
        {!document ? <div className="data-panel designer-empty"><Wrench20Regular /><h2>开始设计一个可复现的仿真场景</h2><p>场景包保存地图、车辆、工作站和任务流，发布前始终经过确定性校验。</p></div> : <>
          <div className="designer-toolbar data-panel">
            <div><strong>{document.warehouseScene.name}</strong><span className="designer-meta">{document.packageId} · revision {String(document.metadata?.revision || 1)} · {document.status}</span></div>
            <div className="designer-actions">
              <Tooltip content="保存当前草稿" relationship="label"><Button icon={<Save20Regular />} onClick={() => void save()} disabled={Boolean(busy) || document.status !== "draft"}>保存</Button></Tooltip>
              <Button icon={<CheckmarkCircle20Regular />} onClick={() => void validate()} disabled={Boolean(busy)}>校验</Button>
              <Button icon={<Play20Regular />} onClick={() => void compile()} disabled={Boolean(busy)}>编译</Button>
              <Button appearance="primary" onClick={() => void publish()} disabled={Boolean(busy) || document.status !== "draft"}>发布仿真版本</Button>
            </div>
          </div>
          <div className="designer-grid">
            <div className="designer-canvas data-panel">
              <div className="panel-heading panel-heading-actions"><div><h2>地图画布</h2><p>拖动节点调整坐标，选择两个节点后添加有向边</p></div><div className="designer-inline-actions"><Button size="small" icon={<Add20Regular />} onClick={addNode} disabled={document.status !== "draft"}>添加安全节点</Button><Select size="small" value={edgeGroup} onChange={(_, data) => setEdgeGroup(data.value as "fork" | "jack")}><option value="fork">叉车边</option><option value="jack">搬运车边</option></Select></div></div>
              <div className="designer-map-stage">
                <svg ref={svgRef} viewBox={`${bounds.minX} ${bounds.minY} ${bounds.maxX - bounds.minX} ${bounds.maxY - bounds.minY}`} onPointerMove={handlePointerMove} onPointerUp={() => setDraggingNodeId(null)} onPointerLeave={() => setDraggingNodeId(null)}>
                  <rect x={bounds.minX} y={bounds.minY} width={bounds.maxX - bounds.minX} height={bounds.maxY - bounds.minY} className="map-floor" />
                  {edges.map((edge) => <path key={edge.id} d={pathForEdge(edge)} className={edge.robotGroup === "fork" ? "designer-edge designer-edge-fork" : "designer-edge designer-edge-jack"}><title>{edge.id}</title></path>)}
                  {nodes.map((node) => <circle key={node.id} cx={node.x} cy={node.y} r={node.id === selectedNodeId ? 1.1 : node.type === "AP" ? 0.55 : 0.38} className={`${node.type === "AP" ? "designer-node-ap" : "designer-node"} ${node.id === selectedNodeId ? "designer-node-selected" : ""}`} onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); setSelectedNodeId(node.id); setDraggingNodeId(node.id); }} onClick={() => setSelectedNodeId(node.id)}><title>{node.id} ({node.type})</title></circle>)}
                </svg>
                <div className="designer-map-stats"><span>{nodes.length}节点</span><span>{edges.length}有向边</span><span>{document.warehouseScene.workstations.length}工作站</span></div>
              </div>
              <div className="designer-edge-form"><Select aria-label="起点" value={edgeStartId} onChange={(_, data) => setEdgeStartId(data.value)}><option value="">选择起点</option>{nodes.map((node) => <option key={node.id} value={node.id}>{node.id}</option>)}</Select><Select aria-label="终点" value={edgeEndId} onChange={(_, data) => setEdgeEndId(data.value)}><option value="">选择终点</option>{nodes.map((node) => <option key={node.id} value={node.id}>{node.id}</option>)}</Select><Button appearance="secondary" onClick={addEdge} disabled={document.status !== "draft" || !edgeStartId || !edgeEndId}>添加有向边</Button></div>
            </div>
            <aside className="designer-inspector">
              <Card className="data-panel inspector-card"><div className="panel-heading"><h2>节点属性</h2><p>{selectedNode?.id || "选择画布节点"}</p></div>{selectedNode ? <div className="inspector-body"><Field label="节点类型"><Select value={selectedNode.type} onChange={(_, data) => updateNode(selectedNode.id, { type: data.value })}><option value="LM">LM</option><option value="AP">AP</option><option value="PP">PP</option><option value="CP">CP</option></Select></Field><div className="two-fields"><Field label="X"><Input type="number" value={String(selectedNode.x)} onChange={(_, data) => updateNode(selectedNode.id, { x: Number(data.value) })} /></Field><Field label="Y"><Input type="number" value={String(selectedNode.y)} onChange={(_, data) => updateNode(selectedNode.id, { y: Number(data.value) })} /></Field></div><Button appearance="subtle" onClick={() => { if (!document) return; const next = clone(document); next.warehouseScene.nodes = next.warehouseScene.nodes.filter((node) => node.id !== selectedNode.id); next.warehouseScene.edges = next.warehouseScene.edges.filter((edge) => edge.startNodeId !== selectedNode.id && edge.endNodeId !== selectedNode.id); updateDocument(next); setSelectedNodeId(null); }} disabled={document.status !== "draft"}>删除节点及关联边</Button></div> : <div className="inspector-body empty-copy">点击地图上的节点查看属性。</div>}</Card>
              <Card className="data-panel inspector-card"><div className="panel-heading"><h2>任务流</h2><p>{document.taskStream.tasks.length}个任务 · 可手工编辑或批量生成</p></div><div className="inspector-body"><div className="task-toolbar"><Button size="small" icon={<Add20Regular />} onClick={addTask} disabled={document.status !== "draft"}>新增任务</Button><Select size="small" value={String(selectedTaskIndex)} onChange={(_, data) => setSelectedTaskIndex(Number(data.value))}>{document.taskStream.tasks.map((task, index) => <option key={task.taskId} value={index}>{task.taskId}</option>)}</Select></div>{document.taskStream.tasks[selectedTaskIndex] && <div className="task-fields"><Field label="优先级"><Input type="number" value={String(document.taskStream.tasks[selectedTaskIndex].priorityClass ?? 0)} onChange={(_, data) => updateTask(selectedTaskIndex, { priorityClass: Number(data.value) })} /></Field><Field label="释放时间(ms)"><Input type="number" value={String(document.taskStream.tasks[selectedTaskIndex].releaseTimeMs ?? 0)} onChange={(_, data) => updateTask(selectedTaskIndex, { releaseTimeMs: Number(data.value) })} /></Field><Field label="取货点"><Input value={String(document.taskStream.tasks[selectedTaskIndex].pickupNodeId || "")} onChange={(_, data) => updateTask(selectedTaskIndex, { pickupNodeId: data.value })} /></Field><Field label="卸货点"><Input value={String(document.taskStream.tasks[selectedTaskIndex].dropoffNodeId || "")} onChange={(_, data) => updateTask(selectedTaskIndex, { dropoffNodeId: data.value })} /></Field></div>}<Textarea value={generationText} onChange={(_, data) => setGenerationText(data.value)} resize="vertical" aria-label="任务生成参数JSON" /><Button appearance="secondary" onClick={() => void generateTasks()} disabled={Boolean(busy) || document.status !== "draft"}>按参数生成任务流</Button></div></Card>
            </aside>
          </div>
          {validation && <div className={`designer-validation ${validation.valid ? "valid" : "invalid"}`}><strong>{validation.valid ? "校验通过" : "校验未通过"}</strong><span>{validation.valid ? `${validation.stats.taskCount || document.taskStream.tasks.length}个任务可进入编译链路` : validation.issues.slice(0, 3).map((item) => `${item.path}: ${item.message}`).join("；")}</span></div>}
        </>}
      </section>
    </div>
  );
}
