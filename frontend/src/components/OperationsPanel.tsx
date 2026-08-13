import { useState } from "react";
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Badge,
  Tab,
  TabList,
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableHeaderCell,
  TableRow,
} from "@fluentui/react-components";
import {
  ClipboardTaskListLtr20Regular,
  History20Regular,
  VehicleTruck20Regular,
} from "@fluentui/react-icons";
import type { AuditEvent, Snapshot } from "../types";

interface OperationsPanelProps {
  snapshot: Snapshot;
  audit: AuditEvent[];
}

type OperationsTab = "tasks" | "vehicles" | "audit";

const eventLabel: Record<string, string> = {
  AGENT_INTENT_PARSED: "意图解析",
  SIMULATION_COMPLETED: "仿真完成",
  SIMULATION_REJECTED: "仿真拒绝",
  APPROVAL_CREATED: "创建审批",
  APPROVAL_DECIDED: "审批决策",
  INTENT_SIMULATION_COMMITTED: "仿真态提交",
};

export function OperationsPanel({ snapshot, audit }: OperationsPanelProps) {
  const [tab, setTab] = useState<OperationsTab>("tasks");
  return (
    <section className="data-panel operations-panel">
      <div className="panel-heading">
        <h2>运营对象与审计</h2>
        <p>所有状态来自当前场景快照，工具调用与人工决策全程留痕</p>
      </div>
      <TabList
        selectedValue={tab}
        onTabSelect={(_, data) => setTab(data.value as OperationsTab)}
        aria-label="运营数据视图"
      >
        <Tab value="tasks" icon={<ClipboardTaskListLtr20Regular />}>任务</Tab>
        <Tab value="vehicles" icon={<VehicleTruck20Regular />}>车辆</Tab>
        <Tab value="audit" icon={<History20Regular />}>审计</Tab>
      </TabList>

      {tab === "tasks" && (
        <div className="table-scroll">
          <Table size="small" aria-label="任务列表">
            <TableHeader><TableRow>
              <TableHeaderCell>任务编号</TableHeaderCell>
              <TableHeaderCell>起点</TableHeaderCell>
              <TableHeaderCell>终点</TableHeaderCell>
              <TableHeaderCell>车型</TableHeaderCell>
              <TableHeaderCell>优先级</TableHeaderCell>
              <TableHeaderCell>状态</TableHeaderCell>
            </TableRow></TableHeader>
            <TableBody>
              {snapshot.tasks.map((task) => (
                <TableRow key={task.taskId}>
                  <TableCell className="mono">{task.taskId}</TableCell>
                  <TableCell className="mono">{task.pickupNodeId}</TableCell>
                  <TableCell className="mono">{task.dropoffNodeId}</TableCell>
                  <TableCell>{task.requiredRobotGroup}</TableCell>
                  <TableCell className="mono">P{task.priorityClass}</TableCell>
                  <TableCell><Badge appearance="tint" color="informative">{task.state}</Badge></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {tab === "vehicles" && (
        <div className="table-scroll">
          <Table size="small" aria-label="车辆列表">
            <TableHeader><TableRow>
              <TableHeaderCell>车辆编号</TableHeaderCell>
              <TableHeaderCell>车型</TableHeaderCell>
              <TableHeaderCell>当前节点</TableHeaderCell>
              <TableHeaderCell>载荷</TableHeaderCell>
              <TableHeaderCell>状态</TableHeaderCell>
            </TableRow></TableHeader>
            <TableBody>
              {snapshot.vehicles.map((vehicle) => (
                <TableRow key={vehicle.vehicleId}>
                  <TableCell className="mono">{vehicle.vehicleId}</TableCell>
                  <TableCell>{vehicle.robotGroup}</TableCell>
                  <TableCell className="mono">{vehicle.currentNodeId}</TableCell>
                  <TableCell>{vehicle.loadState}</TableCell>
                  <TableCell><Badge appearance="tint" color="success">{vehicle.state}</Badge></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {tab === "audit" && (
        <div className="audit-list">
          {audit.map((event) => (
            <Accordion key={event.eventId} collapsible>
              <AccordionItem value={event.eventId}>
                <AccordionHeader>
                  <span className="audit-heading">
                    <strong>{eventLabel[event.eventType] || event.eventType}</strong>
                    <span>{event.actor}</span>
                    <time>{new Date(event.createdAt).toLocaleString("zh-CN", { hour12: false })}</time>
                  </span>
                </AccordionHeader>
                <AccordionPanel>
                  <pre>{JSON.stringify(event.payload, null, 2)}</pre>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          ))}
          {audit.length === 0 && <div className="empty-state">尚无审计事件。</div>}
        </div>
      )}
    </section>
  );
}
