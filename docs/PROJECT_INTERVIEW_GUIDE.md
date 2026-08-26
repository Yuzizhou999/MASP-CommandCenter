# MASP Command Center 项目面试准备指南

这份文档用于熟悉项目实现和准备技术面试。重点不是背诵产品文案，而是能够清楚回答四个问题：

1. 这个项目解决了什么问题？
2. 大模型到底负责什么，为什么不能让它直接调度车辆？
3. 一次用户请求从前端到 MASP 的完整链路是什么？
4. 模型不可靠、服务重启、审批拒绝或仿真失败时，系统如何保证安全和可恢复？

文档中的“当前系统”指本仓库的仿真验证版本。它只向 `simulation` 环境写入意图，不连接真实 WMS、RCS 或车辆控制器。

## 1. 一分钟项目介绍

可以这样介绍：

> 我做的是一个面向多车型智能仓储的调度 Agent。我实现了冻结的 linear v1 和有界 observe-decide-act loop 两种运行时：loop 中 DeepSeek 或本地 Qwen 每轮只选择一个只读工具、请求澄清或提出结构化意图，工具结果和 verifier issues 会进入下一轮决策。服务端负责实体绑定、工具白名单、预算、注入隔离和 MASP 校验，大模型没有权限生成路径、写资源预约或控制车辆。可修复校验问题最多回送两次，权限、审批和过期状态直接阻断。正式评测后 v2 未通过晋级门槛，所以当前默认仍是 linear v1，loop v2 作为候选保留；这也是项目不会为了展示 Agent 概念牺牲默认稳定性的证据。

更短的版本：

> 这是一个“LLM 负责理解，确定性引擎负责决策和执行”的安全闭环调度 Agent。DeepSeek 把自然语言转换为受约束的调度意图，MASP 负责所有路径、资源和冲突计算，R3 高风险操作必须仿真后人工审批。

## 2. 项目定位与边界

### 2.1 解决的问题

仓储调度人员通常需要同时查看车辆、任务、路网、资源占用和安全规则。传统系统要求用户记住大量字段和操作路径，而纯聊天机器人又容易出现以下问题：

- 把不存在的站点、车辆或资源当成真实实体；
- 生成一条看似合理但没有经过连续时间路径规划的路线；
- 在资源冲突或死锁情况下直接给出危险动作；
- 说“已经执行”，但实际上没有可审计的执行记录；
- 模型或网络短暂失败时，整个业务流程不可用；
- 服务重启后丢失正在等待审批的任务。

本项目将自然语言交互和确定性调度引擎组合起来：大模型降低交互门槛，MASP 保留调度正确性和安全控制权。

### 2.2 大模型明确不能做什么

大模型不能：

- 生成或写入车辆路线；
- 直接写资源预约表；
- 解除安全停车或联锁；
- 绕过确定性安全校验；
- 自行补齐缺失的站点、车型或资源 ID；
- 控制真实车辆、WMS、RCS 或现场控制器；
- 把仿真结果描述成真实生产执行。

这些限制不是只写在 Prompt 里，而是由服务端工具白名单、Pydantic 校验、实体权威性复核、MASP 校验、审批策略和仿真环境提交边界共同保证。

### 2.3 当前版本的诚实表述

可以说：

- 支持 DeepSeek API 和本地确定性降级；
- 支持可恢复的目标执行 Agent；
- 支持 MASP 仿真、资源冲突检测和审批闭环；
- 支持证据约束的故障诊断和计划解释；
- 支持群车策略模型提出候选优先级，但候选仍由 MASP 校验；
- 做过统一单动作协议的 QLoRA v2.3 实验，意图层有方向性增益，但闭环门槛未通过，因此没有替换 v1。

不要说：

- DeepSeek 自己保证了调度安全；
- AI 已经控制真实车辆；
- 仿真吞吐就是现场生产收益；
- v2.3 已经稳定或已经上线；
- 系统已经具备企业级身份权限和生产准入。

## 3. 总体架构

```text
React / TypeScript 前端
        |
        | POST /api/v1/agent/runs
        v
FastAPI API 层
        |
        v
AgentRunManager                     持久化：SQLite WAL + append-only events
  |  异步执行、SSE、幂等、取消、超时、恢复
  v
DispatchOrchestrator
  |
  +--> AgentLoopExecutor / AgentProtocol
  |      |-- CALL_TOOL -> observation -> DECIDING 回边
  |      |-- REQUEST_CLARIFICATION（服务端生成问题）
  |      |-- PROPOSE_INTENT -> verifier -> 有界修复
  |
  +--> DeepSeek / local Qwen / deterministic driver
  |      |-- 统一单动作协议
  |      |-- linear v1 兼容模式
  |      |-- 故障解释 / 计划解释
  |      `-- 重试、熔断、Token 和成本统计、确定性降级
  |
  +--> ClarificationResolver        服务端实体解析和多轮澄清
  +--> DispatchAgentTools            只读工具白名单
  +--> AgentMemoryStore              结构化会话记忆
  +--> KnowledgeBase                 SOP 混合检索
  +--> MASP Adapter                  权威世界快照和规则校验
  |
  v
DispatchWorkflowService
  |-- MASP 仿真
  |-- PROCEED / BLOCK 安全门槛
  |-- 仿真关联审批
  `-- simulation-only 提交
```

### 3.1 主要模块

| 模块 | 主要职责 |
|---|---|
| `command_center/api.py` | FastAPI 路由、请求模型转换、异常到 HTTP 状态码的映射 |
| `command_center/provider.py` | DeepSeek/本地 driver、单动作归一化、重试熔断、Fallback、Token 和成本统计 |
| `command_center/agent_protocol.py` | 单动作 Schema、observation、预算和 verifier issue 分类 |
| `command_center/agent_loop.py` | observe-decide-act 回路、工具拒绝观察、修复和安全终局 |
| `command_center/orchestrator.py` | `linear` 与 `loop` 双模式入口及共享安全服务 |
| `command_center/agent_tools.py` | 请求绑定的工具白名单、输入 Schema 和工具执行器 |
| `command_center/agent_runtime.py` | 有界状态机和逐步执行轨迹 |
| `command_center/agent_run_manager.py` | 异步 Agent run、持久化、SSE 事件、审批等待和重启恢复 |
| `command_center/dispatch_workflow.py` | 仿真、推进门槛、审批和仿真提交的共享安全服务 |
| `command_center/clarifications.py` | 从自然语言中解析站点、车型、资源和时长；缺字段时请求补充 |
| `command_center/model_safety.py` | 防止模型改写权威实体和虚构证据 |
| `command_center/engine_adapter.py` | 对 MASP 的统一适配边界 |
| `frontend/src/components/AssistantPanel.tsx` | 调度助手、轨迹、仿真结果、审批和提交状态展示 |
| `command_center/contracts.py` | Pydantic 请求、响应、意图、验证、审批和工作流契约 |

## 4. 一次请求的完整链路

下面以用户输入“创建紧急叉车任务，从 AP1123 运到 AP2121”为例。

### 第 1 步：前端创建目标执行 run

前端不直接调用大模型，而是创建一个 Agent run：

```json
{
  "message": "创建紧急叉车任务，从 AP1123 运到 AP2121",
  "scenarioId": "interactive-multi-fleet",
  "conversationId": "conversation-demo",
  "timeoutSeconds": 30,
  "executionMode": "GOAL_EXECUTION"
}
```

`AgentRunManager.create()` 会：

1. 校验并规范化请求；
2. 根据 `idempotencyKey` 检查是否已有相同请求；
3. 生成 `agent-run-*` ID；
4. 将 run 行和 append-only 事件写入 `data/agent-runs.sqlite3`；
5. 提交到后台线程池；
6. 前端通过查询接口或 SSE 观察状态。

`GOAL_EXECUTION` 和 `ADVISORY` 的区别是：

- `ADVISORY` 只形成建议；高风险意图在原有 Agent 流程中等待人工审批；
- `GOAL_EXECUTION` 会在理解完成后自动运行仿真和推进门槛，低风险目标可以自动提交，高风险目标在仿真后暂停审批。

### 第 2 步：建立有界 Agent 状态机

可选的 `loop` 模式有真实回边；当前默认配置仍使用已通过基线门槛的 `linear` v1：

```text
RECEIVED
  -> PARAMETER_RESOLUTION
  -> PLANNING -> DECIDING
       | CALL_TOOL -> CONTEXT_GATHERING -> OBSERVING --+
       |                                                |
       +----------------------<-------------------------+
       | PROPOSE_INTENT -> INTENT_DRAFTING -> SAFETY_VALIDATION
       |                                      | fixable
       |                                      v
       +------------------------------ REPAIRING
                                              | valid
                                              v
                                         COMPLETED
```

如果字段不足，流程会从 `PARAMETER_RESOLUTION` 转到 `CLARIFICATION_REQUIRED`，不会猜一个默认站点或车型。

每一步都有：

- 序号；
- 状态；
- 标题和详细说明；
- 工具名；
- 是否只读；
- 耗时；
- 成功或失败状态。

运行时分别强制决策、工具、修复、Token、估算成本、延迟和 trace step 预算。任一预算耗尽都进入结构化 `BUDGET_EXCEEDED`，不会抛出未解释的 500。`linear` 模式保留原单向链，专门用于和 loop 做同口径评测。

### 第 3 步：模型逐轮决定一个动作

`decide_agent_action()` 每轮只接受一个动作。模型调用工具后，服务端执行并把结构化 observation 放回下一轮；非法工具、非法参数、多动作和非法 JSON 都作为 rejection observation 返回，不会静默丢弃。当前可被模型选择的工具是：

- `get_world_snapshot`：读取服务端绑定场景的权威世界快照；
- `search_sop`：检索仓储调度、安全和异常处置 SOP；
- `recall_conversation_memory`：读取服务端确认的结构化会话记忆。

`validate_dispatch_intent` 虽然也是 Agent 工具，但 `modelSelectable=false`。它只能由编排器在固定阶段调用，不能由模型决定是否跳过。

循环还有几层确定性保护：

1. 单轮多动作直接拒绝；
2. 工具名、参数和场景绑定都由服务端复核；
3. 提出意图前必须读到权威 `get_world_snapshot`；
4. SOP 正文标为不可信数据，标题和正文先过注入 scanner；
5. 可疑 chunk 隔离并进入 trace，不能进入模型上下文；
6. 模型没有写操作工具定义；
7. 最终必须经过不可由模型选择或跳过的 verifier。

如果 DeepSeek 或本地服务不可用，同一个循环由确定性 driver 驱动，并明确记录 `fallbackUsed`。评测额外要求候选的 `modelDrivenRate=100%`，防止 fallback 托高模型成绩。

### 第 4 步：确定性参数解析和多轮澄清

`ClarificationResolver` 在意图生成前解析业务实体：

- 识别 `fork:AP1123`、`jack:AP1123` 等站点引用；
- 根据显式车型或 MASP 目录解决站点歧义；
- 解析“几分钟”“几秒钟”等封锁时长；
- 识别“共享窄路”“共享通道”等固定资源映射；
- 把已收集的字段写入最小化的澄清存储。

例如用户只说“帮我创建一个紧急任务”，系统会要求补充：

- 取货站点；
- 放货站点；
- 执行车型。

这一层的核心设计是：模型可以理解表达方式，但实体是否存在、是否唯一、属于哪种车型，由服务端目录和 MASP 决定。

### 第 5 步：模型生成结构化调度意图

在参数已解析后，Provider 调用 OpenAI 兼容的：

```text
POST {MODEL_BASE_URL}/chat/completions
```

请求使用：

- `temperature=0`，降低结构化输出的不稳定性；
- DeepSeek 使用 `json_object` 后由服务端解析和校验；
- 本地 loop driver 使用 XGrammar 强制七分支 `AgentAction` JSON Schema，并用 `jsonschema` 终检；
- linear v1 使用 `DispatchIntent`，loop v2 使用单动作 envelope；
- 系统 Prompt 明确禁止生成路径、预约和安全解除；
- 用户内容中包含请求、当前 `worldRevision`、权威参数、检索证据和 Schema。

模型允许形成的意图类型是：

- `QUERY_STATUS`；
- `EXPLAIN_DECISION`；
- `CREATE_TASK`；
- `BLOCK_RESOURCE`；
- `GENERATE_REPORT`。

服务端会覆盖或强制写入：

- `basedOnWorldRevision`；
- `requestedBy`；
- `environment=simulation`。

如果确定性解析器已经给出任务或资源字段，服务端会把这些字段合并回模型结果，并调用 `enforce_intent_authority()` 检查模型没有改写：

- `pickupNodeId`；
- `dropoffNodeId`；
- `requiredRobotGroup`；
- `payloadType`；
- `resourceIds`；
- `startMs`；
- `endMs`。

因此，模型返回一个不存在的站点或尝试把 `fork` 改成 `jack`，都会被拒绝或降级。

### 第 6 步：固定阶段调用安全校验

意图生成完成后，编排器进入 `SAFETY_VALIDATION`，固定调用：

```text
validate_dispatch_intent
```

实际执行由 `MaspAdapter.validate_intent()` 完成。它会检查实体、世界版本、环境、风险等级、资源和任务约束，并给出：

- `valid`；
- `riskLevel`；
- `approvalRequired`；
- 结构化校验问题。

模型不能把这一步改成“通过”，也不能调用完模型后直接提交。安全校验是编排器固定流程的一部分。

### 第 7 步：目标工作流自动仿真

当 `executionMode=GOAL_EXECUTION` 且意图是可执行的 `CREATE_TASK` 或 `BLOCK_RESOURCE` 时，`AgentRunManager` 把已保存的响应交给 `DispatchWorkflowService`。

第一步是 MASP 仿真：

```text
SIMULATE
  -> MASP 路径规划
  -> 连续时间 SIPP
  -> 资源预约
  -> 冲突检测
  -> 完成任务和安全指标
```

仿真结果会持久化到 workflow checkpoint，包含仿真 run ID、任务完成情况、资源冲突和安全字段。服务重启后直接复用这个 checkpoint，不重新调用大模型，也不重复生成意图。

### 第 8 步：确定性推进门槛

`DispatchWorkflowService.recommend()` 根据仿真结果计算四个检查项：

```text
simulationCompleted
conflictFree
allTasksPlanned
simulationOnly
```

全部为真时返回：

```text
PROCEED
```

任意一项为假时返回：

```text
BLOCK
```

推荐理由来自检查结果，而不是来自大模型的自由文本。例如：

- 仿真没有完成；
- 检测到资源预约冲突；
- 仍有任务没有规划；
- 结果不是 simulation-only。

如果是 `BLOCK`，工作流结束为安全阻断，不进入提交步骤。

### 第 9 步：高风险操作仿真后审批

对于 `approvalRequired=true` 的意图，例如通道封闭：

```text
SIMULATE(COMPLETED)
  -> PROCEED
  -> REQUEST_APPROVAL(COMPLETED)
  -> WAITING_APPROVAL
  -> 人工批准 / 拒绝
```

审批请求会绑定：

- `intentId`；
- `approvalId`；
- `simulationRunId`；
- 风险等级；
- 验证结果；
- 请求人和审批人；
- 审批理由。

批准以后，工作流继续从 checkpoint 进入提交；拒绝则将 Agent run 标记为 `REJECTED`，不会提交。

审批等待期间：

- Agent 状态是 `WAITING_APPROVAL`；
- 后台线程阻塞在条件变量上，而不是忙等；
- `_check_control(..., include_deadline=False)` 不让审批等待消耗执行时间；
- 服务重启后保留 approval、simulation 和原始 intent；
- 恢复时复用相同的 `intentId`、仿真 run ID 和审批 ID。

### 第 10 步：只提交到仿真环境

提交前会再次执行确定性校验。高风险意图还必须有：

- 已批准的审批单；
- 与当前意图匹配的 `intentId`；
- 通过门槛的仿真 run；
- 未过期且匹配当前世界版本的提交条件。

最终写入的是本地仿真意图存储和审计日志，不会调用真实设备控制接口。

## 5. 大模型部分的详细实现

### 5.1 Provider 为什么单独封装

`DeepSeekProvider` 不直接散落在业务代码里，而是统一处理：

- API 地址和模型配置；
- API Key；
- OpenAI 兼容请求格式；
- Retry 和 Circuit Breaker；
- JSON 解析和 Pydantic 校验；
- fallback 标记；
- 每个 run 的 Token、成本和调用次数。

这样编排器只关心“得到一个工具计划”或“得到一个解析结果”，不需要知道网络调用细节，也便于未来替换成其他 OpenAI-compatible provider。

### 5.2 重试和熔断

`_post()` 的策略是：

1. 每次调用前检查当前 run 是否被取消或超时；
2. 对请求进行计数；
3. 对瞬时错误最多重试配置的次数；
4. 对 `408`、`429` 和 `5xx` 允许重试；
5. 对明显的 `4xx` 参数错误提前终止；
6. 连续失败达到阈值后打开熔断器；
7. 熔断窗口结束后允许再次尝试。

这避免了模型服务异常时每个请求无限重试，也避免把一个已取消的用户任务继续发送到外部服务。

### 5.3 Fallback 分层

项目不是“调用失败就返回空结果”，而是按能力分层降级：

| 能力 | DeepSeek 正常时 | 失败时 |
|---|---|---|
| 上下文规划 | 模型选择只读工具 | 固定读取世界快照、记忆和 SOP |
| 意图解析 | JSON 结构化输出 | 本地规则解析器 |
| 故障诊断 | 基于证据生成解释 | 确定性诊断报告 |
| 计划解释 | 基于 findings 组织叙述 | 返回确定性解释 |

每次降级都会更新：

- `fallbackUsed`；
- 模型名称，例如 `deepseek-chat:fallback`；
- Provider telemetry；
- Agent 轨迹；
- 审计日志。

所以前端和运维人员可以知道“这次是模型结果”还是“这次是规则降级”，不会把两者混为一谈。

### 5.4 为什么使用 JSON + Schema

自然语言回复不适合直接进入调度系统。这里使用 JSON 不是为了让模型获得更多权限，而是为了让模型输出变成一个可验证的候选对象：

```text
模型文本
  -> JSON 解析
  -> DispatchIntent Pydantic 校验
  -> 权威实体一致性校验
  -> MASP 规则校验
  -> 风险策略
```

任何一层失败都不能直接执行。`extra=forbid` 可以拒绝契约中不存在的额外字段，降低模型偷偷携带未定义动作参数的风险。

XGrammar 解决的是结构合法性，不是业务正确性。v2.3 实测 JSON 合法率达到 100%，但原始 Schema 有效率只有 86%，原因包括模型选择了当前评测不期望的合法动作。因此受约束解码不能替代工具轨迹 gold、权威实体检查和 MASP verifier。

### 5.5 为什么 Prompt 不是唯一安全机制

Prompt 只能表达模型应该怎么做，不能作为最终权限系统。项目使用多层防线：

1. Prompt 明确角色和禁止事项；
2. 工具目录只暴露只读工具；
3. 工具输入使用 Pydantic Schema；
4. 工具执行器服务端绑定场景，模型不能切换场景；
5. 参数解析器确认实体；
6. `model_safety.py` 检查模型没有改写实体；
7. MASP 再做世界、资源、冲突和版本校验；
8. R3 操作强制仿真和人工审批；
9. 提交接口只允许 simulation 环境。

面试时可以明确说：

> Prompt 是意图约束，不是权限边界。真正的权限边界在服务端工具白名单、Schema、确定性校验和提交接口。

## 6. Agent、LLM 和 MASP 的职责边界

这是面试中最重要的设计问题。

| 问题 | 负责组件 | 原因 |
|---|---|---|
| 用户说的是插单还是封路 | DeepSeek + 确定性解析器 | 需要自然语言理解，但实体不能靠猜 |
| 站点和车型是否存在 | MASP 目录 / `ClarificationResolver` | 需要权威数据 |
| 调用哪些只读上下文工具 | DeepSeek 可规划，服务端过滤 | 允许模型参与，但不能越权 |
| 是否必须读取世界快照 | 编排器固定保证 | 不能由模型选择跳过 |
| 生成结构化意图 | DeepSeek | 适合语义分类和字段组织 |
| 意图是否合法 | MASP `validate_intent` | 需要规则、版本和拓扑 |
| 路线怎么走 | MASP | 需要 SIPP、连续时间和运动学约束 |
| 资源是否冲突 | MASP | 需要预约和冲突计算 |
| 是否推进 | `DispatchWorkflowService` | 由固定安全检查项决定 |
| 高风险是否执行 | 人工审批 + 服务端状态 | 需要责任链和可追溯性 |
| 车辆是否真的被控制 | 当前系统不做 | 仿真验证边界 |

### 6.1 为什么不让 LLM 直接生成路线

因为路线不是语言问题，而是约束求解问题。LLM 可能生成：

- 不存在的节点；
- 不符合车辆尺寸的边；
- 忽略时间窗的路径；
- 与其他车辆重叠的资源预约；
- 没有经过安全停车或死锁检查的动作。

MASP 已经具备确定性的路径、资源和冲突计算。正确的架构是让 LLM 生成“我要创建什么意图”，而不是生成“车辆经过哪些边”。

### 6.2 RL 策略模型和 LLM 的区别

项目中还有 MASP Actor-Critic/PPO 群车优先级策略，但它不是 DeepSeek，也不是 Agent 的语言模型：

- LLM 负责自然语言理解和解释；
- PPO 模型负责提出候选车辆/任务优先级顺序；
- MASP 仍负责路径、预约、SIPP 和安全校验；
- checkpoint、版本、动作模式和候选合法性由服务端校验；
- 学习策略失败时自动回退规则基线。

面试中不要把这两个模型混称为“一个大模型”。它们输入、输出、部署方式和安全责任都不同。

## 7. 可恢复 Agent Runtime

### 7.1 为什么要异步 run

一次完整目标执行可能包含：

- 外部模型调用；
- 多个只读工具；
- MASP 仿真；
- 人工等待审批。

如果一直占用 HTTP 请求，容易超时，也不适合服务重启。因此 API 返回 `runId`，后台线程执行，前端用轮询或 SSE 获取状态。

### 7.2 持久化内容

`data/agent-runs.sqlite3` 保存 run 当前文档和 append-only 事件表，并启用 WAL：

- 原始请求；
- 当前状态；
- attempt 和 recovered 标记；
- trace steps；
- ChatResponse；
- provider usage；
- approval checkpoint；
- workflow checkpoint；
- 评价结果；
- 事件时间线；
- deadline、startedAt、completedAt。

目标执行尤其重要的是保存：

```text
原始 intent
simulation.runId
recommendation
approvalRequest.approvalId
commitment.commitId
```

恢复时复用这些 ID，避免重复仿真、重复审批或重复提交。

### 7.3 幂等策略

创建 run 时，如果同一个 `Idempotency-Key` 再次提交完全相同的请求，返回原 run；如果请求内容不同，则拒绝复用这个 key。

仿真、审批和提交服务也使用已有记录做复用判断：

- 已有 simulation checkpoint，不重新生成；
- 已有 approval request，不创建第二张审批单；
- 已有 commitment，不重复写入。

这样可以应对浏览器重试、网络重试和服务重启。

### 7.4 审批等待为什么不超时

执行时限只约束 Agent 的主动计算时间，不应把人工审批等待算成模型执行耗时。否则主管晚几分钟审批，系统会把一个尚未执行的任务标记为超时。

实现上有两层：

- 等待循环调用 `_check_control(include_deadline=False)`；
- `resume()` 根据暂停时长延长 deadline。

服务重启恢复也跳过 `WAITING_APPROVAL` 的截止时间判断。

## 8. 异常和安全场景

### 8.1 DeepSeek 不可用

处理顺序：

1. API 未配置：直接使用确定性策略；
2. 网络错误、超时或可重试状态：有限重试；
3. 连续失败：熔断；
4. `linear` 模式的意图 JSON 缺字段或 Schema 不通过：fallback；
5. `loop` 模式的非法单动作：作为 rejection observation 返回下一轮，不用 fallback 掩盖模型错误；
6. 重复非法动作最终由决策或 step 预算终止，前端和审计保留原因。

### 8.2 参数不完整

不使用默认实体。系统返回 `CLARIFICATION_REQUIRED`，并持久化已收集参数。例如已识别取货点但缺少车型时，只要求补车型，不丢掉已有字段。

### 8.3 模型产生虚构实体

实体必须来自确定性解析器和 MASP 目录。模型返回后，`enforce_intent_authority()` 逐字段比较。如果模型改写权威字段，抛出边界错误并走降级或失败路径。

### 8.4 仿真安全门槛失败

只要仿真未完成、存在资源冲突、有未规划任务或不是仿真模式，推荐就是 `BLOCK`。工作流不创建提交记录。

### 8.5 人工审批拒绝

审批记录写入拒绝决定，Agent run 进入 `REJECTED`，workflow 进入 `BLOCKED`，不会调用 commit。

### 8.6 仿真、审批或提交抛异常

当前运行中的 workflow step 会标记为 `FAILED`，工作流进入 `BLOCKED`，Agent run 记录错误信息和终止事件。这样前端不会显示一个已经失败但仍处于 `RUNNING` 的步骤。

### 8.7 世界版本变化

意图会记录 `basedOnWorldRevision`。仿真和提交阶段会重新检查当前世界版本。世界发生变化时，旧意图不能直接提交，需要重新校验或重新仿真。

## 9. 异常诊断和证据约束

除了调度意图，Provider 还支持两类解释：

### 9.1 故障诊断

模型输入包括：

- Incident 记录；
- 真实 `EV-*` 证据；
- 确定性 findings；
- 当前事件允许的动作；
- 输出 Schema。

模型不能补充不存在的遥测、车辆、任务或根因。输出还会经过 `diagnosis_violation()`：

- 根因和建议引用的证据 ID 必须存在；
- 受影响车辆和任务必须属于 Incident；
- 动作必须在 allowed actions 中；
- 建议必须是 R3、要求仿真和审批。

### 9.2 计划解释

计划解释只组织 MASP 已生成的 findings 和 evidence，不重新计算路线、时间或指标。每条结论需要引用有效证据，并区分 `FACT` 和 `INFERENCE`。

面试时可以回答：

> 解释模块不是让模型重新推理一遍调度，而是让模型把确定性结果翻译成业务语言。事实来源仍是 MASP 原始运行文件。

## 10. 前端如何消费 Agent 结果

前端主要关注三层状态：

1. Agent run 状态：`QUEUED`、`RUNNING`、`WAITING_APPROVAL`、`COMPLETED`、`FAILED` 等；
2. workflow phase：`SIMULATING`、`WAITING_APPROVAL`、`COMMITTING`、`COMPLETED`、`BLOCKED`；
3. 每个 workflow step：`SIMULATE`、`REQUEST_APPROVAL`、`COMMIT`。

在 `GOAL_EXECUTION` 下，手动仿真按钮会隐藏，防止用户在 Agent 已经自动仿真后重复提交一份方案。高风险界面会明确显示“仿真后审批”，而不是让用户误以为审批发生在仿真之前。

## 11. 如何运行和演示

### 11.1 启动

```powershell
cd E:\project\MASP-CommandCenter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
.\scripts\start.ps1
```

浏览器访问：

```text
http://127.0.0.1:8877
```

建议测试环境显式使用锁定 MASP：

```powershell
$env:MASP_TEST_ENGINE_ROOT='E:\project\MASP-locked'
.\.venv\Scripts\python.exe -m pytest -q
```

### 11.2 低风险演示

在调度助手输入：

```text
创建紧急叉车任务，从 AP1123 运到 AP2121
```

预期链路：

```text
结构化意图 CREATE_TASK
-> 风险校验
-> SIMULATE
-> PROCEED
-> COMMIT
-> 目标已完成
```

### 11.3 高风险演示

输入：

```text
共享窄路需要检修，请封闭三分钟并评估影响
```

预期链路：

```text
结构化意图 BLOCK_RESOURCE
-> 风险 R3_HIGH
-> SIMULATE
-> PROCEED
-> REQUEST_APPROVAL
-> WAITING_APPROVAL
-> 人工批准
-> COMMIT
```

如果仿真有冲突或未规划任务，预期是 `BLOCK`，不会出现提交。

## 12. 面试高频问题与建议回答

### Q1：你这个项目中的 Agent 和普通聊天机器人有什么区别？

建议回答：

> 它不是一次请求一次回复，也不是先规划完再顺序执行的流水线。loop 模式下模型每轮只能做一个动作，看到工具结果或 verifier issue 后再决定下一步；非法调用会成为 observation，fixable issue 可以有界修复。终局仍由确定性 verifier 决定，运行还有独立的决策、工具、修复、Token、成本、延迟和步数预算。linear v1 被保留为对照，而不是删掉历史基线。

### Q2：大模型在这里究竟做了什么？

建议回答：

> 大模型主要做策略选择和语义组织：决定是否继续读取只读上下文、是否需要澄清、何时提出结构化意图，以及基于真实证据生成解释。澄清问题文本、权威实体、路径规划、资源预约、冲突判断、verifier 和设备控制都不交给模型。

### Q3：为什么不用 LLM 直接生成路线？

建议回答：

> 路线是带有拓扑、时间、车辆尺寸、资源预约和冲突约束的求解问题，不适合依赖概率性文本输出。项目已经有 MASP 的 SIPP、预约和冲突计算能力，所以 LLM 只生成“要做什么”的意图，MASP 决定“能不能做”和“具体怎么走”。

### Q4：如果模型返回了不存在的站点怎么办？

建议回答：

> 站点首先由确定性解析器和 MASP 目录解析，模型收到 authoritativeParameters。返回后，服务端会逐字段检查模型是否改写了取货点、放货点、车型或资源。如果字段不存在、歧义或被改写，就进入澄清、拒绝或 fallback，不会直接执行。

### Q5：Prompt Injection 怎么处理？

建议回答：

> 我没有把安全寄托在 Prompt 或 1.5B 的拒答能力上。直接注入在模型调用前扫描；检索标题和正文按不可信数据分隔，命中注入模式的 chunk 会隔离并留 trace。之后还有只读工具白名单、Schema、权威实体覆盖、MASP verifier 和审批。评测口径是系统级攻击是否产生可提交/可执行越权动作，冻结攻击集上要求成功率为 0。

### Q6：为什么要把 `validate_dispatch_intent` 设置为不可由模型选择？

建议回答：

> 这是一个必须执行的安全步骤，不应该由模型决定是否调用。模型可以规划只读上下文，但意图校验由编排器在固定状态 `SAFETY_VALIDATION` 中调用。这样不会因为模型漏掉一个 tool call 而跳过安全门槛。

### Q7：DeepSeek 不可用时系统怎么办？

建议回答：

> Provider 对调用做了有限重试、熔断和 telemetry。如果模型没有配置、网络失败、返回非 JSON 或 Schema 不通过，工具规划会退回固定只读计划，意图解析会退回本地确定性解析器，诊断和解释也有确定性版本。结果中会带 `fallbackUsed`、模型名称和审计记录，前端不会把降级结果伪装成模型正常输出。

### Q8：为什么 `temperature=0` 还不能保证确定性？

建议回答：

> `temperature=0` 只能降低采样随机性，不能消除网络、服务版本、模型实现或输出格式问题。所以项目仍然使用 JSON Schema、字段权威性检查、MASP 校验和规则 fallback。最终安全性不依赖模型是否完全确定。

### Q9：模型输出为什么要过 Pydantic？

建议回答：

> Pydantic 把自由文本候选转换成结构化契约，检查枚举、必填字段、字段类型和额外字段。通过后还要做业务层的实体一致性和 MASP 规则校验。Schema 是第一层结构边界，不是全部安全机制。

### Q10：一次目标执行的状态机是什么？

建议回答：

> loop 理解阶段是 `RECEIVED -> PARAMETER_RESOLUTION -> PLANNING -> DECIDING`。`CALL_TOOL` 进入 `CONTEXT_GATHERING -> OBSERVING -> DECIDING` 回边；`PROPOSE_INTENT` 进入 `INTENT_DRAFTING -> SAFETY_VALIDATION`，fixable issue 走 `REPAIRING -> DECIDING`，其余进入 `BLOCKED`。参数不足是 `CLARIFICATION_REQUIRED`，预算耗尽是 `BUDGET_EXCEEDED`。目标工作流再增加 `SIMULATING -> WAITING_APPROVAL/COMMITTING -> COMPLETED/BLOCKED`。

### Q11：为什么需要 `GOAL_EXECUTION`，原来的 chat 不够吗？

建议回答：

> 原来的 chat 只完成意图理解和建议，不能表达“理解后自动完成仿真并在安全范围内提交”的完整目标。`GOAL_EXECUTION` 将后续动作显式建模为可恢复工作流，同时保留原有同步 chat API，避免破坏现有调用方。

### Q12：如何避免重复仿真和重复提交？

建议回答：

> 创建 run 使用幂等 key；目标工作流保存 simulation、approval 和 commitment checkpoint。恢复时优先复用原始响应和这些 ID。已有仿真不重新运行，已有审批不重复创建，已有 commit 不重复写入。

### Q13：服务重启时正在审批怎么办？

建议回答：

> `WAITING_APPROVAL` 的 run、intent、simulationRunId 和 approvalId 都持久化。重启恢复时不会把审批等待按主动执行超时处理。主管继续审批后，Agent 复用同一份意图和仿真结果，从提交步骤继续，不重新调用模型。

### Q14：为什么审批要放在仿真之后？

建议回答：

> 审批人需要看到具体影响，包括任务完成数、资源冲突、受影响资源和安全检查结果。仿真前只能审批一个抽象意图，无法判断真实后果。当前高风险流程是先仿真、再生成确定性推进建议、最后人工审批。

### Q15：`PROCEED/BLOCK` 是模型生成的吗？

建议回答：

> 不是。它由 `DispatchWorkflowService.recommend()` 根据仿真状态、资源冲突、未规划任务和 simulation-only 标记计算。模型可以解释结果，但不能把 `BLOCK` 改成 `PROCEED`。

### Q16：如何保证高风险操作不能绕过审批？

建议回答：

> 校验结果包含 `approvalRequired`。提交服务会再次校验意图；高风险意图如果没有审批 ID、审批状态不对、意图 ID 不匹配或仿真证据不通过，直接拒绝。前端按钮不是安全边界，真正的检查在后端 commit 路径。

### Q17：为什么要保存 world revision？

建议回答：

> 仿真是基于某个世界快照做的。如果车辆、任务或资源状态已经变化，旧仿真可能不再适用。因此意图和仿真都关联 `worldRevision`，提交前需要重新确认版本，避免基于过期状态执行。

### Q18：项目里的结构化记忆是什么？

建议回答：

> 不是保存模型的完整自由文本，而是保存已确认实体、最近意图、风险和工具轨迹。记忆只能帮助下一轮理解“刚才那个任务”，不能替代最新 MASP 世界快照。工具说明也明确要求模型不能把记忆当成最新状态。

### Q19：故障诊断如何避免幻觉？

建议回答：

> 模型输入只包含 Incident、真实 `EV-*` evidence、确定性 findings 和 allowed actions。输出中的根因和建议必须引用存在的 evidence ID，车辆和任务必须属于事故记录，动作必须在允许集合中。任一校验失败，整份报告降级为确定性诊断，不部分采信。

### Q20：Agent 运行为什么要有超时和最大步数？

建议回答：

> 只有一个最大步数不够，因为模型可能用很少 trace step 消耗大量 token 或成本。这里分别限制决策次数、工具调用、修复次数、Token、估算美元成本、延迟和 trace step；超限统一进入可解释的预算终局。审批等待单独暂停 deadline，避免把人工等待误判成 Agent 失控。

### Q21：项目里的 PPO 模型和 DeepSeek 有什么关系？

建议回答：

> 两者职责不同。DeepSeek 是业务大模型，负责自然语言理解和解释；PPO 是群车优先级策略模型，负责提出车辆和任务候选顺序。PPO 只输出候选，仍要经过 MASP 候选评估、SIPP、资源预约和 Top-K guardian。PPO 出问题会回退规则基线，不能直接写车辆路径。

### Q22：当前实现的主要生产化不足是什么？

建议回答：

> 当前仍是单机仿真验证版本。Agent run 已从整份 JSON 重写迁移到 SQLite WAL 和 append-only event；50 并发、12 worker 的本机对照中，JSON 基线为 1.555 runs/s，SQLite WAL 为 6.039 runs/s，总耗时加速 3.884 倍。但生产仍需要 PostgreSQL、分布式任务队列和跨实例租约。企业身份认证、RBAC、CSRF、速率限制、密钥托管、日志脱敏、传输加密和依赖扫描也还没有完成。

### Q23：如果让你继续优化，你会先做什么？

建议回答：

> Agent 模型侧会先扩大真正未见过的轨迹集，并把单动作监督、移除裸意图、受约束解码和恢复样本分别做消融，再决定是否投入三 seed 稳定性训练。系统侧再把 SQLite 迁移到 PostgreSQL 和队列 worker。模型或 prompt 更新必须先做轨迹 replay diff，再过 v1 无退化与系统安全双门槛。

### Q24：v2.3 解决了什么，为什么仍没有上线？

建议回答：

> v2.3 把 1716 条训练样本统一成单个 AgentAction 协议，移除了裸意图监督，并保证 2048 token 下零截断。与控制组在同一 XGrammar 约束下对比，意图 Macro F1 从 0.7457 提升到 0.8294，槽位匹配从 0.60 提升到 0.90；但 18 条闭环 holdout 的目标完成率仍是 0.7222，工具 recall 还从 0.9167 降到 0.8611。独立 claim 审查只给出 partial，资格脚本输出 KEEP_V1，所以我停止了后续 seed，保留 v1 为默认。这说明结构化输出变稳不等于工具策略和闭环目标已经变稳。

## 13. 面试时容易说错的地方

### 不要把“理解”说成“执行”

正确说法：

> 模型形成了结构化调度意图，服务端经过校验后运行仿真。

不要说：

> DeepSeek 直接给车辆下发了路线。

### 不要把“仿真成功”说成“生产成功”

正确说法：

> MASP 仿真在当前场景快照下完成，资源冲突检查通过，并提交为仿真意图。

不要说：

> 已经在仓库现场完成了任务。

### 不要把“规则降级”说成“模型准确率”

正确说法：

> 当模型不可用或输出不合规时，系统切换到确定性解析器，并通过 `fallbackUsed` 标记。

不要说：

> 即使 DeepSeek 挂了，模型效果仍然一样。

### 不要把 PPO 和 LLM 混为一谈

正确说法：

> PPO 只提出候选优先级；DeepSeek 只做业务语言理解和解释；MASP 负责最终确定性调度。

## 14. 复习顺序

建议按以下顺序读代码：

1. [README.md](../README.md)：项目边界、启动方式和接口总览；
2. [contracts.py](../command_center/contracts.py)：先理解所有数据契约；
3. [orchestrator.py](../command_center/orchestrator.py)：理解一次同步 Agent 请求；
4. [provider.py](../command_center/provider.py)：理解 DeepSeek、Tool Calling 和 fallback；
5. [agent_tools.py](../command_center/agent_tools.py)：理解工具权限；
6. [clarifications.py](../command_center/clarifications.py)：理解实体解析和澄清；
7. [agent_runtime.py](../command_center/agent_runtime.py)：理解有界状态机；
8. [agent_run_manager.py](../command_center/agent_run_manager.py)：理解异步、审批和恢复；
9. [dispatch_workflow.py](../command_center/dispatch_workflow.py)：理解仿真、门槛和提交；
10. [model_safety.py](../command_center/model_safety.py)：理解模型边界和证据校验；
11. [MODEL_CARD.md](MODEL_CARD.md) 和 [SECURITY_AND_RIGHTS.md](SECURITY_AND_RIGHTS.md)：理解模型限制和对外表述边界；
12. [test_agent_run_manager.py](../tests/test_agent_run_manager.py)：用测试复习低风险、高风险、恢复和失败路径。

## 15. 最后记忆版

如果面试时间很短，至少记住这五句话：

1. **LLM 只负责理解和解释，MASP 负责路径、资源、冲突和安全计算。**
2. **所有模型输出都要经过 Schema、实体权威性校验和 MASP 确定性校验。**
3. **Agent 是有界、可观测、可恢复的状态机，不是一次性聊天请求。**
4. **高风险操作先仿真，再由确定性门槛给出 `PROCEED/BLOCK`，最后人工审批。**
5. **当前系统只提交到 simulation，不连接真实设备；生产化还需要身份、权限、数据库和合规建设。**
