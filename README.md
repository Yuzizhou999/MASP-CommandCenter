# 保利智仓·灵枢

[![CI](https://github.com/Yuzizhou999/MASP-CommandCenter/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuzizhou999/MASP-CommandCenter/actions/workflows/ci.yml)

面向多车型智能仓储的 AI 调度指挥中心。系统使用 DeepSeek 或本地 Qwen 微调模型将自然语言请求转换为结构化调度意图，再由 MASP 数字孪生完成路径规划、资源预约、安全校验、What-if 仿真和指标计算。

当前版本是仿真验证系统，只允许在 `simulation` 环境提交意图，不连接真实 WMS、RCS 或车辆控制器。

## 核心能力

- 自然语言紧急插单、通道封锁、状态查询和报告生成；
- 单一 v2.3 本地模型服务驱动有界 observe-decide-act `loop`，工具结果会进入下一轮模型决策；
- DeepSeek 原生 Tool Calling、本地 Qwen 单动作 JSON 协议和确定性 fallback 共用同一执行引擎；
- MASP verifier 可修复问题最多回送模型两次，越权、审批、过期版本和未知问题直接阻断；
- 服务端工具白名单，模型只能选择只读上下文工具，安全校验不可跳过；
- 检索内容使用不可信边界、注入扫描和隔离记录，最终仍由权威实体覆盖与 MASP 校验兜底；
- BM25 与字符特征向量混合检索，证据包含稳定 chunk ID、相关度和检索方法；
- 结构化会话记忆，只保存已确认实体、最近意图、风险和工具轨迹；
- Agent 完成率、工具规划率、降级率、P95 延迟和工具分布观测；
- 可恢复异步 Agent run、SSE 实时轨迹、幂等请求、取消/超时和人工审批检查点；
- `GOAL_EXECUTION` 目标执行模式：自动运行 MASP 仿真，由确定性安全门槛给出 `PROCEED/BLOCK` 建议，R3 操作在仿真后暂停等待人工审批；
- DeepSeek 重试与熔断、Token/成本统计和逐条轨迹评测；
- 缺失站点、车型或资源时进入多轮澄清，不使用默认实体补齐；
- DeepSeek API 调用，未配置密钥或 API 异常时自动使用确定性本地解析；
- 基于 Qwen2.5-1.5B-Instruct 的 4-bit 多轮 QLoRA、版本登记、本地兼容 API、轨迹回放和离线晋级评测；
- 可从场景包自动生成或在画布手工配置路网、工位、车辆和任务流；
- 多车型车辆轨迹、任务路径和资源封锁时窗回放；
- 可配置车辆规模、任务负载、车型、策略和随机种子的确定性仿真；
- 候选方案同口径比较与可追溯推荐理由；
- R0-R4 风险分级，R3 高风险操作强制人工审批；
- 世界版本复核、仿真态提交、审计日志和班次报告；
- SOP 知识检索和模型决策依据展示。
- 一键注入紧急任务、车辆故障、工位停用、通道封闭和等待环事件；
- 工位与节点资源冻结、受影响任务识别、等待恢复和暂停任务 What-if；
- MASP 等待图循环检测、受控倒退恢复、不可恢复死锁安全停车；
- 车辆故障、工位停用和死锁的 `EV-*` 证据诊断及 R3 处置审批；
- Actor-Critic/PPO 群车优先级策略接入、Top-K guardian、安全降级和逐轮候选证据。
- 多种子评测矩阵、均值/标准差/95% 置信区间、安全门槛和失败案例留档；
- 运行证据脱敏导出、固定数据划分、质量扫描和版本化资产包。
- 任务分配、等待、路线筛选和策略回退的 `PE-*` 规划证据解释。

## 仓库边界

本项目与 MASP 是两个并列、独立的仓库：

```text
E:\project\MASP                 MASP 核心调度与数字孪生引擎
E:\project\MASP-CommandCenter   灵枢应用、智能体、治理和前端
```

所有 MASP 调用都集中在 `command_center/engine_adapter.py`。场景包编译和任务流生成扩展位于 `command_center/masp/`，由本仓库独立维护。`engine.lock.json` 固定允许使用的 MASP 提交，开发环境可显式允许脏工作区，生产环境必须提交匹配且工作区干净。

## 技术架构

| 层级 | 实现 |
|---|---|
| 交互层 | React 19、TypeScript、Fluent UI、Vite |
| 应用服务 | FastAPI、Pydantic v2、JSON API |
| 智能体 | observe-decide-act 回路、单动作协议、DeepSeek Tool Calling、Qwen QLoRA、Pydantic 强类型工具、确定性 verifier |
| 治理 | 风险分级、审批、世界版本、审计、仿真态提交 |
| 数字孪生 | MASP 在线调度、路径规划、资源预约、事件回放 |
| 数据与评测 | 混合知识检索、结构化记忆、Agent 观测、矩阵评测、安全门槛、脱敏质检 |
| 存储 | SQLite WAL Agent run 与 append-only event、JSONL 审计，仿真结果按 run 和 benchmark 独立落盘 |

大模型只负责理解和解释。路径、资源预约、冲突判断、方案指标全部来自 MASP 确定性引擎。故障诊断中的每条根因与建议必须引用真实 `EV-*` 证据，虚构证据、车辆、任务或越权动作会使整份 AI 报告降级为规则诊断。

车端策略模型只输出任务与车辆候选顺序，不直接生成轨迹。每个学习候选仍须经过 MASP 候选评估、连续时间 SIPP、资源预约和 Top-K guardian 校验；权重缺失、版本不兼容、推理失败或安全评分不占优时，运行自动切回确定性规则候选并记录原因。

## 快速启动

环境要求：Python 3.11 及以上、Node.js 20 及以上、npm、Git，以及并列目录中的 MASP 仓库。

```powershell
cd E:\project\MASP-CommandCenter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
.\scripts\start.ps1
```

`start.ps1` 会启动唯一的 `masp-agent-lora-v2.3` XGrammar 模型服务和 loop 后端。浏览器访问 [http://127.0.0.1:8877](http://127.0.0.1:8877)。首次启动会安装前端依赖并构建生产资源。之后可使用：

```powershell
.\scripts\start.ps1 -SkipBuild
```

演示只使用一个模型端口 `8000` 和一个应用端口 `8877`，不再启动 v1/v2 并排服务。旧模型的对照数据只保留在 `results/`，用于解释模型选择过程，不进入运行路径。

现场网络不稳定或不准备调用 DeepSeek 时，可显式使用离线降级：

```powershell
.\scripts\start.ps1 -SkipBuild -Offline
```

完整检查：

```powershell
.\scripts\check.ps1
```

## 配置 DeepSeek

编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT_SECONDS=30
```

可以直接调用 DeepSeek 官方 API。密钥只由 FastAPI 后端读取，绝不进入浏览器资源。若未配置密钥、请求超时、返回非 JSON 或 Schema 校验失败，系统自动切换到本地解析器，并在界面和审计日志中明确标注降级状态。

## 微调与使用本地意图模型

项目提供从锁定 MASP 场景生成数据、QLoRA 训练、模型卡登记、OpenAI-compatible 本地服务和 holdout 评测的完整链路。默认基座为 `Qwen/Qwen2.5-1.5B-Instruct`，适配 8GB 显存单卡。

当前只保留 `masp-agent-lora-v2.3` adapter。它学习 `CALL_TOOL`、`REQUEST_CLARIFICATION`、`PROPOSE_INTENT` 单动作协议，并通过唯一的本地 OpenAI-compatible 服务接入 loop runtime。历史门禁仍记录为 `KEEP_V1`，因此单模型演示不等价于生产晋级结论；路径、预约、校验、仿真、审批和车辆控制始终不交给模型。完整训练过程和评测限制见 [大模型微调指南](docs/LLM_FINETUNING.md)。

运行模式与预算可通过 `.env` 配置：

```dotenv
AGENT_RUNTIME_MODE=loop
AGENT_MAX_DECISIONS=8
AGENT_MAX_TOOL_CALLS=6
AGENT_MAX_REPAIR_ATTEMPTS=2
AGENT_MAX_TOTAL_TOKENS=8192
AGENT_MAX_ESTIMATED_COST_USD=0.25
AGENT_MAX_LATENCY_MS=30000
AGENT_MAX_STEPS=48
```

## 配置群车策略模型

学习策略使用服务端登记的 MASP Actor-Critic/PPO checkpoint，浏览器不能提交文件路径。源码环境启用模型推理时先安装额外依赖：

```powershell
pip install -r requirements-agent.txt
```

仓库内置的 `models/ppo-priority-v1.pt` 会被自动发现；只有替换模型或版本信息时才需要在 `.env` 中显式配置：

```dotenv
MASP_AGENT_MODEL_ID=masp-ppo-priority
MASP_AGENT_MODEL_VERSION=1.0.0
MASP_AGENT_CHECKPOINT=models/ppo-priority-v1.pt
MASP_AGENT_DEVICE=cpu
MASP_AGENT_TORCH_THREADS=1
```

checkpoint 必须带有 MASP 定义的版本、观测、动作、奖励和优先级前缀元数据。运行前校验失败时不会加载学习策略，界面会显示规则基线及具体降级原因。

使用锁定 MASP 引擎重新训练可执行：

```powershell
.\scripts\train-agent-policy.ps1
```

脚本从 `engine.lock.json` 指定的提交创建临时只读训练副本，训练产物和日志写入 `data/model-training/`，不会改动原 MASP 工作区。正式登记模型的训练信息和文件摘要见 `models/ppo-priority-v1.json`。

## 基本使用流程

1. 选择一个已发布场景运行规则基线，展示多车型调度和资源预约回放。
2. 在调度助手中输入紧急插单或通道封闭目标；前端以 `GOAL_EXECUTION` 创建可恢复 run，大模型只负责理解目标和整理参数。
3. Agent 完成确定性校验后自动调用 MASP 数字孪生，并根据冲突、完成率和安全停车状态给出 `PROCEED/BLOCK` 建议。
4. 低风险方案通过门槛后自动提交到 `simulation`；R3 方案在仿真完成后暂停，由主管批准或拒绝，批准后才提交仿真环境。
5. 在异常诊断页注入车辆故障、工位停用或等待环，比较处置分支并按需送审。
6. 在评测中心选择车辆规模、任务负载、策略和随机种子，运行可重复评测。

## 主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 引擎、模型和安全边界状态 |
| `GET` | `/api/v1/world/snapshot` | 场景世界快照 |
| `GET` | `/api/v1/map` | MASP 统一路网 |
| `GET` | `/api/v1/agent-policy` | 群车策略模型登记与权重状态 |
| `POST` | `/api/v1/agent/chat` | 自然语言到结构化意图 |
| `POST` | `/api/v1/agent/runs` | 创建可恢复的异步 Agent run |
| `GET` | `/api/v1/agent/runs/{runId}` | 查询状态、轨迹、成本和评测 |
| `GET` | `/api/v1/agent/runs/{runId}/events` | 订阅 SSE 实时运行事件 |
| `POST` | `/api/v1/agent/runs/{runId}/resume` | 审批并恢复暂停的 Agent run |
| `POST` | `/api/v1/agent/runs/{runId}/cancel` | 取消运行中的 Agent run |
| `GET` | `/api/v1/agent/tools` | 查看 Agent 工具目录、权限和输入 Schema |
| `GET` | `/api/v1/agent/memory/{conversationId}` | 查看服务端确认的结构化会话记忆 |
| `GET` | `/api/v1/agent/metrics` | 查看 Agent 聚合运行指标和最近轨迹摘要 |
| `GET` | `/api/v1/knowledge/search` | 执行带分数和 chunk ID 的混合知识检索 |
| `GET` | `/api/v1/knowledge/stats` | 查看知识片段数量和检索器版本 |
| `POST` | `/api/v1/intents/validate` | 确定性校验和风险分级 |
| `POST` | `/api/v1/simulations` | 运行数字孪生 |
| `POST` | `/api/v1/simulations/compare` | 比较 2-4 个方案 |
| `POST` | `/api/v1/simulations/{id}/explain` | 按车辆或任务生成规划证据解释 |
| `POST` | `/api/v1/approvals` | 创建高风险审批 |
| `POST` | `/api/v1/approvals/{id}/decision` | 人工审批决策 |
| `POST` | `/api/v1/intents/{id}/commit` | 仅提交到仿真环境 |
| `GET` | `/api/v1/audit` | 审计事件 |
| `GET` | `/api/v1/reports/shift` | 班次仿真运营报告 |
| `POST` | `/api/v1/incidents/inject` | 在已完成运行的安全节点建立故障分支 |
| `POST` | `/api/v1/incidents/inject/workstation` | 注入工位停用并识别关联任务与资源 |
| `POST` | `/api/v1/incidents/inject/deadlock` | 重放 MASP 等待图和死锁恢复事件 |
| `POST` | `/api/v1/incidents/{id}/diagnose` | 基于证据的 AI/规则异常诊断 |
| `POST` | `/api/v1/incidents/{id}/what-if` | 运行异常处置 MASP 推演 |
| `POST` | `/api/v1/incidents/{id}/approvals` | 将已完成处置分支提交 R3 审批 |
| `GET` | `/api/v1/incidents/{id}/report` | 导出异常、证据和分支结果 |
| `POST` | `/api/v1/evaluations/benchmarks` | 运行多配置、多种子评测矩阵 |
| `GET` | `/api/v1/evaluations/benchmarks/{id}` | 获取统计报告与失败案例 |
| `POST` | `/api/v1/dataset-exports` | 生成脱敏、质检后的评测数据资产 |
| `GET` | `/api/v1/dataset-exports/{id}/download` | 下载数据、清单和质量报告 ZIP |

启动后可在 [http://127.0.0.1:8877/docs](http://127.0.0.1:8877/docs) 查看完整 OpenAPI 文档。

## 运行数据

每次仿真写入 `runs/<runId>/`：

- `input-scenario.json`：实际输入场景；
- `planned-scenario.json`：车辆计划和分段路径；
- `planning-summary.json`：规划过程指标；
- `result.json`：事件日志、最终状态和业务指标；
- `manifest.json`：引擎版本、种子、意图和结果摘要；
- `command-center-summary.json`：界面使用的方案摘要。
- `agent-policy-evidence.json`：模型版本、推理与接管计数、逐轮候选和安全边界；
- `incident-context.json`：故障分支、冻结窗口、人工转运和已知限制。

矩阵评测写入 `data/evaluations/<benchmarkId>/`，数据导出写入 `data/dataset-exports/<exportId>/`。`data/` 和 `runs/` 都是可再生成的本地运行目录，不进入 Git。

Agent 会话记忆写入 `data/agent-memories.json`，不保存模型自由文本；匿名化运行指标追加到 `data/agent-metrics.jsonl`，不包含用户提示词和模型回复。

可恢复 Agent 运行写入 `data/agent-runs.sqlite3`。每个 run 独立事务更新，事件进入 append-only 表并启用 WAL；首次启动会自动导入旧 `agent-runs.json`。该数据库属于本地运行数据，不进入 Git。

## 文档

- [模型卡](docs/MODEL_CARD.md)
- [评测方法](docs/EVALUATION.md)
- [安全与权益说明](docs/SECURITY_AND_RIGHTS.md)
- [部署说明](docs/DEPLOYMENT.md)
- [大模型微调指南](docs/LLM_FINETUNING.md)

## 当前安全边界

- `fieldExecutionEnabled` 永远为 `false`；
- 只接受 `environment=simulation` 的提交；
- 大模型不能生成路径、资源预约或安全停车解除指令；
- R3 操作没有已批准且意图匹配的审批单时无法提交；
- 世界状态版本变化后，旧意图必须重新仿真；
- 所有仿真指标必须能追溯到 MASP 原始运行文件。
