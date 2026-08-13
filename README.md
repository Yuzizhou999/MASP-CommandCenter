# 保利智仓·灵枢

面向多车型智能仓储的 AI 调度指挥中心。系统使用 DeepSeek 将自然语言请求转换为结构化调度意图，再由 MASP 数字孪生完成路径规划、资源预约、安全校验、What-if 仿真和指标计算。

当前版本是比赛演示与验证系统，只允许在 `simulation` 环境提交意图，不连接真实 WMS、RCS 或车辆控制器。

## 核心能力

- 自然语言紧急插单、通道封锁、状态查询和报告生成；
- DeepSeek API 调用，未配置密钥或 API 异常时自动使用确定性本地解析；
- 552 个节点、1204 条有向边的真实 MASP 仓储路网展示；
- 多车型车辆轨迹、任务路径和资源封锁时窗回放；
- 14 车 32 任务基线及候选方案确定性仿真；
- 候选方案同口径比较与可追溯推荐理由；
- R0-R4 风险分级，R3 高风险操作强制人工审批；
- 世界版本复核、仿真态提交、审计日志和班次报告；
- SOP 知识检索和模型决策依据展示。
- 车辆故障安全节点注入、`EV-*` 证据链和 DeepSeek 根因解释；
- 等待恢复、隔离重派、安全停车三类可比较的 MASP 故障 What-if。

## 仓库边界

本项目与 MASP 是两个并列、独立的仓库：

```text
E:\project\MASP                 MASP 核心调度与数字孪生引擎
E:\project\MASP-CommandCenter   灵枢应用、智能体、治理、前端和比赛材料
```

所有 MASP 调用都集中在 `command_center/engine_adapter.py`。灵枢不会修改 MASP 源文件，`engine.lock.json` 固定允许使用的 MASP 提交。开发环境可显式允许脏工作区，生产环境必须提交匹配且工作区干净。

## 技术架构

| 层级 | 实现 |
|---|---|
| 交互层 | React 19、TypeScript、Fluent UI、Vite |
| 应用服务 | FastAPI、Pydantic v2、JSON API |
| 智能体 | DeepSeek `deepseek-chat`、结构化 JSON、本地确定性降级 |
| 治理 | 风险分级、审批、世界版本、审计、仿真态提交 |
| 数字孪生 | MASP 在线调度、路径规划、资源预约、事件回放 |
| 存储 | JSON/JSONL 演示存储，运行证据按 run 独立落盘 |

大模型只负责理解和解释。路径、资源预约、冲突判断、方案指标全部来自 MASP 确定性引擎。故障诊断中的每条根因与建议必须引用真实 `EV-*` 证据，虚构证据、车辆、任务或越权动作会使整份 AI 报告降级为规则诊断。

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

浏览器访问 [http://127.0.0.1:8877](http://127.0.0.1:8877)。首次启动会安装前端依赖并构建生产资源。之后可使用：

```powershell
.\scripts\start.ps1 -SkipBuild
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

## 比赛演示流程

1. 点击“14车32任务基线”，展示真实多车型调度和零资源冲突回放。
2. 点击“注入紧急任务”，检查大模型形成的结构化任务草案和 MASP 世界快照依据。
3. 运行数字孪生，将候选方案与基线加入方案比较。
4. 点击“推演通道封闭”，展示 R3 风险识别和封锁资源高亮。
5. 仿真完成后提交主管审批，在审批页批准，再提交到仿真环境。
6. 打开运营审计，核对意图、仿真、审批、提交的完整证据链。
7. 导出运营报告，说明全部数值来自仿真，不冒充真实生产收益。
8. 进入“异常诊断”，选择一个已完成运行注入车辆故障，完成原因分析和至少两个处置分支推演，再跳转方案页比较。

## 主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 引擎、模型和安全边界状态 |
| `GET` | `/api/v1/world/snapshot` | 场景世界快照 |
| `GET` | `/api/v1/map` | MASP 统一路网 |
| `POST` | `/api/v1/agent/chat` | 自然语言到结构化意图 |
| `POST` | `/api/v1/intents/validate` | 确定性校验和风险分级 |
| `POST` | `/api/v1/simulations` | 运行数字孪生 |
| `POST` | `/api/v1/simulations/compare` | 比较 2-4 个方案 |
| `POST` | `/api/v1/approvals` | 创建高风险审批 |
| `POST` | `/api/v1/approvals/{id}/decision` | 人工审批决策 |
| `POST` | `/api/v1/intents/{id}/commit` | 仅提交到仿真环境 |
| `GET` | `/api/v1/audit` | 审计事件 |
| `GET` | `/api/v1/reports/shift` | 班次仿真运营报告 |
| `POST` | `/api/v1/incidents/inject` | 在已完成运行的安全节点建立故障分支 |
| `POST` | `/api/v1/incidents/{id}/diagnose` | 基于证据的 AI/规则故障诊断 |
| `POST` | `/api/v1/incidents/{id}/what-if` | 运行故障处置 MASP 推演 |
| `GET` | `/api/v1/incidents/{id}/report` | 导出异常、证据和分支结果 |

启动后可在 [http://127.0.0.1:8877/docs](http://127.0.0.1:8877/docs) 查看完整 OpenAPI 文档。

## 运行证据

每次仿真写入 `runs/<runId>/`：

- `input-scenario.json`：实际输入场景；
- `planned-scenario.json`：车辆计划和分段路径；
- `planning-summary.json`：规划过程指标；
- `result.json`：事件日志、最终状态和业务指标；
- `manifest.json`：引擎版本、种子、意图和结果摘要；
- `command-center-summary.json`：界面使用的方案摘要。
- `incident-context.json`：故障分支、冻结窗口、人工转运和已知限制。

这些目录默认不进入 Git，避免把运行数据和源代码混在一起。

## 文档

- [参赛作品说明](docs/COMPETITION_SUBMISSION.md)
- [模型卡](docs/MODEL_CARD.md)
- [数据卡](docs/DATA_CARD.md)
- [安全与权益说明](docs/SECURITY_AND_RIGHTS.md)

## 当前安全边界

- `fieldExecutionEnabled` 永远为 `false`；
- 只接受 `environment=simulation` 的提交；
- 大模型不能生成路径、资源预约或安全停车解除指令；
- R3 操作没有已批准且意图匹配的审批单时无法提交；
- 世界状态版本变化后，旧意图必须重新仿真；
- 所有比赛指标必须能追溯到 MASP 原始运行文件。
