# 保利智仓·灵枢

面向多车型智能仓储的 AI 调度指挥中心。系统使用 DeepSeek 将自然语言请求转换为结构化调度意图，再由 MASP 数字孪生完成路径规划、资源预约、安全校验、What-if 仿真和指标计算。

当前版本是比赛演示与验证系统，只允许在 `simulation` 环境提交意图，不连接真实 WMS、RCS 或车辆控制器。

## 核心能力

- 自然语言紧急插单、通道封锁、状态查询和报告生成；
- 缺失站点、车型或资源时进入多轮澄清，不使用默认实体补齐；
- DeepSeek API 调用，未配置密钥或 API 异常时自动使用确定性本地解析；
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
E:\project\MASP-CommandCenter   灵枢应用、智能体、治理、前端和比赛材料
```

所有 MASP 调用都集中在 `command_center/engine_adapter.py`。场景包编译和任务流生成扩展位于`command_center/masp/`，由比赛仓库独立维护；原MASP仓库不承载比赛新增代码。`engine.lock.json`固定允许使用的原MASP提交，开发环境可显式允许脏工作区，生产环境必须提交匹配且工作区干净。

## 技术架构

| 层级 | 实现 |
|---|---|
| 交互层 | React 19、TypeScript、Fluent UI、Vite |
| 应用服务 | FastAPI、Pydantic v2、JSON API |
| 智能体 | DeepSeek `deepseek-chat`、结构化 JSON、本地确定性降级 |
| 治理 | 风险分级、审批、世界版本、审计、仿真态提交 |
| 数字孪生 | MASP 在线调度、路径规划、资源预约、事件回放 |
| 数据与评测 | 可重复矩阵评测、统计汇总、安全门槛、脱敏质检 |
| 存储 | JSON/JSONL 演示存储，运行证据按 run 和 benchmark 独立落盘 |

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

浏览器访问 [http://127.0.0.1:8877](http://127.0.0.1:8877)。首次启动会安装前端依赖并构建生产资源。之后可使用：

```powershell
.\scripts\start.ps1 -SkipBuild
```

现场网络不稳定或不准备调用 DeepSeek 时，可显式使用离线降级：

```powershell
.\scripts\start.ps1 -SkipBuild -Offline
```

完整检查：

```powershell
.\scripts\check.ps1
```

## 生成离线演示包

正式打包前需先提交代码。以下命令会构建前端、从 `engine.lock.json` 指定的提交导出干净 MASP 引擎、下载本机 Python 版本对应的离线依赖，并生成带摘要清单的 ZIP：

```powershell
.\scripts\delivery-check.ps1
```

产物位于 `.delivery/`，不会进入 Git。解压后在目标 Windows 电脑执行：

```powershell
.\scripts\install-demo.ps1
.\scripts\start-demo.ps1
```

离线包默认不调用 DeepSeek，意图解析和诊断自动使用确定性降级；需要联网调用时使用 `start-demo.ps1 -OnlineAI`，并由现场人员在服务端环境中配置密钥。详细要求见[部署说明](docs/DEPLOYMENT.md)、[演示操作手册](docs/DEMO_OPERATIONS.md)和[交付检查表](docs/DELIVERY_CHECKLIST.md)。

## 配置 DeepSeek

编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT_SECONDS=30
```

可以直接调用 DeepSeek 官方 API。密钥只由 FastAPI 后端读取，绝不进入浏览器资源。若未配置密钥、请求超时、返回非 JSON 或 Schema 校验失败，系统自动切换到本地解析器，并在界面和审计日志中明确标注降级状态。

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

## 比赛演示流程

1. 选择一个已发布场景运行规则基线，展示多车型调度和资源预约回放。
2. 点击“注入紧急任务”，检查大模型形成的结构化任务草案和 MASP 世界快照依据。
3. 运行数字孪生，将候选方案与基线加入方案比较。
4. 点击“推演通道封闭”，展示 R3 风险识别和封锁资源高亮。
5. 仿真完成后提交主管审批，在审批页批准，再提交到仿真环境。
6. 打开运营审计，核对意图、仿真、审批、提交的完整证据链。
7. 导出运营报告，说明全部数值来自仿真，不冒充真实生产收益。
8. 进入“异常诊断”，使用一键演示注入车辆故障、工位停用或等待环，完成证据诊断、至少两个处置分支推演和 R3 送审，再跳转方案页比较。
9. 进入“评测中心”，选择车辆规模、任务负载、车型、策略和种子，运行同口径评测并核对安全门槛与置信区间。
10. 生成脱敏评测数据资产包，检查训练/验证/测试集划分和质量报告后下载 ZIP。

## 主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 引擎、模型和安全边界状态 |
| `GET` | `/api/v1/world/snapshot` | 场景世界快照 |
| `GET` | `/api/v1/map` | MASP 统一路网 |
| `GET` | `/api/v1/agent-policy` | 群车策略模型登记与权重状态 |
| `POST` | `/api/v1/agent/chat` | 自然语言到结构化意图 |
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

## 运行证据

每次仿真写入 `runs/<runId>/`：

- `input-scenario.json`：实际输入场景；
- `planned-scenario.json`：车辆计划和分段路径；
- `planning-summary.json`：规划过程指标；
- `result.json`：事件日志、最终状态和业务指标；
- `manifest.json`：引擎版本、种子、意图和结果摘要；
- `command-center-summary.json`：界面使用的方案摘要。
- `agent-policy-evidence.json`：模型版本、推理与接管计数、逐轮候选和安全边界；
- `incident-context.json`：故障分支、冻结窗口、人工转运和已知限制。

每次矩阵评测写入 `data/evaluations/<benchmarkId>/`，保存原始请求、逐用例输入、逐用例结果以及 JSON/Markdown 报告。每次数据导出写入 `data/dataset-exports/<exportId>/`，包含 `manifest.json`、`quality-report.json`、`dataset.jsonl` 和可下载 ZIP。运行产物默认不进入 Git。

这些目录默认不进入 Git，避免把运行数据和源代码混在一起。

## 文档

- [参赛作品说明](docs/COMPETITION_SUBMISSION.md)
- [模型卡](docs/MODEL_CARD.md)
- [数据卡](docs/DATA_CARD.md)
- [评测方法](docs/EVALUATION.md)
- [安全与权益说明](docs/SECURITY_AND_RIGHTS.md)
- [部署说明](docs/DEPLOYMENT.md)
- [演示操作手册](docs/DEMO_OPERATIONS.md)
- [交付检查表](docs/DELIVERY_CHECKLIST.md)

## 当前安全边界

- `fieldExecutionEnabled` 永远为 `false`；
- 只接受 `environment=simulation` 的提交；
- 大模型不能生成路径、资源预约或安全停车解除指令；
- R3 操作没有已批准且意图匹配的审批单时无法提交；
- 世界状态版本变化后，旧意图必须重新仿真；
- 所有比赛指标必须能追溯到 MASP 原始运行文件。
