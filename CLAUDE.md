# MASP-CommandCenter

保利智仓·灵枢：MASP 数字孪生引擎之上的 AI 调度指挥中心。自然语言目标 → 结构化
`DispatchIntent` → 确定性校验与风险分级 → MASP 仿真 → 安全门槛 → 人工审批 →
只提交到 `simulation`。

当前版本是仿真验证系统，不连接真实 WMS、RCS 或车辆控制器。

## 安全边界（不可协商）

- `fieldExecutionEnabled` 恒为 `false`；只接受 `environment=simulation` 的提交。
- 大模型不得生成或写入车辆路线、不得直接写资源预约表、不得解除安全停车或联锁、
  不得绕过确定性安全校验、不得自行补齐缺失的站点/车型/资源 ID。
  实体缺失就进多轮澄清，不要用默认值兜底。
- 工具白名单只增只读工具。`command_center/agent_tools.py` 里现有 4 个工具全部
  `read_only=True`；`validate_dispatch_intent` 额外是 `model_selectable=False`
  （固定阶段调用，模型不能自己选）。新增可写工具或放开 `model_selectable`
  需要先讨论。
- R3 操作没有已批准且意图匹配的审批单时不得提交；世界版本变化后旧意图必须重新仿真。
- 所有仿真指标必须能追溯到 `runs/<runId>/` 下的 MASP 原始文件。故障诊断与计划解释
  必须引用真实 `EV-*` / `PE-*` 证据，禁止编造证据、车辆或任务。
- 鉴权是可选的：未配置 `COMMAND_CENTER_API_TOKEN` 时接口开放、审批人身份沿用
  客户端提交值；配置后变更类 `/api` 请求需要 Bearer token，审批人由服务端覆盖。
  新增的变更类端点会自动被 `command_center/auth.py` 的 `is_protected` 覆盖，
  不要把它们加进 `_OPEN_PREFIXES` 白名单。不要把当前实现表述为企业级身份权限。

## 架构约束

- MASP 只能通过 `command_center/engine_adapter.py` 调用。不要在其他模块
  `import masp.*` 或修改 `sys.path`。
- 契约改动顺序：`command_center/contracts.py` → `command_center/api.py` →
  `frontend/src/types.ts`。
- Pydantic 模型统一 `ConfigDict(extra="forbid", populate_by_name=True)`，字段
  snake_case + camelCase alias；FastAPI 响应加 `response_model_by_alias=True`。
- 配置只经 `command_center/settings.py`（frozen dataclass）读取，不在业务代码里
  直接 `os.getenv`。
- 密钥只由后端读取，绝不进入前端资源。

## 命令

```powershell
# 安装
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt        # 运行 + pytest
pip install -r requirements-agent.txt      # 可选：PPO 策略推理
pip install -r requirements-finetune.txt   # 可选：QLoRA 训练与模型服务
Copy-Item .env.example .env

# 启动（需要 WSL 起 XGrammar 模型服务 :8000，应用 :8877）
.\scripts\start.ps1 [-SkipBuild] [-Offline]

# 测试
python -m pytest -m unit                   # 单元测试，CI 跑这个
.\scripts\check.ps1                        # 完整门禁：锁定引擎副本 + pytest + 前端构建

# 前端
cd frontend; npm ci; npm run build         # tsc -b && vite build
```

直接 `python -m pytest` 会跳过全部 integration 用例：`tests/conftest.py` 要求相邻
MASP 检出的 HEAD 恰好等于 `engine.lock.json` 的 commit 且工作区干净。
跑 integration 请用 `check.ps1`（它按 lock commit 克隆临时只读副本）。
**不要为了让 integration 跑起来去改 `engine.lock.json` 或 `conftest.py` 的门禁。**

## 环境与风格

- Windows + PowerShell，脚本和文档用 PowerShell 语法。
- 仓库没有配置 linter / formatter，跟随现有文件风格：`from __future__ import
  annotations`、内建泛型类型注解（`str | None`、`dict[str, Any]`）、极少 docstring。
- 错误信息中英混用，跟随所在模块既有语言，不要统一改写。
- `models/`、`data/`、`runs/`、`tmp/` 不入库。`masp-agent-lora-v2.3` adapter 被
  gitignore，新克隆的仓库需要本地提供才能跑 `start.ps1`。

## 评测纪律（来自 findings.md，容易踩）

- 不要用单 seed 或小样本套件声称稳定性。比较两个 adapter 前，先确认 prompt、
  response-schema、request-set 三个 hash 一致。
- GPU 实验前先跑 `python -m training.preflight_agent_system` 确认系统可达性，
  避免把确定性系统缺陷算成模型问题。
- 不要把训练/评测 loss 低或 intent-only 指标改善当作晋级理由。
- 当前门禁结论是 `KEEP_V1`：演示使用 v2.3，但**不得表述为生产晋级或稳定优于
  control**。同样不要声称 DeepSeek 保证了调度安全、AI 已控制真实车辆，或把仿真
  吞吐说成现场生产收益。

## 参考文档

- [docs/PROJECT_INTERVIEW_GUIDE.md](docs/PROJECT_INTERVIEW_GUIDE.md) — 最完整的全链路架构说明
- [docs/LLM_FINETUNING.md](docs/LLM_FINETUNING.md) — 数据准备、QLoRA 训练、本地服务、评测命令
- [docs/EVALUATION.md](docs/EVALUATION.md) — 评测矩阵、统计口径、安全门槛
- [findings.md](findings.md) — 实验结论与"不能声称什么"的约束清单

## 语言

中文回复。提交信息中文，可用 `feat:` / `fix:` / `test:` / `experiment:` 前缀。
