# 部署说明

本项目当前仅支持仿真环境，不连接真实 WMS、RCS 或车辆控制器。

## 环境要求

- Windows PowerShell
- Python 3.11 或更高版本
- Node.js 20 或更高版本
- npm 和 Git
- 与本仓库并列的 MASP 仓库

```text
E:\project\MASP
E:\project\MASP-CommandCenter
```

MASP 必须与 `engine.lock.json` 中的提交匹配。可以通过 `MASP_ENGINE_ROOT` 指定其他路径。

## 安装

```powershell
cd E:\project\MASP-CommandCenter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

如需加载 PPO 策略，额外安装：

```powershell
pip install -r requirements-agent.txt
```

## 配置

`.env` 中的常用配置：

```dotenv
APP_HOST=127.0.0.1
APP_PORT=8877
APP_ENV=development
MASP_ENGINE_ROOT=E:\project\MASP
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

`DEEPSEEK_API_KEY` 留空时，系统使用确定性本地解析器。密钥只由后端读取，不应写入前端配置或提交到 Git。

## 启动

```powershell
.\scripts\start.ps1
```

首次启动会构建前端。已有 `frontend/dist` 时可跳过构建：

```powershell
.\scripts\start.ps1 -SkipBuild
```

强制使用离线意图解析：

```powershell
.\scripts\start.ps1 -SkipBuild -Offline
```

启动后访问 `http://127.0.0.1:8877`，OpenAPI 文档位于 `http://127.0.0.1:8877/docs`。

## 检查

```powershell
.\scripts\check.ps1
```

检查包括 Python 测试、前端类型检查和构建。如果健康检查中 `engine.allowed` 为 `false`，先确认 MASP 当前提交与 `engine.lock.json` 一致。

## 运行目录

- `data/`：审批、审计、评测和场景草稿。
- `runs/`：仿真输入、计划和结果。

两个目录均由系统自动创建，不进入 Git，可在不需要保留本地状态时删除。
