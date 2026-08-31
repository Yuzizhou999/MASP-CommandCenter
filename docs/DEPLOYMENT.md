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

## 启用 API token

默认不启用鉴权：服务只绑 `127.0.0.1`，任何能访问该端口的进程都能提交变更，并在审批决策里自称任意 `decidedBy`。只要不止本机可信，就应当配置 token：

```dotenv
COMMAND_CENTER_API_TOKEN=用一段足够长的随机串
COMMAND_CENTER_API_TOKEN_OPERATOR=supervisor-zhang
```

启用后：

- 变更类 `/api` 请求（POST/PUT/PATCH/DELETE）必须带 `Authorization: Bearer <token>`；
- 审批人身份由 `COMMAND_CENTER_API_TOKEN_OPERATOR` 覆盖，客户端提交的 `decidedBy` 被忽略；
- `GET`、`/api/health` 和前端静态资源保持开放，SSE 轨迹订阅（GET）不受影响；
- `APPROVAL_DECIDED` 审计事件带 `authenticated` 字段。

自带前端从 `sessionStorage` 读取 token。启用后在浏览器控制台执行一次：

```javascript
sessionStorage.setItem("masp.apiToken", "你的token");
```

刷新页面即可。关闭标签页后失效，token 不会写进 URL 或浏览器历史。

当前实现只有单一共享 token，没有多用户账号、角色权限、审批人与操作人分离、SSO 或 token 轮换吊销。不要按企业级身份权限系统对外描述。

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
