# 灵枢部署说明

## 1. 交付形态

比赛交付采用 Windows 离线演示包。包内包含 CommandCenter 后端、前端生产资源、知识文件、Schema、群车策略 checkpoint、锁定版本的 MASP 引擎和文件摘要清单。完整包还包含基础依赖及 PyTorch、Gymnasium、NumPy 的 `wheelhouse/`，安装和演示不依赖 npm，也不需要访问 Python 软件源。

DeepSeek 不是启动前提。默认启动方式会清空当前进程中的 `DEEPSEEK_API_KEY`，意图解析和异常诊断使用确定性降级，地图、仿真、SIPP、预约校验、审批和审计均可正常演示。

## 2. 构建电脑要求

- Windows 10/11 或 Windows Server；
- PowerShell 5.1 及以上；
- Python 3.11 及以上；
- Node.js 20 及以上、npm；
- Git；
- `MASP-CommandCenter` 与原 MASP 仓库位于并列目录；
- 当前 Python 版本和目标演示电脑一致，避免离线 wheel 与解释器 ABI 不匹配。

在 CommandCenter 仓库执行：

```powershell
.\scripts\delivery-check.ps1
```

该命令依次检查禁止提交文件、Git 差异格式、MASP 锁定提交、后端测试和前端构建，然后生成演示包并完成一次离线仿真。正式产物位于 `.delivery/`。构建脚本只通过 `git archive` 读取原 MASP 的锁定提交，不复制其工作区改动，也不会向原仓库写入文件或提交。

只验证打包流程、不下载离线依赖时，可执行：

```powershell
.\scripts\delivery-check.ps1 -SkipWheelhouse
```

这种精简包不能作为断网电脑的最终安装介质。

## 3. 演示电脑安装

将 ZIP 解压到普通目录，不要放在系统目录或只读介质。确认 Python 主版本与构建电脑一致，然后执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install-demo.ps1
```

安装脚本创建本地 `.venv`，只从 `wheelhouse/` 安装依赖，并运行最小仿真自检。它不会更改系统级 Python 包。

若拿到的是不含 wheelhouse 的内部测试包，且现场允许联网，可使用：

```powershell
.\scripts\install-demo.ps1 -AllowOnline
```

## 4. 启动与停止

默认离线启动：

```powershell
.\scripts\start-demo.ps1
```

控制台显示自检通过后，访问 `http://127.0.0.1:8877`。服务以前台进程运行，按 `Ctrl+C` 停止。端口被占用时可指定其他端口：

```powershell
.\scripts\start-demo.ps1 -Port 8890
```

需要调用 DeepSeek 时，由现场人员在当前 PowerShell 会话或演示包根目录的 `.env` 中配置密钥，再执行：

```powershell
.\scripts\start-demo.ps1 -OnlineAI
```

`.env` 不属于交付包，也不得回传或加入 Git。联网模式不可用时，后端自动降级，不影响确定性调度链路。

## 5. 生产锁与完整性

演示包固定设置：

- `APP_ENV=production`；
- `MASP_ALLOW_DIRTY_DEVELOPMENT=false`；
- `MASP_ENGINE_ROOT=engine\MASP`；
- `fieldExecutionEnabled=false`。

`delivery-manifest.json` 记录应用文件摘要，`engine/MASP/engine.bundle.json` 记录 MASP 提交和逐文件摘要。启动前任一受管文件缺失或被修改，检查都会失败。运行产生的 `data/`、`runs/`、`.venv/` 和 `__pycache__/` 不在摘要范围内。

## 6. 常见问题

“找不到 Python”：安装与构建包匹配的 Python 3.11 以上版本，并确保 `python` 在 PATH 中。

“离线安装找不到匹配 wheel”：构建电脑和演示电脑的 Python 主次版本或系统架构不一致，需要在与演示电脑一致的环境重新生成完整包。

“MASP 离线文件校验失败”：演示包内容被修改或解压不完整，重新解压原 ZIP，不要手工替换 `engine/MASP` 下的文件。

“端口已占用”：使用 `-Port` 更换端口；不应同时启动两份使用同一数据目录的服务。

“DeepSeek API 不可用”：继续使用离线模式，界面会明确显示确定性降级；不要在前端或截图中填写密钥。
