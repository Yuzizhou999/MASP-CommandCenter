param(
    [switch]$SkipBuild,
    [switch]$Offline,
    # 不启动本地模型服务，只跑确定性链路。适用于没有 GPU、没有 WSL 或没有
    # v2.3 adapter 的机器：MASP 仿真、安全校验、风险分级、审批和评测全部照常，
    # 只是意图理解与解释走确定性解析器，界面会标注降级状态。
    [switch]$Deterministic,
    [string]$WslDistribution = "Ubuntu",
    [string]$ModelPython = "/home/dministrator/.venvs/masp-lora/bin/python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$LogRoot = Join-Path $ProjectRoot "tmp\local-demo"
$ModelId = "masp-agent-lora-v2.3"
$ModelPort = 8000
$AppPort = 8877
$StartedModel = $null

function Test-ListeningPort([int]$Port) {
    $Client = [System.Net.Sockets.TcpClient]::new()
    try {
        $Connection = $Client.ConnectAsync("127.0.0.1", $Port)
        return $Connection.Wait(750) -and $Client.Connected
    }
    catch {
        return $false
    }
    finally {
        $Client.Dispose()
    }
}

function Wait-Endpoint([string]$Url, [string]$Label, [int]$TimeoutSeconds = 180) {
    $Deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        try {
            $Response = Invoke-RestMethod -Uri $Url -TimeoutSec 3
            if ($Response.status -eq "ok") {
                return $Response
            }
        }
        catch {
            Start-Sleep -Milliseconds 750
        }
    } while ([DateTime]::UtcNow -lt $Deadline)
    throw "$Label did not become ready within $TimeoutSeconds seconds. See $LogRoot."
}

function Start-HiddenProcess(
    [string]$FilePath,
    [string[]]$ArgumentList,
    [string]$Name
) {
    $OutputLog = Join-Path $LogRoot "$Name.out.log"
    $ErrorLog = Join-Path $LogRoot "$Name.err.log"
    $Process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $OutputLog `
        -RedirectStandardError $ErrorLog `
        -PassThru
    Write-Host "Started $Name (PID $($Process.Id))."
    return $Process
}

if ($Offline) {
    $env:DEEPSEEK_API_KEY = ""
}

if (-not $Deterministic) {
    $Adapter = Join-Path $ProjectRoot "models\masp-agent-lora-v2.3"
    if (-not (Test-Path -LiteralPath (Join-Path $Adapter "model-card.json"))) {
        throw @"
未找到 v2.3 adapter：$Adapter
adapter 权重体积较大，不随 Git 仓库分发。
没有 adapter 时可以只跑确定性链路：
    .\scripts\start.ps1 -Deterministic
"@
    }

    $Wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if ($null -eq $Wsl) {
        throw @"
XGrammar 模型服务需要 WSL。
没有 WSL 时可以只跑确定性链路：
    .\scripts\start.ps1 -Deterministic
"@
    }
    $PortableProjectRoot = $ProjectRoot.Replace("\", "/")
    $WslPathOutput = & wsl.exe -d $WslDistribution -- wslpath -a $PortableProjectRoot
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($WslPathOutput)) {
        throw "无法在 WSL 发行版 $WslDistribution 中解析项目路径。"
    }
    $WslProjectRoot = $WslPathOutput.Trim()
    $null = & wsl.exe -d $WslDistribution -- test -x $ModelPython
    if ($LASTEXITCODE -ne 0) {
        throw @"
WSL 中未找到模型 Python：$ModelPython
可以改用 -ModelPython 指定，或只跑确定性链路：
    .\scripts\start.ps1 -Deterministic
"@
    }
}

Push-Location $ProjectRoot
try {
    $EngineRoot = (& python -c "from command_center.settings import Settings; print(Settings.load().engine_root)").Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "无法读取应用配置，请先安装 Python 依赖。"
    }
}
finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath (Join-Path $EngineRoot "masp\online.py"))) {
    throw "MASP engine was not found at $EngineRoot. Set MASP_ENGINE_ROOT in .env."
}

if (-not $SkipBuild -or -not (Test-Path -LiteralPath (Join-Path $FrontendRoot "dist\index.html"))) {
    Push-Location $FrontendRoot
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $FrontendRoot "node_modules"))) {
            npm install
        }
        npm run build
    }
    finally {
        Pop-Location
    }
}

New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

try {
    if (-not $Deterministic) {
        if (-not (Test-ListeningPort $ModelPort)) {
            $StartedModel = Start-HiddenProcess "wsl.exe" @(
                "-d", $WslDistribution,
                "--cd", $WslProjectRoot,
                "--", $ModelPython,
                "-m", "training.serve_intent_model",
                "--adapter-dir", "models/masp-agent-lora-v2.3",
                "--host", "0.0.0.0",
                "--port", "$ModelPort",
                "--require-xgrammar"
            ) "model-v23"
        }

        $ModelHealth = Wait-Endpoint "http://127.0.0.1:$ModelPort/health" "v2.3 model"
        if (
            $ModelHealth.model -ne $ModelId -or
            $ModelHealth.structuredOutput -ne "xgrammar"
        ) {
            throw "端口 $ModelPort 上不是 $ModelId 的 XGrammar 服务。"
        }
    }

    if (Test-ListeningPort $AppPort) {
        $AppHealth = Wait-Endpoint "http://127.0.0.1:$AppPort/api/health" "Command Center" 10
        if ($AppHealth.agentRuntime.mode -ne "loop") {
            throw "端口 $AppPort 被另一份 Command Center 配置占用。"
        }
        Write-Host "Command Center 已在运行：http://127.0.0.1:$AppPort"
        return
    }

    $env:APP_PORT = "$AppPort"
    $env:AGENT_RUNTIME_MODE = "loop"

    if ($Deterministic) {
        # 指向一个确定不会有服务监听的地址，让 provider 走既有降级路径，
        # 而不是引入一条新的分支逻辑。
        $env:LLM_PROVIDER = "local"
        $env:LOCAL_LLM_ENABLED = "false"
        $env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:9/v1"
        $env:LOCAL_LLM_TIMEOUT_SECONDS = "2"
        $env:LOCAL_LLM_MODEL_CARD = ""
        $env:DEEPSEEK_API_KEY = ""
        Write-Host "模式:  确定性链路（无模型服务）"
        Write-Host "说明:  MASP 仿真、安全校验、风险分级、审批和评测照常运行；"
        Write-Host "       意图理解与解释走确定性解析器，界面标注降级状态。"
    }
    else {
        $env:LLM_PROVIDER = "local"
        $env:LOCAL_LLM_ENABLED = "true"
        $env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:$ModelPort/v1"
        $env:LOCAL_LLM_MODEL = $ModelId
        $env:LOCAL_LLM_MODEL_CARD = "models/masp-agent-lora-v2.3/model-card.json"
        Write-Host "模式:  $ModelId / XGrammar / loop"
    }

    Write-Host "打开:  http://127.0.0.1:$AppPort"
    Write-Host "日志:  $LogRoot"

    Push-Location $ProjectRoot
    try {
        python -m command_center
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -ne $StartedModel -and -not $StartedModel.HasExited) {
        Stop-Process -Id $StartedModel.Id -ErrorAction SilentlyContinue
    }
}
