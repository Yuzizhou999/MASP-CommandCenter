param(
    [switch]$SkipBuild,
    [switch]$Offline,
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

$Adapter = Join-Path $ProjectRoot "models\masp-agent-lora-v2.3"
if (-not (Test-Path -LiteralPath (Join-Path $Adapter "model-card.json"))) {
    throw "The v2.3 adapter is incomplete: $Adapter"
}

$Wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if ($null -eq $Wsl) {
    throw "WSL is required for the XGrammar model service."
}
$PortableProjectRoot = $ProjectRoot.Replace("\", "/")
$WslPathOutput = & wsl.exe -d $WslDistribution -- wslpath -a $PortableProjectRoot
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($WslPathOutput)) {
    throw "Could not resolve the project path in WSL distribution $WslDistribution."
}
$WslProjectRoot = $WslPathOutput.Trim()
$null = & wsl.exe -d $WslDistribution -- test -x $ModelPython
$ModelPythonReady = $LASTEXITCODE -eq 0
if (-not $ModelPythonReady) {
    throw "Model Python was not found in WSL: $ModelPython"
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
        throw "Port $ModelPort is not the $ModelId XGrammar service."
    }

    if (Test-ListeningPort $AppPort) {
        $AppHealth = Wait-Endpoint "http://127.0.0.1:$AppPort/api/health" "Command Center" 10
        if (
            $AppHealth.model.model -ne $ModelId -or
            $AppHealth.agentRuntime.mode -ne "loop"
        ) {
            throw "Port $AppPort is occupied by a different Command Center configuration."
        }
        Write-Host "Command Center is already running: http://127.0.0.1:$AppPort"
        return
    }

    $env:APP_PORT = "$AppPort"
    $env:LLM_PROVIDER = "local"
    $env:LOCAL_LLM_ENABLED = "true"
    $env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:$ModelPort/v1"
    $env:LOCAL_LLM_MODEL = $ModelId
    $env:LOCAL_LLM_MODEL_CARD = "models/masp-agent-lora-v2.3/model-card.json"
    $env:AGENT_RUNTIME_MODE = "loop"

    Write-Host "Model: $ModelId / XGrammar / loop"
    Write-Host "Open:  http://127.0.0.1:$AppPort"
    Write-Host "Logs:  $LogRoot"

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
