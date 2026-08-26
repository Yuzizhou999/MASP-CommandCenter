param(
    [switch]$SkipBuild,
    [string]$ModelPython = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$LogRoot = Join-Path $ProjectRoot "tmp\dual-demo"

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
    [string]$Name,
    [hashtable]$Environment = @{}
) {
    $OutputLog = Join-Path $LogRoot "$Name.out.log"
    $ErrorLog = Join-Path $LogRoot "$Name.err.log"
    $Options = @{
        FilePath = $FilePath
        ArgumentList = $ArgumentList
        WorkingDirectory = $ProjectRoot
        WindowStyle = "Hidden"
        RedirectStandardOutput = $OutputLog
        RedirectStandardError = $ErrorLog
        PassThru = $true
    }
    if ($Environment.Count -gt 0) {
        $Options.Environment = $Environment
    }
    $Process = Start-Process @Options
    Write-Host "Started $Name (PID $($Process.Id))."
}

if ([string]::IsNullOrWhiteSpace($ModelPython)) {
    $BundledPython = Join-Path $env:USERPROFILE ".conda\envs\masp-lora\python.exe"
    $ModelPython = if (Test-Path -LiteralPath $BundledPython) { $BundledPython } else { "python" }
}

if (-not (Test-Path -LiteralPath $ModelPython) -and $null -eq (Get-Command $ModelPython -ErrorAction SilentlyContinue)) {
    throw "Model Python was not found: $ModelPython. Pass -ModelPython with the finetuning environment Python path."
}

$V1Adapter = Join-Path $ProjectRoot "models\masp-intent-lora"
$V2Adapter = Join-Path $ProjectRoot "models\masp-agent-lora-v2"
foreach ($Adapter in @($V1Adapter, $V2Adapter)) {
    if (-not (Test-Path -LiteralPath (Join-Path $Adapter "model-card.json"))) {
        throw "Model adapter is incomplete: $Adapter"
    }
}

New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

if (-not $SkipBuild -or -not (Test-Path -LiteralPath (Join-Path $FrontendRoot "dist\index.html"))) {
    Push-Location $FrontendRoot
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $FrontendRoot "node_modules"))) {
            npm install
        }
        npm run build
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build failed."
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-ListeningPort 8000)) {
    Start-HiddenProcess $ModelPython @(
        "-m", "training.serve_intent_model",
        "--adapter-dir", $V1Adapter,
        "--port", "8000"
    ) "model-v1"
}

if (-not (Test-ListeningPort 8002)) {
    Start-HiddenProcess $ModelPython @(
        "-m", "training.serve_intent_model",
        "--adapter-dir", $V2Adapter,
        "--port", "8002"
    ) "model-v2"
}

$V1Model = Wait-Endpoint "http://127.0.0.1:8000/health" "v1 model"
$V2Model = Wait-Endpoint "http://127.0.0.1:8002/health" "v2 model"
if ($V1Model.model -ne "masp-intent-lora") {
    throw "Port 8000 serves $($V1Model.model), expected masp-intent-lora."
}
if ($V2Model.model -ne "masp-agent-lora-v2") {
    throw "Port 8002 serves $($V2Model.model), expected masp-agent-lora-v2."
}

$BackendPython = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-ListeningPort 8877)) {
    Start-HiddenProcess $BackendPython @("-m", "command_center") "backend-v1"
}

if (-not (Test-ListeningPort 8878)) {
    $V2Environment = @{
        "APP_PORT" = "8878"
        "LLM_PROVIDER" = "local"
        "LOCAL_LLM_ENABLED" = "true"
        "LOCAL_LLM_BASE_URL" = "http://127.0.0.1:8002/v1"
        "LOCAL_LLM_MODEL" = "masp-agent-lora-v2"
        "LOCAL_LLM_MODEL_CARD" = "models/masp-agent-lora-v2/model-card.json"
        "AGENT_RUNTIME_MODE" = "loop"
        "COMMAND_CENTER_DATA_DIR" = "data/agent-demo-v2"
        "COMMAND_CENTER_RUNS_DIR" = "runs/agent-demo-v2"
    }
    Start-HiddenProcess $BackendPython @("-m", "command_center") "backend-v2" $V2Environment
}

$V1Health = Wait-Endpoint "http://127.0.0.1:8877/api/health" "v1 backend" 60
$V2Health = Wait-Endpoint "http://127.0.0.1:8878/api/health" "v2 backend" 60
if ($V1Health.model.model -ne "masp-intent-lora" -or $V1Health.agentRuntime.mode -ne "linear") {
    throw "Port 8877 is not the stable masp-intent-lora / linear backend."
}
if (
    $V2Health.model.model -ne "masp-agent-lora-v2" -or
    $V2Health.agentRuntime.mode -ne "loop" -or
    $V2Health.agentRuntime.storageNamespace -ne "agent-demo-v2"
) {
    throw "Port 8878 is not the isolated masp-agent-lora-v2 / loop backend."
}

Write-Host ""
Write-Host "Stable demo:    http://127.0.0.1:8877  $($V1Model.model) / $($V1Health.agentRuntime.mode)"
Write-Host "Candidate demo: http://127.0.0.1:8878  $($V2Model.model) / $($V2Health.agentRuntime.mode)"
Write-Host "Logs:           $LogRoot"
