param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8877,
    [switch]$OnlineAI,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EngineRoot = Join-Path $ProjectRoot "engine\MASP"
$FrontendIndex = Join-Path $ProjectRoot "frontend\dist\index.html"

if (-not (Test-Path -LiteralPath (Join-Path $EngineRoot "engine.bundle.json"))) {
    throw "演示包中的 MASP 版本证明不存在，请重新生成交付包。"
}
if (-not (Test-Path -LiteralPath $FrontendIndex)) {
    throw "演示包缺少前端生产资源，请重新生成交付包。"
}

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $BundledPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $BundledPython) {
        $PythonPath = $BundledPython
    }
    else {
        $PythonPath = (Get-Command python -ErrorAction Stop).Source
    }
}

$env:APP_ENV = "production"
$env:APP_HOST = "127.0.0.1"
$env:APP_PORT = [string]$Port
$env:MASP_ENGINE_ROOT = $EngineRoot
$env:MASP_ALLOW_DIRTY_DEVELOPMENT = "false"
if (-not $OnlineAI) {
    $env:DEEPSEEK_API_KEY = ""
}

Push-Location $ProjectRoot
try {
    & $PythonPath -m command_center.delivery
    if ($LASTEXITCODE -ne 0) {
        throw "演示包自检失败，后端未启动。"
    }
    Write-Host "灵枢演示环境已锁定为 simulation-only。"
    Write-Host "访问地址：http://127.0.0.1:$Port"
    if (-not $OnlineAI) {
        Write-Host "当前使用离线确定性降级，不调用 DeepSeek API。"
    }
    & $PythonPath -m command_center
    if ($LASTEXITCODE -ne 0) {
        throw "后端进程异常退出，退出码 $LASTEXITCODE。"
    }
}
finally {
    Pop-Location
}
