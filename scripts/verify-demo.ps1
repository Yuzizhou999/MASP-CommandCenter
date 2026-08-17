param(
    [switch]$SmokeRun,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EngineRoot = Join-Path $ProjectRoot "engine\MASP"

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
$env:APP_PORT = "8877"
$env:MASP_ENGINE_ROOT = $EngineRoot
$env:MASP_ALLOW_DIRTY_DEVELOPMENT = "false"
$env:DEEPSEEK_API_KEY = ""

$Arguments = @("-m", "command_center.delivery")
if ($SmokeRun) {
    $Arguments += "--smoke-run"
}

Push-Location $ProjectRoot
try {
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "交付包自检失败。"
    }
}
finally {
    Pop-Location
}
