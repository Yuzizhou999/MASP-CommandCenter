param(
    [switch]$SkipBuild,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot "frontend"

if ($Offline) {
    $env:DEEPSEEK_API_KEY = ""
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

Push-Location $ProjectRoot
try {
    python -m command_center
}
finally {
    Pop-Location
}
