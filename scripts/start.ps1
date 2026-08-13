param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$EngineRoot = Join-Path (Split-Path -Parent $ProjectRoot) "MASP"

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
