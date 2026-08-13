$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $ProjectRoot
try {
    python -m pytest
    Push-Location (Join-Path $ProjectRoot "frontend")
    try {
        npm run build
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}
