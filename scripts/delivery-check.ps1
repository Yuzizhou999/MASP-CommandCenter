param(
    [switch]$SkipTests,
    [switch]$SkipWheelhouse,
    [switch]$AllowDirtySource
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EngineRoot = Join-Path (Split-Path -Parent $ProjectRoot) "MASP"

$Tracked = @(& git -C $ProjectRoot ls-files)
$Prohibited = @(
    $Tracked | Where-Object {
        $_ -eq ".env" -or
        $_ -eq "docs/LLM_DISPATCH_COPILOT_DESIGN.md" -or
        $_ -like "data/*" -or
        $_ -like "runs/*" -or
        $_ -like "submission/*"
    }
)
if ($Prohibited.Count -gt 0) {
    throw "发现不应提交的文件：$($Prohibited -join ', ')"
}

& git -C $ProjectRoot diff --check
if ($LASTEXITCODE -ne 0) {
    throw "git diff --check 未通过。"
}

$Lock = Get-Content -Raw (Join-Path $ProjectRoot "engine.lock.json") | ConvertFrom-Json
& git -C $EngineRoot cat-file -e "$($Lock.commit)`^{commit}"
if ($LASTEXITCODE -ne 0) {
    throw "MASP 锁定提交不存在。"
}

if (-not $SkipTests) {
    & (Join-Path $PSScriptRoot "check.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "项目检查未通过。"
    }
}

$BuildArguments = @{}
if ($SkipWheelhouse) { $BuildArguments["SkipWheelhouse"] = $true }
if ($AllowDirtySource) { $BuildArguments["AllowDirtySource"] = $true }
$PackageRoot = & (Join-Path $PSScriptRoot "build-demo-package.ps1") @BuildArguments
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace([string]$PackageRoot)) {
    throw "演示包生成失败。"
}
$PackageRoot = ([string]$PackageRoot).Trim()

& (Join-Path $PackageRoot "scripts\verify-demo.ps1") -SmokeRun -PythonPath (Get-Command python -ErrorAction Stop).Source
if ($LASTEXITCODE -ne 0) {
    throw "演示包离线自检未通过。"
}

Write-Host "交付检查通过：$PackageRoot"
