param(
    [string]$EngineRepository = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($EngineRepository)) {
    $EngineRepository = Join-Path (Split-Path -Parent $ProjectRoot) "MASP"
}
$EngineRepository = [System.IO.Path]::GetFullPath($EngineRepository)
$Lock = Get-Content -Raw (Join-Path $ProjectRoot "engine.lock.json") | ConvertFrom-Json
$EngineCommit = [string]$Lock.commit
$TempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$TempRoot = Join-Path $TempBase "masp-command-center-check-$([guid]::NewGuid().ToString('N'))"
$TestEngineRoot = Join-Path $TempRoot "MASP"
$PreviousEngineRoot = $env:MASP_ENGINE_ROOT
$PreviousTestEngineRoot = $env:MASP_TEST_ENGINE_ROOT

try {
    & git -C $EngineRepository cat-file -e "$EngineCommit`^{commit}"
    if ($LASTEXITCODE -ne 0) {
        throw "MASP 仓库中不存在锁定提交 $EngineCommit。"
    }
    New-Item -ItemType Directory -Path $TempRoot | Out-Null
    & git clone --quiet --shared --no-checkout $EngineRepository $TestEngineRoot
    if ($LASTEXITCODE -ne 0) {
        throw "创建锁定引擎测试副本失败。"
    }
    & git -C $TestEngineRoot checkout --quiet --detach $EngineCommit
    if ($LASTEXITCODE -ne 0) {
        throw "检出锁定引擎提交失败。"
    }

    $env:MASP_ENGINE_ROOT = $TestEngineRoot
    $env:MASP_TEST_ENGINE_ROOT = $TestEngineRoot
    Push-Location $ProjectRoot
    try {
        python -m pytest
        if ($LASTEXITCODE -ne 0) { throw "后端测试失败。" }
        Push-Location (Join-Path $ProjectRoot "frontend")
        try {
            npm run build
            if ($LASTEXITCODE -ne 0) { throw "前端构建失败。" }
        }
        finally {
            Pop-Location
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -eq $PreviousEngineRoot) {
        Remove-Item Env:\MASP_ENGINE_ROOT -ErrorAction SilentlyContinue
    }
    else {
        $env:MASP_ENGINE_ROOT = $PreviousEngineRoot
    }
    if ($null -eq $PreviousTestEngineRoot) {
        Remove-Item Env:\MASP_TEST_ENGINE_ROOT -ErrorAction SilentlyContinue
    }
    else {
        $env:MASP_TEST_ENGINE_ROOT = $PreviousTestEngineRoot
    }
    if (Test-Path -LiteralPath $TempRoot) {
        $ResolvedTempRoot = [System.IO.Path]::GetFullPath($TempRoot)
        $ExpectedPrefix = $TempBase.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        if (
            -not $ResolvedTempRoot.StartsWith(
                $ExpectedPrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            ) -or
            -not (Split-Path -Leaf $ResolvedTempRoot).StartsWith("masp-command-center-check-")
        ) {
            throw "拒绝清理非预期测试目录：$ResolvedTempRoot"
        }
        Remove-Item -LiteralPath $ResolvedTempRoot -Recurse -Force
    }
}
