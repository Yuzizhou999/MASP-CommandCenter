param(
    [string]$EngineRepository = "",
    [string]$OutputDirectory = "",
    [ValidateRange(0, 2147483647)]
    [int]$Seed = 0,
    [ValidateRange(0, 1000000)]
    [int]$TrainingSteps = 128
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($EngineRepository)) {
    $EngineRepository = Join-Path (Split-Path -Parent $ProjectRoot) "MASP"
}
$EngineRepository = [System.IO.Path]::GetFullPath($EngineRepository)
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "data\model-training\ppo-priority-v1-seed$Seed"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

$Lock = Get-Content -Raw (Join-Path $ProjectRoot "engine.lock.json") | ConvertFrom-Json
$EngineCommit = [string]$Lock.commit
$TempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$TempRoot = Join-Path $TempBase "masp-agent-training-$([guid]::NewGuid().ToString('N'))"
$TrainingEngine = Join-Path $TempRoot "MASP"

try {
    & git -C $EngineRepository cat-file -e "$EngineCommit`^{commit}"
    if ($LASTEXITCODE -ne 0) {
        throw "MASP 仓库中不存在锁定提交 $EngineCommit。"
    }
    New-Item -ItemType Directory -Path $TempRoot | Out-Null
    & git clone --quiet --shared --no-checkout $EngineRepository $TrainingEngine
    if ($LASTEXITCODE -ne 0) {
        throw "创建锁定引擎训练副本失败。"
    }
    & git -C $TrainingEngine checkout --quiet --detach $EngineCommit
    if ($LASTEXITCODE -ne 0) {
        throw "检出锁定引擎提交失败。"
    }

    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $ScenarioRoot = Join-Path $TrainingEngine "scenarios"
    $Arguments = @(
        (Join-Path $TrainingEngine "tools\train_priority_policy.py"),
        (Join-Path $ScenarioRoot "interactive-multi-fleet.json"),
        (Join-Path $ScenarioRoot "rhpp-long-distance-conflict.json"),
        (Join-Path $ScenarioRoot "realistic-multi-fleet.json"),
        "--state-source", "rolling",
        "--max-training-cases", "96",
        "--evaluation-case-limit", "32",
        "--validation-fraction", "0.2",
        "--oracle-max-evaluations", "240",
        "--behavior-clone-epochs", "30",
        "--steps", [string]$TrainingSteps,
        "--rollout-steps", "32",
        "--epochs", "4",
        "--batch-size", "16",
        "--seed", [string]$Seed,
        "--device", "cpu",
        "--output-dir", $OutputDirectory
    )
    & python @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $OutputDirectory "training.log")
    if ($LASTEXITCODE -ne 0) {
        throw "群车策略模型训练失败。"
    }
    Write-Host "训练产物：$OutputDirectory"
}
finally {
    if (Test-Path -LiteralPath $TempRoot) {
        $ResolvedTempRoot = [System.IO.Path]::GetFullPath($TempRoot)
        $ExpectedPrefix = $TempBase.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        if (
            -not $ResolvedTempRoot.StartsWith(
                $ExpectedPrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            ) -or
            -not (Split-Path -Leaf $ResolvedTempRoot).StartsWith("masp-agent-training-")
        ) {
            throw "拒绝清理非预期训练目录：$ResolvedTempRoot"
        }
        Remove-Item -LiteralPath $ResolvedTempRoot -Recurse -Force
    }
}
