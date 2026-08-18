param(
    [string]$OutputDirectory = "",
    [string]$EngineRoot = "",
    [switch]$SkipWheelhouse,
    [switch]$AllowDirtySource
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot ".delivery"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if ([string]::IsNullOrWhiteSpace($EngineRoot)) {
    $EngineRoot = Join-Path (Split-Path -Parent $ProjectRoot) "MASP"
}
$EngineRoot = [System.IO.Path]::GetFullPath($EngineRoot)

$StatusRows = @(& git -C $ProjectRoot status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 CommandCenter Git 状态。"
}
$MeaningfulStatus = @(
    $StatusRows | Where-Object {
        $_ -notmatch '^\?\? (data/|runs/|submission/)' -and
        $_ -notmatch '^\?\? "submission/' -and
        $_ -notmatch '^\?\? docs/LLM_DISPATCH_COPILOT_DESIGN\.md$'
    }
)
if ($MeaningfulStatus.Count -gt 0 -and -not $AllowDirtySource) {
    throw "CommandCenter 存在未提交代码，请提交后再生成正式演示包；本地验证可使用 -AllowDirtySource。"
}

$Lock = Get-Content -Raw (Join-Path $ProjectRoot "engine.lock.json") | ConvertFrom-Json
$EngineCommit = [string]$Lock.commit
& git -C $EngineRoot cat-file -e "$EngineCommit`^{commit}"
if ($LASTEXITCODE -ne 0) {
    throw "MASP 仓库中不存在 engine.lock.json 指定的提交 $EngineCommit。"
}

$FrontendRoot = Join-Path $ProjectRoot "frontend"
if (-not (Test-Path -LiteralPath (Join-Path $FrontendRoot "node_modules"))) {
    Push-Location $FrontendRoot
    try {
        & npm.cmd ci | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "安装前端依赖失败。" }
    }
    finally { Pop-Location }
}
Push-Location $FrontendRoot
try {
    & npm.cmd run build | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "构建前端生产资源失败。" }
}
finally { Pop-Location }

$SourceCommit = (& git -C $ProjectRoot rev-parse HEAD).Trim()
$BuildStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BundleName = "lingshu-demo-$($SourceCommit.Substring(0, 8))-$BuildStamp"
$PackageRoot = Join-Path $OutputDirectory $BundleName
if (Test-Path -LiteralPath $PackageRoot) {
    throw "交付目录已存在：$PackageRoot"
}
New-Item -ItemType Directory -Path $PackageRoot -Force | Out-Null

function Copy-FilteredTree {
    param([string]$Source, [string]$Destination)
    $SourcePath = [System.IO.Path]::GetFullPath($Source).TrimEnd('\', '/')
    foreach ($File in Get-ChildItem -LiteralPath $SourcePath -Recurse -File -Force) {
        $Relative = $File.FullName.Substring($SourcePath.Length).TrimStart('\', '/')
        if ($Relative -match '(^|[\\/])(__pycache__|node_modules)([\\/]|$)' -or $Relative -match '\.pyc$') {
            continue
        }
        $Target = Join-Path $Destination $Relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
        Copy-Item -LiteralPath $File.FullName -Destination $Target
    }
}

foreach ($FileName in @("README.md", "engine.lock.json", "pyproject.toml", "requirements.txt", "requirements-agent.txt")) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $FileName) -Destination (Join-Path $PackageRoot $FileName)
}
foreach ($DirectoryName in @("command_center", "evals", "knowledge", "schemas")) {
    Copy-FilteredTree (Join-Path $ProjectRoot $DirectoryName) (Join-Path $PackageRoot $DirectoryName)
}
Copy-FilteredTree (Join-Path $ProjectRoot "models") (Join-Path $PackageRoot "models")
New-Item -ItemType Directory -Path (Join-Path $PackageRoot "frontend") -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $FrontendRoot "dist") -Destination (Join-Path $PackageRoot "frontend\dist") -Recurse

$DocumentationFiles = @(
    "COMPETITION_SUBMISSION.md",
    "DATA_CARD.md",
    "DELIVERY_CHECKLIST.md",
    "DEMO_OPERATIONS.md",
    "DEPLOYMENT.md",
    "EVALUATION.md",
    "MODEL_CARD.md",
    "SCENARIO_DESIGNER.md",
    "SCENARIO_DRAFTS.md",
    "SCENARIO_PACKAGE.md",
    "SECURITY_AND_RIGHTS.md",
    "TASK_STREAM_GENERATION.md"
)
New-Item -ItemType Directory -Path (Join-Path $PackageRoot "docs") -Force | Out-Null
foreach ($FileName in $DocumentationFiles) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\$FileName") -Destination (Join-Path $PackageRoot "docs\$FileName")
}

$DemoScripts = @("install-demo.ps1", "start-demo.ps1", "verify-demo.ps1")
New-Item -ItemType Directory -Path (Join-Path $PackageRoot "scripts") -Force | Out-Null
foreach ($FileName in $DemoScripts) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $FileName) -Destination (Join-Path $PackageRoot "scripts\$FileName")
}
New-Item -ItemType Directory -Path (Join-Path $PackageRoot "data") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PackageRoot "runs") -Force | Out-Null

$EnginePackageRoot = Join-Path $PackageRoot "engine\MASP"
New-Item -ItemType Directory -Path $EnginePackageRoot -Force | Out-Null
$TemporaryArchive = Join-Path $OutputDirectory "masp-$([guid]::NewGuid().ToString('N')).zip"
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
& git -C $EngineRoot archive --format=zip --output=$TemporaryArchive $EngineCommit
if ($LASTEXITCODE -ne 0) {
    throw "导出锁定的 MASP 版本失败。"
}
try {
    Expand-Archive -LiteralPath $TemporaryArchive -DestinationPath $EnginePackageRoot
}
finally {
    if (Test-Path -LiteralPath $TemporaryArchive) {
        Remove-Item -LiteralPath $TemporaryArchive -Force
    }
}

function Get-FileHashes {
    param([string]$Root, [string[]]$Exclude = @())
    $RootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $Hashes = [ordered]@{}
    foreach ($File in Get-ChildItem -LiteralPath $RootPath -Recurse -File -Force | Sort-Object FullName) {
        $Relative = $File.FullName.Substring($RootPath.Length).TrimStart('\', '/').Replace('\', '/')
        if ($Exclude -contains $Relative) { continue }
        $Hashes[$Relative] = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $Hashes
}

$EngineManifest = [ordered]@{
    schemaVersion = 1
    name = "MASP"
    commit = $EngineCommit
    files = Get-FileHashes $EnginePackageRoot
}
$EngineManifestJson = $EngineManifest | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText((Join-Path $EnginePackageRoot "engine.bundle.json"), $EngineManifestJson + [Environment]::NewLine, $Utf8NoBom)

if (-not $SkipWheelhouse) {
    $Wheelhouse = Join-Path $PackageRoot "wheelhouse"
    New-Item -ItemType Directory -Path $Wheelhouse -Force | Out-Null
    & python -m pip download --disable-pip-version-check --dest $Wheelhouse -r (Join-Path $ProjectRoot "requirements-agent.txt") | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "下载离线 Python 依赖失败。"
    }
}

$DeliveryManifest = [ordered]@{
    schemaVersion = 1
    project = "保利智仓·灵枢"
    packageMode = "simulation-only"
    fieldExecutionEnabled = $false
    createdAt = (Get-Date).ToUniversalTime().ToString("o")
    sourceCommit = $SourceCommit
    sourceDirty = ($MeaningfulStatus.Count -gt 0)
    engineCommit = $EngineCommit
    wheelhouseIncluded = (-not $SkipWheelhouse)
    files = Get-FileHashes $PackageRoot @("delivery-manifest.json")
}
$DeliveryManifestJson = $DeliveryManifest | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText((Join-Path $PackageRoot "delivery-manifest.json"), $DeliveryManifestJson + [Environment]::NewLine, $Utf8NoBom)

$ArchivePath = "$PackageRoot.zip"
Compress-Archive -LiteralPath $PackageRoot -DestinationPath $ArchivePath -CompressionLevel Optimal
Write-Host "演示包目录：$PackageRoot"
Write-Host "演示包压缩文件：$ArchivePath"
Write-Output $PackageRoot
