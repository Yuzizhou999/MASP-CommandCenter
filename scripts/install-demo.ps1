param(
    [switch]$AllowOnline
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Wheelhouse = Join-Path $ProjectRoot "wheelhouse"
$VirtualEnv = Join-Path $ProjectRoot ".venv"
$Python = (Get-Command python -ErrorAction Stop).Source

if (-not (Test-Path -LiteralPath (Join-Path $VirtualEnv "Scripts\python.exe"))) {
    & $Python -m venv $VirtualEnv
    if ($LASTEXITCODE -ne 0) {
        throw "创建 Python 虚拟环境失败。"
    }
}

$VenvPython = Join-Path $VirtualEnv "Scripts\python.exe"
if (Test-Path -LiteralPath $Wheelhouse) {
    & $VenvPython -m pip install --disable-pip-version-check --no-index --find-links $Wheelhouse -r (Join-Path $ProjectRoot "requirements.txt")
}
elseif ($AllowOnline) {
    & $VenvPython -m pip install --disable-pip-version-check -r (Join-Path $ProjectRoot "requirements.txt")
}
else {
    throw "演示包未包含 wheelhouse。需要联网安装时请使用 -AllowOnline，正式离线交付请重新构建完整包。"
}
if ($LASTEXITCODE -ne 0) {
    throw "安装 Python 依赖失败。"
}

& (Join-Path $PSScriptRoot "verify-demo.ps1") -SmokeRun -PythonPath $VenvPython
if ($LASTEXITCODE -ne 0) {
    throw "安装后自检失败。"
}

Write-Host "安装和离线仿真自检已完成。"
