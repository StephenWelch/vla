[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Config = 'configs/ogbench/eval-ogbench-act.yaml',
    [string]$Python = '',
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)
$ErrorActionPreference = 'Stop'
$Workspace = Split-Path $PSScriptRoot -Parent
if (-not $Python) {
    $DataRoot = if ($env:OGBENCH_DATA_ROOT) { $env:OGBENCH_DATA_ROOT } else { "/mnt/e/vla-ogbench" }
    $Python = "$DataRoot/venv/bin/python"
}
& wsl.exe -e test -x $Python
if ($LASTEXITCODE -ne 0) {
    throw "Missing WSL evaluation Python at $Python. Prepare the evaluation environment or pass -Python; see docs/ogbench-policy-pilots.md."
}
$ConfigPath = (Resolve-Path -LiteralPath $Config).Path
$LinuxConfig = (& wsl.exe -e wslpath -a $ConfigPath).Trim()
$EnvironmentArgs = @('MUJOCO_GL=egl', 'PYTHONUNBUFFERED=1')
$NetrcPath = Join-Path $HOME '.netrc'
if (Test-Path -LiteralPath $NetrcPath) {
    $EnvironmentArgs += "VLA_WANDB_NETRC_PATH=$((& wsl.exe -e wslpath -a $NetrcPath).Trim())"
}
$TokenPath = Join-Path $HOME '.cache/huggingface/token'
if (Test-Path -LiteralPath $TokenPath) {
    $LinuxToken = (& wsl.exe -e wslpath -a $TokenPath).Trim()
    $EnvironmentArgs += "HF_TOKEN_PATH=$LinuxToken"
}
& wsl.exe -e env @EnvironmentArgs $Python -m ogbench_mjwarp.evaluate --config $LinuxConfig @CommandArgs
if ($LASTEXITCODE -ne 0) { throw "OGBench evaluation failed with exit code $LASTEXITCODE" }
