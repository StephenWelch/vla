[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Config = 'configs/ogbench/train-ogbench-act-wandb.yaml',
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
if ($LASTEXITCODE -ne 0) { throw "Missing WSL policy Python: $Python" }
$LinuxConfig = (& wsl.exe -e wslpath -a (Resolve-Path -LiteralPath $Config).Path).Trim()
$EnvironmentArgs = @('MUJOCO_GL=egl', 'PYTHONUNBUFFERED=1')
$NetrcPath = Join-Path $HOME '.netrc'
if (Test-Path -LiteralPath $NetrcPath) {
    $EnvironmentArgs += "VLA_WANDB_NETRC_PATH=$((& wsl.exe -e wslpath -a $NetrcPath).Trim())"
}
$TokenPath = Join-Path $HOME '.cache/huggingface/token'
if (Test-Path -LiteralPath $TokenPath) {
    $EnvironmentArgs += "HF_TOKEN_PATH=$((& wsl.exe -e wslpath -a $TokenPath).Trim())"
}
& wsl.exe -e env @EnvironmentArgs $Python -m ogbench_mjwarp.train --config $LinuxConfig @CommandArgs
if ($LASTEXITCODE -ne 0) { throw "OGBench training failed with exit code $LASTEXITCODE" }
