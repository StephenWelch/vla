[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DataRoot = 'E:\vla-ogbench',
    [string]$Image = 'vla-ogbench-mjwarp:0.1.0',
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force $DataRoot, (Join-Path $DataRoot 'cache') | Out-Null
docker run --rm --gpus all --shm-size 4g `
    --mount "type=bind,source=$DataRoot,target=/data" `
    --mount "type=bind,source=$(Join-Path $DataRoot 'cache'),target=/home/user_lerobot/.cache" `
    --mount "type=bind,source=$(Join-Path $ProjectRoot 'configs'),target=/configs,readonly" `
    $Image @CommandArgs
if ($LASTEXITCODE -ne 0) { throw "OGBench command failed with exit code $LASTEXITCODE" }
