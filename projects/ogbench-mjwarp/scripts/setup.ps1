param(
    [string]$DataRoot = 'E:\vla-ogbench',
    [string]$Image = 'vla-ogbench-mjwarp:0.1.0'
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force $DataRoot, (Join-Path $DataRoot 'datasets'), (Join-Path $DataRoot 'cache') | Out-Null
$RepositoryRoot = Split-Path (Split-Path $ProjectRoot -Parent) -Parent
docker build -f (Join-Path $ProjectRoot 'Dockerfile') -t $Image $RepositoryRoot
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed. Start Docker Desktop with the WSL 2 GPU engine.' }
& (Join-Path $PSScriptRoot 'run.ps1') -DataRoot $DataRoot -Image $Image doctor
if ($LASTEXITCODE -ne 0) { throw 'OGBench GPU/rendering check failed.' }
