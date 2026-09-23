param(
    [string]$EnvRoot = 'E:\vla-smolvla'
)

$ErrorActionPreference = 'Stop'
$envRootPath = [System.IO.Path]::GetFullPath($EnvRoot)
$venvPath = Join-Path $envRootPath 'venv'
$pythonPath = Join-Path $venvPath 'Scripts\python.exe'
$cachePath = Join-Path $envRootPath 'uv-cache'
$checkpointPath = Join-Path $envRootPath 'checkpoint'
$projectPath = Split-Path -Parent $PSScriptRoot
$env:HF_HOME = Join-Path $envRootPath 'hf-cache'
$env:UV_PROJECT_ENVIRONMENT = $venvPath

New-Item -ItemType Directory -Force -Path $envRootPath | Out-Null
uv sync --project $projectPath --frozen --python 3.13 --cache-dir $cachePath
if ($LASTEXITCODE -ne 0) { throw 'Could not install the simulation dependencies.' }

$downloadScript = Join-Path $PSScriptRoot 'download-smolvla.py'
& $pythonPath $downloadScript $checkpointPath
if ($LASTEXITCODE -ne 0) { throw 'Could not download the SmolVLA checkpoint.' }

& $pythonPath -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
