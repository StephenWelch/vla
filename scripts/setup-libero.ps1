param(
    [string]$DataRoot = 'E:\vla-libero',
    [string]$Image = 'vla-smolvla-libero:0.6.1'
)

$ErrorActionPreference = 'Stop'
$dataRootPath = [System.IO.Path]::GetFullPath($DataRoot)
$projectPath = Split-Path -Parent $PSScriptRoot
$env:HF_HOME = Join-Path $dataRootPath 'hf-cache'
$env:UV_CACHE_DIR = Join-Path $dataRootPath 'uv-cache'
New-Item -ItemType Directory -Force -Path $dataRootPath | Out-Null

uv run --no-project --python 3.12 --with 'huggingface-hub>=1.6,<2' `
    (Join-Path $PSScriptRoot 'download-libero.py') $dataRootPath
if ($LASTEXITCODE -ne 0) { throw 'Could not download the LIBERO model and assets.' }

docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'Start Docker Desktop with its WSL 2 engine, then run this script again.'
}

docker build --file (Join-Path $projectPath 'docker\Dockerfile.libero') `
    --tag $Image $projectPath
if ($LASTEXITCODE -ne 0) { throw 'Could not build the LIBERO image.' }

docker run --rm --gpus all $Image nvidia-smi
if ($LASTEXITCODE -ne 0) { throw 'Docker GPU access failed.' }
