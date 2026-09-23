param(
    [string]$DataRoot = 'E:\vla-libero',
    [string]$Image = 'vla-smolvla-libero:0.6.1',
    [switch]$FullBenchmark
)

$ErrorActionPreference = 'Stop'
$dataRootPath = [System.IO.Path]::GetFullPath($DataRoot)
$projectPath = Split-Path -Parent $PSScriptRoot
$modelPath = Join-Path $dataRootPath 'models\smolvla_libero'
$assetsPath = Join-Path $dataRootPath 'libero-assets'
$cachePath = Join-Path $dataRootPath 'container-cache'
$outputPath = Join-Path $projectPath 'outputs\libero'

if (-not (Test-Path -LiteralPath (Join-Path $modelPath 'model.safetensors'))) {
    throw "Missing checkpoint at $modelPath. Run setup-libero.ps1 first."
}
foreach ($subdir in @('articulated_objects', 'stable_scanned_objects', 'turbosquid_objects', 'stable_hope_objects')) {
    if (-not (Test-Path -LiteralPath (Join-Path $assetsPath $subdir))) {
        throw "Missing LIBERO assets at $assetsPath. Run setup-libero.ps1 first."
    }
}
New-Item -ItemType Directory -Force -Path $cachePath, $outputPath | Out-Null

if ($FullBenchmark) {
    $runName = 'full'
    $task = 'libero_spatial,libero_object,libero_goal,libero_10'
    $episodes = 10
} else {
    $runName = 'smoke'
    $task = 'libero_spatial'
    $episodes = 1
}

$evalArgs = @(
    '--policy.path=/models/smolvla',
    '--policy.device=cuda',
    '--env.type=libero',
    "--env.task=$task",
    '--env.control_mode=relative',
    '--env.max_parallel_tasks=1',
    '--eval.batch_size=1',
    "--eval.n_episodes=$episodes",
    '--eval.use_async_envs=false',
    '--seed=1000',
    "--output_dir=/results/$runName"
)
if (-not $FullBenchmark) { $evalArgs += '--env.task_ids=[0]' }

docker run --rm --gpus all --ipc=host `
    --mount "type=bind,source=$modelPath,target=/models/smolvla,readonly" `
    --mount "type=bind,source=$assetsPath,target=/home/user_lerobot/.cache/libero/assets,readonly" `
    --mount "type=bind,source=$cachePath,target=/home/user_lerobot/.cache/huggingface" `
    --mount "type=bind,source=$outputPath,target=/results" `
    --env MUJOCO_GL=egl $Image lerobot-eval @evalArgs
if ($LASTEXITCODE -ne 0) { throw 'LIBERO evaluation failed.' }

Write-Output "Results: $(Join-Path $outputPath $runName)"
