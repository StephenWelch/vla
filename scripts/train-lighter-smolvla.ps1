param(
    [string]$CuratedRoot = 'E:\vla-smolvla\datasets\lighter_left_curated',
    [string]$BasePolicy = 'E:\vla-smolvla\smolvla_base',
    [string]$OutputRoot = 'E:\vla-smolvla\runs\lighter_left',
    [string]$CacheRoot = 'E:\vla-smolvla\hf-cache',
    [int]$BatchSize = 8
)

$train = Join-Path $CuratedRoot 'train'
$manifest = Join-Path $CuratedRoot 'manifest.json'
if (-not (Test-Path -LiteralPath $manifest) -or -not (Test-Path -LiteralPath (Join-Path $train 'meta\info.json'))) {
    throw 'Curate the 60 successful episodes before training.'
}
if (-not (Test-Path -LiteralPath (Join-Path $BasePolicy 'model.safetensors'))) {
    throw "SmolVLA base checkpoint not found at $BasePolicy"
}
$trainer = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $trainer)) { throw 'Run uv sync in the vla workspace first.' }
if ($BatchSize -le 0) { throw 'BatchSize must be positive.' }
if (Test-Path -LiteralPath $OutputRoot) { throw "Output already exists: $OutputRoot" }

& $trainer (Join-Path $PSScriptRoot 'train-smolvla.py') `
    --dataset $train --repo-id telegrip/episode_data_train `
    --policy $BasePolicy --output $OutputRoot `
    --camera-keys observation.images.left_wrist_cam `
    --camera-names observation.images.camera2 `
    --hf-home $CacheRoot `
    --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw 'SmolVLA training failed.' }
