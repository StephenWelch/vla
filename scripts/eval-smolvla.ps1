param(
    [string]$EnvRoot = 'E:\vla-smolvla',
    [ValidateRange(1, 1000)][int]$Episodes = 1
)

$ErrorActionPreference = 'Stop'
$envRootPath = [System.IO.Path]::GetFullPath($EnvRoot)
$evalPath = Join-Path $envRootPath 'venv\Scripts\lerobot-eval.exe'
$checkpointPath = Join-Path $envRootPath 'checkpoint'
$env:HF_HOME = Join-Path $envRootPath 'hf-cache'

if (-not (Test-Path -LiteralPath $evalPath)) { throw "Run .\scripts\setup-smolvla.ps1 -EnvRoot '$envRootPath' first." }
if (-not (Test-Path -LiteralPath (Join-Path $checkpointPath 'model.safetensors'))) { throw "Checkpoint missing: $checkpointPath" }

& $evalPath "--policy.path=$checkpointPath" '--env.type=pusht' '--eval.batch_size=1' "--eval.n_episodes=$Episodes" '--eval.use_async_envs=false' '--policy.device=cuda' '--output_dir=outputs/eval/smolvla_pusht'
if ($LASTEXITCODE -ne 0) { throw 'SmolVLA evaluation failed.' }
