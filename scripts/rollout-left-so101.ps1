param(
    [Parameter(Mandatory = $true)]
    [string]$PolicyPath,
    [Parameter(Mandatory = $true)]
    [string]$Task,
    [string]$RobotId = 'left',
    [string]$CameraName = 'gripper',
    [double]$DurationSeconds = 5,
    [double]$MaxRelativeTarget = 2
)

$projectRoot = Split-Path $PSScriptRoot -Parent
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$rollout = Join-Path $projectRoot '.venv\Scripts\lerobot-rollout.exe'
if (-not (Test-Path -LiteralPath $rollout)) { throw 'Run uv sync first.' }
if ($DurationSeconds -le 0 -or $MaxRelativeTarget -le 0) { throw 'DurationSeconds and MaxRelativeTarget must be positive.' }
if ('COM4' -notin [System.IO.Ports.SerialPort]::GetPortNames()) { throw 'Left follower port COM4 is not connected.' }

$policyDir = (Resolve-Path -LiteralPath $PolicyPath -ErrorAction Stop).Path
$configPath = Join-Path $policyDir 'config.json'
$weightsPath = Join-Path $policyDir 'model.safetensors'
if (-not (Test-Path -LiteralPath $configPath) -or -not (Test-Path -LiteralPath $weightsPath)) {
    throw 'PolicyPath must contain config.json and model.safetensors.'
}
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$visualFeatures = @($config.input_features.PSObject.Properties | Where-Object { $_.Value.type -eq 'VISUAL' })
$expectedCamera = "observation.images.$CameraName"
if ($config.type -ne 'smolvla' -or
    $config.input_features.'observation.state'.shape[0] -ne 6 -or
    $config.output_features.action.shape[0] -ne 6 -or
    $visualFeatures.Count -ne 1 -or
    $visualFeatures[0].Name -ne $expectedCamera) {
    throw "Checkpoint must be a six-joint SmolVLA policy with one camera feature named $expectedCamera."
}

$calibrationRoot = (& $python -c "from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS; print(HF_LEROBOT_CALIBRATION / ROBOTS / 'so_follower')").Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not locate the LeRobot calibration directory.' }
$calibrationPath = Join-Path $calibrationRoot "$RobotId.json"
if (-not (Test-Path -LiteralPath $calibrationPath)) {
    throw "No calibration file at $calibrationPath. Calibrate the left follower with --robot.id=$RobotId first."
}

$cameraConfig = '{ ' + $CameraName + ': {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }'
& $rollout '--strategy.type=base' "--policy.path=$policyDir" '--robot.type=so101_follower' '--robot.port=COM4' "--robot.id=$RobotId" "--robot.cameras=$cameraConfig" "--robot.max_relative_target=$MaxRelativeTarget" "--task=$Task" "--duration=$DurationSeconds" '--device=cuda'
if ($LASTEXITCODE -ne 0) { throw 'Left SO-101 rollout failed.' }
