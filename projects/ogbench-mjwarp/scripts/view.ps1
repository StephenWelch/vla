[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Root = 'E:\vla-ogbench\datasets\vla-ready',
    [int]$Episode = 0,
    [string]$DataRoot = 'E:\vla-ogbench',
    [string]$Distro = 'Ubuntu',
    [switch]$RestartWslg,
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$ViewerArgs
)
$ErrorActionPreference = 'Stop'
if ($RestartWslg) {
    Write-Host 'Restarting the WSLg compositor; Linux GUI windows will close. WSL shells and Docker remain running.'
    & wsl -d $Distro --system --exec pkill -KILL -x weston
    if ($LASTEXITCODE -ne 0) { throw 'Could not restart the WSLg compositor.' }
    # WSLGd restarts Weston asynchronously; wait for the X11 socket to return.
    $DisplayReady = $false
    for ($Attempt = 0; $Attempt -lt 20; $Attempt++) {
        Start-Sleep -Milliseconds 250
        & wsl -d $Distro --exec test -S /tmp/.X11-unix/X0
        if ($LASTEXITCODE -eq 0) { $DisplayReady = $true; break }
    }
    if (-not $DisplayReady) { throw 'WSLg did not restore its X11 display.' }
}
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$RolloutRoot = (Resolve-Path -LiteralPath $Root).Path
$EnvironmentRoot = (Resolve-Path -LiteralPath (Join-Path $DataRoot 'venv')).Path
$LinuxProject = & wsl -d $Distro --exec wslpath -a $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve project path in WSL.' }
$LinuxRoot = & wsl -d $Distro --exec wslpath -a $RolloutRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve rollout path in WSL.' }
$LinuxEnvironment = & wsl -d $Distro --exec wslpath -a $EnvironmentRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve environment path in WSL.' }
$WestonLog = "\\wsl.localhost\$Distro\mnt\wslg\weston.log"
if (Test-Path -LiteralPath $WestonLog) {
    $Transport = Get-Content -LiteralPath $WestonLog | Select-String 'RDP backend: use_gfxredir =' | Select-Object -Last 1
    if ($Transport -and $Transport.Line -match 'use_gfxredir = 0') {
        Write-Warning 'WSLg is in COPY MODE; its window may be invisible. Retry with -RestartWslg to restore the display transport (closes Linux GUI windows).'
    }
}
& wsl -d $Distro --exec env "OGBENCH_ENVIRONMENT=$LinuxEnvironment" `
    bash "$LinuxProject/scripts/view-wsl.sh" --root $LinuxRoot --episode $Episode @ViewerArgs
if ($LASTEXITCODE -ne 0) { throw "MuJoCo viewer failed with exit code $LASTEXITCODE" }
