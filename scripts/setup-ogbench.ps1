param(
    [string]$DataRoot = 'E:\vla-ogbench',
    [string]$Image = 'vla-ogbench-mjwarp:0.1.0'
)
$ErrorActionPreference = 'Stop'
$Setup = Join-Path (Split-Path $PSScriptRoot -Parent) 'projects\ogbench-mjwarp\scripts\setup.ps1'
& $Setup -DataRoot $DataRoot -Image $Image
