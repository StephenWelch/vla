[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Root = 'E:\vla-ogbench\datasets\diversity-demo-v1',
    [int]$Episode = 0,
    [string]$DataRoot = 'E:\vla-ogbench',
    [string]$Distro = 'Ubuntu',
    [switch]$RestartWslg,
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$ViewerArgs
)
$ErrorActionPreference = 'Stop'
$Viewer = Join-Path (Split-Path $PSScriptRoot -Parent) 'projects\ogbench-mjwarp\scripts\view.ps1'
& $Viewer -Root $Root -Episode $Episode -DataRoot $DataRoot -Distro $Distro -RestartWslg:$RestartWslg @ViewerArgs
