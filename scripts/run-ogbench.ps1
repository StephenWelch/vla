[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DataRoot = 'E:\vla-ogbench',
    [string]$Image = 'vla-ogbench-mjwarp:0.1.0',
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)
$ErrorActionPreference = 'Stop'
$Runner = Join-Path (Split-Path $PSScriptRoot -Parent) 'projects\ogbench-mjwarp\scripts\run.ps1'
& $Runner -DataRoot $DataRoot -Image $Image @CommandArgs
