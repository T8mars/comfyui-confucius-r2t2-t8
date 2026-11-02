param(
    [string]$Python = 'py',
    [switch]$SkipModelDownload,
    [switch]$SkipNativeBuild,
    [ValidateSet('auto','cpu','cuda')][string]$Backend = 'auto',
    [ValidateSet('wheel','source')][string]$Native = 'wheel',
    [string]$NativeWheel = ''
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$setupArgs = @((Join-Path $root 'scripts\setup_worker.py'), '--backend', $Backend, '--native', $Native)
if ($SkipModelDownload) { $setupArgs += '--skip-model-download' }
if ($SkipNativeBuild) { $setupArgs += '--skip-native' }
if ($NativeWheel) { $setupArgs += @('--wheel', $NativeWheel) }
if ($Python -eq 'py') { & py -3.12 @setupArgs }
else { & $Python @setupArgs '--python' $Python }
if ($LASTEXITCODE -ne 0) { throw 'Worker setup failed; see the error above' }
