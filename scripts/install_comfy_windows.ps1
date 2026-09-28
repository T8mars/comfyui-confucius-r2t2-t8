param(
    [Parameter(Mandatory=$true)][string]$ComfyDir
)

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$customNodes = Join-Path $ComfyDir 'custom_nodes'
if (-not (Test-Path -LiteralPath (Join-Path $ComfyDir 'main.py'))) {
    throw "Not a ComfyUI checkout: $ComfyDir"
}
if (-not (Test-Path -LiteralPath $customNodes)) {
    throw "custom_nodes directory missing: $customNodes"
}
$target = Join-Path $customNodes 'Confucius4-R2T2'
if (Test-Path -LiteralPath $target) {
    $existing = Get-Item -LiteralPath $target -Force
    if ($existing.LinkType -eq 'Junction' -and $existing.Target -eq $source) {
        Write-Output "Already installed: $target -> $source"
        exit 0
    }
    throw "Target exists and points elsewhere: $target"
}
New-Item -ItemType Junction -Path $target -Target $source | Out-Null
Write-Output "Installed: $target -> $source"
