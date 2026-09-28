param(
    [Parameter(Mandatory = $true)]
    [string]$TorchPython,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$hostPython = (Resolve-Path -LiteralPath $TorchPython).Path
$venv = Join-Path $root '.runtime\bf16'
$referencePython = Join-Path $venv 'Scripts\python.exe'

# The source Python supplies CUDA Torch through read-only system site packages.
# All pip commands below target only the ignored .runtime/bf16 virtual environment.
& $hostPython -c 'import sys, torch; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required"; assert torch.cuda.is_available(), "CUDA Torch is required"'
if ($LASTEXITCODE -ne 0) { throw 'Source Python must provide Python 3.10 and working CUDA Torch' }
if (-not (Test-Path -LiteralPath $referencePython)) {
    & $hostPython -m venv --system-site-packages $venv
    if ($LASTEXITCODE -ne 0) { throw 'BF16 reference venv creation failed' }
}

$expectedBase = (& $hostPython -c 'import sys; print(sys.prefix)').Trim()
$actualBase = (& $referencePython -c 'import sys; print(sys.base_prefix)').Trim()
if ($LASTEXITCODE -ne 0 -or $actualBase -ne $expectedBase) {
    throw "BF16 venv is based on $actualBase, expected $expectedBase"
}

if (-not $SkipInstall) {
    & $referencePython -m pip install --no-deps 'qwen-asr==0.0.6' 'transformers==4.57.6' 'accelerate==1.12.0' 'nagisa==0.2.11' 'soynlp==0.0.493' 'sox==1.5.0' 'dyNET38==2.2' 'scipy==1.14.1' 'opencc-python-reimplemented==0.1.7'
    if ($LASTEXITCODE -ne 0) { throw 'BF16 reference dependency install failed' }
}

& $referencePython -c 'import importlib.metadata as m, torch, qwen_asr, transformers, accelerate, soundfile, scipy, opencc; expected={"qwen-asr":"0.0.6","transformers":"4.57.6","accelerate":"1.12.0","nagisa":"0.2.11","soynlp":"0.0.493","sox":"1.5.0","dyNET38":"2.2","scipy":"1.14.1","opencc-python-reimplemented":"0.1.7"}; assert all(m.version(k)==v for k,v in expected.items()), "BF16 package version mismatch"; assert torch.cuda.is_available(), "CUDA Torch unavailable in BF16 venv"'
if ($LASTEXITCODE -ne 0) { throw 'BF16 reference environment validation failed' }
Write-Output "BF16 reference ready: $referencePython"
