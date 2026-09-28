param(
    [string]$Python = 'py',
    [switch]$SkipModelDownload,
    [switch]$SkipNativeBuild
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$venv = Join-Path $root '.runtime\worker'
$workerPython = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $workerPython)) {
    if ($Python -eq 'py') { & py -3.12 -m venv $venv }
    else { & $Python -m venv $venv }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 worker venv creation failed' }
}
& $workerPython -m pip install -r (Join-Path $root 'requirements-worker.txt')
if ($LASTEXITCODE -ne 0) { throw 'Worker dependency install failed' }

$llamaDir = Join-Path $root '.reference\llama.cpp'
$llamaCommit = 'ad6c66839af3c5646fba8c6c2e2087a1e4e38948'
if (-not (Test-Path -LiteralPath (Join-Path $llamaDir 'CMakeLists.txt'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $llamaDir) | Out-Null
    & git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git $llamaDir
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp clone failed' }
    & git -C $llamaDir checkout --detach $llamaCommit
    if ($LASTEXITCODE -ne 0) { throw 'Pinned llama.cpp checkout failed' }
}
$actualCommit = (& git -C $llamaDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $llamaCommit) { throw "llama.cpp is not at pinned commit $llamaCommit" }
if (-not $SkipModelDownload) {
    & $workerPython (Join-Path $root 'scripts\download_q8.py')
    if ($LASTEXITCODE -ne 0) { throw 'Q8 download/verification failed' }
    & $workerPython (Join-Path $root 'scripts\download_vad.py')
    if ($LASTEXITCODE -ne 0) { throw 'FireRedVAD ONNX download/verification failed' }
    & $workerPython (Join-Path $root 'scripts\public_sample.py')
    if ($LASTEXITCODE -ne 0) { throw 'Official public test audio download/verification failed' }
}
if (-not $SkipNativeBuild) {
    & (Join-Path $root 'scripts\build_native_windows.ps1') -Python $workerPython -LlamaDir $llamaDir
    if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }
}
Write-Output "Worker ready: $workerPython"
