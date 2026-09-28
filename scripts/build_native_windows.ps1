param(
    [string]$Python = (Join-Path $PSScriptRoot '..\.runtime\worker\Scripts\python.exe'),
    [string]$LlamaDir = (Join-Path $PSScriptRoot '..\.reference\llama.cpp'),
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\.runtime\build-native-cu128'),
    [string]$CudaArch = '120',
    [string]$CudaToolkit = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
    [string]$CudaVersion = '12.8'
)

$ErrorActionPreference = 'Stop'
$srcDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\vendor\r2t2_native')).Path
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$llamaPath = (Resolve-Path -LiteralPath $LlamaDir).Path
$expectedLlamaCommit = 'ad6c66839af3c5646fba8c6c2e2087a1e4e38948'
$actualLlamaCommit = (& git -C $llamaPath rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualLlamaCommit -ne $expectedLlamaCommit) {
    throw "Expected llama.cpp commit $expectedLlamaCommit, got $actualLlamaCommit"
}
$buildPath = [IO.Path]::GetFullPath($BuildDir)
$cudaPath = (Resolve-Path -LiteralPath $CudaToolkit).Path
$env:CUDA_PATH = $cudaPath
$pybindDir = (& $pythonPath -m pybind11 --cmakedir).Trim()
if ($LASTEXITCODE -ne 0 -or -not $pybindDir) { throw 'pybind11 CMake directory not found' }
$cmake = (Get-Command cmake.exe -ErrorAction Stop).Source

& $cmake -S $srcDir -B $buildPath -G 'Visual Studio 17 2022' -A x64 -T "cuda=$CudaVersion" `
    "-DLLAMA_CPP_DIR=$llamaPath" "-DPython_EXECUTABLE=$pythonPath" `
    "-Dpybind11_DIR=$pybindDir" '-DGGML_CUDA=ON' `
    "-DCMAKE_CUDA_COMPILER=$cudaPath\bin\nvcc.exe" "-DCUDAToolkit_ROOT=$cudaPath" `
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed: $LASTEXITCODE" }

& $cmake --build $buildPath --config Release --target qwen3asr_native --parallel 8
if ($LASTEXITCODE -ne 0) { throw "Native build failed: $LASTEXITCODE" }

Get-ChildItem -LiteralPath (Join-Path $buildPath 'python') -Filter 'qwen3asr_native*.pyd' |
    Select-Object FullName, Length
