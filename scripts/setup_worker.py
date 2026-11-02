"""Install the same isolated Q8 worker on Windows and Linux; no llama CLI needed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LLAMA_COMMIT = "ad6c66839af3c5646fba8c6c2e2087a1e4e38948"
NATIVE_VERSION = "0.2.0"
RELEASE_BASE = ("https://github.com/T8mars/comfyui-confucius-r2t2-t8/"
                f"releases/download/native-v{NATIVE_VERSION}")


def run(command, **kwargs):
    print("+ " + subprocess.list2cmdline(list(map(str, command))), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def interpreter(base: Path, system: str | None = None) -> Path:
    return base / ("Scripts/python.exe" if (system or sys.platform) == "win32" else "bin/python")


def python312(explicit: str | None = None) -> list[str]:
    if explicit:
        candidates = [[explicit]]
    else:
        candidates = [[sys.executable], ["python3.12"]]
        if sys.platform == "win32":
            candidates.append(["py", "-3.12"])
    for candidate in candidates:
        try:
            probe = subprocess.run(candidate + ["-c", "import sys; print('%s.%s' % sys.version_info[:2])"],
                                   check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        if probe.stdout.strip() == "3.12":
            return candidate
    raise RuntimeError("Install CPython 3.12 first, then run python3.12 scripts/setup_worker.py "
                       "(Windows: py -3.12 scripts/setup_worker.py), or pass --python <path>.")


def detect_backend(selected: str) -> str:
    if selected != "auto":
        return selected
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            probe = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                                   timeout=10, capture_output=True, text=True, check=True)
            if probe.stdout.strip():
                return "cuda"
        except (OSError, subprocess.SubprocessError):
            pass
    return "cpu"


def download(url: str, target: Path, expected: str | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "Confucius4-ComfyUI-setup/0.2.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        while chunk := response.read(4 * 1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
    if expected is not None and digest.hexdigest() != expected:
        raise RuntimeError(f"SHA256 mismatch for {target.name}; download was not installed")
    temporary.replace(target)
    return target


def select_wheel(manifest: dict, tags: set[str], backend: str) -> dict:
    if manifest.get("version") != NATIVE_VERSION or manifest.get("llama_commit") != LLAMA_COMMIT:
        raise ValueError("Wheel manifest does not match the pinned native release")
    for item in manifest.get("wheels", []):
        filename = item.get("filename", "")
        if (not filename.endswith(".whl") or Path(filename).name != filename
                or "/" in filename or "\\" in filename):
            raise ValueError("Invalid wheel asset name")
        if item.get("backend") == backend and tags.intersection(item.get("tags", [])):
            sha = item.get("sha256", "")
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("Invalid wheel SHA256")
            return item
    raise RuntimeError("No compatible prebuilt wheel for this Python/OS/libc/backend. "
                       "Use --native source with a C++ compiler (and CUDA Toolkit for --backend cuda), "
                       "or choose --backend cpu.")


def check_wheel(path: Path, backend: str) -> None:
    with zipfile.ZipFile(path) as package:
        info = json.loads(package.read("r2t2_native/build-info.json"))
        if (info.get("api_version") != 1 or info.get("llama_commit") != LLAMA_COMMIT
                or info.get("cuda") is not (backend == "cuda")):
            raise ValueError("Wheel API, pinned llama commit or CPU/CUDA backend does not match")
        if any(name.lower().endswith(".exe") for name in package.namelist()):
            raise ValueError("Native wheel must contain libraries, not executable programs")


def install_wheel(python: Path, backend: str, wheel: Path | None) -> None:
    if wheel is None:
        cache = ROOT / ".runtime" / "wheels"
        manifest_path = download(RELEASE_BASE + "/native-wheels.json", cache / "native-wheels.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        probe = subprocess.run([str(python), "-c", "import json; from packaging.tags import sys_tags; "
                                "print(json.dumps([str(t) for t in sys_tags()]))"],
                               capture_output=True, text=True, check=True)
        item = select_wheel(manifest, set(json.loads(probe.stdout)), backend)
        wheel = download(RELEASE_BASE + "/" + item["filename"], cache / item["filename"], item["sha256"])
    wheel = wheel.resolve(strict=True)
    check_wheel(wheel, backend)
    run([python, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheel])


def cuda_toolchain(directory: str | None = None) -> tuple[Path, str]:
    location = directory or os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if location:
        root = Path(location).resolve()
        nvcc = root / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
    else:
        found = shutil.which("nvcc")
        if not found:
            raise RuntimeError("CUDA source builds require CUDA Toolkit 12.8; prebuilt wheels do not")
        nvcc = Path(found).resolve()
        root = nvcc.parent.parent
    output = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release (\d+\.\d+)", output)
    if not match or match.group(1) != "12.8":
        raise RuntimeError("Use CUDA Toolkit 12.8 to match the pinned pip CUDA runtimes, "
                           "or install a prebuilt wheel instead")
    return root, match.group(1)


def source_wheel(python: Path, backend: str, cuda_arch: str | None, cuda_directory: str | None = None) -> Path:
    source = ROOT / ".reference" / "llama.cpp"
    if not (source / "CMakeLists.txt").is_file():
        source.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout",
             "https://github.com/ggml-org/llama.cpp.git", source])
        run(["git", "-C", source, "checkout", "--detach", LLAMA_COMMIT])
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != LLAMA_COMMIT:
        raise RuntimeError(f"Expected llama.cpp {LLAMA_COMMIT}, got {commit}; existing checkout left unchanged")
    destination = ROOT / ".runtime" / "wheels" / backend
    destination.mkdir(parents=True, exist_ok=True)
    command = [python, "-m", "pip", "wheel", "--no-deps", ROOT / "vendor/r2t2_native", "-w", destination,
               f"--config-settings=cmake.define.LLAMA_CPP_DIR={source}",
               f"--config-settings=cmake.define.GGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}",
               f"--config-settings=wheel.build-tag=1{'cu128' if backend == 'cuda' else 'cpu'}"]
    if cuda_arch:
        command.append(f"--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES={cuda_arch}")
    if backend == "cuda":
        toolkit, version = cuda_toolchain(cuda_directory)
        command.append(f"--config-settings=cmake.define.CUDAToolkit_ROOT={toolkit}")
        if sys.platform == "win32":
            # VS can keep stale integration files for a newer, absent Toolkit.
            # Select the actual pinned compiler instead of its highest version.
            command.append(f"--config-settings=cmake.args=-Tcuda={version}")
        else:
            command.append(f"--config-settings=cmake.define.CMAKE_CUDA_COMPILER={toolkit / 'bin/nvcc'}")
    run(command)
    wheels = list(destination.glob(f"confucius_r2t2_native-{NATIVE_VERSION}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one native wheel under {destination}, got {len(wheels)}")
    return wheels[0]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", help="CPython 3.12 interpreter; dependencies go only into the new worker")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--native", choices=("wheel", "source"), default="wheel")
    parser.add_argument("--wheel", type=Path, help="Use a local verified native wheel instead of downloading")
    parser.add_argument("--cuda-arch", help="Source build only, e.g. 86;89;120; defaults to the build system's selection")
    parser.add_argument("--cuda-toolkit", help="Source build only: CUDA Toolkit 12.8 directory")
    parser.add_argument("--skip-model-download", action="store_true")
    parser.add_argument("--skip-native", action="store_true", help="Set up worker dependencies only")
    args = parser.parse_args(argv)
    if sys.platform not in ("win32", "linux") or platform.machine().lower() not in ("amd64", "x86_64"):
        parser.error("This release supports Windows x64 and Linux x86_64")
    if args.wheel and args.native == "source":
        parser.error("Choose --wheel or --native source, not both")
    base = ROOT / ".runtime" / "worker"
    python = interpreter(base)
    if not python.is_file():
        foreign = base / ("bin/python" if sys.platform == "win32" else "Scripts/python.exe")
        if foreign.is_file():
            raise RuntimeError(".runtime/worker contains another OS's environment. Rename that local "
                               "directory and rerun setup; Windows environments cannot be copied to Linux.")
        run(python312(args.python) + ["-m", "venv", str(base)])
    version = subprocess.check_output([str(python), "-c", "import sys; print(sys.version_info[:2])"], text=True)
    if version.strip() != "(3, 12)":
        raise RuntimeError("The existing worker must use CPython 3.12; it was not replaced")
    run([python, "-m", "pip", "install", "-r", ROOT / "requirements-worker.txt"])
    backend = detect_backend(args.backend)
    if not args.skip_native:
        print(f"Installing {backend} native Python wheel (no llama executable)", flush=True)
        wheel = source_wheel(python, backend, args.cuda_arch, args.cuda_toolkit) if args.native == "source" else args.wheel
        install_wheel(python, backend, wheel)
        if backend == "cuda":
            run([python, "-m", "pip", "install", "nvidia-cublas-cu12==12.8.4.1", "nvidia-cuda-runtime-cu12==12.8.90"])
        run([python, "-c", "import r2t2_native; print(r2t2_native.build_info()); r2t2_native.load_native()"])
    if not args.skip_model_download:
        for script in ("download_q8.py", "download_vad.py"):
            run([python, ROOT / "scripts" / script])
    print(f"Worker ready: {python}. Restart ComfyUI.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Setup failed: {exc}") from exc
