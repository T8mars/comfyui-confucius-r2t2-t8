"""Load the packaged extension and, for CUDA wheels, NVIDIA's pip runtimes."""

from __future__ import annotations

import ctypes
import importlib
import json
import os
import sys
from pathlib import Path

API_VERSION = 1
LLAMA_COMMIT = "ad6c66839af3c5646fba8c6c2e2087a1e4e38948"
_HANDLES = []
_EXTENSION = None


def build_info() -> dict:
    return json.loads(Path(__file__).with_name("build-info.json").read_text(encoding="utf-8"))


def _cuda_libraries() -> None:
    # The optional NVIDIA wheels contain runtime libraries, not llama executables.
    # Load dependencies by absolute path; never mutate the user's global PATH.
    bases = [Path(p) / "nvidia" for p in sys.path if p]
    if os.name == "nt":
        for base in bases:
            for name in ("cublas", "cuda_runtime"):
                for leaf in ("bin", "lib"):
                    directory = base / name / leaf
                    if directory.is_dir():
                        _HANDLES.append(os.add_dll_directory(str(directory)))
        cuda = os.environ.get("CUDA_PATH")
        if cuda and (Path(cuda) / "bin").is_dir():
            _HANDLES.append(os.add_dll_directory(str(Path(cuda) / "bin")))
    else:
        # cublas depends on cublasLt. Preload in dependency order so pip's
        # relocatable paths work even without LD_LIBRARY_PATH or a Toolkit.
        for component, library in (("cublas", "libcublasLt.so.12"),
                                   ("cublas", "libcublas.so.12"),
                                   ("cuda_runtime", "libcudart.so.12")):
            for base in bases:
                path = base / component / "lib" / library
                if path.is_file():
                    _HANDLES.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
                    break


def load_native():
    global _EXTENSION
    if _EXTENSION is None:
        info = build_info()
        if info["llama_commit"] != LLAMA_COMMIT or info["api_version"] != API_VERSION:
            raise RuntimeError("Incompatible Confucius native wheel; reinstall the pinned release")
        if info["cuda"]:
            _cuda_libraries()
        try:
            _EXTENSION = importlib.import_module(".qwen3asr_native", __name__)
        except (ImportError, OSError) as exc:
            raise ImportError("Cannot load the Confucius native wheel. For CUDA, install "
                              "confucius-r2t2-native[cuda] runtime dependencies, or select "
                              "the CPU wheel. No llama executable is required.") from exc
    return _EXTENSION
