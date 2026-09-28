"""Pinned Q8 decoder + Q8 audio-projector loader for the isolated worker."""

from __future__ import annotations

import hashlib
import importlib
import os
import struct
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GGUF_DIR = ROOT / "models" / "Confucius4-R2T2-GGUF"
MODEL_NAME = "Confucius4-R2T2-Q8_0.gguf"
PROJECTOR_NAME = "mmproj-Confucius4-R2T2-Q8_0.gguf"
OFFICIAL = {
    MODEL_NAME: (1_834_422_208, "151097e43957a19984ea7de66e8144ce69b95039eb31c93da4f58db367e455c3"),
    PROJECTOR_NAME: (348_336_544, "8dc2c67e6a0484114928142d098db7ad94ae9f34c78948ef9d37a9678418cb65"),
}
_DLL_HANDLES: list[Any] = []
SUPPORTED_LANGUAGES = frozenset({"Chinese", "English", "Cantonese", "Japanese", "Korean",
                                 "German", "French", "Russian", "Portuguese", "Spanish", "Italian"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pair(directory: Path = GGUF_DIR, *, hash_files: bool = True) -> tuple[Path, Path]:
    """Require the exact official Q8 files; never auto-select another GGUF."""
    directory = directory.resolve(strict=True)
    paths = []
    for name in (MODEL_NAME, PROJECTOR_NAME):
        path = (directory / name).resolve(strict=True)
        if path.parent != directory:
            raise ValueError(f"Model path escapes its registered directory: {path}")
        expected_size, expected_hash = OFFICIAL[name]
        if path.stat().st_size != expected_size:
            raise ValueError(f"Model size mismatch: {name}")
        with path.open("rb") as f:
            magic, version = struct.unpack("<4sI", f.read(8))
        if magic != b"GGUF" or version != 3:
            raise ValueError(f"Invalid GGUF header: {name}")
        if hash_files and _sha256(path) != expected_hash:
            raise ValueError(f"Model SHA-256 mismatch: {name}")
        paths.append(path)
    return paths[0], paths[1]


def _load_extension(build: Path):
    if os.name != "nt":
        raise RuntimeError("This worker build is for Windows x64")
    build = build.resolve(strict=True)
    pyd_files = list((build / "python").rglob("qwen3asr_native*.pyd"))
    if len(pyd_files) != 1:
        raise FileNotFoundError(f"Expected one qwen3asr_native .pyd under {build / 'python'}; found {len(pyd_files)}")
    pyd_dir = pyd_files[0].parent
    dll_dirs = {p.parent for p in build.rglob("*.dll")}
    cuda_bin = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")) / "bin"
    dll_dirs.add(cuda_bin)
    for dll_dir in sorted(dll_dirs):
        _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
    if str(pyd_dir) not in sys.path:
        sys.path.insert(0, str(pyd_dir))
    return importlib.import_module("qwen3asr_native")


def build_prompt(context: str = "", language: str | None = None, prefix: str = "") -> str:
    if not isinstance(context, str) or not isinstance(prefix, str):
        raise TypeError("Context and prefix must be strings")
    if len(context) > 8192:
        raise ValueError("CONTEXT_LIMIT: context exceeds 8192 characters")
    if language is not None and language not in SUPPORTED_LANGUAGES:
        raise ValueError("Unsupported recognition language")
    prompt = (
        "<|im_start|>system\n" + context + "<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    if language is not None:
        prompt += f"language {language}<asr_text>"
    return prompt + prefix


def parse_raw(raw: str, forced_language: str | None = None) -> tuple[str, str]:
    raw = raw.split("|", 1)[0].replace("\ufffd", "")
    if forced_language is not None:
        return forced_language, raw
    if "<asr_text>" not in raw:
        return "", ""
    meta, text = raw.split("<asr_text>", 1)
    language = ""
    for line in meta.splitlines():
        value = line.strip()
        if value.lower().startswith("language "):
            language = value[9:].strip()
            break
    if language.lower() == "none":
        return "", text
    return language, text


class NativeQ8Engine:
    def __init__(
        self,
        model_dir: Path = GGUF_DIR,
        build_dir: Path = ROOT / ".runtime" / "build-native-cu128",
        *,
        n_ctx: int = 8192,
        n_batch: int = 1024,
        n_threads: int = 8,
        gpu_layers: int = -1,
        hash_files: bool = True,
    ) -> None:
        model, projector = verify_pair(model_dir, hash_files=hash_files)
        native = _load_extension(build_dir)
        self.model_name = model.name
        self.projector_name = projector.name
        self.n_ctx = n_ctx
        self._lock = threading.RLock()
        self._native = native.Qwen3ASRNative(
            str(model), str(projector), n_ctx, n_batch, n_threads,
            gpu_layers != 0, gpu_layers,
        )

    def generate(self, audio: np.ndarray, *, context: str = "", language: str | None = None,
                 prefix: str = "", max_tokens: int = 512) -> dict:
        pcm = np.ascontiguousarray(audio, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0 or not np.isfinite(pcm).all():
            raise ValueError("Expected nonempty finite mono float32 audio")
        if max_tokens < 1 or max_tokens > 4096:
            raise ValueError("max_tokens out of range")
        prompt = build_prompt(context, language, prefix)
        with self._lock:
            result = dict(self._native.generate_once(pcm, prompt, max_tokens))
        return result

    def tokenize(self, text: str) -> list[int]:
        with self._lock:
            return list(self._native.tokenize(text, False, True))

    def detokenize(self, ids: list[int]) -> str:
        with self._lock:
            return str(self._native.detokenize([int(i) for i in ids], True))

    def transcribe(self, audio: np.ndarray, *, context: str = "", language: str | None = None,
                   max_tokens: int = 1024) -> dict:
        result = self.generate(audio, context=context, language=language, max_tokens=max_tokens)
        detected_language, text = parse_raw(result["text"], language)
        return {
            "text": text,
            "language": detected_language,
            "finish_reason": result.get("finish_reason"),
            "truncated": result.get("finish_reason") == "length",
            "prompt_token_count": result.get("prompt_token_count"),
            "prompt_position_count": result.get("prompt_position_count"),
            "model": self.model_name,
            "projector": self.projector_name,
        }
