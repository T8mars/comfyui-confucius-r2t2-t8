"""Run the upstream sample through the Windows Q8 + Q8 native backend.

The script deliberately uses only the isolated worker environment and native
extension, so it can diagnose the first implementation gate without ComfyUI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from public_sample import ensure_sample


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=root / ".runtime/build-native-cu128")
    parser.add_argument("--model", type=Path, default=root / "models/Confucius4-R2T2-GGUF/Confucius4-R2T2-Q8_0.gguf")
    parser.add_argument("--projector", type=Path, default=root / "models/Confucius4-R2T2-GGUF/mmproj-Confucius4-R2T2-Q8_0.gguf")
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--language", default="Chinese", help="Use Auto for model language detection")
    parser.add_argument("--context", default="")
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    args.audio = args.audio or ensure_sample()

    build = args.build.resolve(strict=True)
    pyd_files = list((build / "python").rglob("qwen3asr_native*.pyd"))
    if len(pyd_files) != 1:
        raise FileNotFoundError(f"Expected one Windows extension under {build / 'python'}; found {len(pyd_files)}")
    ext_dir = pyd_files[0].parent

    # Hold handles until the process exits. The Visual Studio output layout has
    # several DLL directories (ggml, ggml-cuda, llama and mtmd).
    handles = []
    dll_dirs = {p.parent for p in build.rglob("*.dll")}
    cuda_bin = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")) / "bin"
    dll_dirs.add(cuda_bin)
    for dll_dir in sorted(dll_dirs):
        handles.append(os.add_dll_directory(str(dll_dir)))
    sys.path.insert(0, str(ext_dir))
    from qwen3asr_native import Qwen3ASRNative  # type: ignore[import-not-found]

    audio, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    if sr != 16000:
        raise ValueError(f"Smoke sample must be 16 kHz; got {sr}")
    pcm = np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)
    if pcm.size == 0 or not np.isfinite(pcm).all():
        raise ValueError("Invalid test audio")

    language = None if args.language.lower() == "auto" else args.language
    prompt = (
        "<|im_start|>system\n" + args.context + "<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    if language:
        prompt += f"language {language}<asr_text>"

    start = time.perf_counter()
    backend = Qwen3ASRNative(
        str(args.model.resolve(strict=True)),
        str(args.projector.resolve(strict=True)),
        args.ctx, args.batch, args.threads, args.gpu_layers != 0, args.gpu_layers,
    )
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = backend.generate_once(pcm, prompt, args.max_tokens)
    infer_seconds = time.perf_counter() - start
    result["whole_sequence_detokenized"] = backend.detokenize(result["token_ids"], True)
    report = {
        "model": args.model.name,
        "projector": args.projector.name,
        "sample_rate": sr,
        "duration_sec": len(pcm) / sr,
        "load_sec": round(load_seconds, 3),
        "infer_sec": round(infer_seconds, 3),
        "rtf": round(infer_seconds / (len(pcm) / sr), 3),
        "language_hint": language,
        "result": result,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
