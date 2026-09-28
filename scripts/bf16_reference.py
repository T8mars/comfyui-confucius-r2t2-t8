"""Produce offline BF16 reference predictions in a separate Transformers environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_corpus import file_sha256, read_jsonl, validate_manifest  # noqa: E402
from r2t2_core.evaluation import BF16_MODEL_SHA256  # noqa: E402

MODEL_DIR = ROOT / "models/Confucius4-R2T2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/evaluation/bf16.jsonl")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / ".runtime").resolve()):
        raise ValueError("BF16 predictions must stay in the Git-ignored .runtime directory")
    entries = validate_manifest(read_jsonl(args.manifest), args.manifest.resolve())
    if file_sha256(MODEL_DIR / "model.safetensors") != BF16_MODEL_SHA256:
        raise ValueError("BF16 source weight SHA-256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 reference requires the separate CUDA-capable environment")
    print("Loading pinned BF16 model in separate Transformers environment", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", max_inference_batch_size=1, max_new_tokens=1024)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".bf16-", suffix=".tmp", dir=output.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for item in entries:
                audio, rate = sf.read(item["audio_path"], dtype="float32", always_2d=True)
                mono = np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)
                language = None if item["language_hint"] == "Auto" else item["language_hint"]
                started = time.perf_counter()
                prediction = model.transcribe(audio=[(mono, rate)], language=[language])[0]
                record = {"id": item["id"], "text": prediction.text,
                          "language": prediction.language, "audio_sha256": item["sha256"],
                          "model_sha256": BF16_MODEL_SHA256,
                          "qwen_asr_version": importlib.metadata.version("qwen-asr"),
                          "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({"id": item["id"], "language": prediction.language,
                                  "elapsed_ms": record["elapsed_ms"]}), flush=True)
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        print(json.dumps({"bf16_predictions": str(output), "cases": len(entries)}, ensure_ascii=True))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
