"""Optional English ASR check against a locally synthesized English WAV."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", type=Path, default=root / ".runtime/english-tts.wav")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("r2t2_english_test", root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    assert spec and spec.loader
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    from r2t2_english_test.bridge import manager

    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    comfy_audio = {"waveform": np.ascontiguousarray(audio.T[None]), "sample_rate": rate}
    model, _ = plugin.NODE_CLASS_MAPPINGS["R2T2GGUFLoader"]().load(8192, 1024, 8, -1)
    node = plugin.NODE_CLASS_MAPPINGS["R2T2Transcribe"]()
    output = {}
    for mode in ("offline", "stream"):
        text, language, details = node.transcribe(model, comfy_audio, mode, "Auto", "", "", "mean")
        output[mode] = {"text": text, "language": language,
                        "audio_samples_16k": json.loads(details)["audio_samples_16k"]}
        assert language == "English", output
        assert "weather" in text.lower() and "train" in text.lower(), output
    print(json.dumps(output, ensure_ascii=True))
    manager.close()


if __name__ == "__main__":
    main()
