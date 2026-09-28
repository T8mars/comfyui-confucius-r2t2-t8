"""Optional bilingual ASR check against locally synthesized Chinese+English WAV."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / ".runtime/mixed-tts.wav"
    spec = importlib.util.spec_from_file_location("r2t2_mixed_test", root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    assert spec and spec.loader
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    from r2t2_mixed_test.bridge import manager

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    comfy_audio = {"waveform": np.ascontiguousarray(audio.T[None]), "sample_rate": rate}
    model, _ = plugin.NODE_CLASS_MAPPINGS["R2T2GGUFLoader"]().load(8192, 1024, 8, -1)
    node = plugin.NODE_CLASS_MAPPINGS["R2T2Transcribe"]()
    output = {}
    for language_hint in ("Auto", "Chinese", "English"):
        for mode in ("offline", "stream"):
            text, language, details = node.transcribe(model, comfy_audio, mode, language_hint, "", "", "mean")
            output[f"{mode}_{language_hint}"] = {"text": text, "language": language,
                "events": len(json.loads(details)["events"])}
    assert "人工智能" in output["stream_Auto"]["text"]
    assert "weather" in output["stream_Auto"]["text"].lower()
    assert "人工智能" in output["offline_Chinese"]["text"]
    assert "weather" in output["offline_Chinese"]["text"].lower()
    print(json.dumps(output, ensure_ascii=True))
    manager.close()


if __name__ == "__main__":
    main()
