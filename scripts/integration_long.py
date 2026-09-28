"""Exercise sample ownership and status on a >30-second public-audio sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from public_sample import ensure_sample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import manager  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silence", action="store_true", help="Test forced-boundary status on 31 s silence")
    parser.add_argument("--silence-seconds", type=int, default=31)
    parser.add_argument("--repeats", type=int, default=5, help="Repeat the public 6.74 s sentence")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error("--repeats must be between 1 and 100")
    if not 1 <= args.silence_seconds <= 600:
        parser.error("--silence-seconds must be between 1 and 600")
    sample, rate = sf.read(ensure_sample(), dtype="float32")
    assert rate == 16000
    pcm = (np.zeros(args.silence_seconds * rate, dtype="<f4") if args.silence
           else np.tile(sample, args.repeats).astype("<f4"))
    result = manager.transcribe(pcm.tobytes(), {"sample_rate": rate, "channels": 1,
        "mode": "stream", "language": "Chinese", "context": "", "hotwords": "", "channel": "mean"},
        {"n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1})
    segments = result["segments"]
    assert result["audio_samples_16k"] == len(pcm)
    assert segments[0]["start_sample"] == 0 and segments[-1]["end_sample"] == len(pcm)
    for previous, current in zip(segments, segments[1:]):
        assert previous["end_sample"] == current["start_sample"]
    text = ""
    for event in result["events"]:
        assert event["stable_text"] == text + event["delta"]
        text = event["stable_text"]
    assert result["events"][-1]["final"]
    if args.silence:
        assert result["status"] == "complete" and result["forced_boundaries"] == 0
        assert sum(segment["skipped_zero_samples"] for segment in segments) == len(pcm)
    else:
        assert result["status"] == "complete" and result["forced_boundaries"] == 0
    print(json.dumps({"seconds": len(pcm) / 16000, "status": result["status"],
                      "segments": len(segments), "forced_boundaries": result["forced_boundaries"],
                      "events": len(result["events"]), "elapsed_ms": result["elapsed_ms"],
                      "text_chars": len(result["text"]), "sentence_count": result["text"].count("之前有顾客"),
                      "text_excerpt": result["text"][:100]}, ensure_ascii=True))
    manager.close()


if __name__ == "__main__":
    main()
