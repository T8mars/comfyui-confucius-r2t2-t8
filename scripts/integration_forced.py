"""Diagnose a real-Q8 duration boundary with VAD boundaries deliberately disabled."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from public_sample import ensure_sample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2t2_core.native import NativeQ8Engine  # noqa: E402
from r2t2_core.segmented import SegmentedStream  # noqa: E402


class NoBoundaryVAD:
    def feed(self, pcm):
        return []


def main() -> None:
    sample, rate = sf.read(ensure_sample(), dtype="float32")
    assert rate == 16000
    pcm = np.tile(sample, 4).astype("<f4")
    stream = SegmentedStream(NativeQ8Engine(), language="Chinese", vad=NoBoundaryVAD())
    events = []
    for start in range(0, len(pcm), 640):
        events.extend(stream.feed(pcm[start:start + 640]))
    events.append(stream.finish())
    assert stream.forced_boundaries == 1
    assert [(s["start_sample"], s["end_sample"]) for s in stream.segments] == [
        (0, 320000), (320000, len(pcm))]
    previous = ""
    for event in events:
        assert event["stable_text"] == previous + event.get("delta", "")
        previous = event["stable_text"]
    print(json.dumps({"seconds": len(pcm) / 16000, "forced_boundaries": stream.forced_boundaries,
                      "status": "requires_review", "sentence_count": stream.stable_text.count("之前有顾客"),
                      "text": stream.stable_text}, ensure_ascii=True))


if __name__ == "__main__":
    main()
