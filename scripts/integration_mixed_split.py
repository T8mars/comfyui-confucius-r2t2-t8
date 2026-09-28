"""Diagnose mixed utterances separated by an exact digital-zero interval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2t2_core.evaluation import score  # noqa: E402
from r2t2_core.native import NativeQ8Engine  # noqa: E402


def exact_zero_gap(pcm: np.ndarray, *, minimum: int = 2400) -> tuple[int, int] | None:
    zeros = np.asarray(pcm == 0, dtype=np.int8)
    changes = np.diff(np.pad(zeros, (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    eligible = [(int(start), int(end)) for start, end in zip(starts, ends)
                if end - start >= minimum and start >= 16000 and len(pcm) - end >= 16000]
    return max(eligible, key=lambda item: item[1] - item[0]) if eligible else None


def main() -> None:
    manifest = ROOT / ".runtime/evaluation/fleurs/fleurs-mixed-only.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    engine = NativeQ8Engine()
    total_errors = total_units = 0
    for row in rows:
        audio, rate = sf.read(manifest.parent / row["audio_path"], dtype="float32")
        if rate != 16000:
            raise ValueError("Expected 16 kHz")
        gap = exact_zero_gap(audio)
        if gap is None:
            raise ValueError(f"No interior digital-zero gap in {row['id']}")
        pieces = [engine.transcribe(np.ascontiguousarray(part), language=None)["text"]
                  for part in (audio[:gap[0]], audio[gap[1]:])]
        result = score(row["reference"], " ".join(pieces), "cer")
        total_errors += result["errors"]
        total_units += result["reference_units"]
        print(json.dumps({"id": row["id"], "gap_samples": gap[1] - gap[0],
                          "errors": result["errors"], "reference_units": result["reference_units"]}),
              flush=True)
    print(json.dumps({"total_errors": total_errors, "reference_units": total_units,
                      "cer": total_errors / total_units}), flush=True)


if __name__ == "__main__":
    main()
