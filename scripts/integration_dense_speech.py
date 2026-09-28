"""Stress-test 90 seconds or 10 minutes of tightly joined public read speech.

This deliberately contracts low-energy pauses between/within FLEURS
utterances. It is synthetic and must not be treated as natural no-pause
quality evidence, but exposes duration-limit behavior and sample ownership.
All audio and reports stay in ignored .runtime/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import manager  # noqa: E402
from r2t2_core.evaluation import score  # noqa: E402
from scripts.evaluate_corpus import CONFIG, file_sha256, read_jsonl, validate_manifest, write_report  # noqa: E402

SAMPLE_RATE = 16000
FRAME = 160  # 10 ms
MANIFEST = ROOT / ".runtime/evaluation/fleurs/fleurs-holdout-100-each.jsonl"
DEST = ROOT / ".runtime/evaluation/dense-speech"


def contract_quiet(pcm: np.ndarray) -> tuple[np.ndarray, int, float]:
    value = np.asarray(pcm, dtype=np.float32)
    framed = value[:len(value) // FRAME * FRAME].reshape(-1, FRAME)
    rms = np.sqrt(np.mean(np.square(framed.astype(np.float64)), axis=1))
    threshold = max(0.0003, 0.05 * float(np.percentile(rms, 90)))
    quiet = rms < threshold
    mask = np.ones(len(value), dtype=bool)
    changes = np.diff(np.pad(quiet.astype(np.int8), (1, 1)))
    for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        # Keep a 30 ms edge on each side of internal pauses; longer than
        # 200 ms is contracted. Leading/trailing background is removed.
        if first == 0:
            mask[:last * FRAME] = False
        elif last == len(rms):
            mask[first * FRAME:] = False
        elif last - first > 20:
            mask[(first + 3) * FRAME:(last - 3) * FRAME] = False
    compact = value[mask]
    if compact.size < SAMPLE_RATE or not np.isfinite(compact).all():
        raise ValueError("Speech contraction removed too much audio")
    return compact, len(value) - len(compact), threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-ms", type=int, choices=(160, 320, 640), default=160)
    parser.add_argument("--target-seconds", type=int, choices=(90, 600), default=90)
    args = parser.parse_args()
    entries = validate_manifest(read_jsonl(MANIFEST), MANIFEST)
    selected = [row for row in entries if row["language"] == "Chinese"]
    DEST.mkdir(parents=True, exist_ok=True)
    pieces = []
    sources = []
    seconds = 0.0
    for item in selected:
        audio, rate = sf.read(item["audio_path"], dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE or audio.shape[1] != 1:
            raise ValueError("Expected mono 16 kHz FLEURS audio")
        compressed, removed, threshold = contract_quiet(audio[:, 0])
        pieces.append(compressed)
        seconds += len(compressed) / SAMPLE_RATE
        sources.append({"id": item["id"], "audio_sha256": item["sha256"],
                        "reference": item["reference"], "input_samples": len(audio),
                        "retained_samples": len(compressed), "removed_samples": removed,
                        "rms_threshold": threshold})
        if seconds >= args.target_seconds:
            break
    if not args.target_seconds <= seconds <= args.target_seconds + 25:
        raise ValueError(f"Dense speech target out of bounds: {seconds:.2f} s")
    pcm = np.concatenate(pieces).astype(np.float32)
    audio_path = DEST / f"fleurs-joined-{args.target_seconds}s.wav"
    sf.write(audio_path, pcm, SAMPLE_RATE, subtype="PCM_24")
    reference = " ".join(row["reference"] for row in sources)
    # The built-in VAD still decides whether a given acoustic interval is a
    # silence; this test claims only deterministic low-energy contraction.
    result = manager.transcribe(pcm.tobytes(), {
        "sample_rate": SAMPLE_RATE, "channels": 1, "mode": "stream",
        "language": "Chinese", "context": "", "hotwords": "", "channel": "mean",
        "auto_gain": False, "stream_chunk_ms": args.chunk_ms}, CONFIG)
    segments = result["segments"]
    if (not segments or segments[0]["start_sample"] != 0 or
            segments[-1]["end_sample"] != len(pcm) or
            any(left["end_sample"] != right["start_sample"]
                for left, right in zip(segments, segments[1:]))):
        raise AssertionError("Dense speech segmentation lost sample ownership")
    stable = ""
    for event in result["events"]:
        if event["stable_text"] != stable + event["delta"]:
            raise AssertionError("Stable text revised during dense speech test")
        stable = event["stable_text"]
    report = {"test": "synthetic_low_energy_contracted_FLEURS",
              "target_seconds": args.target_seconds,
              "stream_chunk_ms": args.chunk_ms,
              "source_manifest_sha256": file_sha256(MANIFEST),
              "audio_sha256": file_sha256(audio_path), "audio_path": str(audio_path),
              "reference": reference, "sources": sources,
              "audio_seconds": len(pcm) / SAMPLE_RATE,
              "q8_score": score(reference, result["text"], "cer"),
              "rtf": result["elapsed_ms"] / (len(pcm) / SAMPLE_RATE * 1000),
              "result": result}
    report_path = DEST / f"q8-dense-{args.target_seconds}s-chunk{args.chunk_ms}.json"
    write_report(report_path, report)
    print(json.dumps({"audio_seconds": report["audio_seconds"],
                      "source_utterances": len(sources), "status": result["status"],
                      "forced_boundaries": result.get("forced_boundaries"),
                      "segments": len(segments), "q8_score": report["q8_score"],
                      "rtf": report["rtf"], "report": str(report_path)},
                     ensure_ascii=False))
    manager.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        manager.close()
