"""Repeated Q8 load/transcribe/unload in one worker with local resource snapshots."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import manager  # noqa: E402
from scripts.evaluate_corpus import CONFIG, write_report  # noqa: E402
from scripts.integration_batch_resources import worker_process_info  # noqa: E402
from scripts.integration_resources import gpu_mib  # noqa: E402
from scripts.public_sample import ensure_sample  # noqa: E402


def snapshot() -> dict:
    process = manager.process
    if process is None or process.poll() is not None:
        raise RuntimeError("Worker exited during reload stress")
    return {**worker_process_info(process.pid), "gpu_total_mib": gpu_mib()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".runtime/evaluation/reload-stress.json")
    args = parser.parse_args()
    if not 1 <= args.cycles <= 50:
        raise ValueError("cycles must be 1..50")
    audio, rate = sf.read(ensure_sample(), dtype="float32", always_2d=True)
    pcm = np.ascontiguousarray(audio, dtype="<f4")
    options = {"sample_rate": rate, "channels": audio.shape[1], "channel": "mean",
               "mode": "offline", "language": "Chinese", "context": "", "hotwords": ""}
    report = {"cycles_requested": args.cycles, "cycles": [], "failure": None,
              "gpu_note": "GPU total includes unrelated processes; worker private bytes use real interpreter child"}
    started = time.perf_counter()
    try:
        manager.ensure()
        report["initial"] = snapshot()
        for index in range(args.cycles):
            try:
                loaded = manager.load(CONFIG)
                after_load = snapshot()
                result = manager.transcribe(pcm.tobytes(), options, CONFIG)
                if result["status"] != "complete" or "顾客" not in result["text"]:
                    raise AssertionError("Q8 sample transcription changed during reload stress")
                manager.request("POST", "/models/unload", value={})
                health = manager.request("GET", "/health")
                if health["loaded"]:
                    raise AssertionError("Worker still has model after unload")
                after_unload = snapshot()
                cycle = {"cycle": index + 1, "generation": loaded["generation"],
                         "after_load": after_load, "after_unload": after_unload,
                         "elapsed_seconds": round(time.perf_counter() - started, 3)}
                report["cycles"].append(cycle)
                write_report(args.output, report)
                print(json.dumps(cycle), flush=True)
            except Exception as exc:
                report["failure"] = {"cycle": index + 1, "error": str(exc)[:500]}
                write_report(args.output, report)
                raise
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_report(args.output, report)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
