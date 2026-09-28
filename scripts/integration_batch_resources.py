"""Exercise repeated Q8 file jobs and record Windows worker memory locally."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import manager  # noqa: E402
from scripts.evaluate_corpus import CONFIG, read_jsonl, validate_manifest, write_report  # noqa: E402


def worker_process_info(launcher_pid: int) -> dict:
    # Windows venv python.exe is a tiny redirector. Query its real interpreter
    # child, otherwise memory samples would misleadingly report <1 MB.
    command = (
        f"$p = Get-CimInstance Win32_Process -Filter 'ParentProcessId = {int(launcher_pid)}' "
        "| Where-Object { $_.Name -eq 'python.exe' } | Select-Object -First 1; "
        "if ($p) { $m = Get-Process -Id $p.ProcessId; "
        "[pscustomobject]@{ worker_pid = $p.ProcessId; private_bytes = $m.PrivateMemorySize64; "
        "working_set_bytes = $m.WorkingSet64 } | ConvertTo-Json -Compress }"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, timeout=10, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else {
        "worker_pid": None, "private_bytes": None, "working_set_bytes": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/evaluation/batch-resources.json")
    args = parser.parse_args()
    if not 1 <= args.passes <= 3:
        raise ValueError("Choose 1-3 passes")
    entries = validate_manifest(read_jsonl(args.manifest), args.manifest.resolve())
    report = {"manifest": str(args.manifest.resolve()), "passes": args.passes,
              "jobs": 0, "snapshots": [], "failure": None}
    started = time.perf_counter()
    try:
        manager.load(CONFIG)
        for repetition in range(args.passes):
            for item in entries:
                audio, rate = sf.read(item["audio_path"], dtype="float32", always_2d=True)
                pcm = np.ascontiguousarray(audio, dtype="<f4")
                try:
                    result = manager.transcribe(pcm.tobytes(), {
                        "sample_rate": rate, "channels": audio.shape[1], "channel": "mean",
                        "mode": "offline", "language": item["language_hint"],
                        "context": "", "hotwords": ""}, CONFIG)
                    if result["status"] != "complete":
                        raise RuntimeError(f"Unexpected status {result['status']}")
                except Exception as exc:
                    report["failure"] = {"pass": repetition + 1, "case_id": item["id"],
                                         "error": str(exc)[:500]}
                    write_report(args.output, report)
                    raise
                report["jobs"] += 1
                if report["jobs"] % 10 == 0:
                    process = manager.process
                    pid = process.pid if process is not None else None
                    snapshot = {"jobs": report["jobs"], "launcher_pid": pid,
                                **(worker_process_info(pid) if pid else {}),
                                "elapsed_seconds": round(time.perf_counter() - started, 3)}
                    report["snapshots"].append(snapshot)
                    write_report(args.output, report)
                    print(json.dumps(snapshot), flush=True)
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_report(args.output, report)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
