"""Sample local GPU usage and worker RSS across Q8 load/unload (diagnostic only)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import manager  # noqa: E402

CONFIG = {"n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1}


def gpu_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(re.search(r"\d+", output).group())


def worker_rss_mib() -> float:
    command = (
        f"$parentPid = {manager.process.pid}; "
        "$children = @(Get-CimInstance Win32_Process -Filter \"ParentProcessId = $parentPid\"); "
        "$pids = @($parentPid) + @($children | ForEach-Object { $_.ProcessId }); "
        "(($pids | ForEach-Object { (Get-Process -Id $_).WorkingSet64 }) | Measure-Object -Sum).Sum"
    )
    output = subprocess.check_output(["powershell", "-NoProfile", "-Command", command], text=True)
    return round(int(output.strip()) / 1048576, 1)


def main() -> None:
    try:
        manager.ensure()
        before = {"gpu_mib": gpu_mib(), "worker_rss_mib": worker_rss_mib()}
        manager.load(CONFIG)
        time.sleep(2)
        loaded = {"gpu_mib": gpu_mib(), "worker_rss_mib": worker_rss_mib()}
        manager.request("POST", "/models/unload", value={})
        time.sleep(3)
        unloaded = {"gpu_mib": gpu_mib(), "worker_rss_mib": worker_rss_mib()}
        assert unloaded["gpu_mib"] < loaded["gpu_mib"], (before, loaded, unloaded)
        print(json.dumps({"before": before, "loaded": loaded, "unloaded": unloaded,
                          "note": "total GPU usage includes other processes"}))
    finally:
        manager.close()


if __name__ == "__main__":
    main()
