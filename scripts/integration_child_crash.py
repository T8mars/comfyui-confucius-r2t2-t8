"""Kill the real Windows venv interpreter child and verify worker recovery."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import WorkerError, manager  # noqa: E402
from scripts.evaluate_corpus import CONFIG  # noqa: E402
from scripts.integration_batch_resources import worker_process_info  # noqa: E402


def main() -> None:
    try:
        first = manager.load(CONFIG)
        stale = manager.start_live(CONFIG, {"language": "Chinese", "context": ""})
        launcher = manager.process
        if launcher is None:
            raise AssertionError("Worker launcher missing")
        child_pid = worker_process_info(launcher.pid)["worker_pid"]
        if not isinstance(child_pid, int) or child_pid == launcher.pid:
            raise AssertionError("Could not identify actual worker interpreter")
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Stop-Process -Id {child_pid} -ErrorAction Stop"],
                       check=True, capture_output=True, text=True, timeout=10)
        # The tiny venv redirector may exit before the next request, or remain
        # briefly alive while its socket disappears. Both must recover.
        time.sleep(0.5)
        transport_failed = False
        try:
            second = manager.load(CONFIG)
        except WorkerError:
            transport_failed = True
            second = manager.load(CONFIG)
        if first["generation"] == second["generation"]:
            raise AssertionError("Real worker child crash did not create a new generation")
        try:
            manager.session_request("GET", stale["session_id"], "result")
        except WorkerError as exc:
            if "previous worker generation" not in str(exc):
                raise
        else:
            raise AssertionError("Old live session survived actual child process crash")
        fresh = manager.start_live(CONFIG, {"language": "Chinese", "context": ""})
        final = manager.session_request("POST", fresh["session_id"], "finish",
                                        value={"last_seq": -1, "total_samples": 0})
        if final["status"] != "finalized":
            raise AssertionError("New worker could not finalize a fresh session")
        print(json.dumps({"actual_child_killed": child_pid,
                          "launcher_pid": launcher.pid,
                          "initial_request_transport_failed": transport_failed,
                          "new_generation": True, "stale_session_rejected": True,
                          "fresh_session": final["status"]}))
    finally:
        manager.close()


if __name__ == "__main__":
    main()
