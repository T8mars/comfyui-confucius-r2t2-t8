"""Verify worker crash recovery and rejection of stale live-session capabilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import WorkerError, manager  # noqa: E402

CONFIG = {"n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1}


def main() -> None:
    try:
        first = manager.load(CONFIG)
        stale = manager.start_live(CONFIG, {"language": "Chinese", "context": ""})
        manager.process.kill()
        manager.process.wait(timeout=15)
        second = manager.load(CONFIG)
        assert first["generation"] != second["generation"]
        try:
            manager.session_request("GET", stale["session_id"], "result")
        except WorkerError as exc:
            assert "previous worker generation" in str(exc)
        else:
            raise AssertionError("Old session survived worker restart")
        fresh = manager.start_live(CONFIG, {"language": "Chinese", "context": ""})
        final = manager.session_request("POST", fresh["session_id"], "finish",
                                        value={"last_seq": -1, "total_samples": 0})
        assert final["status"] == "finalized"
        print(json.dumps({"restart": "passed", "stale_session": "rejected",
                          "fresh_session": final["status"]}))
    finally:
        manager.close()


if __name__ == "__main__":
    main()
