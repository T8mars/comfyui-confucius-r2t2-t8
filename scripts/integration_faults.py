"""Exercise sequence rejection, cancellation, and unload leasing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import WorkerError, manager  # noqa: E402

CONFIG = {"n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1}


def expect_error(code: str, operation) -> None:
    try:
        operation()
    except WorkerError as exc:
        assert code in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected {code}")


def main() -> None:
    manager.load(CONFIG)
    sid = manager.start_live(CONFIG, {"language": "Chinese", "context": ""})["session_id"]
    pcm = np.zeros(640, dtype="<f4").tobytes()
    headers = {"X-R2T2-Seq": "0", "X-R2T2-Start-Sample": "0",
               "Content-Type": "application/octet-stream"}
    first = manager.session_request("POST", sid, "feed", body=pcm, headers=headers)
    assert first["ack_sample"] == 640
    duplicate = manager.session_request("POST", sid, "feed", body=pcm, headers=headers)
    assert duplicate["duplicate"] is True and duplicate["ack_sample"] == 640
    expect_error("SEQUENCE_GAP", lambda: manager.session_request("POST", sid, "feed", body=pcm,
        headers={**headers, "X-R2T2-Seq": "2", "X-R2T2-Start-Sample": "640"}))
    expect_error("BUSY", lambda: manager.request("POST", "/models/unload", value={}))
    cancelled = manager.session_request("POST", sid, "cancel", value={})
    assert cancelled["status"] == "cancelled" and cancelled["audio_samples_16k"] == 640
    expect_error("SESSION_CLOSED", lambda: manager.session_request("POST", sid, "feed", body=pcm,
        headers={**headers, "X-R2T2-Seq": "1", "X-R2T2-Start-Sample": "640"}))
    unloaded = manager.request("POST", "/models/unload", value={})
    assert unloaded["loaded"] is False
    print(json.dumps({"duplicate_ack": duplicate["ack_sample"], "sequence_gap": "rejected",
                      "active_unload": "rejected", "cancel": cancelled["status"],
                      "post_cancel_feed": "rejected", "unload": "complete"}))
    manager.close()


if __name__ == "__main__":
    main()
