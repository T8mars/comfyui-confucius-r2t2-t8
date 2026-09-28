"""Verify the exact 10-minute live PCM limit and final sample watermark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import WorkerError, manager  # noqa: E402

CONFIG = {"n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1}


def main() -> None:
    try:
        sid = manager.start_live(CONFIG, {"language": "Chinese", "context": ""})["session_id"]
        pcm = np.zeros(32000, dtype="<f4").tobytes()
        for seq in range(300):
            answer = manager.session_request("POST", sid, "feed", body=pcm,
                headers={"X-R2T2-Seq": str(seq), "X-R2T2-Start-Sample": str(seq * 32000),
                         "Content-Type": "application/octet-stream"})
            assert answer["ack_sample"] == (seq + 1) * 32000
        try:
            manager.session_request("POST", sid, "feed", body=np.zeros(1, dtype="<f4").tobytes(),
                headers={"X-R2T2-Seq": "300", "X-R2T2-Start-Sample": "9600000",
                         "Content-Type": "application/octet-stream"})
        except WorkerError as exc:
            assert "CONTEXT_LIMIT" in str(exc), str(exc)
        else:
            raise AssertionError("Worker accepted audio past 10 minutes")
        final = manager.session_request("POST", sid, "finish",
                                        value={"last_seq": 299, "total_samples": 9600000})
        assert final["status"] == "finalized" and final["audio_samples_16k"] == 9600000
        assert len(final["segments"]) == 30 and final["forced_boundaries"] == 0
        print(json.dumps({"live_seconds": 600, "frames": 300, "status": final["status"],
                          "segments": len(final["segments"]), "over_limit": "rejected"}))
    finally:
        manager.close()


if __name__ == "__main__":
    main()
