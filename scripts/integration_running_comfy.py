"""Run the public WAV through an already-running ComfyUI instance and workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import uuid
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
from public_sample import ensure_sample


async def main(base: str) -> None:
    root = Path(__file__).resolve().parents[1]
    audio, rate = sf.read(ensure_sample(), dtype="float32")
    assert rate == 16000
    async with aiohttp.ClientSession() as http:
        info = await (await http.get(base + "/object_info/R2T2LiveSession")).json()
        assert "R2T2LiveSession" in info
        start = await http.post(base + "/r2t2/v1/live/start", json={"model_config": {
            "n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1},
            "language": "Chinese", "context": ""}, headers={"Origin": base})
        assert start.status == 200, await start.text()
        created = await start.json()
        sid, token = created["session_id"], created["browser_token"]
        async with http.ws_connect(base.replace("http", "ws", 1) + f"/r2t2/v1/live/{sid}/stream",
                                   headers={"Origin": base}) as ws:
            await ws.send_json({"browser_token": token})
            assert (await ws.receive_json())["type"] == "ready"
            for seq, pos in enumerate(range(0, len(audio), 640)):
                chunk = np.ascontiguousarray(audio[pos:pos + 640], dtype="<f4")
                await ws.send_bytes(struct.pack("<II", seq, pos) + chunk.tobytes())
                ack = await ws.receive_json()
                assert ack["type"] == "ack" and ack["ack_sample"] == pos + len(chunk)
            await ws.send_json({"type": "finish", "last_seq": seq, "total_samples": len(audio)})
            final = await ws.receive_json()
            assert final["type"] == "final" and final["audio_samples_16k"] == len(audio)

        graph = {
            "1": {"class_type": "R2T2GGUFLoader", "inputs": {"n_ctx": 8192, "n_batch": 1024,
                "n_threads": 8, "gpu_layers": -1}},
            "2": {"class_type": "R2T2LiveSession", "inputs": {"model": ["1", 0],
                "language": "Chinese", "context": "", "session_id": sid, "revision": final["revision"]}},
            "3": {"class_type": "R2T2SaveTranscript", "inputs": {"result_json": ["2", 2],
                "format": "json", "prefix": "r2t2_integration"}},
        }
        queued = await http.post(base + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())})
        assert queued.status == 200, await queued.text()
        prompt_id = (await queued.json())["prompt_id"]
        for _ in range(120):
            history = await (await http.get(base + f"/history/{prompt_id}")).json()
            if prompt_id in history:
                record = history[prompt_id]
                assert record["status"]["status_str"] == "success", record["status"]
                print(json.dumps({"comfy_graph": "success", "text": final["text"],
                                  "revision": final["revision"], "prompt_id": prompt_id}, ensure_ascii=True))
                return
            await asyncio.sleep(0.5)
        raise TimeoutError("ComfyUI graph did not finish in 60 seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8197")
    args = parser.parse_args()
    asyncio.run(main(args.base.rstrip("/")))
