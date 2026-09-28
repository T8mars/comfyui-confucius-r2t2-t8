"""Execute actual ComfyUI LoadAudio→Q8→Transcribe→Save graphs."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import uuid
from pathlib import Path

import aiohttp
from public_sample import ensure_sample


async def run(base: str) -> None:
    root = Path(__file__).resolve().parents[1]
    input_dir = root / ".runtime/comfy-input"
    input_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / "r2t2_public_test.wav"
    if not target.exists():
        shutil.copyfile(ensure_sample(), target)
    async with aiohttp.ClientSession() as http:
        for mode in ("offline", "stream"):
            prefix = f"r2t2_file_{mode}_integration_{uuid.uuid4().hex[:8]}"
            graph = {
                "1": {"class_type": "LoadAudio", "inputs": {"audio": "r2t2_public_test.wav"}},
                "2": {"class_type": "R2T2GGUFLoader", "inputs": {"n_ctx": 8192,
                    "n_batch": 1024, "n_threads": 8, "gpu_layers": -1}},
                "3": {"class_type": "R2T2Transcribe", "inputs": {"model": ["2", 0], "audio": ["1", 0],
                    "mode": mode, "language": "Auto", "context": "", "hotwords": "", "channel": "mean",
                    "auto_gain": mode != "offline", "stream_chunk_ms": 320 if mode == "stream" else 160}},
                "4": {"class_type": "R2T2SaveTranscript", "inputs": {"result_json": ["3", 2],
                    "format": "json", "prefix": prefix}},
            }
            queued = await http.post(base + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())})
            assert queued.status == 200, await queued.text()
            prompt_id = (await queued.json())["prompt_id"]
            for _ in range(120):
                history = await (await http.get(base + f"/history/{prompt_id}")).json()
                if prompt_id in history:
                    status = history[prompt_id]["status"]
                    assert status["status_str"] == "success", status
                    break
                await asyncio.sleep(0.5)
            else:
                raise TimeoutError(f"ComfyUI {mode} graph did not finish")
            outputs = list((root / ".runtime/comfy-output").glob(prefix + "_*.json"))
            assert len(outputs) == 1, outputs
            result = json.loads(outputs[0].read_text(encoding="utf-8"))
            assert result["status"] == "complete" and result["audio_samples_16k"] == 107840
            assert result["input_gain"] == 1.0
            assert result["stream_chunk_ms"] == (320 if mode == "stream" else 160)
            assert result["language"] == "Chinese" and result["text"]
            print(json.dumps({"mode": mode, "graph": "success", "text": result["text"],
                              "events": len(result["events"]), "prompt_id": prompt_id}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8197")
    args = parser.parse_args()
    asyncio.run(run(args.base.rstrip("/")))
