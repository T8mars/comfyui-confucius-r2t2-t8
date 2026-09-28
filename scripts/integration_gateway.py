"""Exercise the ComfyUI gateway with a real Q8 worker and public sample audio."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
from public_sample import ensure_sample
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


async def run() -> None:
    root = Path(__file__).resolve().parents[1]
    routes = web.RouteTableDef()
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = type("PromptServer", (), {"instance": types.SimpleNamespace(routes=routes)})
    sys.modules["server"] = fake_server
    spec = importlib.util.spec_from_file_location("r2t2_gateway_test", root / "__init__.py",
                                                   submodule_search_locations=[str(root)])
    assert spec and spec.loader
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    from r2t2_gateway_test.bridge import manager

    app = web.Application()
    app.add_routes(routes)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    origin = str(client.make_url("/")).rstrip("/")
    try:
        bad = await client.post("/r2t2/v1/live/start", json={}, headers={"Origin": "http://attacker.invalid"})
        assert bad.status == 403
        missing = await client.post("/r2t2/v1/live/start", json={})
        assert missing.status == 403
        response = await client.post("/r2t2/v1/live/start", json={"model_config": {
            "n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1},
            "language": "Chinese", "context": ""}, headers={"Origin": origin})
        assert response.status == 200, await response.text()
        created = await response.json()
        assert created["stream_chunk_ms"] == 320
        assert created["min_segment_seconds"] == 8
        assert created["warmup_ms"] >= 0
        sid, token = created["session_id"], created["browser_token"]
        forbidden = await client.get(f"/r2t2/v1/live/{sid}/result",
                                     headers={"Origin": origin, "X-R2T2-Session-Token": "wrong"})
        assert forbidden.status == 403
        audio, sr = sf.read(ensure_sample(), dtype="float32")
        assert sr == 16000
        ws = await client.ws_connect(f"/r2t2/v1/live/{sid}/stream", headers={"Origin": origin})
        await ws.send_json({"browser_token": token})
        assert (await ws.receive_json())["type"] == "ready"
        for seq, start in enumerate(range(0, len(audio), 640)):
            chunk = np.ascontiguousarray(audio[start:start + 640], dtype="<f4")
            await ws.send_bytes(struct.pack("<II", seq, start) + chunk.tobytes())
            ack = await ws.receive_json()
            assert ack["type"] == "ack" and ack["ack_sample"] == start + len(chunk)
        await ws.send_json({"type": "finish", "last_seq": seq, "total_samples": len(audio)})
        final = await ws.receive_json()
        assert final["type"] == "final" and final["audio_samples_16k"] == len(audio)
        assert final["quality_status"] == "standard"
        result = await client.get(f"/r2t2/v1/live/{sid}/result",
                                  headers={"Origin": origin, "X-R2T2-Session-Token": token})
        assert result.status == 200 and (await result.json())["revision"] == final["revision"]
        tail = np.concatenate((audio, np.zeros(20 * sr, dtype=np.float32)))
        tail_start = await client.post("/r2t2/v1/live/start", json={"model_config": {
            "n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1},
            "language": "Chinese", "context": ""}, headers={"Origin": origin})
        assert tail_start.status == 200, await tail_start.text()
        tail_created = await tail_start.json()
        tail_sid, tail_token = tail_created["session_id"], tail_created["browser_token"]
        tail_ws = await client.ws_connect(f"/r2t2/v1/live/{tail_sid}/stream",
                                          headers={"Origin": origin})
        await tail_ws.send_json({"browser_token": tail_token})
        assert (await tail_ws.receive_json())["type"] == "ready"
        for tail_seq, tail_pos in enumerate(range(0, len(tail), 640)):
            chunk = np.ascontiguousarray(tail[tail_pos:tail_pos + 640], dtype="<f4")
            await tail_ws.send_bytes(struct.pack("<II", tail_seq, tail_pos) + chunk.tobytes())
            ack = await tail_ws.receive_json()
            assert ack["type"] == "ack" and ack["ack_sample"] == tail_pos + len(chunk)
        await tail_ws.send_json({"type": "finish", "last_seq": tail_seq,
                                 "total_samples": len(tail)})
        tail_final = await tail_ws.receive_json()
        assert tail_final["type"] == "final" and tail_final["status"] == "finalized"
        assert tail_final["audio_samples_16k"] == len(tail)
        assert tail_final["forced_boundaries"] == 0
        assert any(segment["end_reason"] == "silence" for segment in tail_final["segments"])
        assert tail_final["segments"][0]["start_sample"] == 0
        assert tail_final["segments"][-1]["end_sample"] == len(tail)
        assert all(left["end_sample"] == right["start_sample"]
                   for left, right in zip(tail_final["segments"], tail_final["segments"][1:]))
        await tail_ws.close()
        invalid = await client.post("/r2t2/v1/live/start", json={"model_config": {
            "n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1},
            "language": "Chinese", "stream_chunk_ms": 123}, headers={"Origin": origin})
        assert invalid.status == 409 and "stream_chunk_ms" in await invalid.text()
        invalid_minimum = await client.post("/r2t2/v1/live/start", json={"model_config": {
            "n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1},
            "language": "Chinese", "min_segment_seconds": 3}, headers={"Origin": origin})
        assert invalid_minimum.status == 409 and "min_segment_seconds" in await invalid_minimum.text()
        print(json.dumps({"gateway": "passed", "text": final["text"],
                          "segments": len(final["segments"]), "revision": final["revision"],
                          "silence_tail_segments": len(tail_final["segments"])}, ensure_ascii=True))
        await ws.close()
    finally:
        await client.close()
        if manager.process is not None:
            manager.process.terminate()
            manager.process.wait(timeout=10)


if __name__ == "__main__":
    asyncio.run(run())
