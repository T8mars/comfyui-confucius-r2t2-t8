"""Feed public 16 kHz PCM through the real live gateway at microphone pace.

This is an injected-audio transport/throughput diagnostic. It never requests
physical microphone access and does not validate AudioWorklet capture.
"""

from __future__ import annotations

import asyncio
import argparse
import importlib.util
import json
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from public_sample import ensure_sample

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / ".runtime/evaluation/dense-speech/fleurs-joined-90s.wav"
REPORT = ROOT / ".runtime/evaluation/dense-speech/q8-live-paced-90s.json"
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 640


async def run(*, seconds: int, warmup: bool, chunk_ms: int,
              min_segment_seconds: int, audio_path: Path) -> None:
    audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or audio.shape[1] != 1 or len(audio) + SAMPLE_RATE < seconds * SAMPLE_RATE:
        raise ValueError("Expected a previously generated mono 16 kHz public test WAV with enough audio")
    pcm = audio[:seconds * SAMPLE_RATE, 0]

    routes = web.RouteTableDef()
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = type("PromptServer", (), {"instance": types.SimpleNamespace(routes=routes)})
    sys.modules["server"] = fake_server
    spec = importlib.util.spec_from_file_location(
        "r2t2_live_paced_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    assert spec and spec.loader
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    from r2t2_live_paced_test.bridge import manager

    app = web.Application()
    app.add_routes(routes)
    client = TestClient(TestServer(app))
    await client.start_server()
    origin = str(client.make_url("/")).rstrip("/")
    ws = None
    try:
        if warmup:
            sample, sample_rate = sf.read(ensure_sample(), dtype="float32")
            assert sample_rate == SAMPLE_RATE
            await asyncio.to_thread(manager.transcribe,
                                    np.ascontiguousarray(sample, dtype="<f4").tobytes(),
                                    {"sample_rate": SAMPLE_RATE, "channels": 1,
                                     "mode": "offline", "language": "Chinese", "context": "",
                                     "hotwords": "", "channel": "mean", "auto_gain": False},
                                    {"n_ctx": 8192, "n_batch": 1024,
                                     "n_threads": 8, "gpu_layers": -1})
        response = await client.post("/r2t2/v1/live/start", json={"model_config": {
            "n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1},
            "language": "Chinese", "context": "", "stream_chunk_ms": chunk_ms,
            "min_segment_seconds": min_segment_seconds},
            headers={"Origin": origin})
        if response.status != 200:
            raise RuntimeError(f"Live start failed: {response.status} {await response.text()}")
        created = await response.json()
        assert created["stream_chunk_ms"] == chunk_ms
        assert created["min_segment_seconds"] == min_segment_seconds
        sid, token = created["session_id"], created["browser_token"]
        ws = await client.ws_connect(f"/r2t2/v1/live/{sid}/stream", headers={"Origin": origin},
                                     receive_timeout=240, max_msg_size=0)
        await ws.send_json({"browser_token": token})
        ready = await ws.receive_json()
        assert ready["type"] == "ready"

        started = time.perf_counter()
        sent = 0
        acknowledged = 0
        received_events = 0
        stable = ""
        max_outstanding = 0
        max_ack_delay_seconds = 0.0
        first_overload_at_seconds = None
        final = None

        async def produce() -> None:
            nonlocal sent, max_outstanding, first_overload_at_seconds
            for seq, start in enumerate(range(0, len(pcm), FRAME_SAMPLES)):
                target = started + start / SAMPLE_RATE
                await asyncio.sleep(max(0.0, target - time.perf_counter()))
                payload = np.ascontiguousarray(pcm[start:start + FRAME_SAMPLES], dtype="<f4")
                await ws.send_bytes(struct.pack("<II", seq, start) + payload.tobytes())
                sent = start + len(payload)
                outstanding = sent - acknowledged
                max_outstanding = max(max_outstanding, outstanding)
                # Match the UI's check: it examines the previous outstanding
                # count before sending the next 40 ms packet.
                if first_overload_at_seconds is None and outstanding > 3 * SAMPLE_RATE:
                    first_overload_at_seconds = round(time.perf_counter() - started, 3)
                if start and start % (20 * SAMPLE_RATE) == 0:
                    print(json.dumps({"sent_seconds": round(sent / SAMPLE_RATE, 2),
                                      "ack_seconds": round(acknowledged / SAMPLE_RATE, 2),
                                      "backlog_seconds": round(outstanding / SAMPLE_RATE, 2)}), flush=True)
            await ws.send_json({"type": "finish", "last_seq": seq, "total_samples": len(pcm)})

        async def consume() -> None:
            nonlocal acknowledged, received_events, stable, max_ack_delay_seconds, final
            async for message in ws:
                if message.type != web.WSMsgType.TEXT:
                    raise RuntimeError(f"Unexpected WebSocket message: {message.type}")
                value = json.loads(message.data)
                if value["type"] == "error":
                    raise RuntimeError(f"Gateway error: {value}")
                if value["type"] == "ack":
                    acknowledged = value["ack_sample"]
                    max_ack_delay_seconds = max(
                        max_ack_delay_seconds,
                        time.perf_counter() - (started + acknowledged / SAMPLE_RATE))
                    for event in value.get("events", []):
                        if event["stable_text"] != stable + event["delta"]:
                            raise AssertionError("Live stable text was revised")
                        stable = event["stable_text"]
                        received_events += 1
                elif value["type"] == "final":
                    final = value
                    return
            raise RuntimeError("Live WebSocket closed without final snapshot")

        await asyncio.wait_for(asyncio.gather(produce(), consume()), timeout=seconds + 240)
        elapsed = time.perf_counter() - started
        assert final is not None
        assert acknowledged == len(pcm) == final["audio_samples_16k"]
        # The worker may commit a final suffix only when the finish watermark
        # arrives; ACKs still must form a prefix of that final snapshot.
        assert final["text"].startswith(stable)
        assert final["status"] == "finalized"
        assert final["min_segment_seconds"] == min_segment_seconds
        segments = final["segments"]
        assert segments and segments[0]["start_sample"] == 0
        assert segments[-1]["end_sample"] == len(pcm)
        assert all(a["end_sample"] == b["start_sample"] for a, b in zip(segments, segments[1:]))
        report = {"test": "injected_public_pcm_real_time_gateway", "audio_path": str(audio_path),
                  "external_warmup": warmup, "stream_chunk_ms": chunk_ms,
                  "min_segment_seconds": min_segment_seconds,
                  "live_start_warmup_ms": created["warmup_ms"],
                  "audio_seconds": len(pcm) / SAMPLE_RATE, "wall_seconds": elapsed,
                  "max_outstanding_seconds": max_outstanding / SAMPLE_RATE,
                  "max_ack_delay_seconds": max_ack_delay_seconds,
                  "first_ui_overload_at_seconds": first_overload_at_seconds,
                  "sent_samples": sent, "acknowledged_samples": acknowledged,
                  "received_events": received_events, "final": final}
        report_path = REPORT.with_name(
            f"q8-live-paced-{seconds}s-chunk{chunk_ms}-min{min_segment_seconds}-"
            f"{'warm' if warmup else 'cold'}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "final"},
                         ensure_ascii=False), flush=True)
    finally:
        if ws is not None and not ws.closed:
            await ws.close()
        await client.close()
        manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=95)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--chunk-ms", type=int, choices=(160, 320, 480, 640), default=160)
    parser.add_argument("--min-segment-seconds", type=int, choices=(0, 4, 8), default=8)
    parser.add_argument("--audio-path", type=Path, default=AUDIO)
    arguments = parser.parse_args()
    asyncio.run(run(seconds=arguments.seconds, warmup=arguments.warmup,
                    chunk_ms=arguments.chunk_ms,
                    min_segment_seconds=arguments.min_segment_seconds,
                    audio_path=arguments.audio_path))
