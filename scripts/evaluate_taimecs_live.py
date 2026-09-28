"""Evaluate TaiMECS human clips through the real Live WebSocket gateway.

Uses saved, hash-pinned public audio. It never requests a microphone. Frames
are ACK-paced for recognition quality; this does not measure live throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2t2_core.evaluation import BF16_MODEL_SHA256, score  # noqa: E402
from scripts.evaluate_corpus import (CONFIG, file_sha256, read_jsonl,  # noqa: E402
                                     validate_baseline, validate_manifest, write_report)

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 640
EVAL_DIR = ROOT / ".runtime/evaluation/taimecs"


async def run(hint: str, min_segment_seconds: int) -> None:
    suffix = "-chinese" if hint == "Chinese" else ""
    manifest = EVAL_DIR / f"human-20{suffix}.jsonl"
    baseline_path = EVAL_DIR / f"bf16-human-20{suffix}.jsonl"
    minimum = f"-min{min_segment_seconds}" if min_segment_seconds else ""
    output = EVAL_DIR / f"q8-bf16-human-20-live320{suffix}{minimum}.json"
    partial = output.with_name(output.stem + ".partial.json")
    entries = validate_manifest(read_jsonl(manifest), manifest)
    if len(entries) != 20 or any(item["language_hint"] != hint for item in entries):
        raise ValueError("Expected the pinned TaiMECS 20-case manifest with one language hint")
    baseline = validate_baseline(entries, baseline_path)
    if len(baseline) != len(entries):
        raise ValueError("Missing paired BF16 prediction")

    routes = web.RouteTableDef()
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = type("PromptServer", (), {"instance": types.SimpleNamespace(routes=routes)})
    sys.modules["server"] = fake_server
    spec = importlib.util.spec_from_file_location(
        "r2t2_taimecs_live_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    assert spec and spec.loader
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    from r2t2_taimecs_live_test.bridge import manager

    app = web.Application()
    app.add_routes(routes)
    client = TestClient(TestServer(app))
    await client.start_server()
    origin = str(client.make_url("/")).rstrip("/")
    cases: list[dict] = []
    worker_failures = 0
    try:
        loaded = await asyncio.to_thread(manager.load, CONFIG)
        identity = {"manifest_sha256": file_sha256(manifest),
                    "bf16_sha256": file_sha256(baseline_path), "mode": "stream",
                    "route": "live_gateway", "stream_chunk_ms": 320,
                    "min_segment_seconds": min_segment_seconds, "hint": hint,
                    "model_sha256": loaded["model_sha256"],
                    "projector_sha256": loaded["projector_sha256"],
                    "bf16_model_sha256": BF16_MODEL_SHA256, "build_id": loaded["build_id"]}
        if partial.exists():
            checkpoint = json.loads(partial.read_text(encoding="utf-8"))
            if any(checkpoint.get(key) != value for key, value in identity.items()):
                raise ValueError("Live evaluation checkpoint identity mismatch")
            cases = checkpoint["cases"]
            worker_failures = checkpoint.get("worker_failures", 0)
            if len(cases) > len(entries) or any(
                case["id"] != item["id"] or case["audio_sha256"] != item["sha256"]
                for case, item in zip(cases, entries)
            ):
                raise ValueError("Live evaluation checkpoint case mismatch")
            print(json.dumps({"resumed": len(cases), "hint": hint}), flush=True)

        for item in entries[len(cases):]:
            audio, rate = sf.read(item["audio_path"], dtype="float32", always_2d=True)
            if rate != SAMPLE_RATE or audio.shape[1] != 1 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid 16 kHz mono audio in {item['id']}")
            pcm = audio[:, 0]
            ws = None
            try:
                response = await client.post("/r2t2/v1/live/start", json={
                    "model_config": CONFIG, "language": hint, "context": "",
                    "stream_chunk_ms": 320,
                    "min_segment_seconds": min_segment_seconds}, headers={"Origin": origin})
                if response.status != 200:
                    raise RuntimeError(f"Live start {item['id']}: {response.status} {await response.text()}")
                created = await response.json()
                if (created["stream_chunk_ms"] != 320 or
                        created["min_segment_seconds"] != min_segment_seconds):
                    raise AssertionError("Gateway did not apply live decode settings")
                sid, token = created["session_id"], created["browser_token"]
                ws = await client.ws_connect(f"/r2t2/v1/live/{sid}/stream",
                                             headers={"Origin": origin}, max_msg_size=0,
                                             receive_timeout=180)
                await ws.send_json({"browser_token": token})
                if (await ws.receive_json())["type"] != "ready":
                    raise RuntimeError("Live WebSocket was not ready")
                stable = ""
                event_count = 0
                for seq, start in enumerate(range(0, len(pcm), FRAME_SAMPLES)):
                    frame = np.ascontiguousarray(pcm[start:start + FRAME_SAMPLES], dtype="<f4")
                    await ws.send_bytes(struct.pack("<II", seq, start) + frame.tobytes())
                    ack = await ws.receive_json()
                    if (ack["type"] != "ack" or ack["ack_seq"] != seq or
                            ack["ack_sample"] != start + len(frame)):
                        raise AssertionError(f"Live ACK mismatch in {item['id']}: {ack}")
                    for event in ack["events"]:
                        if event["stable_text"] != stable + event["delta"]:
                            raise AssertionError(f"Live text rollback in {item['id']}")
                        stable = event["stable_text"]
                        event_count += 1
                await ws.send_json({"type": "finish", "last_seq": seq,
                                    "total_samples": len(pcm)})
                final = await ws.receive_json()
                if (final["type"] != "final" or final["status"] != "finalized" or
                        final["audio_samples_16k"] != len(pcm) or
                        not final["text"].startswith(stable) or
                        final["stream_chunk_ms"] != 320 or
                        final["min_segment_seconds"] != min_segment_seconds):
                    raise AssertionError(f"Live final mismatch in {item['id']}")
                segments = final["segments"]
                if (not segments or segments[0]["start_sample"] != 0 or
                        segments[-1]["end_sample"] != len(pcm) or
                        any(left["end_sample"] != right["start_sample"]
                            for left, right in zip(segments, segments[1:]))):
                    raise AssertionError(f"Live sample ownership mismatch in {item['id']}")
                status = ("complete" if final["quality_status"] == "standard" and
                          final["forced_boundaries"] == 0 else "requires_review")
                bf16 = baseline[item["id"]]
                cases.append({"id": item["id"], "language": item["language"],
                              "metric": item["metric"], "tags": item["tags"],
                              "reference": item["reference"], "audio_sha256": item["sha256"],
                              "audio_seconds": len(pcm) / SAMPLE_RATE,
                              "q8_text": final["text"], "detected_language": final["language"],
                              "q8_score": score(item["reference"], final["text"], "cer"),
                              "bf16_text": bf16["text"],
                              "bf16_score": score(item["reference"], bf16["text"], "cer"),
                              "status": status, "gateway_status": final["status"],
                              "quality_status": final["quality_status"],
                              "forced_boundaries": final["forced_boundaries"],
                              "segments": segments, "event_count": event_count,
                              "revision": final["revision"]})
                write_report(partial, {**identity, "cases": cases,
                                       "worker_failures": worker_failures})
                print(json.dumps({"hint": hint, "processed": len(cases),
                                  "total": len(entries), "id": item["id"],
                                  "status": status}), flush=True)
            except Exception:
                worker_failures += 1
                write_report(partial, {**identity, "cases": cases,
                                       "worker_failures": worker_failures})
                raise
            finally:
                if ws is not None and not ws.closed:
                    await ws.close()
        report = {**identity, "dataset": "JacobLinCool/TaiMECS human only",
                  "cases": cases, "worker_failures": worker_failures,
                  "audio_seconds": sum(row["audio_seconds"] for row in cases),
                  "release_gate": "not_applicable_single_speaker_20_cases"}
        write_report(output, report)
        partial.unlink(missing_ok=True)
        print(json.dumps({"output": str(output), "hint": hint,
                          "min_segment_seconds": min_segment_seconds,
                          "cases": len(cases), "worker_failures": worker_failures}), flush=True)
    finally:
        await client.close()
        manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hint", choices=("Auto", "Chinese"), default="Auto")
    parser.add_argument("--min-segment-seconds", type=int, choices=(0, 4, 8), default=8)
    arguments = parser.parse_args()
    asyncio.run(run(arguments.hint, arguments.min_segment_seconds))
