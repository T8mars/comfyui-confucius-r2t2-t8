"""Real-model worker and ComfyUI-node integration using the official public WAV."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from public_sample import ensure_sample


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "r2t2_plugin", root / "__init__.py", submodule_search_locations=[str(root)])
    assert spec and spec.loader
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    from r2t2_plugin.bridge import manager

    audio, rate = sf.read(ensure_sample(), dtype="float32", always_2d=True)
    assert rate == 16000
    loader = plugin.NODE_CLASS_MAPPINGS["R2T2GGUFLoader"]()
    model, status = loader.load(8192, 1024, 8, -1)
    transcriber = plugin.NODE_CLASS_MAPPINGS["R2T2Transcribe"]()
    assert transcriber.INPUT_TYPES()["optional"]["auto_gain"][1]["default"] is True
    assert transcriber.INPUT_TYPES()["optional"]["stream_chunk_ms"][1]["default"] == 160
    comfy_audio = {"waveform": np.ascontiguousarray(audio.T[None]), "sample_rate": rate}
    offline = transcriber.transcribe(model, comfy_audio, "offline", "Chinese", "", "", "mean")
    stream = transcriber.transcribe(model, comfy_audio, "stream", "Chinese", "", "", "mean")
    auto_offline = transcriber.transcribe(model, comfy_audio, "offline", "Auto", "", "", "mean")
    auto_stream = transcriber.transcribe(model, comfy_audio, "stream", "Auto", "", "", "mean")
    hinted = transcriber.transcribe(model, comfy_audio, "offline", "Auto", "餐饮场景", "顾客 酒水", "mean")
    no_gain = json.loads(transcriber.transcribe(model, comfy_audio, "offline", "Chinese", "", "", "mean", False)[2])
    wider_stream = json.loads(transcriber.transcribe(model, comfy_audio, "stream", "Chinese", "", "",
                                                     "mean", True, 320)[2])
    stream_result = json.loads(stream[2])
    assert offline[0], "Offline Q8 transcription is empty"
    assert stream[0], "Streaming Q8 transcription is empty"
    assert auto_offline[1] == "Chinese" and auto_offline[0]
    assert auto_stream[1] == "Chinese" and auto_stream[0], f"Auto stream: {auto_stream[:2]!r}"
    assert "顾客" in hinted[0]
    assert no_gain["input_gain"] == 1.0 and no_gain["status"] == "complete"
    assert wider_stream["stream_chunk_ms"] == 320 and wider_stream["status"] == "complete"
    assert stream_result["events"][-1]["final"] is True
    assert stream_result["events"][-1]["audio_end_sample"] == len(audio)
    live = manager.start_live(model["config"], {"language": "Chinese", "context": ""})
    sid = live["session_id"]
    for seq, pos in enumerate(range(0, len(audio), 640)):
        chunk = np.ascontiguousarray(audio[pos:pos + 640, 0], dtype="<f4")
        ack = manager.session_request("POST", sid, "feed", body=chunk.tobytes(),
                                      headers={"X-R2T2-Seq": str(seq), "X-R2T2-Start-Sample": str(pos),
                                               "Content-Type": "application/octet-stream"})
        assert ack["ack_sample"] == pos + len(chunk)
    live_result = manager.session_request("POST", sid, "finish", value={"last_seq": seq, "total_samples": len(audio)})
    assert live_result["status"] == "finalized"
    assert live_result["audio_samples_16k"] == len(audio)
    assert live_result["forced_boundaries"] == 0
    assert live_result["segments"][0]["start_sample"] == 0
    assert live_result["segments"][-1]["end_sample"] == len(audio)
    for previous, current in zip(live_result["segments"], live_result["segments"][1:]):
        assert previous["end_sample"] == current["start_sample"]
    published = ""
    for event in live_result["events"]:
        assert event["stable_text"] == published + event["delta"]
        published = event["stable_text"]
    assert manager.session_request("POST", sid, "finish", value={"last_seq": seq, "total_samples": len(audio)})["revision"] == live_result["revision"]
    print(json.dumps({"loaded": status, "offline": offline[0], "stream": stream[0],
                      "stream_events": len(stream_result["events"]),
                      "stream_elapsed_ms": stream_result["elapsed_ms"],
                      "stream_truncated": stream_result["truncated"],
                      "live": live_result["text"], "live_revision": live_result["revision"],
                      "auto_offline": auto_offline[0], "auto_stream": auto_stream[0],
                      "hinted": hinted[0]}, ensure_ascii=True))
    if manager.process is not None:
        manager.process.terminate()
        manager.process.wait(timeout=10)


if __name__ == "__main__":
    main()
