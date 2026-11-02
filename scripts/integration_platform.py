"""Real Q8 node/worker/SRT/Live validation for an installed platform wheel."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import types

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.public_sample import ensure_sample
from subtitles import build_subtitles, render_subtitle
import r2t2_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/platform-check.json")
    args = parser.parse_args()
    started = time.perf_counter()
    name = "confucius_platform_check"
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py",
                                                 submodule_search_locations=[str(ROOT)])
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[name] = plugin
    spec.loader.exec_module(plugin)
    manager = sys.modules[name + ".nodes"].manager
    mapping = plugin.NODE_CLASS_MAPPINGS
    sample = ensure_sample()
    pcm, rate = sf.read(sample, dtype="float32")
    assert rate == 16000 and pcm.ndim == 1 and 0 < len(pcm) <= rate * 120
    report = {"platform": sys.platform, "machine": platform.machine(),
              "python": platform.python_version(), "native": r2t2_native.build_info(),
              "audio_seconds": len(pcm) / rate, "gpu_layers": args.gpu_layers}
    try:
        # This is the same node interface ComfyUI calls; the node then talks to
        # the separately spawned worker through its authenticated HTTP bridge.
        model, _ = mapping["R2T2GGUFLoader"]().load(2048, 256, args.threads, args.gpu_layers)
        report["build_id"] = model["build_id"]
        audio = {"waveform": pcm.reshape(1, 1, -1), "sample_rate": rate}
        transcript = mapping["R2T2Transcribe"]()
        # Finalize Live first, then prove automatic release + reload preserves it.
        live = manager.start_live(model["config"], {"language": "Chinese", "stream_chunk_ms": 640})
        sid = live["session_id"]
        seq = 0
        for pos in range(0, len(pcm), 16000):
            chunk = np.ascontiguousarray(pcm[pos:pos + 16000], dtype="<f4")
            manager.session_request("POST", sid, "feed", body=chunk.tobytes(), headers={
                "X-R2T2-Seq": str(seq), "X-R2T2-Start-Sample": str(pos)})
            seq += 1
        original = manager.session_request("POST", sid, "finish", value={
            "last_seq": seq - 1, "total_samples": len(pcm)})
        assert original["status"] == "finalized" and original["text"].strip()
        live_srt = render_subtitle(build_subtitles(original))
        assert "-->" in live_srt
        results = []
        for release in (True, True, False):
            values = transcript.transcribe(model, audio, "offline", "Chinese", "", "", "mean",
                                           subtitle_timings=True, unload_after=release)
            result = json.loads(values[2])
            assert result["text"].strip() and not result["truncated"]
            assert values[3].strip() and "-->" in values[3]
            assert result["model_released"] is release
            assert manager.request("GET", "/health")["loaded"] is not release
            assert manager.session_request("GET", sid, "result") == original
            results.append(result)
        assert len({result["text"] for result in results}) == 1
        # Exercise the file-stream path too, with the complete public sample.
        streamed = transcript.transcribe(model, audio, "stream", "Chinese", "", "", "mean",
                                         subtitle_timings=True, stream_chunk_ms=640, unload_after=True)
        stream_result = json.loads(streamed[2])
        assert stream_result["text"].strip() and not stream_result["truncated"]
        assert "-->" in streamed[3]
        snapshot_values = mapping["R2T2LiveSession"]().read_snapshot(
            model, "Chinese", "", sid, original["revision"])
        assert snapshot_values[0] == original["text"] and snapshot_values[3] == live_srt
        # The real save nodes, without installing all of ComfyUI into the worker.
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_output_directory = lambda: str(ROOT / ".runtime/platform-outputs")
        sys.modules["folder_paths"] = folder_paths
        for node_type in ("R2T2SaveTranscript", "R2T2Subtitle"):
            saved = mapping[node_type]().save(json.dumps(results[0], ensure_ascii=False), "srt", "platform")
            saved_path = Path(saved["result"][0])
            assert saved_path.is_file() and "-->" in saved_path.read_text(encoding="utf-8")
        report.update({"success": True, "offline_release_reload_runs": len(results),
                       "offline_text": results[0]["text"], "stream_text": stream_result["text"],
                       "live_text": original["text"], "live_srt_retained": True,
                       "srt_save_nodes": True})
    finally:
        manager.close()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # English Windows terminals may use cp1252 even when recognition is Chinese.
    # Keep the report file UTF-8 and the console JSON encoding-independent.
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
