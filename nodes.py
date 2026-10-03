"""Thin ComfyUI nodes. Model dependencies live only in the separate worker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from .bridge import ROOT, WorkerError, manager
from .r2t2_core.subtitle import render_subtitle

LANGUAGES = ["Auto", "Chinese", "English", "Cantonese", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]


class R2T2GGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768, "step": 1024}),
            "n_batch": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 256}),
            "n_threads": ("INT", {"default": 8, "min": 1, "max": 64}),
            "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 99}),
        }}

    RETURN_TYPES = ("R2T2_MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "load"
    CATEGORY = "Confucius4-R2T2"

    def load(self, n_ctx, n_batch, n_threads, gpu_layers):
        config = {"n_ctx": n_ctx, "n_batch": n_batch, "n_threads": n_threads, "gpu_layers": gpu_layers}
        status = manager.load(config)
        return ({"config": config, "generation": status["generation"], "model": status["model"],
                 "projector": status["projector"], "build_id": status["build_id"],
                 "fingerprint": status["model_fingerprint"],
                 "model_sha256": status["model_sha256"],
                 "projector_sha256": status["projector_sha256"]},
                f"{status['model']} + {status['projector']} loaded")


class R2T2Transcribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("R2T2_MODEL",),
            "audio": ("AUDIO",),
            "mode": (["offline", "stream"],),
            "language": (LANGUAGES,),
            "context": ("STRING", {"default": "", "multiline": True}),
            "hotwords": ("STRING", {"default": "", "multiline": True}),
            "channel": (["mean", "left", "right"],),
        }, "optional": {
            "auto_gain": ("BOOLEAN", {"default": True}),
            "stream_chunk_ms": ("INT", {"default": 160, "min": 160, "max": 640, "step": 160}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "language", "result_json")
    FUNCTION = "transcribe"
    CATEGORY = "Confucius4-R2T2"

    def transcribe(self, model, audio, mode, language, context, hotwords, channel,
                   auto_gain=True, stream_chunk_ms=160):
        import numpy as np

        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("Expected ComfyUI AUDIO with waveform and sample_rate")
        waveform = audio["waveform"]
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().float().numpy()
        waveform = np.asarray(waveform)
        if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] < 1 or waveform.shape[1] > 8:
            raise ValueError(f"Expected AUDIO [1, channels, samples], got {waveform.shape}")
        if waveform.shape[2] == 0 or not np.isfinite(waveform).all():
            raise ValueError("Audio must be nonempty and finite")
        pcm = np.ascontiguousarray(waveform[0].T, dtype="<f4")
        options = {"sample_rate": int(audio["sample_rate"]), "channels": waveform.shape[1],
                   "mode": mode, "language": language, "context": context,
                   "hotwords": hotwords, "channel": channel, "auto_gain": auto_gain,
                   "stream_chunk_ms": stream_chunk_ms}
        result = manager.transcribe(pcm.tobytes(), options, model["config"])
        return (result["text"], result.get("language", ""), json.dumps(result, ensure_ascii=False))


class R2T2LiveSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("R2T2_MODEL",),
            "language": (LANGUAGES,),
            "context": ("STRING", {"default": "", "multiline": True}),
            "session_id": ("STRING", {"default": ""}),
            "revision": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
        }, "optional": {
            "stream_chunk_ms": ("INT", {"default": 320, "min": 160, "max": 640, "step": 160}),
            "min_segment_seconds": ("INT", {"default": 8, "min": 0, "max": 8, "step": 4}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "language", "result_json")
    FUNCTION = "read_snapshot"
    CATEGORY = "Confucius4-R2T2"

    @classmethod
    def IS_CHANGED(cls, model, language, context, session_id, revision,
                   stream_chunk_ms=320, min_segment_seconds=8):
        return f"{session_id}:{revision}:{stream_chunk_ms}:{min_segment_seconds}:{model.get('generation', '')}"

    def read_snapshot(self, model, language, context, session_id, revision,
                      stream_chunk_ms=320, min_segment_seconds=8):
        if not session_id:
            raise ValueError("Click Start and Stop in the Live Session node before running the workflow")
        result = manager.session_request("GET", session_id, "result")
        if result["generation"] != manager.generation:
            raise WorkerError("Live snapshot belongs to a previous worker generation")
        if result["status"] != "finalized":
            raise WorkerError(f"Live session is {result['status']}; click Stop before workflow execution")
        if int(revision) != result["revision"]:
            raise WorkerError(f"Snapshot revision mismatch: node={revision}, worker={result['revision']}")
        return (result["text"], result.get("language", ""), json.dumps(result, ensure_ascii=False))


def _write_transcript(prefix: str, suffix: str, content: str, result_json: str) -> str:
    """Write one transcript file into the ComfyUI output directory without racing."""
    import folder_paths

    safe_prefix = re.sub(r"[^A-Za-z0-9_-]", "_", prefix).strip("_")[:64] or "r2t2_transcript"
    digest = hashlib.sha256(result_json.encode("utf-8")).hexdigest()[:16]
    output_dir = Path(folder_paths.get_output_directory()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_prefix}_{digest}.{suffix}"
    if path.is_symlink():
        raise FileExistsError(f"Refusing to follow a transcript symlink: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Different content already exists at {path}")
    else:
        temp_path = None
        try:
            descriptor, temp_name = tempfile.mkstemp(prefix=".r2t2-", suffix=".tmp", dir=output_dir)
            temp_path = Path(temp_name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp_path, path)
            except FileExistsError:
                if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                    raise FileExistsError(f"Different content already exists at {path}")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    return str(path)


def _savable(result: dict) -> dict:
    if result.get("status") not in ("complete", "finalized", "requires_review", "truncated"):
        raise ValueError("Only finalized transcripts or reviewable partial results can be saved")
    return result


class R2T2SaveTranscript:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "result_json": ("STRING", {"forceInput": True}),
            "format": (["txt", "json", "srt", "vtt"],),
            "prefix": ("STRING", {"default": "r2t2_transcript"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "Confucius4-R2T2"
    OUTPUT_NODE = True

    def save(self, result_json, format, prefix):
        if format not in ("txt", "json", "srt", "vtt"):
            raise ValueError("Transcript format must be txt, json, srt or vtt")
        result = _savable(json.loads(result_json))
        if format == "json":
            content = json.dumps(result, ensure_ascii=False, indent=2)
        elif format in ("srt", "vtt"):
            content = render_subtitle(result, format)
        else:
            content = result.get("text", "")
        path = _write_transcript(prefix, format, content, result_json)
        return {"ui": {"text": (path,)}, "result": (path,)}


class R2T2Subtitle:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "result_json": ("STRING", {"forceInput": True}),
            "format": (["srt", "vtt"],),
            "prefix": ("STRING", {"default": "r2t2_subtitle"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "Confucius4-R2T2"
    OUTPUT_NODE = True

    def save(self, result_json, format, prefix):
        if format not in ("srt", "vtt"):
            raise ValueError("Subtitle format must be srt or vtt")
        result = _savable(json.loads(result_json))
        segments = result.get("segments", [])
        if segments and not any(segment.get("text") for segment in segments):
            raise ValueError("Transcript has segment boundaries but no per-segment text; "
                             "re-run the workflow to capture it")
        content = render_subtitle(result, format)
        if not content.strip():
            raise ValueError("Transcript carries no timed text to turn into cues")
        path = _write_transcript(prefix, format, content, result_json)
        return {"ui": {"text": (path,)}, "result": (path,)}


class R2T2Unload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("R2T2_MODEL",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "Confucius4-R2T2"
    OUTPUT_NODE = True

    def unload(self, model):
        return (json.dumps(manager.unload(), ensure_ascii=False),)


NODE_CLASS_MAPPINGS = {name: cls for name, cls in (
    ("R2T2GGUFLoader", R2T2GGUFLoader),
    ("R2T2Transcribe", R2T2Transcribe),
    ("R2T2LiveSession", R2T2LiveSession),
    ("R2T2SaveTranscript", R2T2SaveTranscript),
    ("R2T2Subtitle", R2T2Subtitle),
    ("R2T2Unload", R2T2Unload),
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "R2T2GGUFLoader": "Confucius4 Q8 Loader",
    "R2T2Transcribe": "Confucius4 Transcribe",
    "R2T2LiveSession": "Confucius4 Live Microphone",
    "R2T2SaveTranscript": "Confucius4 Save Transcript",
    "R2T2Subtitle": "Confucius4 Save Subtitle",
    "R2T2Unload": "Confucius4 Unload Q8",
}
