"""Isolated localhost inference process; never imported by ComfyUI's Python."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from aiohttp import web

from .audio import limited_quiet_gain, to_mono_16k
from .native import NativeQ8Engine, ROOT, OFFICIAL, MODEL_NAME, PROJECTOR_NAME, SUPPORTED_LANGUAGES
from .segmented import PresetBoundaryVAD, SegmentedStream, exact_zero_boundaries, join_segment_text
from .streaming import StreamingSession, StabilityViolation

VERSION = 1
MAX_UPLOAD = 512 * 1024 * 1024
MAX_LIVE_SAMPLES = 10 * 60 * 16000
ACTIVE_IDLE_SECONDS = 30
AWAITING_FIRST_AUDIO_SECONDS = 120
TERMINAL_RETENTION_SECONDS = 3600
_PYD = list((ROOT / ".runtime/build-native-cu128/python").rglob("qwen3asr_native*.pyd"))
_NATIVE_SHA = hashlib.sha256(_PYD[0].read_bytes()).hexdigest()[:12] if len(_PYD) == 1 else "unbuilt"
BUILD_ID = "llama-ad6c66839af3-cu128-" + _NATIVE_SHA


def error(code: str, message: str, status: int = 400) -> web.HTTPException:
    return web.HTTPBadRequest(text=json.dumps({"code": code, "message": message})) if status == 400 else web.HTTPConflict(text=json.dumps({"code": code, "message": message}))


def decode_pcm(data: bytes, sample_rate: int, channels: int, channel: str) -> np.ndarray:
    if channels < 1 or channels > 8 or len(data) % (4 * channels):
        raise ValueError("Invalid float32 PCM shape")
    pcm = np.frombuffer(data, dtype="<f4").reshape(-1, channels)
    return to_mono_16k(pcm, sample_rate, channel)


@dataclass
class LiveState:
    session: SegmentedStream
    owner: str
    sid: str
    next_seq: int = 0
    samples: int = 0
    last_digest: str = ""
    status: str = "active"
    revision: int = 0
    events: list[dict] = field(default_factory=list)
    updated: float = field(default_factory=time.monotonic)


class Service:
    def __init__(self) -> None:
        self.generation = secrets.token_hex(12)
        self.engine: NativeQ8Engine | None = None
        self.sessions: dict[str, LiveState] = {}
        self.lock = asyncio.Lock()
        self.model_config: dict | None = None
        self.live_warmed = False

    async def reap_idle(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        async with self.lock:
            for sid, state in list(self.sessions.items()):
                idle = current - state.updated
                if state.status == "active":
                    limit = AWAITING_FIRST_AUDIO_SECONDS if state.samples == 0 else ACTIVE_IDLE_SECONDS
                    if idle > limit:
                        state.status = "interrupted"
                        state.revision += 1
                        state.updated = current
                elif idle > TERMINAL_RETENTION_SECONDS:
                    del self.sessions[sid]

    async def load(self, config: dict) -> dict:
        allowed = {"n_ctx", "n_batch", "n_threads", "gpu_layers"}
        if set(config) - allowed:
            raise ValueError("Unknown model setting")
        normalized = {
            "n_ctx": int(config.get("n_ctx", 8192)),
            "n_batch": int(config.get("n_batch", 1024)),
            "n_threads": int(config.get("n_threads", 8)),
            "gpu_layers": int(config.get("gpu_layers", -1)),
        }
        if not 2048 <= normalized["n_ctx"] <= 32768 or not 256 <= normalized["n_batch"] <= 4096:
            raise ValueError("Invalid native context or batch size")
        if not 1 <= normalized["n_threads"] <= 64 or not -1 <= normalized["gpu_layers"] <= 99:
            raise ValueError("Invalid native thread or GPU layer count")
        async with self.lock:
            if self.engine is not None and self.model_config != normalized and any(s.status == "active" for s in self.sessions.values()):
                raise error("BUSY", "Cannot change model while a live session is active", 409)
            if self.engine is None or self.model_config != normalized:
                had_engine = self.engine is not None
                self.sessions.clear()
                self.engine = None
                self.model_config = None
                self.live_warmed = False
                if had_engine:
                    # The old model and terminal session streams must release
                    # their native VRAM before constructing a replacement.
                    gc.collect()
                replacement = await asyncio.to_thread(NativeQ8Engine, **normalized)
                self.engine = replacement
                self.model_config = normalized
            fingerprint = hashlib.sha256(json.dumps({"build": BUILD_ID, "config": normalized,
                "model_sha256": OFFICIAL[MODEL_NAME][1],
                "projector_sha256": OFFICIAL[PROJECTOR_NAME][1]}, sort_keys=True).encode()).hexdigest()
            return {"loaded": True, "model": self.engine.model_name, "projector": self.engine.projector_name,
                    "generation": self.generation, "config": normalized,
                    "build_id": BUILD_ID, "model_fingerprint": fingerprint,
                    "model_sha256": OFFICIAL[MODEL_NAME][1], "projector_sha256": OFFICIAL[PROJECTOR_NAME][1]}

    def require_engine(self) -> NativeQ8Engine:
        if self.engine is None:
            raise error("MODEL_NOT_LOADED", "Load the official Q8 pair first", 409)
        return self.engine


def service(request: web.Request) -> Service:
    return request.app["service"]


def require_owner(request: web.Request, state: LiveState) -> None:
    owner = request.headers.get("X-R2T2-Owner", "")
    if not owner or not secrets.compare_digest(owner, state.owner):
        raise web.HTTPForbidden(text='{"code":"FORBIDDEN"}')


def get_state(request: web.Request) -> LiveState:
    state = service(request).sessions.get(request.match_info["sid"])
    if state is None:
        raise web.HTTPNotFound(text='{"code":"SESSION_NOT_FOUND"}')
    require_owner(request, state)
    return state


@web.middleware
async def auth(request: web.Request, handler):
    token = request.app["token"]
    value = request.headers.get("Authorization", "")
    if not secrets.compare_digest(value, "Bearer " + token):
        raise web.HTTPUnauthorized(text='{"code":"UNAUTHORIZED"}')
    try:
        return await handler(request)
    except StabilityViolation as exc:
        raise web.HTTPConflict(text=json.dumps({"code": "STABILITY_VIOLATION",
                                               "message": "Previously committed text would change; session stopped"})) from exc
    except (ValueError, TypeError) as exc:
        code = "CONTEXT_LIMIT" if str(exc).startswith("CONTEXT_LIMIT") else "INVALID_INPUT"
        raise error(code, str(exc)) from exc


async def health(request: web.Request) -> web.Response:
    svc = service(request)
    return web.json_response({"protocol_version": VERSION, "ready": True, "loaded": svc.engine is not None,
                              "generation": svc.generation, "build_id": BUILD_ID,
                              "active_sessions": sum(s.status == "active" for s in svc.sessions.values()),
                              "capabilities": ["gguf_q8_offline", "gguf_q8_file_stream", "gguf_q8_live_pcm"]})


async def load(request: web.Request) -> web.Response:
    value = await request.json()
    return web.json_response(await service(request).load(value))


def transcribe_segmented_offline(engine: NativeQ8Engine, pcm: np.ndarray,
                                 language: str | None, context: str) -> dict:
    # Reuse the live/file-stream boundary ownership and quality flags, but
    # decode only once per finalized segment and retain no streaming events.
    session = SegmentedStream(engine, language=language, context=context, offline=True)
    for pos in range(0, len(pcm), 640):
        session.feed(pcm[pos:pos + 640])
    terminal = session.finish()
    return {"text": session.stable_text, "language": session.detected_language,
            "finish_reason": terminal.get("finish_reason"), "truncated": terminal.get("truncated", False),
            "events": [], "model": engine.model_name, "projector": engine.projector_name,
            "mode_executed": "segmented_offline", "forced_boundaries": session.forced_boundaries,
            "segments": session.segments}


async def transcribe(request: web.Request) -> web.Response:
    svc = service(request)
    data = await request.read()
    if len(data) < 4:
        raise ValueError("Missing request metadata")
    option_size = struct.unpack_from("<I", data)[0]
    if option_size > 16_384 or len(data) < 4 + option_size:
        raise ValueError("Invalid request metadata size")
    options = json.loads(data[4:4 + option_size].decode("utf-8"))
    data = data[4 + option_size:]
    sample_rate = int(options["sample_rate"])
    channels = int(options["channels"])
    channel = options.get("channel", "mean")
    mode = options.get("mode", "offline")
    stream_chunk_ms = options.get("stream_chunk_ms", 160)
    if type(stream_chunk_ms) is not int or stream_chunk_ms not in (160, 320, 480, 640):
        raise ValueError("stream_chunk_ms must be 160, 320, 480 or 640")
    language = options.get("language", "Auto")
    language = None if language == "Auto" else language
    context = options.get("context", "")
    hotwords = options.get("hotwords", "")
    if hotwords:
        context = (context + "\n" if context else "") + "Hotwords: " + hotwords
    if len(context) > 8192:
        raise ValueError("Context/hotwords too long")
    pcm = await asyncio.to_thread(decode_pcm, data, sample_rate, channels, channel)
    auto_gain = options.get("auto_gain", True)
    if not isinstance(auto_gain, bool):
        raise ValueError("auto_gain must be boolean")
    input_gain = 1.0
    if auto_gain:
        pcm, input_gain = limited_quiet_gain(pcm)
    start = time.perf_counter()
    zero_boundaries = exact_zero_boundaries(pcm) if len(pcm) <= 30 * 16000 else []
    async with svc.lock:
        engine = svc.require_engine()
        if mode == "offline" and zero_boundaries:
            stops = zero_boundaries + [len(pcm)]
            starts = [0] + zero_boundaries
            pieces = []
            segments = []
            for index, (first, last) in enumerate(zip(starts, stops)):
                piece = await asyncio.to_thread(engine.transcribe, pcm[first:last],
                                                context=context, language=language)
                pieces.append(piece)
                segments.append({"segment_id": index, "start_sample": first,
                                 "end_sample": last, "end_reason": "digital_zero" if index < len(zero_boundaries) else "input_end",
                                 "language": piece["language"], "finish_reason": piece["finish_reason"],
                                 "truncated": piece["truncated"]})
            languages = {piece["language"] for piece in pieces if piece["language"]}
            joined_text = ""
            for piece in pieces:
                joined_text = join_segment_text(joined_text, piece["text"].strip(),
                                                piece["language"] or language)
            result = {"text": joined_text,
                      "language": "Mixed" if len(languages) > 1 else next(iter(languages), ""),
                      "finish_reason": "length" if any(piece["truncated"] for piece in pieces) else "stop",
                      "truncated": any(piece["truncated"] for piece in pieces),
                      "events": [], "model": engine.model_name, "projector": engine.projector_name,
                      "mode_executed": "offline_zero_split", "segments": segments,
                      "zero_split_boundaries": zero_boundaries}
        elif mode == "offline" and len(pcm) <= 30 * 16000:
            result = await asyncio.to_thread(engine.transcribe, pcm, context=context, language=language)
            result["events"] = []
        elif mode == "offline":
            result = await asyncio.to_thread(transcribe_segmented_offline, engine, pcm, language, context)
        elif mode == "stream":
            session = (SegmentedStream(engine, language=language, context=context,
                                       chunk_ms=stream_chunk_ms,
                                       vad=PresetBoundaryVAD(zero_boundaries)) if zero_boundaries else
                       StreamingSession(engine, language=language, context=context,
                                        chunk_ms=stream_chunk_ms)
                       if len(pcm) <= 30 * 16000 else
                       SegmentedStream(engine, language=language, context=context,
                                       chunk_ms=stream_chunk_ms))
            events = []
            for pos in range(0, len(pcm), 640):
                events.extend(await asyncio.to_thread(session.feed, pcm[pos:pos + 640]))
            events.append(await asyncio.to_thread(session.finish))
            result = {"text": session.stable_text, "language": session.detected_language,
                      "finish_reason": events[-1].get("finish_reason"), "truncated": events[-1].get("truncated", False),
                      "events": events, "model": engine.model_name, "projector": engine.projector_name,
                      "mode_executed": "stream_zero_split" if zero_boundaries else
                                       "segmented_stream" if isinstance(session, SegmentedStream) else "stream",
                      "forced_boundaries": getattr(session, "forced_boundaries", 0),
                      "segments": getattr(session, "segments", []),
                      "zero_split_boundaries": zero_boundaries}
        else:
            raise ValueError("mode must be offline or stream")
    quality_status = ("truncated" if result["truncated"] else
                      "forced_boundaries_unverified" if result.get("forced_boundaries", 0) else
                      "digital_zero_split" if zero_boundaries else "standard")
    result.update({"protocol_version": VERSION, "generation": svc.generation,
                   "status": "truncated" if result["truncated"] else "requires_review" if result.get("forced_boundaries", 0) else "complete",
                   "quality_status": quality_status,
                   "input_sample_rate": sample_rate, "input_samples": len(data) // (4 * channels),
                   "input_gain": round(input_gain, 6),
                   "stream_chunk_ms": stream_chunk_ms,
                   "audio_samples_16k": len(pcm), "audio_seconds": len(pcm) / 16000,
                   "elapsed_ms": round((time.perf_counter() - start) * 1000, 1), "mode": mode})
    return web.json_response(result)


async def create_session(request: web.Request) -> web.Response:
    svc = service(request)
    value = await request.json()
    stream_chunk_ms = value.get("stream_chunk_ms", 320)
    if type(stream_chunk_ms) is not int or stream_chunk_ms not in (160, 320, 480, 640):
        raise ValueError("stream_chunk_ms must be 160, 320, 480 or 640")
    min_segment_seconds = value.get("min_segment_seconds", 8)
    if type(min_segment_seconds) is not int or min_segment_seconds not in (0, 4, 8):
        raise ValueError("min_segment_seconds must be 0, 4 or 8")
    owner = request.headers.get("X-R2T2-Owner", "")
    if len(owner) < 32:
        raise ValueError("Owner credential missing")
    language = value.get("language", "Auto")
    if language != "Auto" and language not in SUPPORTED_LANGUAGES:
        raise ValueError("Unsupported recognition language")
    context = value.get("context", "")
    if not isinstance(context, str) or len(context) > 8192:
        raise ValueError("CONTEXT_LIMIT: context exceeds 8192 characters")
    async with svc.lock:
        engine = svc.require_engine()
        if any(s.status == "active" for s in svc.sessions.values()):
            raise error("BUSY", "One live microphone session is already active", 409)
        warmup_ms = 0.0
        if not svc.live_warmed:
            # Run the first native audio decode before the browser opens its
            # microphone. A cold first decode can otherwise exceed the UI's
            # three-second backlog guard while real-time PCM keeps arriving.
            start = time.perf_counter()
            await asyncio.to_thread(engine.generate, np.zeros(5120, dtype=np.float32),
                                    language="Chinese", max_tokens=1)
            warmup_ms = round((time.perf_counter() - start) * 1000, 1)
            svc.live_warmed = True
        sid = secrets.token_urlsafe(24)
        stream = SegmentedStream(engine, language=None if language == "Auto" else language,
                                 context=context, chunk_ms=stream_chunk_ms,
                                 min_segment_seconds=min_segment_seconds)
        svc.sessions[sid] = LiveState(stream, owner, sid)
    return web.json_response({"protocol_version": VERSION, "session_id": sid, "generation": svc.generation,
                              "status": "active", "sample_rate": 16000, "max_segment_seconds": 30,
                              "stream_chunk_ms": stream_chunk_ms,
                              "min_segment_seconds": min_segment_seconds,
                              "warmup_ms": warmup_ms})


async def feed(request: web.Request) -> web.Response:
    svc = service(request)
    state = get_state(request)
    seq = int(request.headers["X-R2T2-Seq"])
    start_sample = int(request.headers["X-R2T2-Start-Sample"])
    data = await request.read()
    if not data or len(data) % 4 or len(data) > 16000 * 4 * 2:
        raise ValueError("Feed must contain 1 to 32000 float32 samples")
    digest = hashlib.sha256(data).hexdigest()
    pcm = np.frombuffer(data, dtype="<f4").copy()
    if not np.isfinite(pcm).all():
        raise ValueError("Audio contains NaN or infinity")
    async with svc.lock:
        if state.status != "active":
            raise error("SESSION_CLOSED", "Session is no longer active", 409)
        if seq == state.next_seq - 1 and start_sample + len(pcm) == state.samples and digest == state.last_digest:
            return web.json_response({"ack_seq": seq, "ack_sample": state.samples, "events": [], "duplicate": True})
        if seq != state.next_seq or start_sample != state.samples:
            raise error("SEQUENCE_GAP", "Out-of-order or missing PCM samples", 409)
        if state.samples + len(pcm) > MAX_LIVE_SAMPLES:
            raise error("CONTEXT_LIMIT", "Live session exceeds 10 minutes of audio", 409)
        try:
            events = await asyncio.to_thread(state.session.feed, pcm)
        except Exception:
            state.status = "failed"
            raise
        state.next_seq += 1
        state.samples += len(pcm)
        state.last_digest = digest
        state.events.extend(events)
        state.revision += len(events)
        state.updated = time.monotonic()
        return web.json_response({"ack_seq": seq, "ack_sample": state.samples, "events": events, "revision": state.revision})


async def finish(request: web.Request) -> web.Response:
    svc = service(request)
    state = get_state(request)
    value = await request.json()
    async with svc.lock:
        if state.status == "finalized":
            return web.json_response(snapshot(state, svc))
        if state.status != "active":
            raise error("SESSION_CLOSED", "Session cannot be finished", 409)
        if int(value.get("last_seq", -2)) != state.next_seq - 1 or int(value.get("total_samples", -1)) != state.samples:
            raise error("SEQUENCE_GAP", "Finish watermark does not match acknowledged audio", 409)
        state.status = "draining"
        try:
            event = await asyncio.to_thread(state.session.finish)
        except Exception:
            state.status = "failed"
            raise
        state.events.append(event)
        state.revision += 1
        state.status = "finalized"
        state.updated = time.monotonic()
        return web.json_response(snapshot(state, svc))


async def cancel(request: web.Request) -> web.Response:
    svc = service(request)
    state = get_state(request)
    async with svc.lock:
        if state.status in ("active", "draining"):
            state.status = "cancelled"
            state.updated = time.monotonic()
        return web.json_response(snapshot(state, svc))


def snapshot(state: LiveState, svc: Service) -> dict:
    truncated = bool(state.session.truncated)
    return {"protocol_version": VERSION, "session_id": state.sid, "generation": svc.generation,
            "status": state.status, "revision": state.revision, "text": state.session.stable_text,
            "preview_text": state.session.preview_text, "language": state.session.detected_language,
            "audio_samples_16k": state.samples, "events": state.events,
            "stream_chunk_ms": state.session.chunk_ms,
            "min_segment_seconds": state.session.min_segment_seconds,
            "truncated": truncated,
            "quality_status": "truncated" if truncated else
                              "forced_boundaries_unverified" if state.session.forced_boundaries else "standard",
            "forced_boundaries": state.session.forced_boundaries, "segments": state.session.segments}


async def result(request: web.Request) -> web.Response:
    state = get_state(request)
    return web.json_response(snapshot(state, service(request)))


async def session_status(request: web.Request) -> web.Response:
    state = get_state(request)
    return web.json_response({"protocol_version": VERSION, "session_id": state.sid,
                              "status": state.status, "revision": state.revision,
                              "audio_samples_16k": state.samples})


async def unload(request: web.Request) -> web.Response:
    svc = service(request)
    async with svc.lock:
        if any(s.status == "active" for s in svc.sessions.values()):
            raise error("BUSY", "Live session has a model lease", 409)
        svc.sessions.clear()
        svc.engine = None
        svc.model_config = None
    return web.json_response({"loaded": False, "generation": svc.generation})


def make_app(token: str) -> web.Application:
    app = web.Application(middlewares=[auth], client_max_size=MAX_UPLOAD)
    app["token"] = token
    app["service"] = Service()
    async def idle_reaper(application):
        async def run():
            while True:
                await asyncio.sleep(5)
                await application["service"].reap_idle()
        task = asyncio.create_task(run())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    app.cleanup_ctx.append(idle_reaper)
    app.router.add_get("/health", health)
    app.router.add_post("/models/load", load)
    app.router.add_post("/models/unload", unload)
    app.router.add_post("/transcribe", transcribe)
    app.router.add_post("/sessions", create_session)
    app.router.add_post("/sessions/{sid}/feed", feed)
    app.router.add_post("/sessions/{sid}/finish", finish)
    app.router.add_post("/sessions/{sid}/cancel", cancel)
    app.router.add_get("/sessions/{sid}/result", result)
    app.router.add_get("/sessions/{sid}/status", session_status)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    token = os.environ.get("R2T2_WORKER_TOKEN", "")
    if len(token) < 32:
        raise SystemExit("R2T2_WORKER_TOKEN missing")
    os.environ.pop("DEBUG_PRINT", None)
    os.environ.pop("QWEN3ASR_DUMP_DECODE_EMBD", None)
    web.run_app(make_app(token), host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
