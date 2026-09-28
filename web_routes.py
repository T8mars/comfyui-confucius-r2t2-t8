"""ComfyUI-origin live microphone gateway; worker credentials stay server-side."""

from __future__ import annotations

import asyncio
import json
import os
import struct
import threading
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from .bridge import WorkerError, manager


_stream_leases: dict[str, web.WebSocketResponse] = {}
_stream_leases_lock = threading.Lock()
_MAX_START_BODY = 64 * 1024


def _worker_error_response(exc: WorkerError) -> web.Response:
    status = (exc.http_status if exc.http_status in (400, 404, 409, 413) else
              503 if exc.http_status is None else 502)
    code = exc.worker_code or ("WORKER_UNAVAILABLE" if status == 503 else "WORKER_ERROR")
    return web.json_response({"code": code, "message": str(exc)}, status=status)


def _input_error(code: str, message: str, status: int) -> web.Response:
    return web.json_response({"code": code, "message": message}, status=status)


def _claim_stream(sid: str, ws: web.WebSocketResponse) -> bool:
    with _stream_leases_lock:
        if sid in _stream_leases:
            return False
        _stream_leases[sid] = ws
        return True


def _release_stream(sid: str, ws: web.WebSocketResponse) -> None:
    with _stream_leases_lock:
        if _stream_leases.get(sid) is ws:
            del _stream_leases[sid]


def same_origin(request: web.Request) -> None:
    origin = request.headers.get("Origin", "")
    host = urlsplit("//" + request.host).hostname
    allowed = {"localhost", "127.0.0.1", "::1"}
    allowed.update(item.strip().lower() for item in os.environ.get("R2T2_ALLOWED_HOSTS", "").split(",") if item.strip())
    if not origin or host is None or host.lower() not in allowed:
        raise web.HTTPForbidden(text='{"code":"BAD_ORIGIN"}')
    parsed = urlsplit(origin)
    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() != request.host.lower():
        raise web.HTTPForbidden(text='{"code":"BAD_ORIGIN"}')


def browser_auth(request: web.Request, sid: str, token: str) -> None:
    same_origin(request)
    try:
        manager.check_browser(sid, token)
    except WorkerError as exc:
        raise web.HTTPForbidden(text=json.dumps({"code": "FORBIDDEN", "message": str(exc)})) from exc


try:
    from server import PromptServer
except ImportError:
    PromptServer = None


if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    routes = PromptServer.instance.routes

    @routes.post("/r2t2/v1/live/start")
    async def live_start(request):
        same_origin(request)
        if request.content_type != "application/json":
            return _input_error("UNSUPPORTED_MEDIA_TYPE", "Expected application/json", 415)
        if request.content_length is not None and request.content_length > _MAX_START_BODY:
            return _input_error("PAYLOAD_TOO_LARGE", "Live start request exceeds 64 KiB", 413)
        chunks = []
        size = 0
        async for chunk in request.content.iter_chunked(8192):
            size += len(chunk)
            if size > _MAX_START_BODY:
                return _input_error("PAYLOAD_TOO_LARGE", "Live start request exceeds 64 KiB", 413)
            chunks.append(chunk)
        try:
            value = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _input_error("INVALID_INPUT", "Invalid JSON request body", 400)
        if not isinstance(value, dict):
            return _input_error("INVALID_INPUT", "Live start request must be a JSON object", 400)
        config = value.get("model_config", {})
        if not isinstance(config, dict):
            return _input_error("INVALID_INPUT", "model_config must be a JSON object", 400)
        options = {"language": value.get("language", "Auto"), "context": value.get("context", ""),
                   "stream_chunk_ms": value.get("stream_chunk_ms", 320),
                   "min_segment_seconds": value.get("min_segment_seconds", 8)}
        try:
            result = await asyncio.to_thread(manager.start_live, config, options)
            return web.json_response(result)
        except WorkerError as exc:
            return _worker_error_response(exc)

    @routes.get("/r2t2/v1/live/{sid}/result")
    async def live_result(request):
        sid = request.match_info["sid"]
        browser_auth(request, sid, request.headers.get("X-R2T2-Session-Token", ""))
        try:
            result = await asyncio.to_thread(manager.session_request, "GET", sid, "result")
            return web.json_response(result)
        except WorkerError as exc:
            return _worker_error_response(exc)

    @routes.post("/r2t2/v1/live/{sid}/cancel")
    async def live_cancel(request):
        sid = request.match_info["sid"]
        browser_auth(request, sid, request.headers.get("X-R2T2-Session-Token", ""))
        try:
            result = await asyncio.to_thread(manager.session_request, "POST", sid, "cancel", value={})
            return web.json_response(result)
        except WorkerError as exc:
            return _worker_error_response(exc)

    @routes.get("/r2t2/v1/live/{sid}/stream")
    async def live_stream(request):
        same_origin(request)
        sid = request.match_info["sid"]
        ws = web.WebSocketResponse(max_msg_size=16000 * 4 * 2 + 8, heartbeat=15)
        await ws.prepare(request)
        authenticated = False
        terminal = False
        monitor_task = None

        async def monitor_worker():
            while not ws.closed and not terminal:
                await asyncio.sleep(5)
                if ws.closed or terminal:
                    return
                try:
                    current = await asyncio.to_thread(manager.session_request, "GET", sid, "status")
                except WorkerError:
                    current = {"status": "worker_restarted"}
                if current["status"] in ("interrupted", "failed", "worker_restarted"):
                    if not ws.closed:
                        try:
                            await ws.send_json({"type": "error", "code": current["status"].upper(),
                                                "message": "Live session stopped; start a new session"})
                        except ConnectionResetError:
                            pass
                        await ws.close()
                    return

        try:
            async for msg in ws:
                if not authenticated:
                    if msg.type != WSMsgType.TEXT:
                        await ws.close(code=1008, message=b"auth required")
                        break
                    hello = json.loads(msg.data)
                    if not isinstance(hello, dict) or not isinstance(hello.get("browser_token", ""), str):
                        raise ValueError("Invalid stream hello")
                    try:
                        browser_auth(request, sid, hello.get("browser_token", ""))
                    except web.HTTPForbidden:
                        await ws.send_json({"type": "error", "code": "FORBIDDEN",
                                            "message": "Invalid live-session credential"})
                        await ws.close(code=1008)
                        break
                    if not _claim_stream(sid, ws):
                        await ws.send_json({"type": "error", "code": "ALREADY_CONNECTED",
                                            "message": "Live session already has a stream connection"})
                        await ws.close(code=1008)
                        break
                    authenticated = True
                    monitor_task = asyncio.create_task(monitor_worker())
                    await ws.send_json({"type": "ready", "protocol_version": 1})
                    continue
                if msg.type == WSMsgType.BINARY:
                    if len(msg.data) < 12 or (len(msg.data) - 8) % 4:
                        await ws.send_json({"type": "error", "code": "INVALID_FRAME"})
                        break
                    seq, start_sample = struct.unpack_from("<II", msg.data)
                    answer = await asyncio.to_thread(
                        manager.session_request, "POST", sid, "feed", body=msg.data[8:],
                        headers={"X-R2T2-Seq": str(seq), "X-R2T2-Start-Sample": str(start_sample),
                                 "Content-Type": "application/octet-stream"},
                    )
                    await ws.send_json({"type": "ack", **answer})
                elif msg.type == WSMsgType.TEXT:
                    value = json.loads(msg.data)
                    if not isinstance(value, dict):
                        raise ValueError("Invalid stream command")
                    if value.get("type") == "finish":
                        answer = await asyncio.to_thread(manager.session_request, "POST", sid, "finish",
                                                         value={"last_seq": value["last_seq"], "total_samples": value["total_samples"]})
                        terminal = True
                        await ws.send_json({"type": "final", **answer})
                        await ws.close()
                        break
                    if value.get("type") == "cancel":
                        answer = await asyncio.to_thread(manager.session_request, "POST", sid, "cancel", value={})
                        terminal = True
                        await ws.send_json({"type": "cancelled", **answer})
                        await ws.close()
                        break
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        except (WorkerError, KeyError, ValueError) as exc:
            if not ws.closed:
                await ws.send_json({"type": "error", "code": "STREAM_ERROR", "message": str(exc)})
        finally:
            try:
                if monitor_task is not None:
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass
            finally:
                try:
                    if authenticated and not terminal:
                        try:
                            await asyncio.to_thread(manager.session_request, "POST", sid, "cancel", value={})
                        except WorkerError:
                            pass
                finally:
                    if authenticated:
                        _release_stream(sid, ws)
                    if not ws.closed:
                        await ws.close()
        return ws
