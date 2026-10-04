"""ComfyUI-side transport to the isolated Python 3.12 inference worker."""

from __future__ import annotations

import atexit
import json
import os
import secrets
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SESSION_CREDENTIAL_TTL_SECONDS = 75 * 60


class WorkerError(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None = None,
                 worker_code: str | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.worker_code = worker_code


class WorkerManager:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.port: int | None = None
        self.token: str | None = None
        self.generation: str | None = None
        self._lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._log_file = None
        # The host may set a system HTTP proxy. Worker traffic must go directly
        # to the loopback listener even if proxy-bypass settings later change.
        self._local_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._owners: dict[str, str] = {}
        self._browser_tokens: dict[str, str] = {}
        self._session_activity: dict[str, float] = {}

    def _forget_credentials(self, sid: str) -> None:
        with self._lock:
            self._owners.pop(sid, None)
            self._browser_tokens.pop(sid, None)
            self._session_activity.pop(sid, None)

    def _prune_credentials(self) -> None:
        cutoff = time.monotonic() - SESSION_CREDENTIAL_TTL_SECONDS
        with self._lock:
            for sid, last_activity in list(self._session_activity.items()):
                if last_activity < cutoff:
                    self._owners.pop(sid, None)
                    self._browser_tokens.pop(sid, None)
                    del self._session_activity[sid]

    @staticmethod
    def _port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def _start(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        python = Path(os.environ.get("R2T2_WORKER_PYTHON", ROOT / ".runtime/worker/Scripts/python.exe"))
        if not python.is_file():
            raise WorkerError(f"Isolated worker Python not found: {python}. Run scripts/setup_worker_windows.ps1")
        self.port = self._port()
        self.token = secrets.token_urlsafe(48)
        env_keys = ("SYSTEMROOT", "WINDIR", "PATH", "USERPROFILE", "TEMP", "TMP", "CUDA_PATH", "CUDA_VISIBLE_DEVICES")
        env = {key: os.environ[key] for key in env_keys if key in os.environ}
        env["R2T2_WORKER_TOKEN"] = self.token
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONFAULTHANDLER"] = "1"
        env["PYTHONPATH"] = str(ROOT)
        log_path = ROOT / ".runtime/worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = log_path.open("ab", buffering=0)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [str(python), "-m", "r2t2_core.worker", "--port", str(self.port)],
            cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
            stdout=self._log_file, stderr=subprocess.STDOUT, creationflags=creationflags,
        )
        self._owners.clear()
        self._browser_tokens.clear()
        self._session_activity.clear()
        for _ in range(100):
            if self.process.poll() is not None:
                raise WorkerError(f"Worker exited early; see {log_path}")
            try:
                info = self._request("GET", "/health", timeout=1, invalidate_on_failure=False)
                self.generation = info["generation"]
                return
            except (WorkerError, urllib.error.URLError, TimeoutError):
                time.sleep(0.1)
        self.process.terminate()
        raise WorkerError(f"Worker startup timed out; see {log_path}")

    def ensure(self) -> None:
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                self._start()

    def _invalidate(self, failed_process: subprocess.Popen | None = None) -> None:
        with self._lock:
            if failed_process is not None and self.process is not failed_process:
                return
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
            self.port = None
            self.token = None
            self.generation = None
            self._owners.clear()
            self._browser_tokens.clear()
            self._session_activity.clear()

    def _request(self, method: str, path: str, *, body: bytes | None = None,
                 headers: dict[str, str] | None = None, timeout: int = 120,
                 invalidate_on_failure: bool = True) -> dict:
        if self.port is None or self.token is None:
            raise WorkerError("Worker is not running")
        request_process = self.process
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=body, method=method,
            headers={"Authorization": "Bearer " + self.token, **(headers or {})},
        )
        try:
            with self._local_http.open(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            try:
                parsed = json.loads(detail)
                worker_code = parsed.get("code") if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                worker_code = None
            raise WorkerError(f"Worker {exc.code}: {detail}", http_status=exc.code,
                              worker_code=worker_code if isinstance(worker_code, str) else None) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            exit_code = request_process.poll() if request_process is not None else None
            process_state = ("unknown" if request_process is None else
                             "alive" if exit_code is None else f"exited:{exit_code}")
            if invalidate_on_failure:
                self._invalidate(request_process)
            raise WorkerError(f"Worker transport failure (process {process_state}): {exc}") from exc

    def request(self, method: str, path: str, *, value: dict | None = None,
                body: bytes | None = None, headers: dict[str, str] | None = None,
                timeout: int = 120) -> dict:
        self.ensure()
        if value is not None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json", **(headers or {})}
        return self._request(method, path, body=body, headers=headers, timeout=timeout)

    def load(self, config: dict) -> dict:
        with self._model_lock:
            return self.request("POST", "/models/load", value=config, timeout=180)

    def transcribe(self, audio: bytes, options: dict, config: dict) -> dict:
        metadata = json.dumps(options, ensure_ascii=False).encode("utf-8")
        if len(metadata) > 65_536:
            raise ValueError("Transcription options exceed 64 KiB")
        payload = struct.pack("<I", len(metadata)) + metadata + audio
        with self._model_lock:
            self.load(config)
            return self.request("POST", "/transcribe", body=payload,
                                headers={"Content-Type": "application/octet-stream"},
                                timeout=1800)

    def start_live(self, config: dict, options: dict) -> dict:
        self._prune_credentials()
        with self._model_lock:
            self.load(config)
            owner = secrets.token_urlsafe(48)
            browser_token = secrets.token_urlsafe(32)
            result = self.request("POST", "/sessions", value=options, headers={"X-R2T2-Owner": owner})
            sid = result["session_id"]
            with self._lock:
                self._owners[sid] = owner
                self._browser_tokens[sid] = browser_token
                self._session_activity[sid] = time.monotonic()
            return {**result, "browser_token": browser_token}

    def unload(self) -> dict:
        with self._model_lock:
            return self.request("POST", "/models/unload", value={})

    def check_browser(self, sid: str, token: str) -> None:
        self._prune_credentials()
        with self._lock:
            expected = self._browser_tokens.get(sid)
        if not expected or not secrets.compare_digest(expected, token):
            raise WorkerError("Invalid live-session credential")

    def session_request(self, method: str, sid: str, action: str, *, value: dict | None = None,
                        body: bytes | None = None, headers: dict | None = None) -> dict:
        self._prune_credentials()
        with self._lock:
            owner = self._owners.get(sid)
        if not owner:
            raise WorkerError("Session belongs to a previous worker generation or is unknown")
        try:
            answer = self.request(method, f"/sessions/{sid}/{action}", value=value, body=body,
                                  headers={"X-R2T2-Owner": owner, **(headers or {})},
                                  timeout=180 if action == "finish" else 120 if action == "feed" else 30)
            with self._lock:
                if self._owners.get(sid) == owner:
                    self._session_activity[sid] = time.monotonic()
            return answer
        except WorkerError as exc:
            if exc.http_status == 404 and exc.worker_code == "SESSION_NOT_FOUND":
                self._forget_credentials(sid)
            raise

    def close(self) -> None:
        with self._lock:
            self._invalidate()
            if self._log_file is not None:
                self._log_file.close()
            self._log_file = None


manager = WorkerManager()
atexit.register(manager.close)
