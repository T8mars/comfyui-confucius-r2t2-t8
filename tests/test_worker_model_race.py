"""Deterministic model-switch tests without loading real GGUF weights."""

import asyncio
import json
import struct
import types
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from r2t2_core import worker


class FakeEngine:
    def __init__(self, **config):
        self.model_name = f"fake-{config['n_ctx']}"
        self.projector_name = "fake-projector"

    def transcribe(self, pcm, **kwargs):
        return {"text": self.model_name, "language": "English",
                "finish_reason": "stop", "truncated": False,
                "model": self.model_name, "projector": self.projector_name}


class PausedRequest:
    def __init__(self, service):
        self.app = {"service": service}
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        options = json.dumps({"sample_rate": 16000, "channels": 1,
                              "language": "English", "mode": "offline"}).encode()
        audio = np.full(100, 0.1, dtype="<f4").tobytes()
        self.payload = struct.pack("<I", len(options)) + options + audio

    async def read(self):
        self.started.set()
        await self.release.wait()
        return self.payload


class WorkerModelRaceTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def loaded_service():
        service = worker.Service()
        service.engine = FakeEngine(n_ctx=2048)
        service.model_config = {"n_ctx": 2048, "n_batch": 1024,
                                "n_threads": 8, "gpu_layers": -1}
        return service

    async def test_transcribe_uses_model_current_when_inference_lock_is_acquired(self):
        service = self.loaded_service()
        request = PausedRequest(service)
        with patch.object(worker, "NativeQ8Engine", FakeEngine):
            task = asyncio.create_task(worker.transcribe(request))
            await request.started.wait()
            try:
                loaded = await service.load({"n_ctx": 4096})
            finally:
                request.release.set()
            response = json.loads((await task).text)
        self.assertEqual(loaded["model"], "fake-4096")
        self.assertEqual(response["model"], "fake-4096")
        self.assertEqual(response["text"], "fake-4096")

    async def test_model_switch_releases_old_engine_and_terminal_sessions_before_construct(self):
        service = self.loaded_service()
        service.sessions["old"] = worker.LiveState(
            types.SimpleNamespace(engine=service.engine), "owner", "old", status="finalized")
        old_ref = weakref.ref(service.engine)
        old_alive_at_construct = []

        def construct(**config):
            old_alive_at_construct.append(old_ref() is not None)
            return FakeEngine(**config)

        with patch.object(worker, "NativeQ8Engine", construct):
            await service.load({"n_ctx": 4096})
        self.assertEqual(old_alive_at_construct, [False])
        self.assertEqual(service.sessions, {})
        self.assertEqual(service.engine.model_name, "fake-4096")

    async def test_failed_replacement_leaves_service_unloaded(self):
        service = self.loaded_service()
        service.sessions["old"] = worker.LiveState(
            types.SimpleNamespace(engine=service.engine), "owner", "old", status="finalized")

        def fail(**config):
            raise RuntimeError("synthetic model-load failure")

        with patch.object(worker, "NativeQ8Engine", fail):
            with self.assertRaisesRegex(RuntimeError, "synthetic model-load failure"):
                await service.load({"n_ctx": 4096})
        self.assertIsNone(service.engine)
        self.assertIsNone(service.model_config)
        self.assertEqual(service.sessions, {})


if __name__ == "__main__":
    unittest.main()
