"""Model-release HTTP lifecycle, native ownership, Live snapshots and node ABI."""

import asyncio
import importlib.util
import json
import struct
import sys
import threading
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from r2t2_core import worker
from r2t2_core.segmented import PresetBoundaryVAD, SegmentedStream
from subtitles import build_subtitles, render_subtitle


class FakeEngine:
    def __init__(self, **config):
        self.model_name = f"fake-{config['n_ctx']}"
        self.projector_name = "projector"

    def transcribe(self, pcm, **options):
        return {"text": "hello", "language": "English", "truncated": False,
                "finish_reason": "stop", "model": self.model_name,
                "projector": self.projector_name}

    def generate(self, pcm, **options):
        text = "hello"
        prefix = options.get("prefix", "")
        return {"text": text[len(prefix):] if text.startswith(prefix) else text,
                "finish_reason": "stop"}

    def tokenize(self, text):
        return list(map(ord, text))

    def detokenize(self, ids):
        return "".join(map(chr, ids))


class TestStream(SegmentedStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vad=PresetBoundaryVAD([]), **kwargs)


class AutoReleaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.token = "release-test-token-" * 3
        self.engine_patch = patch.object(worker, "NativeQ8Engine", FakeEngine)
        self.engine_patch.start()
        self.addCleanup(self.engine_patch.stop)
        self.stream_patch = patch.object(worker, "SegmentedStream", TestStream)
        self.stream_patch.start()
        self.addCleanup(self.stream_patch.stop)
        self.app = worker.make_app(self.token)
        self.svc = self.app["service"]
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer " + self.token}
        await self.svc.load({"n_ctx": 2048})

    async def send(self, **options):
        pcm = np.full(6400, .1, dtype="<f4")
        metadata = json.dumps({"sample_rate": 16000, "channels": 1,
                               "language": "English", "auto_gain": False,
                               **options}).encode()
        return await self.client.post("/transcribe", headers=self.headers,
            data=struct.pack("<I", len(metadata)) + metadata + pcm.tobytes())

    async def terminal_live(self, status="finalized"):
        stream = TestStream(self.svc.engine, language="English")
        stream.feed(np.full(6400, .1, dtype=np.float32))
        stream.finish()
        state = worker.LiveState(stream, "owner-" * 8, "live-" + status,
                                 samples=6400, status=status, revision=7)
        self.svc.sessions[state.sid] = state
        return state

    async def read_live(self, state):
        response = await self.client.get(f"/sessions/{state.sid}/result",
            headers={**self.headers, "X-R2T2-Owner": state.owner})
        self.assertEqual(response.status, 200, await response.text())
        return await response.json()

    async def test_omitted_and_false_keep_the_loaded_engine(self):
        engine_ref = weakref.ref(self.svc.engine)
        for options in ({}, {"unload_after": False}):
            response = await self.send(**options)
            self.assertEqual(response.status, 200, await response.text())
            self.assertFalse((await response.json())["model_released"])
            self.assertIs(self.svc.engine, engine_ref())

    async def test_true_releases_native_owner_and_next_request_can_reload(self):
        engine_ref = weakref.ref(self.svc.engine)
        first = await self.send(unload_after=True)
        self.assertEqual(first.status, 200, await first.text())
        first_result = await first.json()
        self.assertTrue(first_result["model_released"])
        self.assertIsNone(engine_ref())
        self.assertIsNone(self.svc.engine)
        self.assertIsNone(self.svc.model_config)
        self.assertFalse(self.svc.live_warmed)
        await self.svc.load({"n_ctx": 2048})
        second = await self.send(unload_after=True)
        self.assertEqual(second.status, 200, await second.text())
        second_result = await second.json()
        self.assertEqual(first_result["text"], second_result["text"])
        self.assertTrue(second_result["model_released"])
        self.assertIsNone(self.svc.engine)

    async def test_nonboolean_option_rejected_before_inference_or_release(self):
        engine_ref = weakref.ref(self.svc.engine)
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value):
                response = await self.send(unload_after=value)
                self.assertEqual(response.status, 400, await response.text())
                self.assertIn("unload_after must be boolean",
                              (await response.json(content_type=None))["message"])
                self.assertIs(self.svc.engine, engine_ref())

    async def test_finalized_live_result_and_srt_survive_release_and_reload(self):
        state = await self.terminal_live()
        original = await self.read_live(state)
        srt = render_subtitle(build_subtitles(original))
        self.assertIn("hello", srt)
        engine_ref = weakref.ref(self.svc.engine)
        result = await self.send(unload_after=True)
        self.assertTrue((await result.json())["model_released"])
        self.assertIsNone(state.session)
        self.assertIsNone(engine_ref(), "terminal Live stream retained the native engine")
        self.assertEqual(await self.read_live(state), original)
        await self.svc.load({"n_ctx": 2048})
        self.assertEqual(await self.read_live(state), original)
        self.assertEqual(render_subtitle(build_subtitles(await self.read_live(state))), srt)
        # A repeated finish still returns the existing finalized revision.
        response = await self.client.post(f"/sessions/{state.sid}/finish", json={},
            headers={**self.headers, "X-R2T2-Owner": state.owner})
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(await response.json(), original)
        forbidden = await self.client.get(f"/sessions/{state.sid}/result", headers=self.headers)
        self.assertEqual(forbidden.status, 403)

    async def test_other_terminal_states_preserved_and_reaped_normally(self):
        states = [await self.terminal_live(status) for status in
                  ("interrupted", "cancelled", "failed")]
        originals = [await self.read_live(state) for state in states]
        engine_ref = weakref.ref(self.svc.engine)
        response = await self.send(unload_after=True)
        self.assertTrue((await response.json())["model_released"])
        self.assertIsNone(engine_ref())
        for state, original in zip(states, originals):
            self.assertIsNone(state.session)
            self.assertEqual(await self.read_live(state), original)
        await self.svc.reap_idle(max(s.updated for s in states) + worker.TERMINAL_RETENTION_SECONDS + 1)
        self.assertEqual(self.svc.sessions, {})

    async def test_active_and_draining_leases_prevent_automatic_and_explicit_release(self):
        state = await self.terminal_live()
        engine_ref = weakref.ref(self.svc.engine)
        for status in ("active", "draining"):
            state.status = status
            response = await self.send(unload_after=True)
            self.assertEqual(response.status, 200, await response.text())
            self.assertFalse((await response.json())["model_released"])
            self.assertIs(self.svc.engine, engine_ref())
            self.assertIsNotNone(state.session)
            unloaded = await self.client.post("/models/unload", json={}, headers=self.headers)
            self.assertEqual(unloaded.status, 409)

    async def test_explicit_unload_still_resets_preserved_sessions(self):
        state = await self.terminal_live()
        await self.send(unload_after=True)
        unloaded = await self.client.post("/models/unload", json={}, headers=self.headers)
        self.assertEqual(unloaded.status, 200, await unloaded.text())
        self.assertEqual(self.svc.sessions, {})
        response = await self.client.get(f"/sessions/{state.sid}/result",
            headers={**self.headers, "X-R2T2-Owner": state.owner})
        self.assertEqual(response.status, 404)

    async def test_failed_reload_does_not_discard_preserved_live_result(self):
        state = await self.terminal_live()
        original = await self.read_live(state)
        await self.send(unload_after=True)
        with patch.object(worker, "NativeQ8Engine", side_effect=RuntimeError("load failed")):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                await self.svc.load({"n_ctx": 2048})
        self.assertIsNone(self.svc.engine)
        self.assertEqual(await self.read_live(state), original)
        await self.svc.load({"n_ctx": 2048})
        self.assertEqual(await self.read_live(state), original)

    async def test_file_stream_drops_its_local_session_before_native_release(self):
        engine_ref = weakref.ref(self.svc.engine)
        response = await self.send(mode="stream", unload_after=True)
        self.assertEqual(response.status, 200, await response.text())
        self.assertTrue((await response.json())["model_released"])
        self.assertIsNone(engine_ref(), "file stream kept an engine reference")

    async def test_subtitle_offline_path_can_release_and_keep_exportable_segments(self):
        response = await self.send(subtitle_timings=True, unload_after=True)
        self.assertEqual(response.status, 200, await response.text())
        result = await response.json()
        self.assertTrue(result["model_released"])
        self.assertTrue(result["subtitle_timings"])
        self.assertIn("hello", render_subtitle(build_subtitles(result)))

    async def test_release_occurs_under_inference_lock_before_queued_model_load(self):
        entered, finish_inference = threading.Event(), threading.Event()
        original_release = self.svc.release
        release_lock_states = []

        def transcribe(pcm, **options):
            entered.set()
            if not finish_inference.wait(5):
                raise TimeoutError("test did not finish inference")
            return {"text": "first", "language": "English", "truncated": False,
                    "finish_reason": "stop"}

        def release(**options):
            release_lock_states.append(self.svc.lock.locked())
            return original_release(**options)

        self.svc.engine.transcribe = transcribe
        engine_ref = weakref.ref(self.svc.engine)
        with patch.object(self.svc, "release", release):
            file_task = asyncio.create_task(self.send(unload_after=True))
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            load_task = asyncio.create_task(self.svc.load({"n_ctx": 4096}))
            try:
                await asyncio.sleep(0)
                self.assertFalse(load_task.done())
            finally:
                finish_inference.set()
            file_response, loaded = await asyncio.gather(file_task, load_task)
        self.assertTrue((await file_response.json())["model_released"])
        self.assertEqual(release_lock_states, [True])
        self.assertIsNone(engine_ref())
        self.assertEqual(loaded["config"]["n_ctx"], 4096)
        self.assertEqual(self.svc.engine.model_name, "fake-4096")
        response = await self.send()
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual((await response.json())["model"], "fake-4096")


class AutoReleaseNodeTests(unittest.TestCase):
    def test_option_appended_after_subtitle_widget_and_legacy_default_is_false(self):
        root = Path(__file__).resolve().parents[1]
        name = "auto_release_nodes"
        spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                     submodule_search_locations=[str(root)])
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[name] = plugin
        spec.loader.exec_module(plugin)
        module = sys.modules[name + ".nodes"]
        node = plugin.NODE_CLASS_MAPPINGS["R2T2Transcribe"]()
        options = node.INPUT_TYPES()["optional"]
        self.assertEqual(list(options), ["auto_gain", "stream_chunk_ms", "subtitle_timings", "unload_after"])
        self.assertFalse(options["unload_after"][1]["default"])
        self.assertEqual(node.RETURN_NAMES, ("text", "language", "result_json", "srt"))
        calls = []
        manager = types.SimpleNamespace(transcribe=lambda pcm, opts, config:
            calls.append(opts) or {"text": "hello", "language": "English"})
        audio = {"sample_rate": 16000, "waveform": np.full((1, 1, 6400), .1, np.float32)}
        with patch.object(module, "manager", manager):
            # Exact old positional call, including subtitle_timings.
            node.transcribe({"config": {}}, audio, "offline", "English", "", "", "mean", True, 160, True)
            node.transcribe({"config": {}}, audio, "offline", "English", "", "", "mean", unload_after=True)
        self.assertTrue(calls[0]["subtitle_timings"])
        self.assertFalse(calls[0]["unload_after"])
        self.assertTrue(calls[1]["unload_after"])


if __name__ == "__main__":
    unittest.main()
