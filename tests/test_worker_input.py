"""Validate file requests through the real worker HTTP middleware, without CUDA."""

import importlib.util
import json
import struct
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from r2t2_core import worker
from r2t2_core.audio import comfy_audio_to_16k
from r2t2_core.native import build_prompt
from subtitles import build_subtitles, render_subtitle


class RecordingEngine:
    model_name = "fake-q8"
    projector_name = "fake-projector"

    def __init__(self):
        self.calls = []

    def tokenize(self, text):
        return list(map(ord, text))

    def detokenize(self, ids):
        return "".join(map(chr, ids))

    def transcribe(self, pcm, **options):
        build_prompt(options["context"], options["language"])
        self.calls.append(("offline", pcm.copy(), options))
        return {"text": "recognition", "language": "English", "truncated": False,
                "finish_reason": "stop"}

    def generate(self, pcm, **options):
        build_prompt(options["context"], options["language"])
        self.calls.append(("stream", pcm.copy(), options))
        target = "recognition"
        prefix = options["prefix"]
        return {"text": target[len(prefix):] if target.startswith(prefix) else target,
                "finish_reason": "stop"}


class WorkerInputTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.token = "test-token-" * 4
        self.app = worker.make_app(self.token)
        self.engine = RecordingEngine()
        self.app["service"].engine = self.engine
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def send(self, options, pcm=None):
        if pcm is None:
            pcm = np.full(1600, 0.1, dtype=np.float32)
        metadata = json.dumps(options, ensure_ascii=False).encode("utf-8")
        payload = struct.pack("<I", len(metadata)) + metadata + pcm.astype("<f4").tobytes()
        return await self.client.post("/transcribe", data=payload,
                                      headers={"Authorization": "Bearer " + self.token})

    def options(self, **changes):
        return {"sample_rate": 16000, "channels": 1, "language": "English",
                "auto_gain": False, **changes}

    async def assert_invalid(self, options, pcm=None, message=None):
        response = await self.send(options, pcm)
        self.assertEqual(response.status, 400, await response.text())
        body = await response.json(content_type=None)
        self.assertEqual(body["code"], "INVALID_INPUT")
        if message is not None:
            self.assertIn(message, body["message"])

    async def test_options_must_be_object(self):
        for options in (None, [], 3, "options"):
            with self.subTest(options=options):
                await self.assert_invalid(options, message="JSON object")
        self.assertEqual(self.engine.calls, [])

    async def test_required_integer_metadata_is_not_coerced(self):
        for name in ("sample_rate", "channels"):
            for value in (None, True, False, "16000", 1.5, [], {}):
                with self.subTest(name=name, value=value):
                    await self.assert_invalid(self.options(**{name: value}),
                                              message=name + " must be an integer")
            missing = self.options()
            del missing[name]
            await self.assert_invalid(missing, message=name + " must be an integer")
        self.assertEqual(self.engine.calls, [])

    async def test_valid_file_request_preserves_merged_context(self):
        response = await self.send(self.options(context="Interview", hotwords="New York; New York"))
        self.assertEqual(response.status, 200, await response.text())
        result = await response.json()
        self.assertEqual(result["text"], "recognition")
        self.assertEqual(len(self.engine.calls), 1)
        np.testing.assert_array_equal(self.engine.calls[0][1], np.full(1600, .1, np.float32))
        self.assertEqual(self.engine.calls[0][2],
                         {"language": "English", "context": "Interview\nHotwords: New York"})

    async def test_digital_silence_has_no_decode_or_subtitle_in_all_file_modes(self):
        pcm = np.zeros(16000, dtype=np.float32)
        for mode in ("offline", "stream"):
            for timings in (False, True):
                with self.subTest(mode=mode, timings=timings):
                    response = await self.send(self.options(mode=mode, subtitle_timings=timings), pcm)
                    self.assertEqual(response.status, 200, await response.text())
                    result = await response.json()
                    self.assertEqual(result["text"], "")
                    self.assertEqual(result["status"], "complete")
                    self.assertFalse(result["truncated"])
                    self.assertEqual(result["mode"], mode)
                    self.assertEqual(result["mode_executed"], "digital_silence")
                    self.assertEqual(result["audio_samples_16k"], 16000)
                    self.assertEqual(result["audio_seconds"], 1.0)
                    self.assertEqual(result["events"], [])
                    self.assertEqual(result["segments"], [])
                    document = build_subtitles(result)
                    self.assertEqual(document["cue_count"], 0)
                    self.assertEqual(render_subtitle(document), "")
        self.assertEqual(self.engine.calls, [])

    async def test_nonzero_low_amplitude_is_still_decoded(self):
        for mode in ("offline", "stream"):
            for auto_gain in (False, True):
                with self.subTest(mode=mode, auto_gain=auto_gain):
                    self.engine.calls.clear()
                    response = await self.send(self.options(mode=mode, auto_gain=auto_gain),
                                               np.full(1600, 1e-10, np.float32))
                    self.assertEqual(response.status, 200, await response.text())
                    result = await response.json()
                    self.assertEqual(result["text"], "recognition")
                    self.assertEqual(result["input_gain"], 10.0 if auto_gain else 1.0)
                    self.assertTrue(self.engine.calls)
                    self.assertTrue(np.any(self.engine.calls[0][1]))

    async def test_zero_shortcut_applies_to_selected_audio_channel(self):
        stereo = np.column_stack((np.full(1600, 0.1, np.float32), np.full(1600, -0.1, np.float32)))
        response = await self.send(self.options(channels=2, channel="mean"), stereo)
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual((await response.json())["text"], "")
        self.assertEqual(self.engine.calls, [])
        for channel in ("left", "right"):
            response = await self.send(self.options(channels=2, channel=channel), stereo)
            self.assertEqual(response.status, 200, await response.text())
            self.assertEqual((await response.json())["text"], "recognition")
        self.assertEqual(len(self.engine.calls), 2)

    async def test_silence_cannot_bypass_mode_language_or_channel_validation(self):
        pcm = np.zeros(1600, np.float32)
        for key, invalid in (("mode", "invalid"), ("mode", None), ("mode", []),
                             ("language", "invalid"), ("language", []),
                             ("channel", "invalid"), ("channel", None), ("channel", [])):
            with self.subTest(key=key, invalid=invalid):
                await self.assert_invalid(self.options(**{key: invalid}), pcm)
        await self.assert_invalid(self.options(channel="right"), pcm)
        self.assertEqual(self.engine.calls, [])

    async def test_null_language_preserves_legacy_auto_alias(self):
        response = await self.send(self.options(language=None), np.zeros(1600, np.float32))
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual((await response.json())["text"], "")
        self.assertEqual(self.engine.calls, [])

    async def test_silence_still_requires_a_loaded_model(self):
        self.app["service"].engine = None
        response = await self.send(self.options(), np.zeros(1600, np.float32))
        self.assertEqual(response.status, 409)
        self.assertEqual((await response.json(content_type=None))["code"], "MODEL_NOT_LOADED")

    async def test_silence_does_not_bypass_pcm_shape_range_or_finite_checks(self):
        pcm = np.zeros(1600, np.float32)
        for options, samples in ((self.options(sample_rate=7999), pcm),
                                 (self.options(sample_rate=192001), pcm),
                                 (self.options(channels=0), pcm),
                                 (self.options(channels=9), pcm),
                                 (self.options(), np.empty(0, np.float32)),
                                 (self.options(), np.array([float("nan")], np.float32)),
                                 (self.options(), np.array([float("inf")], np.float32))):
            with self.subTest(options=options, samples=samples):
                await self.assert_invalid(options, samples)
        metadata = json.dumps(self.options(channels=2)).encode()
        response = await self.client.post("/transcribe",
            data=struct.pack("<I", len(metadata)) + metadata + b"abc",
            headers={"Authorization": "Bearer " + self.token})
        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json(content_type=None))["code"], "INVALID_INPUT")
        self.assertEqual(self.engine.calls, [])

    async def test_tiny_nonzero_audio_cannot_be_mistaken_for_silence_after_resampling(self):
        await self.assert_invalid(self.options(sample_rate=192000), np.full(1, .1, np.float32),
                                  message="too short")
        self.assertEqual(self.engine.calls, [])

    async def test_resampled_silence_keeps_input_and_model_sample_counts(self):
        for sample_rate in (8000, 44100, 192000):
            with self.subTest(sample_rate=sample_rate):
                response = await self.send(self.options(sample_rate=sample_rate),
                                           np.zeros(sample_rate, np.float32))
                self.assertEqual(response.status, 200, await response.text())
                result = await response.json()
                self.assertEqual(result["input_sample_rate"], sample_rate)
                self.assertEqual(result["input_samples"], sample_rate)
                self.assertEqual(result["audio_samples_16k"], 16000)
                self.assertEqual(result["audio_seconds"], 1)
        self.assertEqual(self.engine.calls, [])


class NodeInputTests(unittest.TestCase):
    def test_audio_helper_rejects_sample_rate_coercion(self):
        audio = {"waveform": np.zeros((1, 1, 4410), np.float32), "sample_rate": 44100}
        for rate in (44100.5, "44100", True, None):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "sample_rate"):
                comfy_audio_to_16k({**audio, "sample_rate": rate})
        self.assertEqual(len(comfy_audio_to_16k(audio)), 1600)

    def test_node_rejects_invalid_sample_rate_without_calling_worker(self):
        root = Path(__file__).resolve().parents[1]
        name = "worker_input_node_tests"
        spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                     submodule_search_locations=[str(root)])
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[name] = plugin
        spec.loader.exec_module(plugin)
        module = sys.modules[name + ".nodes"]
        calls = []
        manager = types.SimpleNamespace(transcribe=lambda pcm, options, config:
            calls.append(options) or {"status": "complete", "text": "ok", "language": "English"})
        node = plugin.NODE_CLASS_MAPPINGS["R2T2Transcribe"]()
        audio = {"waveform": np.full((1, 1, 4410), .1, np.float32), "sample_rate": 44100}
        with patch.object(module, "manager", manager):
            for rate in (44100.5, "44100", True, None):
                with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "sample_rate"):
                    node.transcribe({"config": {}}, {**audio, "sample_rate": rate},
                                    "offline", "English", "", "", "mean")
            self.assertEqual(calls, [])
            node.transcribe({"config": {}}, audio, "offline", "English", "", "", "mean")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["sample_rate"], 44100)


if __name__ == "__main__":
    unittest.main()
