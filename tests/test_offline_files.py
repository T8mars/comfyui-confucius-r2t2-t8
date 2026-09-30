"""Long file offline requests must decode each bounded segment only once."""

import json
import struct
import unittest
from unittest.mock import patch

import numpy as np

from r2t2_core import worker
from r2t2_core.segmented import PresetBoundaryVAD, SegmentedStream


class OfflineEngine:
    model_name = "fake-q8"
    projector_name = "fake-projector"

    def __init__(self, truncate_first=False):
        self.calls = []
        self.truncate_first = truncate_first

    def generate(self, *args, **kwargs):
        raise AssertionError("offline file must not use streaming prefix generation")

    def transcribe(self, pcm, **options):
        self.calls.append((len(pcm), options))
        truncated = self.truncate_first and len(self.calls) == 1
        return {"text": f"part{len(self.calls)}", "language": "English",
                "truncated": truncated, "finish_reason": "length" if truncated else "stop"}


class FileRequest:
    def __init__(self, svc, pcm, **options):
        self.app = {"service": svc}
        metadata = json.dumps({"sample_rate": 16000, "channels": 1,
                               "mode": "offline", "language": "English",
                               "context": "test context", "hotwords": "example",
                               **options}).encode()
        self.payload = struct.pack("<I", len(metadata)) + metadata + pcm.astype("<f4").tobytes()

    async def read(self):
        return self.payload


class OfflineFileTests(unittest.IsolatedAsyncioTestCase):
    async def run_file(self, seconds, *, boundaries=(), truncate_first=False, silence=False):
        svc = worker.Service()
        engine = OfflineEngine(truncate_first)
        svc.engine = engine
        pcm = np.full(seconds * 16000, 0 if silence else 0.1, dtype=np.float32)

        class TestSegments(SegmentedStream):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, vad=PresetBoundaryVAD(list(boundaries)), **kwargs)

        with patch.object(worker, "SegmentedStream", TestSegments):
            response = await worker.transcribe(FileRequest(svc, pcm))
        return json.loads(response.text), engine

    async def test_over_30_seconds_stays_offline_and_decodes_once_per_segment(self):
        result, engine = await self.run_file(31)
        self.assertEqual(result["mode_executed"], "segmented_offline")
        self.assertEqual([count for count, _ in engine.calls], [20 * 16000, 11 * 16000])
        self.assertEqual(result["text"], "part1 part2")
        self.assertEqual(result["events"], [])
        self.assertEqual(result["status"], "requires_review")
        for _, options in engine.calls:
            self.assertEqual(options, {"language": "English", "context": "test context\nHotwords: example"})

    async def test_eight_minutes_have_contiguous_coverage_and_bounded_decodes(self):
        result, engine = await self.run_file(480)
        self.assertEqual(result["audio_seconds"], 480)
        self.assertEqual(len(engine.calls), 24)
        self.assertEqual(sum(size for size, _ in engine.calls), 480 * 16000)
        segments = result["segments"]
        self.assertEqual(segments[0]["start_sample"], 0)
        self.assertEqual(segments[-1]["end_sample"], 480 * 16000)
        for previous, current in zip(segments, segments[1:]):
            self.assertEqual(previous["end_sample"], current["start_sample"])
        self.assertEqual(result["events"], [])
        self.assertFalse(result["truncated"])

    async def test_silence_boundaries_preserve_all_samples_and_truncation(self):
        result, engine = await self.run_file(31, boundaries=(10 * 16000, 22 * 16000),
                                             truncate_first=True)
        self.assertEqual([size for size, _ in engine.calls], [10 * 16000, 12 * 16000, 9 * 16000])
        self.assertEqual(result["forced_boundaries"], 0)
        self.assertEqual(result["status"], "truncated")
        self.assertTrue(result["segments"][0]["truncated"])
        self.assertFalse(result["segments"][-1]["truncated"])

    async def test_digital_silence_skips_asr_without_forced_quality_warning(self):
        result, engine = await self.run_file(31, silence=True)
        self.assertEqual(engine.calls, [])
        self.assertEqual(result["mode_executed"], "segmented_offline")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["forced_boundaries"], 0)
        self.assertEqual(sum(s["skipped_zero_samples"] for s in result["segments"]), 31 * 16000)


if __name__ == "__main__":
    unittest.main()
