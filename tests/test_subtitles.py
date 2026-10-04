import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from subtitles import build_subtitles, cue_timestamp, render_subtitle, wrap_text
from r2t2_core.segmented import PresetBoundaryVAD, SegmentedStream
from r2t2_core.worker import Service, transcribe
from test_offline_files import FileRequest, OfflineEngine
from test_segmented import SegmentWords, NoBoundaryVAD

ROOT = Path(__file__).resolve().parents[1]


def payload(text="字幕 New York", seconds=2):
    return {"status": "complete", "text": text, "audio_samples_16k": seconds * 16000,
            "segments": [{"segment_id": 0, "start_sample": 0, "end_sample": seconds * 16000,
                          "text": text, "timing_method": "vad_segment"}]}


class SubtitleTests(unittest.TestCase):
    def test_stream_saves_entire_segment_not_final_delta(self):
        stream = SegmentedStream(SegmentWords("already committed caption", "unused"), vad=NoBoundaryVAD())
        events = stream.feed(np.full(16000, .1, dtype=np.float32))
        final = stream.finish()
        self.assertNotEqual(final["delta"], final["stable_text"])
        self.assertEqual(stream.segments[0]["text"], "already committed caption")
        result = {"status": "finalized", "text": final["stable_text"],
                  "segments": stream.segments, "audio_samples_16k": 16000}
        self.assertIn("already committed caption", render_subtitle(build_subtitles(result)))
        self.assertEqual("".join(e["delta"] for e in events) + final["delta"], result["text"])

    def test_speech_bounds_do_not_change_sample_ownership(self):
        class BoundsVAD(NoBoundaryVAD):
            def speech_bounds(self, start, end):
                return 4000, 12000
        stream = SegmentedStream(SegmentWords("hello", "unused"), vad=BoundsVAD())
        stream.feed(np.full(16000, .1, dtype=np.float32))
        stream.finish()
        segment = stream.segments[0]
        self.assertEqual((segment["start_sample"], segment["end_sample"]), (0, 16000))
        result = {"status": "complete", "text": stream.stable_text,
                  "segments": stream.segments, "audio_samples_16k": 16000}
        cue = build_subtitles(result)["cues"][0]
        self.assertEqual((cue["start_ms"], cue["end_ms"]), (250, 750))

    def test_draft_never_invents_time_splits_or_loses_words(self):
        result = payload("A. " + "unmapped " * 35, 30)
        doc = build_subtitles(result)
        self.assertEqual(doc["cue_count"], 1)
        self.assertEqual((doc["cues"][0]["start_ms"], doc["cues"][0]["end_ms"]), (0, 30000))
        self.assertIn("long_cue_needs_alignment_or_editing", doc["warnings"])
        self.assertEqual(" ".join(doc["cues"][0]["lines"]), result["text"].strip())
        self.assertEqual(doc["subtitle_status"], "requires_review")
        self.assertEqual(doc["asr_status"], "complete")

    def test_offset_applied_once_clips_and_changes_export(self):
        doc = build_subtitles(payload(), offset_ms=500)
        self.assertIn("00:00:00,500 --> 00:00:02,500", render_subtitle(doc))
        negative = build_subtitles(payload(), offset_ms=-500)
        self.assertEqual(negative["clipped_cues"], 1)
        self.assertEqual(negative["cues"][0]["start_ms"], 0)
        dropped = build_subtitles(payload(), offset_ms=-2000)
        self.assertEqual(dropped["subtitle_status"], "empty")
        self.assertEqual(dropped["dropped_cues"], 1)
        self.assertEqual(render_subtitle(dropped, "vtt"), "")

    def test_bad_timing_and_missing_text_are_rejected(self):
        for change in ({"start_sample": -1}, {"start_sample": 32000},
                       {"end_sample": 0}, {"end_sample": 32001},
                       {"start_sample": float("nan")}, {"end_sample": "bad"},
                       {"speech_start_sample": 40000}, {"text": "missing words"}):
            result = payload()
            result["segments"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_subtitles(result)
        result = payload()
        result["segments"][0].pop("text")
        with self.assertRaisesRegex(ValueError, "Re-run"):
            build_subtitles(result)
        result = payload("字幕字幕")
        result["segments"] = [{"start_sample": 0, "end_sample": 20000, "text": "字幕"},
                              {"start_sample": 16000, "end_sample": 32000, "text": "字幕"}]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            build_subtitles(result)

    def test_legacy_empty_and_partial_are_explicit(self):
        legacy = {"status": "complete", "text": "hello world", "audio_seconds": 2}
        with self.assertRaisesRegex(ValueError, "subtitle_timings"):
            build_subtitles(legacy)
        self.assertEqual(build_subtitles(legacy, whole_audio_draft=True)["cue_count"], 1)
        empty = build_subtitles({"status": "complete", "text": "", "audio_samples_16k": 32000})
        for kind in ("srt", "vtt"):
            self.assertEqual(render_subtitle(empty, kind), "")
        partial = payload()
        partial["truncated"] = True
        with self.assertRaisesRegex(ValueError, "allow_partial"):
            build_subtitles(partial)
        self.assertEqual(build_subtitles(partial, allow_partial=True)["subtitle_status"], "partial")

    def test_time_formats_and_unicode_wrapping(self):
        self.assertEqual(cue_timestamp(3600000 + 59999, "srt"), "01:00:59,999")
        self.assertEqual(cue_timestamp(3600000 + 60000, "vtt"), "01:01:00.000")
        self.assertEqual(" ".join(wrap_text("New York international subtitles", english_chars=12)),
                         "New York international subtitles")
        self.assertIn("international", wrap_text("New York international subtitles", english_chars=12))
        text = "繁體字幕你好世界這是一個測試"
        self.assertEqual("".join(wrap_text(text, chinese_chars=4)), text)
        self.assertIn("e\u0301", "".join(wrap_text("e\u0301 long words", english_chars=4)))
        self.assertTrue(render_subtitle(build_subtitles(payload()), "vtt").startswith("WEBVTT\n\n"))

    def test_nodes_share_bytes_cache_and_empty_behavior(self):
        name = "subtitle_node_tests"
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as directory:
            folders = types.SimpleNamespace(get_output_directory=lambda: directory)
            with patch.dict(sys.modules, folder_paths=folders):
                source = json.dumps(payload(), ensure_ascii=False)
                new = module.NODE_CLASS_MAPPINGS["R2T2Subtitle"]()
                old = module.NODE_CLASS_MAPPINGS["R2T2SaveTranscript"]()
                first = new.save(source, "srt", "safe")
                same = old.save(source, "srt", "safe")
                self.assertEqual(first["result"][0], same["result"][0])
                self.assertEqual(Path(first["result"][0]).read_bytes(), first["result"][1].encode())
                self.assertNotEqual(new.save(source, "srt", "safe", offset_ms=500)["result"][0], first["result"][0])
                for kind in ("srt", "vtt"):
                    empty = json.dumps({"status": "complete", "text": "", "audio_samples_16k": 32000})
                    self.assertEqual(new.save(empty, kind, "empty")["ui"]["files"], [])
                    self.assertEqual(old.save(empty, kind, "empty")["result"], ("",))
                    legacy = json.dumps({"status": "complete", "text": "hello", "audio_seconds": 2})
                    for node in (new, old):
                        with self.assertRaises(ValueError):
                            node.save(legacy, kind, "legacy")
                input_file = Path(directory) / "source.json"
                input_file.write_text(source, encoding="utf-8")
                output = Path(directory) / "cli.srt"
                process = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/json_to_subtitle.py"),
                                          str(input_file), "-o", str(output)], capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(output.read_bytes(), Path(first["result"][0]).read_bytes())


class TimingRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_offline_collects_subtitle_segments_only_when_requested(self):
        pcm = np.full(16000, .1, dtype=np.float32)
        svc = Service()
        svc.engine = OfflineEngine()
        class TestSegments(SegmentedStream):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, vad=PresetBoundaryVAD([]), **kwargs)
        with patch("r2t2_core.worker.SegmentedStream", TestSegments):
            result = json.loads((await transcribe(FileRequest(svc, pcm, subtitle_timings=True))).text)
        self.assertEqual(result["segments"][0]["text"], "part1")
        self.assertEqual(build_subtitles(result)["cue_count"], 1)
        legacy = json.loads((await transcribe(FileRequest(svc, pcm))).text)
        self.assertNotIn("segments", legacy)


if __name__ == "__main__":
    unittest.main()
