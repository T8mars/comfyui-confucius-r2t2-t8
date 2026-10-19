import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from subtitles import build_subtitles, cue_timestamp, render_subtitle, sample_milliseconds, wrap_text
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

    def test_tiny_fallback_uses_ownership_and_does_not_create_zero_ms_cue(self):
        for first, last in ((8000, 8001), (24, 40)):
            with self.subTest(first=first, last=last):
                stream = SegmentedStream(OfflineEngine(), vad=NoBoundaryVAD(), offline=True)
                pcm = np.zeros(16000, dtype=np.float32)
                pcm[first:last] = .1
                stream.feed(pcm)
                stream.finish()
                self.assertNotIn("speech_start_sample", stream.segments[0])
                result = {"status": "complete", "text": stream.stable_text,
                          "segments": stream.segments, "audio_samples_16k": 16000}
                self.assertEqual(build_subtitles(result)["cues"][0]["end_ms"], 1000)
                self.assertEqual(result["segments"][0]["text"], "part1")

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

    def test_srt_preserves_plain_symbols_and_vtt_uses_character_references(self):
        document = build_subtitles(payload("Tom & Jerry / A > B / <i>literal</i>"))
        srt = render_subtitle(document, "srt")
        self.assertIn("Tom & Jerry / A > B", srt)
        self.assertNotIn("&amp;", srt)
        self.assertIn("&amp;", render_subtitle(document, "vtt"))
        self.assertIn("&lt;i&gt;literal&lt;/i&gt;", render_subtitle(document, "vtt"))
        self.assertEqual(srt.count(" --> "), 1)

    def test_malformed_json_fields_fail_clearly_and_large_integer_timing_is_exact(self):
        for change in ({"status": []}, {"status": {}}, {"truncated": "true"},
                       {"forced_boundaries": "0"}, {"audio_seconds": 1e308}, {"text": "\ud800"}):
            source = {"status": "complete", "text": "hello", "audio_seconds": 2}
            source.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_subtitles(source, whole_audio_draft=True)
        source = payload()
        source["segments"][0]["truncated"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            build_subtitles(source)
        self.assertEqual(build_subtitles({"status": "complete", "text": "hello", "audio_seconds": 10 ** 1000},
                                        whole_audio_draft=True)["cue_count"], 1)
        for sample, expected in ((8, 0), (24, 2), (40, 2), (56, 4)):
            source = payload("x")
            source["segments"][0]["start_sample"] = sample
            self.assertEqual(build_subtitles(source)["cues"][0]["start_ms"], expected)
            self.assertEqual(sample_milliseconds(sample), expected)

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
                for node in (new, old):
                    first_token = node.IS_CHANGED(result_json=None, format="srt", prefix="safe")
                    second_token = node.IS_CHANGED(result_json=None, format="srt", prefix="safe")
                    self.assertTrue(math.isnan(first_token))
                    self.assertIsNot(first_token, second_token)
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

    def test_cli_rejects_malformed_json_without_traceback_or_file(self):
        changes = ({"status": []}, {"truncated": "true"}, {"forced_boundaries": []},
                   {"audio_seconds": 1e308}, {"text": "\ud800"})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            output = Path(directory) / "invalid.srt"
            for change in changes:
                result = {"status": "complete", "text": "hello", "audio_seconds": 2}
                result.update(change)
                source.write_text(json.dumps(result), encoding="utf-8")
                with self.subTest(change=change):
                    process = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/json_to_subtitle.py"),
                                              str(source), "-o", str(output), "--whole-audio-draft"],
                                             capture_output=True, text=True)
                    self.assertEqual(process.returncode, 1)
                    self.assertIn("Subtitle export failed:", process.stderr)
                    self.assertNotIn("Traceback", process.stderr)
                    self.assertFalse(output.exists())

    def test_convenience_srt_failure_keeps_transcribe_and_live_text_json(self):
        name = "subtitle_convenience_tests"
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        invalid = payload("hello")
        invalid["segments"][0]["end_sample"] = 1
        with patch.object(module.nodes.manager, "transcribe", return_value=invalid):
            outputs = module.nodes.R2T2Transcribe().transcribe(
                {"config": {}}, {"waveform": np.zeros((1, 1, 16000)), "sample_rate": 16000},
                "offline", "English", "", "", "mean", subtitle_timings=True)
        self.assertEqual((outputs[0], outputs[3]), ("hello", ""))
        saved = json.loads(outputs[2])
        self.assertEqual(saved["status"], "complete")
        self.assertIn("millisecond", saved["subtitle_export_error"])
        with self.assertRaises(ValueError):
            build_subtitles(saved)
        live = {**invalid, "status": "finalized", "generation": 7, "revision": 3}
        with patch.object(module.nodes.manager, "session_request", return_value=live), patch.object(module.nodes.manager, "generation", 7):
            outputs = module.nodes.R2T2LiveSession().read_snapshot({}, "English", "", "sid", 3)
        self.assertEqual((outputs[0], outputs[3]), ("hello", ""))
        self.assertIn("subtitle_export_error", json.loads(outputs[2]))

    def test_live_linked_model_fingerprint_and_invalidated_session(self):
        name = "subtitle_live_cache_tests"
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        node = module.nodes.R2T2LiveSession()
        # The real ComfyUI fingerprint path passes None for linked inputs.
        # It must re-read snapshots deliberately, without an AttributeError.
        first = node.IS_CHANGED(None, "Auto", "", "sid", 3)
        second = node.IS_CHANGED(None, "Auto", "", "sid", 3)
        self.assertTrue(math.isnan(first) and math.isnan(second))
        self.assertIsNot(first, second)
        live = {**payload("current text"), "status": "finalized", "generation": 7, "revision": 3}
        missing = module.nodes.WorkerError("Worker 404: session removed", http_status=404,
                                          worker_code="SESSION_NOT_FOUND")
        with patch.object(module.nodes.manager, "generation", 7), patch.object(
                module.nodes.manager, "session_request", side_effect=[live, missing]) as request:
            outputs = node.read_snapshot({"generation": 7}, "Auto", "", "sid", 3)
            self.assertEqual(outputs[0], "current text")
            self.assertTrue(outputs[3])
            # Unload may remove sessions without changing worker generation.
            # A second read must propagate that loss, not return the old text.
            with self.assertRaisesRegex(module.nodes.WorkerError, "session removed"):
                node.read_snapshot({"generation": 7}, "Auto", "", "sid", 3)
            self.assertEqual(request.call_count, 2)
            request.assert_called_with("GET", "sid", "result")


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
