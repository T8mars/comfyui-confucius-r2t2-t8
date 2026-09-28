import unittest

import numpy as np

from r2t2_core.segmented import PresetBoundaryVAD, SegmentedStream, exact_zero_boundaries


class FakeEngine:
    def generate(self, audio, **kwargs):
        return {"text": "", "finish_reason": "stop"}

    def tokenize(self, text):
        return []

    def detokenize(self, ids):
        return ""


class FirstSegmentTruncates(FakeEngine):
    def __init__(self):
        self.finishes = 0

    def generate(self, audio, **kwargs):
        if kwargs["max_tokens"] == 512:
            self.finishes += 1
            return {"text": "", "finish_reason": "length" if self.finishes == 1 else "stop"}
        return {"text": "", "finish_reason": "stop"}


class SegmentWords(FakeEngine):
    def __init__(self, first, second, language="English"):
        self.words = (first, second)
        self.language = language

    def generate(self, audio, **kwargs):
        word = self.words[0 if audio[0] < 0.5 else 1]
        target = (f"language {self.language}<asr_text>" + word
                  if kwargs["language"] is None else word)
        prefix = kwargs.get("prefix", "")
        return {"text": target[len(prefix):] if target.startswith(prefix) else target,
                "finish_reason": "stop"}

    def tokenize(self, text):
        return list(text)

    def detokenize(self, ids):
        return "".join(ids)


class NoBoundaryVAD:
    def feed(self, pcm):
        return []


class IndependentDurationVAD:
    def __init__(self):
        self.seen = 0
        self.speech_samples = 0
        self.reset_count = 0

    def feed(self, pcm):
        self.seen += len(pcm)
        self.speech_samples += len(pcm)
        if self.speech_samples >= 17600:
            self.speech_samples = 0
            return [{"sample": self.seen, "reason": "forced"}]
        return []

    def reset_speech_duration(self):
        self.speech_samples = 0
        self.reset_count += 1


class SegmentedTests(unittest.TestCase):
    def test_late_auto_language_does_not_revise_segment_join(self):
        stream = SegmentedStream(FakeEngine(), vad=NoBoundaryVAD())
        stream.completed_text = "hello"
        stream.stable_text = "hello"
        first = stream._decorate({"stable_text": "world", "preview_text": "world",
                                  "language": "", "audio_end_sample": 640})
        second = stream._decorate({"stable_text": "worlds", "preview_text": "worlds",
                                   "language": "English", "audio_end_sample": 1280})
        self.assertEqual(first["stable_text"], "hello world")
        self.assertEqual(first["delta"], " world")
        self.assertEqual(second["stable_text"], "hello worlds")
        self.assertEqual(first["delta"] + second["delta"], " worlds")
        self.assertTrue(first["preview_text"].startswith(first["stable_text"]))
        self.assertTrue(second["preview_text"].startswith(second["stable_text"]))

    def test_word_spaced_segments_preserve_boundaries_and_append_only_events(self):
        cases = (("hello", "world", "English", "hello world"),
                 ("hello ", "world", "English", "hello world"),
                 ("hello", " world", "English", "hello world"),
                 ("Hello.", "World", "English", "Hello. World"),
                 ("hello", ", world", "English", "hello, world"),
                 ("你好", "世界", "Chinese", "你好世界"))
        pcm = np.concatenate((np.full(8000, 0.1, dtype=np.float32),
                              np.full(8000, 0.9, dtype=np.float32)))
        for first, second, language, expected in cases:
            with self.subTest(first=first, second=second, language=language):
                stream = SegmentedStream(SegmentWords(first, second, language),
                                         vad=PresetBoundaryVAD([8000]))
                events = []
                for start in range(0, len(pcm), 640):
                    events.extend(stream.feed(pcm[start:start + 640]))
                events.append(stream.finish())
                self.assertEqual(events[-1]["stable_text"], expected)
                self.assertEqual(events[-1]["preview_text"], expected)
                assembled = ""
                for event in events:
                    assembled += event["delta"]
                    self.assertEqual(assembled, event["stable_text"])
                    self.assertTrue(event["preview_text"].startswith(event["stable_text"]))
                self.assertEqual(assembled, expected)

    def test_interior_digital_zero_boundary_preserves_both_sides(self):
        pcm = np.concatenate((np.full(8000, 0.1, dtype=np.float32),
                              np.zeros(2400, dtype=np.float32),
                              np.full(8000, 0.1, dtype=np.float32)))
        boundaries = exact_zero_boundaries(pcm)
        self.assertEqual(boundaries, [10400])
        self.assertEqual(exact_zero_boundaries(np.full(18400, 0.001, dtype=np.float32)), [])
        stream = SegmentedStream(FakeEngine(), language="Chinese",
                                 vad=PresetBoundaryVAD(boundaries))
        for first in range(0, len(pcm), 640):
            stream.feed(pcm[first:first + 640])
        final = stream.finish()
        self.assertEqual([(s["start_sample"], s["end_sample"]) for s in stream.segments],
                         [(0, 10400), (10400, len(pcm))])
        self.assertEqual(stream.forced_boundaries, 0)
        self.assertEqual(final["audio_end_sample"], len(pcm))

    def test_forced_boundary_and_sample_ownership(self):
        stream = SegmentedStream(FakeEngine(), language="Chinese", vad=NoBoundaryVAD(), segment_seconds=1)
        stream.feed(np.full(24000, 0.1, dtype=np.float32))
        final = stream.finish()
        self.assertEqual([(s["start_sample"], s["end_sample"]) for s in stream.segments],
                         [(0, 16000), (16000, 24000)])
        self.assertEqual(stream.forced_boundaries, 1)
        self.assertEqual(final["audio_end_sample"], 24000)

    def test_exact_zero_silence_is_owned_without_asr(self):
        engine = FakeEngine()
        stream = SegmentedStream(engine, language="Chinese", vad=NoBoundaryVAD(), segment_seconds=1)
        events = stream.feed(np.zeros(24000, dtype=np.float32))
        self.assertEqual([event["end_reason"] for event in events], ["silence_rollover"])
        final = stream.finish()
        self.assertEqual(sum(s["skipped_zero_samples"] for s in stream.segments), 24000)
        self.assertEqual(stream.forced_boundaries, 0)
        self.assertEqual(final["audio_end_sample"], 24000)

    def test_duration_cut_resets_vad_hard_clock(self):
        vad = IndependentDurationVAD()
        stream = SegmentedStream(FakeEngine(), language="Chinese", vad=vad, segment_seconds=1)
        pcm = np.full(35200, 0.1, dtype=np.float32)
        for first in range(0, len(pcm), 640):
            stream.feed(pcm[first:first + 640])
        stream.finish()
        self.assertEqual([segment["end_reason"] for segment in stream.segments],
                         ["duration_limit", "duration_limit", "input_end"])
        self.assertEqual(vad.reset_count, 2)
        self.assertEqual(stream.segments[-1]["end_sample"], len(pcm))

    def test_minimum_segment_defers_early_silence_only(self):
        pcm = np.full(160000, 0.1, dtype=np.float32)
        vad = PresetBoundaryVAD([3 * 16000, 5 * 16000, 8 * 16000])
        stream = SegmentedStream(FakeEngine(), language="Chinese", vad=vad,
                                 min_segment_seconds=8)
        for first in range(0, len(pcm), 640):
            stream.feed(pcm[first:first + 640])
        stream.finish()
        self.assertEqual([(s["start_sample"], s["end_sample"]) for s in stream.segments],
                         [(0, 8 * 16000), (8 * 16000, len(pcm))])
        self.assertEqual(stream.forced_boundaries, 0)

    def test_early_silence_is_closed_at_minimum_instead_of_forced_at_hard_limit(self):
        pcm = np.concatenate((np.full(3 * 16000, 0.1, dtype=np.float32),
                              np.zeros(17 * 16000, dtype=np.float32)))
        stream = SegmentedStream(FakeEngine(), language="Chinese",
                                 vad=PresetBoundaryVAD([3 * 16000]),
                                 min_segment_seconds=8)
        for first in range(0, len(pcm), 640):
            stream.feed(pcm[first:first + 640])
        stream.finish()
        self.assertEqual([(s["start_sample"], s["end_sample"], s["end_reason"])
                          for s in stream.segments],
                         [(0, 8 * 16000, "silence"),
                          (8 * 16000, 20 * 16000, "input_end")])
        self.assertEqual(stream.forced_boundaries, 0)

    def test_earlier_segment_truncation_survives_later_normal_finish(self):
        from r2t2_core.worker import LiveState, Service, snapshot

        pcm = np.full(2 * 16000, 0.1, dtype=np.float32)
        stream = SegmentedStream(FirstSegmentTruncates(), language="Chinese",
                                 vad=PresetBoundaryVAD([16000]))
        stream.feed(pcm)
        final = stream.finish()
        self.assertEqual([segment["truncated"] for segment in stream.segments], [True, False])
        self.assertTrue(final["truncated"])
        self.assertEqual(final["finish_reason"], "length")
        state = LiveState(stream, "owner", "sid", samples=len(pcm), status="finalized")
        live = snapshot(state, Service())
        self.assertTrue(live["truncated"])
        self.assertEqual(live["quality_status"], "truncated")


if __name__ == "__main__":
    unittest.main()
