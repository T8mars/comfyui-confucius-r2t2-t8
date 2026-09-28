"""State-machine contract tests independent of CUDA and model weights."""

import unittest

import numpy as np

from r2t2_core.audio import limited_quiet_gain, to_mono_16k
from r2t2_core.streaming import StreamingSession


class FakeEngine:
    def tokenize(self, text):
        return [ord(c) for c in text]

    def detokenize(self, ids):
        return "".join(chr(i) for i in ids)

    def generate(self, audio, *, context, language, prefix, max_tokens):
        assert language == "English"
        target = "hello" if len(audio) <= 5120 else "hello world"
        if target.startswith(prefix):
            text = target[len(prefix):]
        else:
            text = target
        return {"text": text, "finish_reason": "stop"}


class TruncatingEngine(FakeEngine):
    def generate(self, audio, *, context, language, prefix, max_tokens):
        result = super().generate(audio, context=context, language=language,
                                  prefix=prefix, max_tokens=max_tokens)
        result["finish_reason"] = "length" if max_tokens == 512 else "stop"
        return result


class StreamingTests(unittest.TestCase):
    def test_repeated_finish_preserves_terminal_state_without_duplicate_delta(self):
        session = StreamingSession(TruncatingEngine(), language="English")
        session.feed(np.zeros(100, np.float32))
        final = session.finish()
        repeated = session.finish()
        self.assertEqual(final["finish_reason"], "length")
        self.assertTrue(final["truncated"])
        for key in ("truncated", "finish_reason", "stable_text", "preview_text",
                    "language", "audio_end_sample"):
            self.assertEqual(repeated[key], final[key])
        self.assertEqual(repeated["delta"], "")
        self.assertTrue(repeated["idempotent"])

    def test_initial_lookahead_and_final_commit(self):
        session = StreamingSession(FakeEngine(), language="English")
        self.assertEqual(session.feed(np.zeros(5119, np.float32)), [])
        events = session.feed(np.zeros(1, np.float32))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["audio_end_sample"], 5120)
        self.assertEqual(events[0]["stable_text"], "hell")
        events = session.feed(np.zeros(2560, np.float32))
        self.assertEqual(events[0]["stable_text"], "hello worl")
        final = session.finish()
        self.assertEqual(final["stable_text"], "hello world")
        self.assertEqual(final["delta"], "d")
        self.assertEqual(final["audio_end_sample"], 7680)
        self.assertTrue(session.finish()["idempotent"])

    def test_short_tail_is_processed(self):
        session = StreamingSession(FakeEngine(), language="English")
        self.assertEqual(session.feed(np.zeros(100, np.float32)), [])
        final = session.finish()
        self.assertEqual(final["audio_end_sample"], 100)
        self.assertEqual(final["stable_text"], "hello")

    def test_empty_audio_and_invalid_pcm(self):
        session = StreamingSession(FakeEngine(), language="English")
        self.assertEqual(session.finish()["stable_text"], "")
        with self.assertRaises(RuntimeError):
            session.feed(np.zeros(1, np.float32))
        with self.assertRaises(ValueError):
            StreamingSession(FakeEngine()).feed(np.array([np.nan], np.float32))

    def test_resampling_preserves_duration(self):
        stereo = np.zeros((44_100, 2), np.float32)
        self.assertEqual(len(to_mono_16k(stereo, 44_100)), 16_000)

    def test_quiet_gain_is_limited_and_never_boosts_digital_zero(self):
        quiet = np.full(16000, 0.00028, np.float32)
        raised, gain = limited_quiet_gain(quiet)
        self.assertEqual(gain, 10.0)
        self.assertAlmostEqual(float(raised[0]), 0.0028, places=6)
        self.assertEqual(limited_quiet_gain(np.zeros(16000, np.float32))[1], 1.0)
        self.assertEqual(limited_quiet_gain(np.full(16000, 0.1, np.float32))[1], 1.0)


if __name__ == "__main__":
    unittest.main()
