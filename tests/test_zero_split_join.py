import asyncio
import json
import struct
import unittest

import numpy as np

from r2t2_core.worker import Service, transcribe


class TwoPieceEngine:
    model_name = "test-model"
    projector_name = "test-projector"

    def __init__(self, pieces):
        self.pieces = iter(pieces)

    def transcribe(self, audio, **kwargs):
        text, language = next(self.pieces)
        return {"text": text, "language": language,
                "finish_reason": "stop", "truncated": False}


class FakeRequest:
    def __init__(self, service, pcm, language):
        options = {"sample_rate": 16000, "channels": 1, "channel": "mean",
                   "mode": "offline", "language": language, "auto_gain": False}
        metadata = json.dumps(options).encode("utf-8")
        self.payload = struct.pack("<I", len(metadata)) + metadata + pcm.tobytes()
        self.app = {"service": service}

    async def read(self):
        return self.payload


class OfflineZeroSplitJoinTests(unittest.TestCase):
    def test_chinese_segments_do_not_gain_english_spaces(self):
        pcm = np.concatenate((np.full(8000, 0.1, dtype="<f4"),
                              np.zeros(2400, dtype="<f4"),
                              np.full(8000, 0.1, dtype="<f4")))
        service = Service()
        service.engine = TwoPieceEngine([("你好", "Chinese"), ("世界", "Chinese")])
        response = asyncio.run(transcribe(FakeRequest(service, pcm, "Chinese")))
        result = json.loads(response.text)
        self.assertEqual(result["mode_executed"], "offline_zero_split")
        self.assertEqual(result["text"], "你好世界")

    def test_english_segments_get_word_boundary(self):
        pcm = np.concatenate((np.full(8000, 0.1, dtype="<f4"),
                              np.zeros(2400, dtype="<f4"),
                              np.full(8000, 0.1, dtype="<f4")))
        service = Service()
        service.engine = TwoPieceEngine([("hello", "English"), ("world", "English")])
        response = asyncio.run(transcribe(FakeRequest(service, pcm, "English")))
        self.assertEqual(json.loads(response.text)["text"], "hello world")


if __name__ == "__main__":
    unittest.main()
