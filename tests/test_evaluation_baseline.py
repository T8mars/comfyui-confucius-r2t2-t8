import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from r2t2_core.evaluation import BF16_MODEL_SHA256
from scripts.evaluate_corpus import validate_baseline, validate_manifest


class BaselineValidationTests(unittest.TestCase):
    def test_duplicate_audio_does_not_inflate_corpus_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            audio.write_bytes(b"fixed audio")
            digest = hashlib.sha256(audio.read_bytes()).hexdigest()
            base = {"audio_path": str(audio), "sha256": digest, "reference": "你好",
                    "language": "Chinese", "metric": "cer", "tags": []}
            with self.assertRaisesRegex(ValueError, "Duplicate audio"):
                validate_manifest([{**base, "id": "first"}, {**base, "id": "second"}],
                                  root / "manifest.jsonl")

    def test_paired_identity_is_checked_before_inference(self):
        entries = [{"id": "sample", "sha256": "a" * 64}]
        valid = {"id": "sample", "text": "你好", "audio_sha256": "a" * 64,
                 "model_sha256": BF16_MODEL_SHA256}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bf16.jsonl"
            path.write_text(json.dumps(valid, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(validate_baseline(entries, path)["sample"]["text"], "你好")
            for changed in ({**valid, "audio_sha256": "b" * 64},
                            {**valid, "model_sha256": "b" * 64},
                            {**valid, "id": "other"}):
                path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_baseline(entries, path)


if __name__ == "__main__":
    unittest.main()
