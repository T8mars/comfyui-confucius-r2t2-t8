import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from bridge import WorkerManager
from hotwords import hotword_context, normalize_hotwords, read_hotword_file
from r2t2_core import worker


ROOT = Path(__file__).resolve().parents[1]


class HotwordTests(unittest.TestCase):
    def test_batch_delimiters_deduplicate_without_splitting_phrases(self):
        words, count = normalize_hotwords("\ufeff 网易有道\nConfucius4，网易有道; New York；繁體中文、\n")
        self.assertEqual(words, "网易有道, Confucius4, New York, 繁體中文")
        self.assertEqual(count, 4)
        self.assertEqual(normalize_hotwords(" ,\n； "), ("", 0))

    def test_context_budget_is_checked_after_merging(self):
        self.assertEqual(hotword_context("产品访谈", "品牌甲\n品牌乙"),
                         "产品访谈\nHotwords: 品牌甲, 品牌乙")
        self.assertEqual(hotword_context("x" * 8192, ""), "x" * 8192)
        with self.assertRaisesRegex(ValueError, "CONTEXT_LIMIT"):
            hotword_context("x" * 8192, "品牌")
        for value in (None, ["品牌"], 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_hotwords(value)

    def test_local_txt_is_bounded_and_cannot_escape_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "input"
            inside.mkdir()
            (inside / "words.txt").write_bytes("\ufeff品牌甲\nNew York".encode("utf-8"))
            self.assertIn("New York", read_hotword_file(inside, "words.txt").decode("utf-8-sig"))
            (root / "outside.txt").write_text("private", encoding="utf-8")
            for filename in ("../outside.txt", str(root / "outside.txt")):
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    read_hotword_file(inside, filename)
            (inside / "oversize.txt").write_bytes(b"a" * (65536 + 1))
            with self.assertRaisesRegex(ValueError, "64 KiB"):
                read_hotword_file(inside, "oversize.txt")
            (inside / "gbk.txt").write_bytes("中文".encode("gbk"))
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                read_hotword_file(inside, "gbk.txt")

    def test_file_edits_invalidate_node_cache_and_reach_transcription(self):
        name = "r2t2_hotwords_node_test"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[name] = plugin
        spec.loader.exec_module(plugin)
        module = sys.modules[name + ".nodes"]
        node = plugin.NODE_CLASS_MAPPINGS["R2T2Hotwords"]()
        folder_paths = types.ModuleType("folder_paths")
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"folder_paths": folder_paths}):
            folder_paths.get_input_directory = lambda: directory
            path = Path(directory) / "words.txt"
            path.write_text("品牌甲\nNew York", encoding="utf-8")
            before = node.IS_CHANGED("品牌甲;额外词", "words.txt")
            self.assertEqual(node.build("品牌甲;额外词", "words.txt"),
                             ("品牌甲, 额外词, New York", 3))
            path.write_text("品牌乙\nNew York", encoding="utf-8")
            self.assertNotEqual(before, node.IS_CHANGED("品牌甲;额外词", "words.txt"))
            hotwords, count = node.build("品牌甲;额外词", "words.txt")
            self.assertEqual(count, 4)
            import numpy as np

            fake = types.SimpleNamespace(transcribe=lambda audio, options, config: {
                "text": options["hotwords"], "language": "Chinese", "status": "complete"})
            with patch.object(module, "manager", fake):
                text, _, _, _ = plugin.NODE_CLASS_MAPPINGS["R2T2Transcribe"]().transcribe(
                    {"config": {}}, {"waveform": np.zeros((1, 1, 16000)), "sample_rate": 16000},
                    "offline", "Chinese", "产品访谈", hotwords, "mean")
            self.assertEqual(text, "品牌甲, 额外词, 品牌乙, New York")

    def test_valid_non_ascii_prompt_fits_bridge_metadata(self):
        manager = WorkerManager()
        captured = {}
        with patch.object(manager, "load"), patch.object(manager, "request",
                side_effect=lambda method, path, **kw: captured.update(kw) or {}):
            manager.transcribe(b"\0" * 4, {"hotwords": "词" * 8000}, {})
        body = captured["body"]
        size = struct.unpack_from("<I", body)[0]
        self.assertGreater(size, 16384)
        self.assertLessEqual(size, 65536)
        self.assertEqual(json.loads(body[4:4 + size])["hotwords"], "词" * 8000)


class HotwordWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_ascii_batch_reaches_offline_prompt_with_large_metadata(self):
        calls = []
        svc = worker.Service()
        svc.engine = types.SimpleNamespace(model_name="fake-q8", projector_name="fake-projector",
            transcribe=lambda pcm, **options: calls.append(options) or {
                "text": "test", "language": "Chinese", "truncated": False, "finish_reason": "stop"})
        metadata = json.dumps({"sample_rate": 16000, "channels": 1, "language": "Chinese",
                               "hotwords": "词" * 8000, "auto_gain": False}, ensure_ascii=False).encode("utf-8")
        self.assertGreater(len(metadata), 16384)
        request = types.SimpleNamespace(app={"service": svc})

        async def read():
            return struct.pack("<I", len(metadata)) + metadata + struct.pack("<f", 0.1) * 1600

        request.read = read
        result = await worker.transcribe(request)
        self.assertEqual(json.loads(result.text)["text"], "test")
        self.assertEqual(calls[0]["context"], "Hotwords: " + "词" * 8000)


if __name__ == "__main__":
    unittest.main()
