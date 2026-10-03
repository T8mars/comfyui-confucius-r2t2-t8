import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


class SaveTests(unittest.TestCase):
    def test_atomic_idempotent_save_and_collision(self):
        root = Path(__file__).resolve().parents[1]
        name = "r2t2_save_test"
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)])
        assert spec and spec.loader
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[name] = plugin
        spec.loader.exec_module(plugin)
        node = plugin.NODE_CLASS_MAPPINGS["R2T2SaveTranscript"]()
        folder_paths = types.ModuleType("folder_paths")
        previous = sys.modules.get("folder_paths")
        sys.modules["folder_paths"] = folder_paths
        try:
            with tempfile.TemporaryDirectory(dir=(root / ".runtime") if (root / ".runtime").is_dir() else None) as directory:
                folder_paths.get_output_directory = lambda: directory
                payload = json.dumps({"status": "complete", "text": "测试 transcript"}, ensure_ascii=False)
                saved = node.save(payload, "txt", "../unsafe")
                first = Path(saved["result"][0])
                self.assertEqual(saved["ui"]["text"], (str(first),))
                self.assertEqual(first.parent, Path(directory).resolve())
                self.assertEqual(first.read_text(encoding="utf-8"), "测试 transcript")
                self.assertEqual(node.save(payload, "txt", "../unsafe")["result"][0], str(first))
                self.assertFalse(list(Path(directory).glob(".r2t2-*.tmp")))
                first.write_text("collision", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    node.save(payload, "txt", "../unsafe")
                with self.assertRaises(ValueError):
                    node.save(payload, "../escape", "safe")
                review = json.dumps({"status": "requires_review", "quality_status":
                                     "forced_boundaries_unverified", "text": "边界待核对"}, ensure_ascii=False)
                reviewed_path = Path(node.save(review, "json", "reviewable")["result"][0])
                self.assertEqual(json.loads(reviewed_path.read_text(encoding="utf-8"))["status"],
                                 "requires_review")
                with self.assertRaises(ValueError):
                    node.save(json.dumps({"status": "active", "text": "unfinished"}), "txt", "unfinished")
        finally:
            if previous is None:
                del sys.modules["folder_paths"]
            else:
                sys.modules["folder_paths"] = previous


if __name__ == "__main__":
    unittest.main()
