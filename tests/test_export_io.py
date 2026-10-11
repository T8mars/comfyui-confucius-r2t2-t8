import errno
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from export_io import atomic_write_no_overwrite


class ExportIOTests(unittest.TestCase):
    def test_idempotent_and_refuses_different_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "字幕.srt"
            data = "繁體字幕 & English\n".encode("utf-8")
            self.assertEqual(atomic_write_no_overwrite(path, data), path)
            atomic_write_no_overwrite(path, data)
            with self.assertRaises(FileExistsError):
                atomic_write_no_overwrite(path, b"different")
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse(list(Path(directory).glob(".r2t2-*.tmp")))

    @unittest.skipUnless(os.name == "nt", "Windows export should not require hard links")
    def test_windows_works_without_hard_link_support(self):
        with tempfile.TemporaryDirectory() as directory, patch("export_io.os.link", side_effect=OSError(errno.ENOTSUP, "unsupported")) as link:
            path = Path(directory) / "portable.srt"
            atomic_write_no_overwrite(path, b"caption")
            self.assertEqual(path.read_bytes(), b"caption")
            link.assert_not_called()

    def test_permission_failure_leaves_no_export_or_temporary_file(self):
        publisher = "export_io.os.rename" if os.name == "nt" else "export_io.os.link"
        with tempfile.TemporaryDirectory() as directory, patch(publisher, side_effect=PermissionError("read-only output")):
            path = Path(directory) / "unwritable.srt"
            with self.assertRaises(PermissionError):
                atomic_write_no_overwrite(path, b"caption")
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_concurrent_publication_has_one_content_and_no_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "race.srt"
            def publish(data):
                try:
                    atomic_write_no_overwrite(path, data)
                    return data
                except FileExistsError:
                    return None
            values = [b"A" * 20000, b"B" * 20000] * 6
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(publish, values))
            winner = path.read_bytes()
            self.assertIn(winner, values)
            self.assertTrue(any(value == winner for value in results))
            self.assertTrue(all(value in (None, winner) for value in results))
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_file_arriving_during_publish_is_compared_without_replacement(self):
        publisher = "export_io.os.rename" if os.name == "nt" else "export_io.os.link"
        for arrival in (b"same", b"different"):
            with self.subTest(arrival=arrival), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "arrival.srt"
                def race(source, target):
                    Path(target).write_bytes(arrival)
                    raise FileExistsError("race")
                with patch(publisher, side_effect=race):
                    if arrival == b"same":
                        atomic_write_no_overwrite(path, b"same")
                    else:
                        with self.assertRaises(FileExistsError):
                            atomic_write_no_overwrite(path, b"same")
                self.assertEqual(path.read_bytes(), arrival)
                self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
