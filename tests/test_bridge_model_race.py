import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from bridge import WorkerManager


class BridgeModelRaceTests(unittest.TestCase):
    def test_config_change_waits_for_transcription_request(self):
        manager = WorkerManager()
        transcribe_entered = threading.Event()
        release_transcribe = threading.Event()
        changed_model = threading.Event()
        calls = []

        def fake_request(method, path, **kwargs):
            if path == "/models/load":
                config = kwargs["value"]
                calls.append((path, config["n_ctx"]))
                if config["n_ctx"] == 4096:
                    changed_model.set()
                return {"config": config}
            calls.append((path, None))
            transcribe_entered.set()
            if not release_transcribe.wait(2):
                raise TimeoutError("test did not release transcribe request")
            return {"text": "ok"}

        manager.request = fake_request
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(manager.transcribe, b"\x00\x00\x00\x00", {}, {"n_ctx": 2048})
            self.assertTrue(transcribe_entered.wait(2))
            second = pool.submit(manager.load, {"n_ctx": 4096})
            self.assertFalse(changed_model.wait(0.1), "Model changed during an active transcription")
            release_transcribe.set()
            self.assertEqual(first.result(timeout=2)["text"], "ok")
            second.result(timeout=2)
        self.assertEqual(calls, [("/models/load", 2048), ("/transcribe", None),
                                 ("/models/load", 4096)])


if __name__ == "__main__":
    unittest.main()
