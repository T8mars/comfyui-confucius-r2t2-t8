"""Broken HTTP responses must use worker transport recovery, not raw errors."""

import http.client
import io
import unittest
import urllib.error
from unittest.mock import patch

from bridge import WorkerError, WorkerManager


class Process:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


class Response(io.BytesIO):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure

    def read(self, *args):
        raise self.failure


class BridgeTransportTests(unittest.TestCase):
    def manager(self):
        manager = WorkerManager()
        manager.port = 12345
        manager.token = "test-only-token"
        manager.generation = "old"
        manager.process = Process()
        manager._owners["old-session"] = "test-owner"
        return manager

    def test_broken_response_invalidates_process_and_credentials(self):
        for failure in (http.client.IncompleteRead(b'{"text":', 20),
                        http.client.RemoteDisconnected("peer closed"),
                        ConnectionResetError("reset"), BrokenPipeError("pipe closed")):
            with self.subTest(failure=type(failure).__name__):
                manager = self.manager()
                original = manager.process
                with patch.object(manager._local_http, "open", return_value=Response(failure)):
                    with self.assertRaisesRegex(WorkerError, "transport failure"):
                        manager._request("GET", "/sessions/old-session/result")
                self.assertTrue(original.terminated)
                self.assertIsNone(manager.process)
                self.assertIsNone(manager.generation)
                self.assertEqual(manager._owners, {})

    def test_health_probe_failure_keeps_process_for_startup_retry(self):
        manager = self.manager()
        original = manager.process
        with patch.object(manager._local_http, "open", return_value=Response(http.client.IncompleteRead(b"", 2))):
            with self.assertRaises(WorkerError):
                manager._request("GET", "/health", invalidate_on_failure=False)
        self.assertIs(manager.process, original)
        self.assertFalse(original.terminated)
        self.assertIn("old-session", manager._owners)

    def test_error_body_disconnect_also_recovers_transport(self):
        manager = self.manager()
        failure = urllib.error.HTTPError("http://127.0.0.1", 409, "busy", {},
                                         Response(http.client.IncompleteRead(b'{"code":', 10)))
        with patch.object(manager._local_http, "open", side_effect=failure):
            with self.assertRaisesRegex(WorkerError, "transport failure"):
                manager._request("POST", "/models/unload")
        self.assertIsNone(manager.process)

    def test_complete_validation_errors_keep_model_and_structured_code(self):
        for status in (400, 409):
            manager = self.manager()
            original = manager.process
            failure = urllib.error.HTTPError("http://127.0.0.1", status, "busy", {},
                                             io.BytesIO(b'{"code":"BUSY"}'))
            with patch.object(manager._local_http, "open", side_effect=failure):
                with self.assertRaises(WorkerError) as raised:
                    manager._request("POST", "/models/unload")
            self.assertEqual(raised.exception.http_status, status)
            self.assertEqual(raised.exception.worker_code, "BUSY")
            self.assertIs(manager.process, original)
            self.assertFalse(original.terminated)

    def test_old_request_failure_does_not_clear_replacement_worker(self):
        manager = self.manager()
        replacement = Process()

        def replaced_before_failure(*args, **kwargs):
            manager.process = replacement
            manager.generation = "new"
            manager._owners = {"new-session": "new-owner"}
            raise ConnectionResetError("old request broke")

        with patch.object(manager._local_http, "open", side_effect=replaced_before_failure):
            with self.assertRaises(WorkerError):
                manager._request("GET", "/health")
        self.assertIs(manager.process, replacement)
        self.assertFalse(replacement.terminated)
        self.assertEqual(manager.generation, "new")
        self.assertEqual(manager._owners, {"new-session": "new-owner"})


if __name__ == "__main__":
    unittest.main()
