"""Credential retention follows the worker's live-session lifetime."""

import io
import unittest
import urllib.error
from unittest.mock import patch

import bridge
from bridge import WorkerError, WorkerManager


class FakeManager(WorkerManager):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.failure = None

    def load(self, config):
        return {"loaded": True}

    def request(self, method, path, **kwargs):
        if path == "/sessions":
            self.count += 1
            return {"session_id": f"sid-{self.count}"}
        if self.failure:
            raise self.failure
        return {"status": "finalized", "text": "done"}


class CredentialRetentionTests(unittest.TestCase):
    def test_worker_confirmed_session_not_found_removes_only_that_sid(self):
        manager = FakeManager()
        first = manager.start_live({}, {})
        second = manager.start_live({}, {})
        manager.failure = WorkerError("Worker 404", http_status=404,
                                      worker_code="SESSION_NOT_FOUND")
        with self.assertRaises(WorkerError):
            manager.session_request("GET", first["session_id"], "result")
        self.assertNotIn(first["session_id"], manager._owners)
        self.assertNotIn(first["session_id"], manager._browser_tokens)
        self.assertIn(second["session_id"], manager._owners)

    def test_other_404_keeps_credentials(self):
        manager = FakeManager()
        created = manager.start_live({}, {})
        manager.failure = WorkerError("Worker 404: unrelated", http_status=404,
                                      worker_code="OTHER_NOT_FOUND")
        with self.assertRaises(WorkerError):
            manager.session_request("GET", created["session_id"], "result")
        self.assertIn(created["session_id"], manager._owners)
        self.assertIn(created["session_id"], manager._browser_tokens)

    def test_finalized_snapshot_remains_readable_within_retention_then_ttl_prunes(self):
        manager = FakeManager()
        with patch.object(bridge.time, "monotonic", return_value=1000):
            created = manager.start_live({}, {})
        sid = created["session_id"]
        with patch.object(bridge.time, "monotonic", return_value=1000 + 3600):
            self.assertEqual(manager.session_request("GET", sid, "result")["text"], "done")
            manager.check_browser(sid, created["browser_token"])
        self.assertIn(sid, manager._owners)
        with patch.object(bridge.time, "monotonic", return_value=1000 + 3600 + 75 * 60 + 1):
            with self.assertRaises(WorkerError):
                manager.check_browser(sid, created["browser_token"])
        self.assertNotIn(sid, manager._owners)
        self.assertNotIn(sid, manager._browser_tokens)

    def test_http_error_exposes_structured_code(self):
        manager = WorkerManager()
        manager.port = 12345
        manager.token = "test-token"
        body = io.BytesIO(b'{"code":"SESSION_NOT_FOUND"}')
        failure = urllib.error.HTTPError("http://127.0.0.1", 404, "missing", {}, body)

        class Opener:
            def open(self, request, timeout):
                raise failure

        manager._local_http = Opener()
        with self.assertRaises(WorkerError) as raised:
            manager._request("GET", "/sessions/sid/result")
        self.assertEqual(raised.exception.http_status, 404)
        self.assertEqual(raised.exception.worker_code, "SESSION_NOT_FOUND")

    def test_sparse_but_active_session_keeps_browser_credential(self):
        manager = FakeManager()
        with patch.object(bridge.time, "monotonic", return_value=0):
            created = manager.start_live({}, {})
        sid = created["session_id"]
        manager.request = lambda *args, **kwargs: {"status": "active"}
        # The worker limits PCM samples and inactivity, not wall time since
        # Start. Sparse feeds can keep a session active for over 75 minutes.
        with patch.object(bridge.time, "monotonic", return_value=4400):
            self.assertEqual(manager.session_request("GET", sid, "status")["status"], "active")
        with patch.object(bridge.time, "monotonic", return_value=4501):
            manager.check_browser(sid, created["browser_token"])
        self.assertIn(sid, manager._owners)


if __name__ == "__main__":
    unittest.main()
