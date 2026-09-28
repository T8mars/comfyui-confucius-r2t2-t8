"""Browser WebSocket gateway tests without ComfyUI or a running worker."""

import asyncio
import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


class FakeManager:
    def __init__(self, error_type):
        self.calls = []
        self.error_type = error_type
        self.failure = None

    def start_live(self, config, options):
        self.calls.append(("start_live", config, options))
        if self.failure is not None:
            raise self.failure
        return {"session_id": "sid", "browser_token": "secret"}

    def check_browser(self, sid, token):
        if (sid, token) != ("sid", "secret"):
            raise self.error_type("Invalid live-session credential")

    def session_request(self, method, sid, action, **kwargs):
        self.calls.append((method, sid, action))
        if action == "status":
            return {"status": "active"}
        if action == "finish":
            return {"status": "finalized"}
        if action == "cancel":
            return {"status": "cancelled"}
        return {}


class WebSocketLeaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        root = Path(__file__).resolve().parents[1]
        server_module = types.ModuleType("server")
        server_module.PromptServer = types.SimpleNamespace(
            instance=types.SimpleNamespace(routes=web.RouteTableDef()))
        self.old_server = sys.modules.get("server")
        sys.modules["server"] = server_module
        name = "r2t2_ws_lease_test_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)])
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[name] = plugin
        spec.loader.exec_module(plugin)
        self.plugin = plugin
        self.manager = FakeManager(plugin._web_routes.WorkerError)
        plugin._web_routes.manager = self.manager
        app = web.Application()
        app.add_routes(server_module.PromptServer.instance.routes)
        self.server = TestServer(app)
        self.client = TestClient(self.server)
        await self.client.start_server()
        self.origin = str(self.server.make_url("/")).rstrip("/")

    async def asyncTearDown(self):
        await self.client.close()
        if self.old_server is None:
            sys.modules.pop("server", None)
        else:
            sys.modules["server"] = self.old_server

    async def connect(self):
        ws = await self.client.ws_connect(
            "/r2t2/v1/live/sid/stream", headers={"Origin": self.origin})
        await ws.send_json({"browser_token": "secret"})
        return ws

    async def post_start(self, data, *, content_type="application/json", origin=None):
        return await self.client.post("/r2t2/v1/live/start", data=data,
                                      headers={"Origin": origin or self.origin,
                                               "Content-Type": content_type})

    async def test_start_rejects_invalid_json_types_and_media(self):
        cases = (("[]", "application/json", 400),
                 ("null", "application/json", 400),
                 ("{", "application/json", 400),
                 ('{"model_config":[]}', "application/json", 400),
                 ("{}", "text/plain", 415))
        for body, content_type, expected_status in cases:
            with self.subTest(body=body, content_type=content_type):
                response = await self.post_start(body, content_type=content_type)
                self.assertEqual(response.status, expected_status)
                payload = await response.json()
                self.assertIn("code", payload)
                self.assertIn("message", payload)
        self.assertFalse(self.manager.calls)

    async def test_start_rejects_oversize_body_before_worker(self):
        response = await self.post_start('{"context":"' + "a" * (64 * 1024) + '"}')
        self.assertEqual(response.status, 413)
        self.assertEqual((await response.json())["code"], "PAYLOAD_TOO_LARGE")
        self.assertFalse(self.manager.calls)

    async def test_start_preserves_origin_and_worker_error_status(self):
        rejected = await self.post_start("{}", origin="http://evil.invalid")
        self.assertEqual(rejected.status, 403)
        self.assertFalse(self.manager.calls)
        self.manager.failure = self.plugin._web_routes.WorkerError(
            "bad input", http_status=400, worker_code="INVALID_INPUT")
        invalid = await self.post_start("{}")
        self.assertEqual(invalid.status, 400)
        self.assertEqual((await invalid.json())["code"], "INVALID_INPUT")
        self.manager.failure = self.plugin._web_routes.WorkerError("offline")
        unavailable = await self.post_start("{}")
        self.assertEqual(unavailable.status, 503)
        self.manager.failure = None
        success = await self.post_start("{}")
        self.assertEqual(success.status, 200)
        self.assertEqual((await success.json())["session_id"], "sid")

    async def test_http_cancel_requires_origin_and_browser_token(self):
        url = "/r2t2/v1/live/sid/cancel"
        bad_origin = await self.client.post(url, headers={
            "Origin": "http://evil.invalid", "X-R2T2-Session-Token": "secret"})
        self.assertEqual(bad_origin.status, 403)
        missing = await self.client.post(url, headers={"Origin": self.origin})
        self.assertEqual(missing.status, 403)
        wrong = await self.client.post(url, headers={
            "Origin": self.origin, "X-R2T2-Session-Token": "wrong"})
        self.assertEqual(wrong.status, 403)
        self.assertFalse(self.manager.calls)
        authorized = await self.client.post(url, headers={
            "Origin": self.origin, "X-R2T2-Session-Token": "secret"})
        self.assertEqual(authorized.status, 200)
        self.assertEqual((await authorized.json())["status"], "cancelled")
        self.assertEqual(self.manager.calls, [("POST", "sid", "cancel")])

    async def test_second_connection_cannot_cancel_primary_session(self):
        first = await self.connect()
        try:
            self.assertEqual((await first.receive_json())["type"], "ready")
            second = await self.connect()
            try:
                rejected = await second.receive_json()
                self.assertEqual(rejected["type"], "error")
                self.assertEqual(rejected["code"], "ALREADY_CONNECTED")
                await second.close()
                await asyncio.sleep(0.05)
                self.assertFalse(first.closed)
                self.assertFalse(any(action == "cancel" for _, _, action in self.manager.calls))
            finally:
                await second.close()
        finally:
            await first.close()
        await asyncio.sleep(0.05)
        self.assertEqual(sum(action == "cancel" for _, _, action in self.manager.calls), 1)

    async def test_finish_releases_lease_without_cancelling(self):
        first = await self.connect()
        self.assertEqual((await first.receive_json())["type"], "ready")
        await first.send_json({"type": "finish", "last_seq": -1, "total_samples": 0})
        self.assertEqual((await first.receive_json())["type"], "final")
        await first.close()
        for _ in range(50):
            if "sid" not in self.plugin._web_routes._stream_leases:
                break
            await asyncio.sleep(0.01)
        self.assertNotIn("sid", self.plugin._web_routes._stream_leases)
        self.assertFalse(any(action == "cancel" for _, _, action in self.manager.calls))
        second = await self.connect()
        try:
            self.assertEqual((await second.receive_json())["type"], "ready")
        finally:
            await second.close()

    async def test_bad_hello_never_claims_or_cancels_session(self):
        malformed = await self.client.ws_connect(
            "/r2t2/v1/live/sid/stream", headers={"Origin": self.origin})
        await malformed.send_json([])
        self.assertEqual((await malformed.receive_json())["code"], "STREAM_ERROR")
        await malformed.close()
        unauthorized = await self.client.ws_connect(
            "/r2t2/v1/live/sid/stream", headers={"Origin": self.origin})
        await unauthorized.send_json({"browser_token": "wrong"})
        self.assertEqual((await unauthorized.receive_json())["code"], "FORBIDDEN")
        await unauthorized.close()
        self.assertNotIn("sid", self.plugin._web_routes._stream_leases)
        self.assertFalse(any(action == "cancel" for _, _, action in self.manager.calls))
        valid = await self.connect()
        self.assertEqual((await valid.receive_json())["type"], "ready")
        await valid.close()

    async def test_bad_authenticated_command_cancels_and_releases(self):
        first = await self.connect()
        self.assertEqual((await first.receive_json())["type"], "ready")
        await first.send_json([])
        self.assertEqual((await first.receive_json())["code"], "STREAM_ERROR")
        await first.close()
        for _ in range(50):
            if "sid" not in self.plugin._web_routes._stream_leases:
                break
            await asyncio.sleep(0.01)
        self.assertNotIn("sid", self.plugin._web_routes._stream_leases)
        self.assertEqual(sum(action == "cancel" for _, _, action in self.manager.calls), 1)


if __name__ == "__main__":
    unittest.main()
