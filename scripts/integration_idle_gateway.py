"""Check that a silent live WebSocket learns about worker idle expiration."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


async def main() -> None:
    root = Path(__file__).resolve().parents[1]
    routes = web.RouteTableDef()
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = type("PromptServer", (), {"instance": types.SimpleNamespace(routes=routes)})
    sys.modules["server"] = fake_server
    spec = importlib.util.spec_from_file_location("r2t2_idle_gateway_test", root / "__init__.py",
                                                   submodule_search_locations=[str(root)])
    assert spec and spec.loader
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    from r2t2_idle_gateway_test.bridge import manager

    app = web.Application()
    app.add_routes(routes)
    client = TestClient(TestServer(app))
    await client.start_server()
    origin = str(client.make_url("/")).rstrip("/")
    try:
        start = await client.post("/r2t2/v1/live/start", json={"model_config": {
            "n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1},
            "language": "Chinese", "context": ""}, headers={"Origin": origin})
        assert start.status == 200, await start.text()
        created = await start.json()
        sid, token = created["session_id"], created["browser_token"]
        ws = await client.ws_connect(f"/r2t2/v1/live/{sid}/stream", headers={"Origin": origin})
        await ws.send_json({"browser_token": token})
        assert (await ws.receive_json())["type"] == "ready"
        await ws.send_bytes(struct.pack("<II", 0, 0) + np.zeros(640, dtype="<f4").tobytes())
        ack = await ws.receive_json()
        assert ack["type"] == "ack" and ack["ack_sample"] == 640
        expired = await ws.receive_json(timeout=45)
        assert expired["type"] == "error" and expired["code"] == "INTERRUPTED", expired
        result = await client.get(f"/r2t2/v1/live/{sid}/result",
                                  headers={"Origin": origin, "X-R2T2-Session-Token": token})
        assert result.status == 200 and (await result.json())["status"] == "interrupted"
        print(json.dumps({"idle_gateway": "passed", "ack_sample": 640,
                          "worker_status": "interrupted", "browser_event": expired["code"]}))
        await ws.close()
    finally:
        await client.close()
        manager.close()


if __name__ == "__main__":
    asyncio.run(main())
