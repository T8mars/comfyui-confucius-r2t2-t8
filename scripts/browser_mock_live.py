"""Exercise the real ComfyUI live-node UI with injected PCM, without microphone access."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
from playwright.async_api import async_playwright

from public_sample import ensure_sample

BASE = "http://127.0.0.1:8197"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ROOT = Path(__file__).resolve().parents[1]


async def wait_default_canvas(page):
    await page.wait_for_function("""async () => {
        try {
            const {app} = await import('/scripts/app.js');
            return !!window.LiteGraph?.registered_node_types?.R2T2LiveSession &&
                !!app.canvas && app.graph?._nodes.some(n => n.type === 'SaveImage');
        } catch { return false; }
    }""", timeout=60000)
    await page.wait_for_timeout(2000)


async def main() -> None:
    audio, rate = sf.read(ensure_sample(), dtype="float32")
    assert rate == 16000
    encoded = base64.b64encode(np.ascontiguousarray(audio, dtype="<f4").tobytes()).decode("ascii")
    prefix = f"r2t2_live_ui_{uuid.uuid4().hex[:8]}"
    output_dir = ROOT / ".runtime" / "comfy-output"
    assert not list(output_dir.glob(prefix + "_*.json")), "Run prefix already has saved output"
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=CHROME, headless=True)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            page_errors = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            await page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)
            await wait_default_canvas(page)
            workflow = (ROOT / "workflows" / "confucius4_q8_live.json").read_bytes()
            await page.evaluate("""({bytes, name}) => {
                const file = new File([Uint8Array.from(bytes)], name, {type: 'application/json'});
                const transfer = new DataTransfer();
                transfer.items.add(file);
                const target = document.querySelector('canvas') || document.body;
                for (const type of ['dragenter', 'dragover', 'drop'])
                    target.dispatchEvent(new DragEvent(type,
                        {bubbles: true, cancelable: true, dataTransfer: transfer}));
            }""", {"bytes": list(workflow), "name": "confucius4_q8_live.json"})
            await page.wait_for_function("""async () => {
                const {app} = await import('/scripts/app.js');
                return app.graph._nodes.some(n => n.type === 'R2T2LiveSession') &&
                       app.graph._nodes.some(n => n.type === 'R2T2SaveTranscript');
            }""", timeout=10000)
            selected_prefix = await page.evaluate("""async ({encoded, prefix}) => {
                const raw = atob(encoded);
                const bytes = new Uint8Array(raw.length);
                for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
                window.__mockSamples = new Float32Array(bytes.buffer);
                window.__mockTracksStopped = false;
                const media = {getTracks: () => [{stop: () => {window.__mockTracksStopped = true;}}]};
                Object.defineProperty(navigator, 'mediaDevices', {configurable: true,
                    value: {getUserMedia: async () => media}});
                class MockAudioContext {
                    constructor() { this.audioWorklet = {addModule: async () => {}}; this.destination = {}; }
                    createMediaStreamSource() { return {connect() {}, disconnect() {}}; }
                    createGain() { return {gain: {value: 1}, connect(destination) {return destination;}, disconnect() {}}; }
                    async close() { window.__mockContextClosed = true; }
                }
                class MockAudioWorkletNode {
                    constructor() {
                        this.port = {onmessage: null, postMessage: (value) => {
                            if (value.type === 'flush') queueMicrotask(() =>
                                this.port.onmessage?.({data: {type: 'flushed'}}));
                        }};
                        window.__mockWorklet = this;
                    }
                    connect(destination) { return destination; }
                    disconnect() { window.__mockWorkletDisconnected = true; }
                }
                window.AudioContext = MockAudioContext;
                window.AudioWorkletNode = MockAudioWorkletNode;
                const {app} = await import('/scripts/app.js');
                const live = app.graph._nodes.find(n => n.type === 'R2T2LiveSession');
                if (app.graph._nodes.length !== 3 || Object.values(app.graph.links || {}).length !== 2)
                    throw new Error('Dragged Live workflow graph does not have 3 nodes and 2 links');
                const minimum = live.widgets.find(w => w.name === 'min_segment_seconds');
                if (!minimum || minimum.value !== 8) throw new Error('Live minimum segment default mismatch');
                minimum.value = 4;
                const save = app.graph._nodes.find(n => n.type === 'R2T2SaveTranscript');
                const savePrefix = save.widgets.find(w => w.name === 'prefix');
                if (!savePrefix) throw new Error('Dragged Live Save node has no prefix widget');
                savePrefix.value = prefix;
                window.__r2t2MockLive = live;
                return savePrefix.value;
            }""", {"encoded": encoded, "prefix": prefix})
            assert selected_prefix == prefix
            await page.evaluate("""async () => {
                const node = window.__r2t2MockLive;
                await node.widgets.find(w => w.name === 'Start microphone').callback();
            }""")
            start = await page.evaluate("""() => {
                const node = window.__r2t2MockLive;
                const status = node.widgets.find(w => w.name === 'R2T2 status').value;
                return {status, sid: node.r2t2?.sid, token: node.r2t2?.browserToken,
                        worklet: !!window.__mockWorklet};
            }""")
            assert start["status"] == "recording" and start["sid"] and start["worklet"], start

            for pos in range(0, len(audio), 640):
                end = min(len(audio), pos + 640)
                await page.evaluate("""({start, end}) => {
                    const samples = new Float32Array(window.__mockSamples.subarray(start, end));
                    window.__mockWorklet.port.onmessage({data: {type: 'pcm', samples}});
                }""", {"start": pos, "end": end})
                if end % (640 * 12) == 0 or end == len(audio):
                    await page.wait_for_function("end => window.__r2t2MockLive.r2t2?.acked >= end",
                                                 arg=end, timeout=60000)

            await page.evaluate("""async () => {
                const node = window.__r2t2MockLive;
                await node.widgets.find(w => w.name === 'Stop and finalize').callback();
            }""")
            await page.wait_for_function("""() => {
                const node = window.__r2t2MockLive;
                return node.widgets.find(w => w.name === 'revision').value > 0;
            }""", timeout=60000)
            final = await page.evaluate("""() => {
                const node = window.__r2t2MockLive;
                const value = name => node.widgets.find(w => w.name === name)?.value;
                return {status: value('R2T2 status'), text: value('Stable text'),
                        panel: node.r2t2Panel?.stable.textContent,
                        session_id: value('session_id'), revision: value('revision'),
                        tracks_stopped: window.__mockTracksStopped,
                        context_closed: window.__mockContextClosed,
                        worklet_disconnected: window.__mockWorkletDisconnected};
            }""")
            assert final["status"].startswith("complete"), final
            assert final["session_id"] == start["sid"] and final["revision"] > 0
            assert final["text"] == final["panel"] and "之前有顾客" in final["text"]
            assert final["tracks_stopped"] and final["context_closed"] and final["worklet_disconnected"]
            async with page.expect_response(
                lambda response: response.request.method == "POST" and response.url.rstrip("/").endswith("/prompt"),
                timeout=20000,
            ) as queued_response:
                await page.get_by_role("button", name="运行", exact=True).click()
            queued = await queued_response.value
            assert queued.status == 200, await queued.text()
            prompt_id = (await queued.json())["prompt_id"]
            async with aiohttp.ClientSession() as http:
                for _ in range(60):
                    history = await (await http.get(BASE + f"/history/{prompt_id}")).json()
                    if prompt_id in history:
                        assert history[prompt_id]["status"]["status_str"] == "success", history[prompt_id]
                        break
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError("Dragged Live workflow did not finish after UI Run")
            output_files = list(output_dir.glob(prefix + "_*.json"))
            assert len(output_files) == 1, "Dragged Live workflow did not save one fresh transcript"
            saved = json.loads(output_files[0].read_text(encoding="utf-8"))
            assert (saved["text"] == final["text"] and saved["status"] == "finalized" and
                    saved["session_id"] == start["sid"] and saved["revision"] == final["revision"]), saved
            async with aiohttp.ClientSession() as http:
                snapshot_response = await http.get(
                    BASE + f"/r2t2/v1/live/{start['sid']}/result",
                    headers={"Origin": BASE, "X-R2T2-Session-Token": start["token"]})
                snapshot = await snapshot_response.json()
                assert snapshot_response.status == 200 and snapshot["min_segment_seconds"] == 4, snapshot
            await page.wait_for_function("() => window.__r2t2MockLive.r2t2 === null", timeout=10000)
            await page.evaluate("""() => {
                const previousFetch = window.fetch.bind(window);
                window.fetch = async (...args) => {
                    const response = await previousFetch(...args);
                    if (String(args[0]) === '/r2t2/v1/live/start' && response.ok) {
                        window.__deniedSession = await response.clone().json();
                    }
                    return response;
                };
                Object.defineProperty(navigator, 'mediaDevices', {configurable: true,
                    value: {getUserMedia: async () => {
                        throw new DOMException('Permission denied', 'NotAllowedError');
                    }}});
            }""")
            await page.evaluate("""async () => {
                const node = window.__r2t2MockLive;
                await node.widgets.find(w => w.name === 'Start microphone').callback();
            }""")
            denied = await page.evaluate("""() => {
                const node = window.__r2t2MockLive;
                return {status: node.widgets.find(w => w.name === 'R2T2 status').value,
                        sid: window.__deniedSession?.session_id,
                        token: window.__deniedSession?.browser_token,
                        cleared: node.r2t2 === null};
            }""")
            assert denied["status"].startswith("error:") and denied["sid"] and denied["cleared"], denied
            await page.wait_for_function("""async ({sid, token}) => {
                const response = await fetch(`/r2t2/v1/live/${sid}/result`,
                    {headers: {'X-R2T2-Session-Token': token}});
                return response.ok && (await response.json()).status === 'cancelled';
            }""", arg={"sid": denied["sid"], "token": denied["token"]}, timeout=10000)
            await page.evaluate("""() => {
                const media = {getTracks: () => [{stop() {}}]};
                Object.defineProperty(navigator, 'mediaDevices', {configurable: true,
                    value: {getUserMedia: async () => media}});
            }""")
            await page.evaluate("""async () => {
                const node = window.__r2t2MockLive;
                await node.widgets.find(w => w.name === 'Start microphone').callback();
            }""")
            overload = await page.evaluate("""() => {
                const node = window.__r2t2MockLive;
                const samples = new Float32Array(32000);
                for (let i = 0; i < 3; i++) {
                    window.__mockWorklet.port.onmessage({data: {type: 'pcm', samples}});
                }
                return {status: node.widgets.find(w => w.name === 'R2T2 status').value,
                        sid: window.__deniedSession?.session_id,
                        token: window.__deniedSession?.browser_token};
            }""")
            assert overload["status"].startswith("OVERLOAD") and overload["sid"], overload
            await page.wait_for_function("""async ({sid, token}) => {
                const response = await fetch(`/r2t2/v1/live/${sid}/result`,
                    {headers: {'X-R2T2-Session-Token': token}});
                return response.ok && (await response.json()).status === 'cancelled';
            }""", arg={"sid": overload["sid"], "token": overload["token"]}, timeout=10000)
            await page.wait_for_function("() => window.__r2t2MockLive.r2t2 === null", timeout=10000)
            await page.evaluate("""async () => {
                const node = window.__r2t2MockLive;
                await node.widgets.find(w => w.name === 'Start microphone').callback();
                await node.widgets.find(w => w.name === 'Cancel').callback();
            }""")
            cancelled = await page.evaluate("""() => window.__deniedSession""")
            await page.wait_for_function("""async ({sid, token}) => {
                const response = await fetch(`/r2t2/v1/live/${sid}/result`,
                    {headers: {'X-R2T2-Session-Token': token}});
                return response.ok && (await response.json()).status === 'cancelled';
            }""", arg={"sid": cancelled["session_id"],
                        "token": cancelled["browser_token"]}, timeout=10000)
            await page.wait_for_function("() => window.__r2t2MockLive.r2t2 === null", timeout=10000)
            await page.evaluate("""async () => {
                const node = window.__r2t2MockLive;
                await node.widgets.find(w => w.name === 'Start microphone').callback();
                node.r2t2.ws.close();
            }""")
            disconnected = await page.evaluate("""() => window.__deniedSession""")
            await page.wait_for_function("""() => {
                const node = window.__r2t2MockLive;
                return node.r2t2 === null &&
                    node.widgets.find(w => w.name === 'R2T2 status').value.startsWith('interrupted');
            }""", timeout=10000)
            await page.wait_for_function("""async ({sid, token}) => {
                const response = await fetch(`/r2t2/v1/live/${sid}/result`,
                    {headers: {'X-R2T2-Session-Token': token}});
                return response.ok && (await response.json()).status === 'cancelled';
            }""", arg={"sid": disconnected["session_id"],
                        "token": disconnected["browser_token"]}, timeout=10000)
            await page.evaluate("""async () => {
                const node = window.__r2t2MockLive;
                await node.widgets.find(w => w.name === 'Start microphone').callback();
            }""")
            closed_page = await page.evaluate("""() => window.__deniedSession""")
            await page.close()
            async with aiohttp.ClientSession() as http:
                for _ in range(50):
                    response = await http.get(
                        BASE + f"/r2t2/v1/live/{closed_page['session_id']}/result",
                        headers={"Origin": BASE, "X-R2T2-Session-Token": closed_page["browser_token"]})
                    if response.status == 200 and (await response.json())["status"] == "cancelled":
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("Closing browser page left an active worker session")
            assert not page_errors, page_errors
            print(json.dumps({"browser_mock_live": "passed", "samples": len(audio),
                              "text": final["text"], "revision": final["revision"],
                              "cleanup": True, "permission_denied": "cancelled",
                              "overload": "cancelled", "cancel_button": "cancelled",
                              "disconnect": "cancelled", "page_close": "cancelled"}, ensure_ascii=True))
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
