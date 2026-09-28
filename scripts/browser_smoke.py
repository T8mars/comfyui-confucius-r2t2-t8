"""Headless Chrome test of the actual ComfyUI frontend and fake microphone."""

from __future__ import annotations

import asyncio
import argparse
import json
from pathlib import Path

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8197"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


async def main(frontend_only: bool = False) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=CHROME, headless=True,
            args=[] if frontend_only else ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                  "--autoplay-policy=no-user-gesture-required"])
        context = await browser.new_context(permissions=[] if frontend_only else ["microphone"])
        page = await context.new_page()
        errors = []
        network = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda response: network.append(f"{response.status} {response.url}") if "/r2t2/" in response.url or "worklet" in response.url else None)
        page.on("requestfailed", lambda request: network.append(f"failed {request.url} {request.failure}"))
        def observe_ws(ws):
            network.append(f"ws {ws.url}")
            if "/r2t2/" in ws.url:
                ws.on("framesent", lambda payload: network.append(f"sent {str(payload)[:180]}"))
                ws.on("framereceived", lambda payload: network.append(f"recv {str(payload)[:180]}"))
                ws.on("close", lambda: network.append("ws closed"))
                ws.on("socketerror", lambda error: network.append(f"ws error {error}"))
        page.on("websocket", observe_ws)
        print("browser: navigate", flush=True)
        await page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)
        print("browser: loaded", flush=True)
        await page.wait_for_function("""async () => {
            const {app} = await import('/scripts/app.js');
            return !!app.graph && !!app.graph._nodes_by_id;
        }""", timeout=60000)
        probe = await page.evaluate("""async () => {
            const {app} = await import('/scripts/app.js');
            return {graph: !!app.graph, litegraph: !!window.LiteGraph,
                    nodes: Object.keys(app.graph?._nodes_by_id || {}).length,
                    liveType: !!window.LiteGraph?.registered_node_types?.R2T2LiveSession};
        }""")
        module_probe = None if frontend_only else await page.evaluate("""async () => {
            const ctx = new AudioContext();
            const file = await Promise.race([
                ctx.audioWorklet.addModule('/extensions/Confucius4-R2T2/r2t2-worklet.js').then(() => 'loaded').catch(e => 'error: ' + e.message),
                new Promise(resolve => setTimeout(() => resolve('timeout'), 5000)),
            ]);
            const blob = URL.createObjectURL(new Blob(["registerProcessor('r2t2-probe', class extends AudioWorkletProcessor { process() { return true; } });"], {type:'application/javascript'}));
            const minimal = await Promise.race([
                ctx.audioWorklet.addModule(blob).then(() => 'loaded').catch(e => 'error: ' + e.message),
                new Promise(resolve => setTimeout(() => resolve('timeout'), 5000)),
            ]);
            URL.revokeObjectURL(blob);
            await ctx.close();
            return {file, minimal};
        }""")
        print("browser: worklet module=" + repr(module_probe), flush=True)
        widgets = await page.evaluate("""async () => {
            const {app} = await import('/scripts/app.js');
            const loader = window.LiteGraph.createNode('R2T2GGUFLoader');
            const live = window.LiteGraph.createNode('R2T2LiveSession');
            app.graph.add(loader);
            app.graph.add(live);
            loader.connect(0, live, 0);
            window.__r2t2LiveTest = live;
            return live.widgets.map(w => w.name);
        }""")
        print("browser: widgets=" + repr(widgets), flush=True)
        assert "Start microphone" in widgets and "Stop and finalize" in widgets, widgets
        panel = await page.evaluate("""() => {
            const node = window.__r2t2LiveTest;
            return {exists: !!node.r2t2Panel, status: node.r2t2Panel?.status?.textContent,
                    domWidget: node.widgets.some(w => w.name === 'R2T2 captions')};
        }""")
        assert panel == {"exists": True, "status": "idle", "domWidget": True}, panel
        if frontend_only:
            print(json.dumps({"frontend": probe, "widgets": widgets, "panel": panel,
                              "page_errors": errors[:10]}, ensure_ascii=True))
            assert not errors, errors
            await browser.close()
            return
        await page.evaluate("""() => {
            const node = window.__r2t2LiveTest;
            node.widgets.find(w => w.name === 'Start microphone').callback();
            return true;
        }""")
        await page.wait_for_timeout(4000)
        state = await page.evaluate("""() => {
            const node = window.__r2t2LiveTest;
            return {status: node.widgets.find(w => w.name === 'R2T2 status')?.value,
                    sid: node.r2t2?.sid, ws: node.r2t2?.ws?.readyState,
                    media: !!node.r2t2?.media, context: !!node.r2t2?.context,
                    contextState: node.r2t2?.context?.state,
                    worklet: !!node.r2t2?.worklet};
        }""")
        print("browser: after start=" + repr(state) + " network=" + repr(network[-10:]), flush=True)
        await page.wait_for_function("""() => {
            const value = window.__r2t2LiveTest?.widgets?.find(w => w.name === 'R2T2 status')?.value || '';
            return value === 'recording' || value.startsWith('error:');
        }""", timeout=20000)
        current = await page.evaluate("window.__r2t2LiveTest.widgets.find(w => w.name === 'R2T2 status').value")
        if current != "recording":
            print(json.dumps({"browser_mic": "not_started", "status": current,
                              "network": network[-12:], "page_errors": errors[:10]}, ensure_ascii=True), flush=True)
            raise RuntimeError(current)
        print("browser: recording", flush=True)
        await page.wait_for_timeout(1500)
        await page.evaluate("""async () => {
            const node = window.__r2t2LiveTest;
            await node.widgets.find(w => w.name === 'Stop and finalize').callback();
        }""")
        await page.wait_for_function("""() => {
            const node = window.__r2t2LiveTest;
            return node?.widgets?.find(w => w.name === 'session_id')?.value &&
                   node?.widgets?.find(w => w.name === 'revision')?.value > 0;
        }""", timeout=30000)
        live = await page.evaluate("""() => {
            const node = window.__r2t2LiveTest;
            const value = name => node.widgets.find(w => w.name === name)?.value;
            return {status: value('R2T2 status'), stable: value('Stable text'),
                    session_id: value('session_id'), revision: value('revision')};
        }""")
        screenshot = Path(__file__).resolve().parents[1] / ".runtime" / "browser-smoke.png"
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(screenshot))
        print(json.dumps({"frontend": probe, "widgets": widgets, "live": live,
                          "url": page.url, "title": await page.title(),
                          "page_errors": errors[:10]}, ensure_ascii=True))
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-only", action="store_true")
    asyncio.run(main(parser.parse_args().frontend_only))
