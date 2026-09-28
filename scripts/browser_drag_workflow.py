"""Drag a saved UI workflow into ComfyUI and run it through the frontend."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

import aiohttp
from playwright.async_api import async_playwright

from public_sample import ensure_sample


ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8197"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
WORKFLOW = ROOT / "workflows" / "confucius4_q8_file.json"
LIVE_WORKFLOW = ROOT / "workflows" / "confucius4_q8_live.json"
INPUT = ROOT / ".runtime" / "comfy-input" / "r2t2_public_test.wav"
OUTPUT = ROOT / ".runtime" / "comfy-output"


async def wait_default_canvas(page):
    await page.wait_for_function("""async () => {
        try {
            const {app} = await import('/scripts/app.js');
            return !!window.LiteGraph?.registered_node_types?.R2T2Transcribe &&
                !!app.canvas && app.graph?._nodes.some(n => n.type === 'SaveImage');
        } catch { return false; }
    }""", timeout=60000)
    await page.wait_for_timeout(2000)


async def ready(page):
    await page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
    await page.wait_for_timeout(3000)
    await wait_default_canvas(page)


async def main():
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ensure_sample(), INPUT)
    if not WORKFLOW.is_file() or not LIVE_WORKFLOW.is_file():
        raise FileNotFoundError("Both saved UI workflow examples are required")
    prefix = f"r2t2_ui_drag_{uuid.uuid4().hex[:8]}"
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=CHROME, headless=True)
        try:
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await ready(page)
            before = await page.evaluate("""async () => {
                const {app} = await import('/scripts/app.js');
                return app.graph._nodes.map(n => n.type);
            }""")
            payload = WORKFLOW.read_bytes()
            drop = await page.evaluate("""({bytes, name}) => {
                const data = Uint8Array.from(bytes);
                const file = new File([data], name, {type: 'application/json'});
                const transfer = new DataTransfer();
                transfer.items.add(file);
                const target = document.querySelector('canvas') || document.body;
                for (const type of ['dragenter', 'dragover', 'drop']) {
                    target.dispatchEvent(new DragEvent(type, {bubbles: true, cancelable: true,
                        dataTransfer: transfer}));
                }
                return {target: target.tagName, file: file.name, size: file.size};
            }""", {"bytes": list(payload), "name": WORKFLOW.name})
            await page.wait_for_timeout(3000)
            after = await page.evaluate("""async () => {
                const {app} = await import('/scripts/app.js');
                return {nodes: app.graph._nodes.map(n => ({type: n.type,
                    widgets: Object.fromEntries((n.widgets || []).map(w => [w.name, w.value]))})),
                    links: Object.values(app.graph.links || {}).length};
            }""")
            print(json.dumps({"before": before, "drop": drop, "after": after,
                "buttons": await page.locator('button').all_text_contents(), "errors": errors},
                ensure_ascii=True), flush=True)
            assert sorted(n["type"] for n in after["nodes"]) == sorted(
                ["LoadAudio", "R2T2GGUFLoader", "R2T2Transcribe", "R2T2SaveTranscript"]), after
            assert after["links"] == 3, after
            assert next(n for n in after["nodes"] if n["type"] == "LoadAudio")["widgets"]["audio"] == INPUT.name
            selected = await page.evaluate("""async (prefix) => {
                const {app} = await import('/scripts/app.js');
                const save = app.graph._nodes.find(n => n.type === 'R2T2SaveTranscript');
                save.widgets.find(w => w.name === 'prefix').value = prefix;
                return save.widgets.find(w => w.name === 'prefix').value;
            }""", prefix)
            assert selected == prefix
            await page.screenshot(path=str(ROOT / ".runtime" / "comfy-drag-loaded.png"), full_page=True)
            async with page.expect_response(
                lambda response: response.request.method == "POST" and response.url.rstrip("/").endswith("/prompt"),
                timeout=20000,
            ) as queued_response:
                await page.get_by_role("button", name="运行", exact=True).click()
            queued = await queued_response.value
            assert queued.status == 200, await queued.text()
            prompt_id = (await queued.json())["prompt_id"]
            async with aiohttp.ClientSession() as http:
                for _ in range(120):
                    response = await http.get(f"{BASE}/history/{prompt_id}")
                    history = await response.json()
                    if prompt_id in history:
                        status = history[prompt_id]["status"]
                        assert status["status_str"] == "success", status
                        break
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError(f"UI queued workflow {prompt_id} did not finish within 120 seconds")
            outputs = list(OUTPUT.glob(prefix + "_*.json"))
            assert outputs, "No transcript from dragged workflow"
            result = json.loads(max(outputs, key=lambda item: item.stat().st_mtime).read_text(encoding="utf-8"))
            assert result["status"] == "complete" and result["audio_samples_16k"] == 107840
            assert "之前有顾客" in result["text"], result
            await page.screenshot(path=str(ROOT / ".runtime" / "comfy-drag-complete.png"), full_page=True)
            print(json.dumps({"ui_queued_prompt": prompt_id, "history": status["status_str"],
                "output": str(max(outputs, key=lambda item: item.stat().st_mtime)),
                "text": result["text"], "page_errors": errors}, ensure_ascii=True), flush=True)
            await page.reload(wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)
            await wait_default_canvas(page)
            live_payload = LIVE_WORKFLOW.read_bytes()
            await page.evaluate("""({bytes, name}) => {
                const file = new File([Uint8Array.from(bytes)], name, {type: 'application/json'});
                const transfer = new DataTransfer();
                transfer.items.add(file);
                const target = document.querySelector('canvas') || document.body;
                for (const type of ['dragenter', 'dragover', 'drop'])
                    target.dispatchEvent(new DragEvent(type, {bubbles: true,
                        cancelable: true, dataTransfer: transfer}));
            }""", {"bytes": list(live_payload), "name": LIVE_WORKFLOW.name})
            await page.wait_for_timeout(3000)
            live_loaded = await page.evaluate("""async () => {
                const {app} = await import('/scripts/app.js');
                return {nodes: app.graph._nodes.map(n => n.type),
                    links: Object.values(app.graph.links || {}).length,
                    liveWidgets: app.graph._nodes.find(n => n.type === 'R2T2LiveSession')?.widgets
                        .map(w => ({name: w.name, value: w.value}))};
            }""")
            assert sorted(live_loaded["nodes"]) == sorted(
                ["R2T2GGUFLoader", "R2T2LiveSession", "R2T2SaveTranscript"]), live_loaded
            assert live_loaded["links"] == 2, live_loaded
            assert next(w for w in live_loaded["liveWidgets"] if w["name"] == "min_segment_seconds")["value"] == 8
            await page.screenshot(path=str(ROOT / ".runtime" / "comfy-drag-live.png"), full_page=True)
            print(json.dumps({"live_drag": live_loaded, "page_errors": errors}, ensure_ascii=True), flush=True)
            assert not errors, errors
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
