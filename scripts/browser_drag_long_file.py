"""Run a 94-second reviewable stream from a workflow dragged into ComfyUI."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

import aiohttp
from playwright.async_api import async_playwright


ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8197"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
WORKFLOW = ROOT / "workflows" / "confucius4_q8_file.json"
SOURCE = ROOT / ".runtime/evaluation/dense-speech/fleurs-joined-90s.wav"
INPUT = ROOT / ".runtime/comfy-input/r2t2_dense_94s.wav"
OUTPUT = ROOT / ".runtime/comfy-output"


async def main() -> None:
    if not WORKFLOW.is_file() or not SOURCE.is_file():
        raise FileNotFoundError("Saved UI workflow and prepared 94-second public audio are required")
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, INPUT)
    prefix = f"r2t2_ui_long94_{uuid.uuid4().hex[:8]}"
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=CHROME, headless=True)
        try:
            page = await browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)
            await page.wait_for_function("""async () => {
                try {
                    const {app} = await import('/scripts/app.js');
                    return !!window.LiteGraph?.registered_node_types?.R2T2Transcribe &&
                        !!app.canvas && app.graph?._nodes.some(n => n.type === 'SaveImage');
                } catch { return false; }
            }""", timeout=60000)
            await page.wait_for_timeout(2000)
            payload = WORKFLOW.read_bytes()
            await page.evaluate("""({bytes, name}) => {
                const file = new File([Uint8Array.from(bytes)], name, {type: 'application/json'});
                const transfer = new DataTransfer();
                transfer.items.add(file);
                const target = document.querySelector('canvas') || document.body;
                for (const type of ['dragenter', 'dragover', 'drop'])
                    target.dispatchEvent(new DragEvent(type,
                        {bubbles: true, cancelable: true, dataTransfer: transfer}));
            }""", {"bytes": list(payload), "name": WORKFLOW.name})
            await page.wait_for_function("""async () => {
                try {
                    const {app} = await import('/scripts/app.js');
                    const types = app.graph?._nodes.map(n => n.type).sort().join(',');
                    return types === 'LoadAudio,R2T2GGUFLoader,R2T2SaveTranscript,R2T2Transcribe' &&
                        Object.values(app.graph.links || {}).length === 3;
                } catch { return false; }
            }""", timeout=10000)
            async def select_long_audio():
                return await page.evaluate("""async ({filename, prefix}) => {
                const {app} = await import('/scripts/app.js');
                const audio = app.graph._nodes.find(n => n.type === 'LoadAudio');
                const transcribe = app.graph._nodes.find(n => n.type === 'R2T2Transcribe');
                const save = app.graph._nodes.find(n => n.type === 'R2T2SaveTranscript');
                if (!audio || !transcribe || !save)
                    throw new Error(`Dragged graph changed: ${app.graph._nodes.map(n => n.type)}`);
                audio.widgets.find(w => w.name === 'audio').value = filename;
                transcribe.widgets.find(w => w.name === 'mode').value = 'stream';
                transcribe.widgets.find(w => w.name === 'language').value = 'Chinese';
                transcribe.widgets.find(w => w.name === 'stream_chunk_ms').value = 320;
                save.widgets.find(w => w.name === 'prefix').value = prefix;
                app.canvas?.setDirty(true, true);
                return {audio: audio.widgets.find(w => w.name === 'audio').value,
                    mode: transcribe.widgets.find(w => w.name === 'mode').value,
                    chunk_ms: transcribe.widgets.find(w => w.name === 'stream_chunk_ms').value,
                    prefix: save.widgets.find(w => w.name === 'prefix').value};
                }""", {"filename": INPUT.name, "prefix": prefix})
            for attempt in range(30):
                try:
                    selected = await select_long_audio()
                    break
                except Exception:
                    if attempt == 29:
                        raise
                    await page.wait_for_timeout(500)
            assert selected == {"audio": INPUT.name, "mode": "stream", "chunk_ms": 320,
                                "prefix": prefix}, selected
            await page.screenshot(path=str(ROOT / ".runtime/comfy-drag-long94-loaded.png"), full_page=True)
            await page.locator('div[data-pc-section="mask"]').first.wait_for(state="hidden", timeout=30000)
            async with page.expect_response(
                lambda response: response.request.method == "POST" and response.url.rstrip("/").endswith("/prompt"),
                timeout=20000,
            ) as queued_response:
                await page.get_by_role("button", name="运行", exact=True).click()
            queued = await queued_response.value
            assert queued.status == 200, await queued.text()
            prompt_id = (await queued.json())["prompt_id"]
            print(json.dumps({"ui_queued_prompt": prompt_id, "audio_seconds": 94.17},
                             ensure_ascii=True), flush=True)
            async with aiohttp.ClientSession() as http:
                for _ in range(480):
                    history = await (await http.get(f"{BASE}/history/{prompt_id}")).json()
                    if prompt_id in history:
                        status = history[prompt_id]["status"]
                        assert status["status_str"] == "success", status
                        break
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError("Dragged 94-second workflow did not finish within 480 seconds")
            outputs = list(OUTPUT.glob(prefix + "_*.json"))
            assert outputs, "Save Transcript did not export the reviewable result"
            target = max(outputs, key=lambda path: path.stat().st_mtime)
            result = json.loads(target.read_text(encoding="utf-8"))
            assert result["status"] == "requires_review", result["status"]
            assert result["quality_status"] == "forced_boundaries_unverified"
            assert result["forced_boundaries"] == 1
            assert result["audio_samples_16k"] == 1_506_720
            assert result["segments"][0]["start_sample"] == 0
            assert result["segments"][-1]["end_sample"] == result["audio_samples_16k"]
            assert all(a["end_sample"] == b["start_sample"] for a, b in
                       zip(result["segments"], result["segments"][1:]))
            assert not result["truncated"] and result["text"]
            assert not errors, errors
            await page.screenshot(path=str(ROOT / ".runtime/comfy-drag-long94-complete.png"), full_page=True)
            print(json.dumps({"prompt_id": prompt_id, "history": status["status_str"],
                              "saved": str(target), "status": result["status"],
                              "forced_boundaries": result["forced_boundaries"],
                              "audio_seconds": result["audio_seconds"],
                              "elapsed_ms": result["elapsed_ms"]}, ensure_ascii=True), flush=True)
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
