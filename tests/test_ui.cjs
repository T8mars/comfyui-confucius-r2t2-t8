const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

async function main() {
    let extension;
    globalThis.__testApp = {registerExtension(value) { extension = value; }};
    globalThis.location = {protocol: "http:", host: "127.0.0.1"};
    globalThis.document = {createElement() {
        return {style: {}, textContent: "", children: [], append(...items) { this.children.push(...items); }};
    }};
    globalThis.WebSocket = {OPEN: 1};
    const source = fs.readFileSync(path.join(__dirname, "../web/r2t2.js"), "utf8")
        .replace('import { app } from "../../scripts/app.js";', "const app = globalThis.__testApp;");
    await import("data:text/javascript," + encodeURIComponent(source));
    assert.equal(extension.name, "confucius4.r2t2.live");

    class FakeNode {
        constructor() { this.widgets = [["language", "Auto"], ["context", ""], ["session_id", ""],
            ["revision", 0], ["stream_chunk_ms", 320], ["min_segment_seconds", 8]]
            .map(([name, value]) => ({name, value})); this.size = [200, 100]; }
        addWidget(type, name, value, callback, options) {
            const widget = {type, name, value, callback, options};
            this.widgets.push(widget);
            return widget;
        }
        addDOMWidget(name, type, element, options) {
            this.panel = element;
            const widget = {name, type, options};
            this.widgets.push(widget);
            return widget;
        }
        configure(info) {
            // The current frontend removes a genuine model forceInput value
            // only when the stored count equals its complete input mask.
            const values = info.widgets_values.length === 7 ? info.widgets_values.slice(1) : info.widgets_values;
            let index = 0;
            for (const widget of this.widgets) {
                if (widget.serialize === false) continue;
                if (index < values.length) widget.value = values[index++];
            }
        }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(FakeNode, {name: "R2T2LiveSession"});
    const node = new FakeNode();
    node.onNodeCreated();
    assert.deepEqual(node.panel.children.map((item) => item.textContent), ["idle", "", ""]);
    assert.equal(node.widgets.filter((item) => item.name === "R2T2 captions").length, 1);
    const values = node => node.widgets.filter(widget => widget.serialize !== false).map(widget => widget.value ?? null);
    const expected = ["English", "context", "sid-finalized", 12, 480, 4];
    for (const saved of [expected, [...expected, null, null, null, "idle", "old stable", "old preview", ""],
        [null, ...expected]]) {
        node.configure({widgets_values: saved});
        assert.deepEqual(values(node), expected);
        assert.equal(node.widgets.filter(widget => widget.serialize === false).length, 7);
        assert.equal(node.panel.children[1].textContent, "");
    }
    let restored = node;
    for (let cycle = 0; cycle < 5; cycle++) {
        const saved = JSON.parse(JSON.stringify({widgets_values: values(restored)}));
        restored = new FakeNode();
        restored.onNodeCreated();
        restored.configure(saved);
        assert.deepEqual(values(restored), expected);
    }
    const liveWorkflow = JSON.parse(fs.readFileSync(path.join(__dirname, "../workflows/confucius4_q8_live.json"), "utf8"));
    const publicLive = liveWorkflow.nodes.find(node => node.type === "R2T2LiveSession");
    assert.equal(publicLive.widgets_values.length, 6);
    assert.deepEqual(Object.keys(publicLive.widgets_values_named),
        ["language", "context", "session_id", "revision", "stream_chunk_ms", "min_segment_seconds"]);
    await node.widgets.find((item) => item.name === "Start microphone").callback();
    assert.match(node.panel.children[0].textContent, /Connect a Confucius4 Q8 Loader/);

    const sent = [];
    node.r2t2 = {stopping: false, ws: {readyState: 1, send(value) { sent.push(JSON.parse(value)); }, close() {}},
        worklet: {port: {postMessage() {}}, disconnect() {}}, source: null, silent: null,
        media: null, context: null};
    node.r2t2Owner = node.r2t2;
    const originalTimeout = globalThis.setTimeout;
    globalThis.setTimeout = (callback) => { queueMicrotask(callback); return 1; };
    try {
        const stop = node.widgets.find((item) => item.name === "Stop and finalize").callback;
        const first = stop();
        await stop();
        await first;
    } finally {
        globalThis.setTimeout = originalTimeout;
    }
    assert.match(node.panel.children[0].textContent, /AudioWorklet flush timed out/);
    assert.equal(node.r2t2.stopping, true);
    assert.deepEqual(sent, [{type: "cancel"}]);
    node.onRemoved();
    assert.equal(node.r2t2, null);
    console.log("Live UI caption panel and flush timeout checks passed");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
