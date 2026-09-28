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
        constructor() { this.widgets = []; this.size = [200, 100]; }
        addWidget(type, name, value, callback) {
            const widget = {type, name, value, callback};
            this.widgets.push(widget);
            return widget;
        }
        addDOMWidget(name, type, element) {
            this.panel = element;
            this.widgets.push({name, type});
        }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(FakeNode, {name: "R2T2LiveSession"});
    const node = new FakeNode();
    node.onNodeCreated();
    assert.deepEqual(node.panel.children.map((item) => item.textContent), ["idle", "", ""]);
    assert.equal(node.widgets.filter((item) => item.name === "R2T2 captions").length, 1);
    await node.widgets.find((item) => item.name === "Start microphone").callback();
    assert.match(node.panel.children[0].textContent, /Connect a Confucius4 Q8 Loader/);

    const sent = [];
    node.r2t2 = {stopping: false, ws: {readyState: 1, send(value) { sent.push(JSON.parse(value)); }, close() {}},
        worklet: {port: {postMessage() {}}, disconnect() {}}, source: null, silent: null,
        media: null, context: null};
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
