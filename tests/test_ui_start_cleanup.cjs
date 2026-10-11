const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

async function main() {
    let extension;
    globalThis.__testApp = {
        graph: {links: {1: {origin_id: 42}}, getNodeById() { return {
            type: "R2T2GGUFLoader", widgets: [
                {name: "n_ctx", value: 8192}, {name: "n_batch", value: 1024},
                {name: "n_threads", value: 8}, {name: "gpu_layers", value: -1},
            ],
        }; }},
        registerExtension(value) { extension = value; },
    };
    globalThis.location = {protocol: "http:", host: "127.0.0.1"};
    globalThis.document = {createElement() {
        return {style: {}, textContent: "", append() {}};
    }};
    let active = false;
    let startCalls = 0;
    let cancelCalls = 0;
    let busyResponses = 0;
    let deferredStart = null;
    let socketConstructions = 0;
    globalThis.fetch = async (url, options) => {
        if (url === "/r2t2/v1/live/start") {
            startCalls++;
            if (deferredStart) await deferredStart;
            if (active) {
                busyResponses++;
                return {ok: false, json: async () => ({message: "BUSY"})};
            }
            active = true;
            return {ok: true, json: async () => ({session_id: "sid", browser_token: "secret"})};
        }
        if (url === "/r2t2/v1/live/sid/cancel") {
            assert.equal(options.method, "POST");
            assert.equal(options.headers["X-R2T2-Session-Token"], "secret");
            assert.ok(options.signal);
            cancelCalls++;
            active = false;
            return {ok: true, json: async () => ({status: "cancelled"})};
        }
        throw new Error("Unexpected URL: " + url);
    };
    globalThis.WebSocket = class {
        static OPEN = 1;
        constructor() { socketConstructions++; throw new Error("WebSocket blocked by CSP"); }
    };
    const source = fs.readFileSync(path.join(__dirname, "../web/r2t2.js"), "utf8")
        .replace('import { app } from "../../scripts/app.js";',
            "const app = globalThis.__testApp;");
    await import("data:text/javascript," + encodeURIComponent(source));
    class FakeNode {
        constructor() { this.widgets = [{name: "session_id", value: "previous-finalized"},
            {name: "revision", value: 7}]; this.inputs = [{name: "model", link: 1}]; this.size = [200, 100]; }
        addWidget(type, name, value, callback) {
            const item = {type, name, value, callback};
            this.widgets.push(item);
            return item;
        }
        addDOMWidget() {}
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(FakeNode, {name: "R2T2LiveSession"});
    const node = new FakeNode();
    node.onNodeCreated();
    node.widgets.find(item => item.name === "Stable text").value = "previous text";
    node.r2t2Panel.stable.textContent = "previous text";
    node.r2t2Panel.preview.textContent = "previous preview";
    const start = node.widgets.find((item) => item.name === "Start microphone").callback;
    const status = () => node.widgets.find((item) => item.name === "R2T2 status").value;
    await start();
    assert.equal(active, false);
    assert.equal(cancelCalls, 1);
    assert.match(status(), /WebSocket blocked by CSP/);
    assert.equal(node.widgets.find(item => item.name === "session_id").value, "");
    assert.equal(node.widgets.find(item => item.name === "revision").value, 0);
    assert.equal(node.r2t2Panel.stable.textContent, "");
    assert.equal(node.r2t2Panel.preview.textContent, "");
    await start();
    assert.equal(active, false);
    assert.equal(startCalls, 2);
    assert.equal(cancelCalls, 2);
    assert.equal(busyResponses, 0);

    let releaseStart;
    deferredStart = new Promise(resolve => { releaseStart = resolve; });
    const firstLoading = start();
    await start();
    assert.equal(startCalls, 3, "Double-click while the model loads must issue one request");
    assert.ok(node.r2t2, "Ignored Start must retain the first startup's state");
    releaseStart();
    await firstLoading;
    assert.equal(cancelCalls, 3);
    assert.equal(busyResponses, 0);

    deferredStart = new Promise(resolve => { releaseStart = resolve; });
    const removedNode = new FakeNode();
    removedNode.onNodeCreated();
    const removedStart = removedNode.widgets.find(item => item.name === "Start microphone").callback();
    const socketsBeforeRemoval = socketConstructions;
    removedNode.onRemoved();
    releaseStart();
    await removedStart;
    assert.equal(socketConstructions, socketsBeforeRemoval,
        "A node removed during model loading must cancel the returned session before opening a socket");
    assert.equal(active, false);
    assert.equal(cancelCalls, 4);
    assert.equal(removedNode.r2t2, null);
    console.log("Live UI start failure cleanup checks passed");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
