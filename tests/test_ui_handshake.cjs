const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

async function main() {
    let extension;
    const loader = {type: "R2T2GGUFLoader", widgets: [
        {name: "n_ctx", value: 8192}, {name: "n_batch", value: 1024},
        {name: "n_threads", value: 8}, {name: "gpu_layers", value: -1},
    ]};
    globalThis.__testApp = {
        graph: {links: {1: {origin_id: 42}}, getNodeById() { return loader; }},
        registerExtension(value) { extension = value; },
    };
    globalThis.location = {protocol: "http:", host: "127.0.0.1"};
    globalThis.document = {createElement() {
        return {style: {}, textContent: "", append() {}};
    }};
    globalThis.fetch = async () => ({ok: true, json: async () => (
        {session_id: "sid", browser_token: "secret"})});

    const sockets = [];
    let scenario = null;
    class FakeWebSocket {
        static OPEN = 1;
        constructor() {
            this.readyState = 0;
            this.bufferedAmount = 0;
            sockets.push(this);
            queueMicrotask(() => { this.readyState = 1; this.onopen?.(); });
        }
        send(value) {
            if (!JSON.parse(value).browser_token) return;
            if (scenario.type === "before_ready") {
                queueMicrotask(() => {
                    this.emitError(scenario.code);
                    this.close();
                });
            } else if (scenario.type === "after_ready") {
                queueMicrotask(() => this.onmessage?.({data: JSON.stringify({type: "ready"})}));
            }
        }
        emitError(code) {
            this.onmessage?.({data: JSON.stringify({type: "error", code,
                message: `gateway rejected ${code}`})});
        }
        close() {
            if (this.readyState === 3) return;
            this.readyState = 3;
            this.onclose?.();
        }
    }
    globalThis.WebSocket = FakeWebSocket;
    const source = fs.readFileSync(path.join(__dirname, "../web/r2t2.js"), "utf8")
        .replace('import { app } from "../../scripts/app.js";',
            "const app = globalThis.__testApp;");
    await import("data:text/javascript," + encodeURIComponent(source));
    class FakeNode {
        constructor() { this.widgets = []; this.inputs = [{name: "model", link: 1}]; this.size = [200, 100]; }
        addWidget(type, name, value, callback) {
            const item = {type, name, value, callback};
            this.widgets.push(item);
            return item;
        }
        addDOMWidget() {}
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(FakeNode, {name: "R2T2LiveSession"});
    const makeNode = () => { const node = new FakeNode(); node.onNodeCreated(); return node; };
    const status = (node) => node.widgets.find((item) => item.name === "R2T2 status").value;
    const start = (node) => node.widgets.find((item) => item.name === "Start microphone").callback();

    for (const code of ["ALREADY_CONNECTED", "FORBIDDEN", "STREAM_ERROR"]) {
        scenario = {type: "before_ready", code};
        const node = makeNode();
        await start(node);
        assert.match(status(node), new RegExp(code));
        assert.match(status(node), new RegExp(`gateway rejected ${code}`));
        assert.doesNotMatch(status(node), /WebSocket closed before microphone was ready/);
    }

    let mediaRequested;
    let releaseMedia;
    const requested = new Promise((resolve) => { mediaRequested = resolve; });
    Object.defineProperty(globalThis, "navigator", {configurable: true, value: {mediaDevices: {getUserMedia() {
        mediaRequested();
        return new Promise((resolve) => { releaseMedia = resolve; });
    }}}});
    scenario = {type: "after_ready"};
    const node = makeNode();
    const started = start(node);
    await Promise.race([requested, new Promise((_, reject) =>
        setTimeout(() => reject(new Error("getUserMedia was not reached")), 1000))]);
    const ws = sockets.at(-1);
    ws.emitError("STREAM_ERROR");
    ws.close();
    releaseMedia({getTracks() { return [{stop() {}}]; }});
    await started;
    assert.match(status(node), /STREAM_ERROR/);
    assert.match(status(node), /gateway rejected STREAM_ERROR/);
    assert.doesNotMatch(status(node), /Live connection closed during microphone setup/);
    console.log("Live UI gateway error propagation checks passed");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
