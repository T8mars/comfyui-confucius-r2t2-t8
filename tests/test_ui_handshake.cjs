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
            "const app = globalThis.__testApp;")
        .replace('new URL("./r2t2-worklet.js", import.meta.url)', '"http://localhost/worklet.js"');
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
    const releaseOldMedia = releaseMedia;
    ws.emitError("STREAM_ERROR");
    ws.close();
    // A later Start can begin while the old getUserMedia promise is unresolved.
    // That old catch/onclose must not clear or repaint the replacement state.
    const replacementStart = start(node);
    await new Promise(resolve => setImmediate(resolve));
    const replacementState = node.r2t2;
    const replacementSocket = sockets.at(-1);
    const releaseReplacementMedia = releaseMedia;
    assert.notEqual(replacementSocket, ws);
    ws.onmessage?.({data: JSON.stringify({type: "final", revision: 99, text: "stale final"})});
    assert.equal(node.widgets.find(item => item.name === "session_id").value, "");
    releaseOldMedia({getTracks() { return [{stop() {}}]; }});
    await started;
    assert.equal(node.r2t2, replacementState);
    assert.doesNotMatch(status(node), /gateway rejected STREAM_ERROR/);
    replacementSocket.emitError("STREAM_ERROR");
    replacementSocket.close();
    releaseReplacementMedia({getTracks() { return [{stop() {}}]; }});
    await replacementStart;
    assert.match(status(node), /STREAM_ERROR/);
    assert.match(status(node), /gateway rejected STREAM_ERROR/);
    assert.doesNotMatch(status(node), /Live connection closed during microphone setup/);

    Object.defineProperty(globalThis, "navigator", {configurable: true, value: {
        mediaDevices: {async getUserMedia() {return {getTracks() {return [{stop() {}}];}};}}}});
    globalThis.AudioContext = class {
        constructor() {this.audioWorklet = {async addModule() {}}; this.destination = {};}
        createMediaStreamSource() {return {connect() {}, disconnect() {}};}
        createGain() {return {gain: {}, connect() {}, disconnect() {}};}
        async close() {}
    };
    globalThis.AudioWorkletNode = class {
        constructor() {this.port = {};}
        connect(gain) {return gain;}
        disconnect() {}
    };
    const completedNode = makeNode();
    await start(completedNode);
    assert.equal(completedNode.widgets.find(item => item.name === "session_id").value, "");
    assert.equal(completedNode.widgets.find(item => item.name === "revision").value, 0);
    const finalSocket = sockets.at(-1);
    finalSocket.onmessage({data: JSON.stringify({type: "final", revision: 11,
        text: "current finalized text", quality_status: "standard"})});
    assert.equal(completedNode.widgets.find(item => item.name === "session_id").value, "sid");
    assert.equal(completedNode.widgets.find(item => item.name === "revision").value, 11);
    assert.equal(completedNode.r2t2Panel.stable.textContent, "current finalized text");
    finalSocket.close();
    console.log("Live UI gateway error propagation checks passed");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
