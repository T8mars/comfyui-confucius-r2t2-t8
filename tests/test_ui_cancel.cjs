const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const deferred = () => {
    let resolve;
    const promise = new Promise(value => {resolve = value;});
    return {promise, resolve};
};

async function main() {
    let extension;
    let scenario;
    const sockets = [];
    const cancellations = [];
    const tracks = [];
    const contexts = [];
    const loader = {type: "R2T2GGUFLoader", widgets: [
        {name: "n_ctx", value: 8192}, {name: "n_batch", value: 1024},
        {name: "n_threads", value: 8}, {name: "gpu_layers", value: -1},
    ]};
    globalThis.__testApp = {graph: {links: {1: {origin_id: 1}}, getNodeById() {return loader;}},
        registerExtension(value) {extension = value;}};
    globalThis.location = {protocol: "http:", host: "localhost"};
    globalThis.document = {createElement() {return {style: {}, textContent: "", append() {}};}};
    globalThis.fetch = async (url) => {
        if (url === "/r2t2/v1/live/start") {
            if (scenario.stage === "fetch") {scenario.reached.resolve(); await scenario.release.promise;}
            return {ok: true, json: async () => ({session_id: "sid", browser_token: "secret"})};
        }
        assert.equal(url, "/r2t2/v1/live/sid/cancel");
        cancellations.push(url);
        return {ok: true, json: async () => ({status: "cancelled"})};
    };
    class Socket {
        static OPEN = 1;
        constructor() {
            this.readyState = 0;
            this.sent = [];
            this.scenario = scenario;
            sockets.push(this);
            queueMicrotask(() => {
                if (this.readyState === 3) return;
                this.readyState = 1;
                this.onopen?.();
            });
        }
        send(raw) {
            const message = JSON.parse(raw);
            this.sent.push(message);
            if (message.browser_token) {
                if (this.scenario.stage === "ready") {this.scenario.reached.resolve(); return;}
                queueMicrotask(() => this.onmessage?.({data: JSON.stringify({type: "ready"})}));
            }
            if (message.type === "cancel") queueMicrotask(() => this.close());
        }
        close() {if (this.readyState === 3) return; this.readyState = 3; this.onclose?.();}
    }
    globalThis.WebSocket = Socket;
    Object.defineProperty(globalThis, "navigator", {configurable: true, value: {mediaDevices: {
        async getUserMedia() {
            const track = {stopped: false, stop() {this.stopped = true;}};
            tracks.push(track);
            if (scenario.stage === "media") {scenario.reached.resolve(); await scenario.release.promise;}
            return {getTracks() {return [track];}};
        },
    }}});
    globalThis.AudioContext = class {
        constructor() {
            contexts.push(this);
            this.closed = false;
            this.audioWorklet = {async addModule() {
                if (scenario.stage === "worklet") {scenario.reached.resolve(); await scenario.release.promise;}
            }};
        }
        async close() {this.closed = true;}
        createMediaStreamSource() {throw new Error("Cancelled startup must not connect audio nodes");}
    };
    const source = fs.readFileSync(path.join(__dirname, "../web/r2t2.js"), "utf8")
        .replace('import { app } from "../../scripts/app.js";', "const app = globalThis.__testApp;")
        .replace('new URL("./r2t2-worklet.js", import.meta.url)', '"http://localhost/worklet.js"');
    await import("data:text/javascript," + encodeURIComponent(source));
    class Node {
        constructor() {
            this.widgets = [{name: "session_id", value: "previous"}, {name: "revision", value: 17}];
            this.inputs = [{name: "model", link: 1}]; this.size = [200, 100];
        }
        addWidget(type, name, value, callback) {const item = {type, name, value, callback}; this.widgets.push(item); return item;}
        addDOMWidget() {}
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Node, {name: "R2T2LiveSession"});
    const action = (node, name) => node.widgets.find(widget => widget.name === name).callback();
    const saved = (node, name) => node.widgets.find(widget => widget.name === name).value;
    const originalTimeout = globalThis.setTimeout;
    const originalClearTimeout = globalThis.clearTimeout;
    // These cases explicitly release their deferred operations; no deadline
    // callback is needed, and leaving worklet deadlines alive slows this test.
    globalThis.setTimeout = () => 1;
    globalThis.clearTimeout = () => {};
    try {
        for (const stage of ["fetch", "ready", "media", "worklet"]) {
            scenario = {stage, reached: deferred(), release: deferred()};
            const node = new Node(); node.onNodeCreated();
            const beforeCancels = cancellations.length;
            const beforeSockets = sockets.length;
            const started = action(node, "Start microphone");
            await scenario.reached.promise;
            await action(node, "Stop and finalize");
            assert.equal(cancellations.length, beforeCancels, `${stage}: Stop must not cancel setup`);
            assert.equal(node.r2t2.stopping, false);
            await action(node, "Cancel");
            assert.equal(saved(node, "session_id"), "");
            assert.equal(saved(node, "revision"), 0);
            scenario.release.resolve();
            await started;
            assert.equal(node.r2t2, null, `${stage}: cancelled state must be released`);
            assert.equal(saved(node, "R2T2 status"), "cancelled");
            assert.equal(cancellations.length, beforeCancels + 1);
            if (stage === "fetch") assert.equal(sockets.length, beforeSockets);
            assert.ok(tracks.every(track => track.stopped), `${stage}: no microphone track may leak`);
            assert.ok(contexts.every(context => context.closed), `${stage}: no audio context may leak`);
            assert.ok(sockets.every(socket => socket.sent.every(message => message.type !== "finish")));
        }
        for (const closed of [false, true]) {
            scenario = {stage: "media", reached: deferred(), release: deferred()};
            const node = new Node(); node.onNodeCreated();
            const oldStart = action(node, "Start microphone");
            await scenario.reached.promise;
            await action(node, "Cancel");
            await new Promise(resolve => setImmediate(resolve));
            assert.equal(node.r2t2, null);
            const replacement = {ws: {readyState: closed ? 3 : 1}, marker: "replacement"};
            // Match a newer Start's persistent owner, including after its
            // successful final event closes the socket and clears r2t2.
            node.r2t2 = closed ? null : replacement;
            node.r2t2Owner = replacement;
            const expectedStatus = closed ? "complete; run workflow to use the snapshot" : "replacement recording";
            const expectedSid = closed ? "new-finalized-sid" : "";
            node.widgets.find(widget => widget.name === "R2T2 status").value = expectedStatus;
            node.widgets.find(widget => widget.name === "session_id").value = expectedSid;
            scenario.release.resolve();
            await oldStart;
            assert.equal(node.r2t2, closed ? null : replacement);
            assert.equal(node.r2t2Owner, replacement);
            assert.equal(saved(node, "R2T2 status"), expectedStatus);
            assert.equal(saved(node, "session_id"), expectedSid);
            assert.ok(tracks.every(track => track.stopped));
        }

        for (const replace of ["same", "active", "closed"]) {
            const finishingNode = new Node(); finishingNode.onNodeCreated();
            const flushRequested = deferred();
            const messages = [];
            const socket = {readyState: 1, send(raw) {
                messages.push(JSON.parse(raw)); this.readyState = 3;
            }};
            const finishingState = {ws: socket, seq: 2, sent: 320, stopping: false, finalizing: false,
                cancelled: false, disposed: false, worklet: {
                    port: {postMessage(message) {assert.equal(message.type, "flush"); flushRequested.resolve();}},
                    disconnect() {},
                }};
            finishingNode.r2t2 = finishingState;
            finishingNode.r2t2Owner = finishingState;
            const stopped = action(finishingNode, "Stop and finalize");
            await flushRequested.promise;
            const releaseFlush = finishingState.flushed;
            await action(finishingNode, "Cancel");
            const nextState = {marker: "next recording"};
            const expectedStatus = replace === "closed" ? "complete; run workflow to use the snapshot" : "next recording";
            if (replace !== "same") {
                finishingNode.r2t2 = replace === "closed" ? null : nextState;
                finishingNode.r2t2Owner = nextState;
                finishingNode.widgets.find(widget => widget.name === "R2T2 status").value = expectedStatus;
            }
            releaseFlush();
            await stopped;
            assert.deepEqual(messages, [{type: "cancel"}], "Cancelled pending flush must never send finish");
            assert.equal(finishingState.flushed, null);
            assert.equal(finishingState.finalizing, false);
            assert.equal(finishingNode.r2t2, replace === "same" ? finishingState : replace === "closed" ? null : nextState);
            assert.equal(saved(finishingNode, "R2T2 status"), replace === "same" ? "cancelled" : expectedStatus);
        }
    } finally {
        globalThis.setTimeout = originalTimeout;
        globalThis.clearTimeout = originalClearTimeout;
    }
    console.log("Live Cancel during model, handshake, microphone and worklet setup checks passed");
}

main().catch(error => {console.error(error); process.exitCode = 1;});
