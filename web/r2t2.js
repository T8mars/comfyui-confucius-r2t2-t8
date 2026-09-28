import { app } from "../../scripts/app.js";

const PROTOCOL = location.protocol === "https:" ? "wss:" : "ws:";

function widget(node, name) {
    return node.widgets?.find((entry) => entry.name === name);
}

function modelConfig(node) {
    const input = node.inputs?.find((item) => item.name === "model");
    const link = input?.link != null ? app.graph.links[input.link] : null;
    const loader = link ? app.graph.getNodeById(link.origin_id) : null;
    if (!loader || loader.type !== "R2T2GGUFLoader") {
        throw new Error("Connect a Confucius4 Q8 Loader to the model input first.");
    }
    const names = ["n_ctx", "n_batch", "n_threads", "gpu_layers"];
    return Object.fromEntries(names.map((name) => [name, Number(widget(loader, name)?.value)]));
}

function setText(node, name, value) {
    const target = widget(node, name);
    if (target) target.value = String(value);
    const panelKey = {"R2T2 status": "status", "Stable text": "stable", "Preview": "preview"}[name];
    if (panelKey && node.r2t2Panel?.[panelKey]) node.r2t2Panel[panelKey].textContent = String(value);
    node.setDirtyCanvas?.(true, true);
}

function addCaptionPanel(node) {
    if (typeof node.addDOMWidget !== "function") return;
    const root = document.createElement("div");
    Object.assign(root.style, {boxSizing: "border-box", width: "100%", minHeight: "150px",
        padding: "10px", background: "#1b2028", color: "#f2f4f8", borderRadius: "8px",
        fontFamily: "system-ui, sans-serif", fontSize: "14px", overflowWrap: "anywhere"});
    const status = document.createElement("div");
    Object.assign(status.style, {fontSize: "12px", color: "#8ed3c6", marginBottom: "8px"});
    status.textContent = "idle";
    const stable = document.createElement("div");
    Object.assign(stable.style, {whiteSpace: "pre-wrap", minHeight: "55px", lineHeight: "1.5"});
    const preview = document.createElement("div");
    Object.assign(preview.style, {whiteSpace: "pre-wrap", color: "#a6afbe", lineHeight: "1.5"});
    root.append(status, stable, preview);
    node.r2t2Panel = {status, stable, preview};
    node.addDOMWidget("R2T2 captions", "div", root, {serialize: false});
}

function cleanup(state) {
    state.worklet?.disconnect();
    state.source?.disconnect();
    state.silent?.disconnect();
    state.media?.getTracks().forEach((track) => track.stop());
    state.context?.close().catch(() => {});
    state.worklet = null;
    state.source = null;
    state.silent = null;
    state.media = null;
    state.context = null;
}

function requireOpen(state) {
    if (state.ws?.readyState !== WebSocket.OPEN || state.stopping || state.disposed) {
        throw state.connectionError || new Error("Live connection closed during microphone setup");
    }
}

function sendPCM(node, state, pcm) {
    if (state.ws?.readyState !== WebSocket.OPEN || state.stopping) return;
    const outstanding = state.sent - state.acked;
    if (outstanding > 3 * 16000 || state.ws.bufferedAmount > 3 * 16000 * 4) {
        setText(node, "R2T2 status", "OVERLOAD: microphone stopped; audio backlog exceeded 3 seconds");
        cleanup(state);
        state.ws.send(JSON.stringify({type: "cancel"}));
        state.stopping = true;
        return;
    }
    const payload = new ArrayBuffer(8 + pcm.byteLength);
    const view = new DataView(payload);
    view.setUint32(0, state.seq, true);
    view.setUint32(4, state.sent, true);
    new Float32Array(payload, 8).set(pcm);
    state.ws.send(payload);
    state.sent += pcm.length;
    state.seq++;
}

async function start(node) {
    if (node.r2t2?.ws) return;
    const state = {seq: 0, sent: 0, acked: 0, stopping: false, finalizing: false, worklet: null, source: null,
        media: null, context: null, ws: null, flushed: null, silent: null, disposed: false,
        connectionError: null};
    node.r2t2 = state;
    setText(node, "R2T2 status", "loading Q8 model...");
    try {
        const config = modelConfig(node);
        const response = await fetch("/r2t2/v1/live/start", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({model_config: config, language: widget(node, "language")?.value || "Auto",
                context: widget(node, "context")?.value || "",
                stream_chunk_ms: Number(widget(node, "stream_chunk_ms")?.value ?? 320),
                min_segment_seconds: Number(widget(node, "min_segment_seconds")?.value ?? 8)}),
        });
        const created = await response.json();
        if (!response.ok) throw new Error(created.message || JSON.stringify(created));
        state.sid = created.session_id;
        state.browserToken = created.browser_token;
        const ws = new WebSocket(`${PROTOCOL}//${location.host}/r2t2/v1/live/${state.sid}/stream`);
        state.ws = ws;
        ws.binaryType = "arraybuffer";
        const ready = new Promise((resolve, reject) => {
            let readyResolved = false;
            ws.onopen = () => ws.send(JSON.stringify({browser_token: state.browserToken}));
            ws.onerror = () => {
                state.connectionError ||= new Error("WebSocket connection failed");
                reject(state.connectionError);
            };
            ws.onmessage = (message) => {
                const event = JSON.parse(message.data);
                if (event.type === "ready") {
                    readyResolved = true;
                    resolve();
                }
                if (event.type === "ack") {
                    state.acked = event.ack_sample;
                    for (const update of event.events || []) {
                        setText(node, "Stable text", update.stable_text || "");
                        setText(node, "Preview", update.preview_text || "");
                    }
                    if (state.sent - state.acked > 16000) setText(node, "R2T2 status", "audio backlog over 1 second");
                    else if (!state.stopping) setText(node, "R2T2 status", "recording");
                }
                if (event.type === "final") {
                    widget(node, "session_id").value = state.sid;
                    widget(node, "revision").value = event.revision;
                    setText(node, "Stable text", event.text || "");
                    setText(node, "Preview", "");
                    setText(node, "R2T2 status", event.truncated
                        ? "truncated: model output limit; review transcript before use"
                        : event.quality_status === "standard"
                            ? "complete; run workflow to use the snapshot"
                            : "requires review: forced segment boundary; run workflow for text");
                    state.stopping = true;
                }
                if (event.type === "error" || event.type === "cancelled") {
                    const reason = event.type === "error"
                        ? `${event.code || "STREAM_ERROR"}${event.message ? `: ${event.message}` : ""}`
                        : event.message || event.code || "cancelled";
                    state.connectionError = new Error(reason);
                    setText(node, "R2T2 status", `${event.type}: ${reason}`);
                    state.stopping = true;
                    if (!readyResolved) reject(state.connectionError);
                }
            };
            ws.onclose = () => {
                cleanup(state);
                reject(state.connectionError || new Error("WebSocket closed before microphone was ready"));
                if (!state.stopping) setText(node, "R2T2 status", state.connectionError
                    ? `error: ${state.connectionError.message}` : "interrupted: WebSocket closed");
                node.r2t2 = null;
            };
        });
        await ready;
        requireOpen(state);
        state.media = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
        requireOpen(state);
        state.context = new AudioContext();
        setText(node, "R2T2 status", "loading audio worklet...");
        await Promise.race([
            state.context.audioWorklet.addModule(new URL("./r2t2-worklet.js", import.meta.url)),
            new Promise((_, reject) => setTimeout(() => reject(new Error("AudioWorklet load timed out")), 10000)),
        ]);
        requireOpen(state);
        state.source = state.context.createMediaStreamSource(state.media);
        state.worklet = new AudioWorkletNode(state.context, "r2t2-capture");
        state.silent = state.context.createGain();
        state.silent.gain.value = 0;
        state.worklet.port.onmessage = (message) => {
            if (message.data?.type === "pcm") sendPCM(node, state, message.data.samples);
            if (message.data?.type === "flushed") state.flushed?.();
        };
        state.source.connect(state.worklet);
        state.worklet.connect(state.silent).connect(state.context.destination);
        setText(node, "R2T2 status", "recording");
    } catch (error) {
        state.stopping = true;
        cleanup(state);
        state.ws?.close();
        if (state.sid && state.browserToken) {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 5000);
            try {
                await fetch(`/r2t2/v1/live/${encodeURIComponent(state.sid)}/cancel`, {
                    method: "POST", headers: {"X-R2T2-Session-Token": state.browserToken},
                    signal: controller.signal,
                });
            } catch (_) {
                // A closed authenticated WebSocket may already have cancelled
                // the worker session; preserve the original startup error.
            } finally {
                clearTimeout(timeout);
            }
        }
        node.r2t2 = null;
        setText(node, "R2T2 status", `error: ${error.message}`);
    }
}

async function stop(node, cancel = false) {
    const state = node.r2t2;
    if (!state || state.stopping || state.finalizing || state.ws?.readyState !== WebSocket.OPEN) return;
    if (cancel) {
        state.stopping = true;
        cleanup(state);
        state.ws.send(JSON.stringify({type: "cancel"}));
        setText(node, "R2T2 status", "cancelling");
        return;
    }
    state.finalizing = true;
    setText(node, "R2T2 status", "finishing");
    try {
        if (!state.worklet) throw new Error("AudioWorklet is not recording");
        await Promise.race([
            new Promise((resolve) => {
                state.flushed = resolve;
                state.worklet.port.postMessage({type: "flush"});
            }),
            new Promise((_, reject) => setTimeout(() => reject(new Error("AudioWorklet flush timed out")), 5000)),
        ]);
        requireOpen(state);
    } catch (error) {
        state.finalizing = false;
        state.stopping = true;
        cleanup(state);
        if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({type: "cancel"}));
        setText(node, "R2T2 status", `error: ${error.message}`);
        return;
    } finally {
        state.flushed = null;
    }
    cleanup(state);
    state.finalizing = false;
    state.stopping = true;
    state.ws.send(JSON.stringify({type: "finish", last_seq: state.seq - 1, total_samples: state.sent}));
}

app.registerExtension({
    name: "confucius4.r2t2.live",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "R2T2LiveSession") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            original?.apply(this, args);
            this.addWidget("button", "Start microphone", null, () => start(this));
            this.addWidget("button", "Stop and finalize", null, () => stop(this));
            this.addWidget("button", "Cancel", null, () => stop(this, true));
            this.addWidget("text", "R2T2 status", "idle", () => {}, {serialize: false});
            this.addWidget("text", "Stable text", "", () => {}, {serialize: false});
            this.addWidget("text", "Preview", "", () => {}, {serialize: false});
            addCaptionPanel(this);
            this.size = [Math.max(this.size[0], 420), Math.max(this.size[1], 560)];
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            if (this.r2t2) {
                this.r2t2.disposed = true;
                if (this.r2t2.ws?.readyState === WebSocket.OPEN) {
                    this.r2t2.ws.send(JSON.stringify({type: "cancel"}));
                }
                cleanup(this.r2t2);
                this.r2t2.ws?.close();
                this.r2t2 = null;
            }
            removed?.apply(this, args);
        };
    },
});
