const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

(async () => {
    let extension;
    const listeners = {};
    globalThis.__subtitleApp = {registerExtension: value => {extension = value;}, graph: {_nodes: []}};
    globalThis.__subtitleApi = {apiURL: value => "http://localhost" + value,
        addEventListener: (name, fn) => {listeners[name] = fn;}};
    globalThis.document = {createElement: () => ({style: {}, children: [], textContent: "",
        append(...items) {this.children.push(...items);}, removeAttribute(name) {delete this[name];}})};
    const source = fs.readFileSync(path.join(__dirname, "../web/subtitles.js"), "utf8")
        .replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.__subtitleApp;')
        .replace('import { api } from "../../scripts/api.js";', 'const api = globalThis.__subtitleApi;');
    await import("data:text/javascript," + encodeURIComponent(source));
    const chained = [];
    class Node {
        constructor() {this.size = [200, 100];}
        addDOMWidget(name, type, element, options) {assert.equal(options.serialize, false);}
        setDirtyCanvas() {}
        onNodeCreated(...args) {chained.push(["created", ...args]);}
        onExecuted(message, ...args) {chained.push(["executed", message, ...args]);}
        onExecutionStart(...args) {chained.push(["start", ...args]); return "original-start";}
        onExecutionError(...args) {chained.push(["error", ...args]); return "original-error";}
    }
    extension.setup();
    await extension.beforeRegisterNodeDef(Node, {name: "R2T2Subtitle"});
    const node = new Node();
    node.onNodeCreated();
    const other = new Node();
    other.onNodeCreated();
    node.size = [200, 100]; // A pre-subtitle workflow restores its old size.
    globalThis.__subtitleApp.graph._nodes.push(node, other);
    const message = {cue_count: [2], subtitle_status: ["requires_review"],
        subtitle_preview: ["<script>caption</script>"], warnings: ["approximate_segment_timing"],
        files: [{filename: "字幕 & 1.srt", subfolder: "", type: "output"}]};
    const event = promptId => ({detail: {prompt_id: promptId, node_id: "unrelated-node"}});
    listeners.execution_start(event("first"));
    node.onExecuted(message);
    assert.deepEqual(node.size, [430, 600]);
    assert.equal(node.r2t2SubtitlePanel.preview.textContent, message.subtitle_preview[0]);
    const url = new URL(node.r2t2SubtitlePanel.download.href);
    assert.equal(url.searchParams.get("filename"), "字幕 & 1.srt");
    assert.match(node.r2t2SubtitlePanel.status.textContent, /请校对/);
    // A later failure in another branch must preserve this run's successful
    // download, while an unfinished export gets a failure message.
    const successfulUrl = node.r2t2SubtitlePanel.download.href;
    listeners.execution_error(event("first"));
    assert.equal(node.r2t2SubtitlePanel.download.href, successfulUrl);
    assert.equal(node.r2t2SubtitlePanel.preview.textContent, message.subtitle_preview[0]);
    assert.match(other.r2t2SubtitlePanel.status.textContent, /failed/);

    listeners.execution_start(event("second"));
    assert.equal(node.r2t2SubtitlePanel.download.href, undefined);
    assert.equal(node.r2t2SubtitlePanel.preview.textContent, "");
    // Cached onExecuted uses the same message as a fresh export. An empty
    // export is also successful and must retain its empty-state message.
    node.onExecuted(message);
    other.onExecuted({cue_count: [0], subtitle_status: ["empty"], subtitle_preview: [""], files: []});
    listeners.execution_error(event("second"));
    assert.equal(node.r2t2SubtitlePanel.download.href, successfulUrl);
    assert.match(other.r2t2SubtitlePanel.status.textContent, /No subtitles/);

    listeners.execution_start(event("third"));
    for (const name of ["execution_error", "execution_interrupted"]) {
        listeners[name](event("second"));
        assert.equal(node.r2t2SubtitlePanel.status.textContent, "");
        listeners[name](); // Unscoped/broadcast failures do not own this prompt.
        assert.equal(node.r2t2SubtitlePanel.status.textContent, "");
    }
    listeners.execution_interrupted(event("third"));
    assert.equal(node.r2t2SubtitlePanel.download.style.display, "none");
    assert.equal(node.r2t2SubtitlePanel.preview.textContent, "");
    assert.match(node.r2t2SubtitlePanel.status.textContent, /cancelled/);

    listeners.execution_start(event("fourth"));
    node.onExecuted(message);
    listeners.execution_interrupted(event("fourth"));
    assert.equal(node.r2t2SubtitlePanel.download.href, successfulUrl);
    assert.match(other.r2t2SubtitlePanel.status.textContent, /cancelled/);

    // Legacy node-local hooks have no event/prompt ID but remain scoped to the
    // node. A new start cannot retain the preceding run's success.
    assert.equal(node.onExecutionStart("local-start"), "original-start");
    assert.equal(node.onExecutionError("local-error"), "original-error");
    assert.match(node.r2t2SubtitlePanel.status.textContent, /failed/);
    assert.equal(node.r2t2SubtitlePanel.download.href, undefined);
    listeners.execution_start();
    node.onExecuted(message);
    listeners.execution_interrupted(event("another-client"));
    listeners.execution_error(event("another-client"));
    assert.equal(other.r2t2SubtitlePanel.status.textContent, "");
    listeners.execution_error();
    assert.equal(node.r2t2SubtitlePanel.download.href, successfulUrl);
    assert.match(other.r2t2SubtitlePanel.status.textContent, /failed/);

    listeners.execution_start(event("fifth"));
    node.onExecuted(message);
    listeners.execution_success(event("fifth"));
    listeners.execution_error(event("fifth"));
    assert.equal(node.r2t2SubtitlePanel.download.href, successfulUrl);
    assert.equal(other.r2t2SubtitlePanel.status.textContent, "");
    assert.equal(chained.filter(call => call[0] === "created").length, 2);
    assert(chained.some(call => call[0] === "executed" && call[1] === message));
    assert.deepEqual(chained.find(call => call[0] === "start"), ["start", "local-start"]);
    assert.deepEqual(chained.find(call => call[0] === "error"), ["error", "local-error"]);
    console.log("subtitle preview, download, empty, cached and prompt-scoped failure/interrupt handling pass");
})();
