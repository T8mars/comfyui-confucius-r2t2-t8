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
    class Node {
        constructor() {this.size = [200, 100];}
        addDOMWidget(name, type, element, options) {assert.equal(options.serialize, false);}
        setDirtyCanvas() {}
    }
    extension.setup();
    await extension.beforeRegisterNodeDef(Node, {name: "R2T2Subtitle"});
    const node = new Node();
    node.onNodeCreated();
    node.size = [200, 100]; // A pre-subtitle workflow restores its old size.
    globalThis.__subtitleApp.graph._nodes.push(node);
    const message = {cue_count: [2], subtitle_status: ["requires_review"],
        subtitle_preview: ["<script>caption</script>"], warnings: ["approximate_segment_timing"],
        files: [{filename: "字幕 & 1.srt", subfolder: "", type: "output"}]};
    node.onExecuted(message);
    assert.deepEqual(node.size, [430, 600]);
    assert.equal(node.r2t2SubtitlePanel.preview.textContent, message.subtitle_preview[0]);
    const url = new URL(node.r2t2SubtitlePanel.download.href);
    assert.equal(url.searchParams.get("filename"), "字幕 & 1.srt");
    assert.match(node.r2t2SubtitlePanel.status.textContent, /请校对/);
    listeners.execution_start();
    assert.equal(node.r2t2SubtitlePanel.download.href, undefined);
    node.onExecuted({cue_count: [0], subtitle_status: ["empty"], subtitle_preview: [""], files: []});
    assert.match(node.r2t2SubtitlePanel.status.textContent, /No subtitles/);
    node.onExecuted(message);
    listeners.execution_error();
    assert.equal(node.r2t2SubtitlePanel.download.style.display, "none");
    assert.equal(node.r2t2SubtitlePanel.preview.textContent, "");
    assert.match(node.r2t2SubtitlePanel.status.textContent, /failed/);
    console.log("subtitle preview, download, empty and stale-error cleanup pass");
})();
