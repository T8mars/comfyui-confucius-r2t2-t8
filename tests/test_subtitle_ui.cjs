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
    // Match frontend 1.53.6: workflow persistence skips widget.serialize,
    // not options.serialize. Its forceInput migration runs before restoration.
    function serializeWidgets(widgets) {
        const saved = {widgets_values: [], widgets_values_named: {}};
        for (const widget of widgets) {
            if (widget.serialize === false) continue;
            saved.widgets_values.push(widget.value ?? null);
            saved.widgets_values_named[widget.name] = widget.value ?? null;
        }
        return saved;
    }
    function migrateForceInput(values, count) {
        const forceInputMask = [true, ...Array(count).fill(false)];
        return values.length === forceInputMask.length
            ? values.filter((_, index) => !forceInputMask[index]) : values;
    }
    class BaseNode {
        constructor() {
            this.size = [200, 100];
            this.widgets = this.constructor.initialValues.map(([name, value]) => ({name, value}));
        }
        addDOMWidget(name, type, element, options) {
            assert.equal(options.serialize, false);
            const widget = {name, type, element, options, value: ""};
            this.widgets.push(widget);
            return widget;
        }
        configure(info) {
            this.lastConfigureInfo = info;
            const values = migrateForceInput(info.widgets_values, this.constructor.initialValues.length);
            let index = 0;
            for (const widget of this.widgets) {
                if (widget.serialize === false) continue;
                if (index < values.length) widget.value = values[index];
                index++;
            }
            return "original-configure";
        }
        setDirtyCanvas() {}
        onNodeCreated(...args) {chained.push(["created", ...args]);}
        onExecuted(message, ...args) {chained.push(["executed", message, ...args]);}
        onExecutionStart(...args) {chained.push(["start", ...args]); return "original-start";}
        onExecutionError(...args) {chained.push(["error", ...args]); return "original-error";}
    }
    class Node extends BaseNode {
        static initialValues = [["format", "srt"], ["prefix", "subtitle"], ["offset_ms", 0],
            ["chinese_chars", 16], ["english_chars", 42], ["allow_partial", false], ["whole_audio_draft", false]];
    }
    class TranscriptNode extends BaseNode {
        static initialValues = [["format", "json"], ["prefix", "transcript"], ["offset_ms", 0],
            ["allow_partial", false], ["whole_audio_draft", false]];
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
    await extension.beforeRegisterNodeDef(TranscriptNode, {name: "R2T2SaveTranscript"});
    for (const Type of [Node, TranscriptNode]) {
        let current = new Type();
        current.onNodeCreated();
        const expected = Type.initialValues.map(([, value]) => value);
        const previewWidget = current.widgets.at(-1);
        assert.equal(previewWidget.name, "R2T2 subtitle preview");
        assert.equal(previewWidget.serialize, false);
        assert.deepEqual(serializeWidgets(current.widgets).widgets_values, expected);
        assert.equal(serializeWidgets(current.widgets).widgets_values_named[previewWidget.name], undefined);

        // Real old exports contained one trailing empty DOM value. The core
        // migration alone deletes format; the extension must strip the tail
        // before that core migration runs, without changing source JSON.
        const oldSaved = {widgets_values: [...expected, ""],
            widgets_values_named: {...serializeWidgets(current.widgets).widgets_values_named,
                "R2T2 subtitle preview": ""}};
        assert.equal(migrateForceInput(oldSaved.widgets_values, expected.length)[0], expected[1]);
        assert.equal(current.configure(oldSaved), "original-configure");
        assert.deepEqual(serializeWidgets(current.widgets).widgets_values, expected);
        assert.deepEqual(current.lastConfigureInfo.widgets_values, expected);
        assert.deepEqual(oldSaved.widgets_values, [...expected, ""]);

        // An actual legacy forceInput placeholder is at the START. It must
        // reach ComfyUI's migration intact, not be mistaken for our preview.
        const forceInputLegacy = {widgets_values: [null, ...expected]};
        current.configure(forceInputLegacy);
        assert.equal(current.lastConfigureInfo, forceInputLegacy);
        assert.deepEqual(serializeWidgets(current.widgets).widgets_values, expected);

        // Repeat workflow tab save/reload cycles using positional restoration
        // (the named-values feature is not required for this fix).
        for (let cycle = 0; cycle < 5; cycle++) {
            const saved = JSON.parse(JSON.stringify(serializeWidgets(current.widgets)));
            current = new Type();
            current.onNodeCreated();
            current.configure(saved);
            assert.deepEqual(serializeWidgets(current.widgets).widgets_values, expected);
        }
        // Already damaged files have lost format. Preserve their error instead
        // of guessing which format the user originally selected.
        const damaged = {widgets_values: [...expected.slice(1), "", ""]};
        current.configure(damaged);
        assert.equal(current.lastConfigureInfo, damaged);
        assert.equal(current.lastConfigureInfo.widgets_values[0], expected[1]);
    }
    const legacyTranscript = new TranscriptNode();
    legacyTranscript.onNodeCreated();
    legacyTranscript.configure({widgets_values: ["txt", "legacy-prefix"]});
    assert.deepEqual(serializeWidgets(legacyTranscript.widgets).widgets_values,
        ["txt", "legacy-prefix", 0, false, false]);
    console.log("subtitle UI state, workflow persistence, legacy migration and repeated tab reload checks pass");
})();
