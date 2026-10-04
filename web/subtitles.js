import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let activeRun = null;
const PREVIEW_WIDGET = "R2T2 subtitle preview";

function clearPanel(node) {
    const panel = node.r2t2SubtitlePanel;
    if (!panel) return;
    panel.status.textContent = "";
    panel.preview.textContent = "";
    panel.download.style.display = "none";
    panel.download.removeAttribute("href");
}

function pendingPanel(node) {
    clearPanel(node);
    const panel = node.r2t2SubtitlePanel;
    if (!panel) return;
    panel.run = activeRun;
    panel.completed = false;
}

function matchesRun(event, requirePromptId = false) {
    if (!activeRun) return false;
    const promptId = event?.detail?.prompt_id;
    // Interrupts are broadcast to other clients. Never infer their owner from
    // a missing ID. Legacy local hooks are handled on the node itself.
    if (promptId == null) return !requirePromptId && activeRun.promptId == null;
    return activeRun.promptId != null && String(promptId) === activeRun.promptId;
}

function failPendingPanels(message) {
    for (const node of app.graph?._nodes || []) {
        const panel = node.r2t2SubtitlePanel;
        if (!panel || panel.run !== activeRun || panel.completed) continue;
        clearPanel(node);
        panel.status.textContent = message;
    }
}

app.registerExtension({
    name: "confucius4.r2t2.subtitles",
    setup() {
        api.addEventListener("execution_start", event => {
            const promptId = event?.detail?.prompt_id;
            activeRun = {promptId: promptId == null ? null : String(promptId)};
            for (const node of app.graph?._nodes || []) pendingPanel(node);
        });
        api.addEventListener("execution_error", event => {
            if (matchesRun(event)) failPendingPanels("Export failed / 导出失败");
        });
        api.addEventListener("execution_interrupted", event => {
            if (matchesRun(event, true)) failPendingPanels("Export cancelled / 导出已取消");
        });
        api.addEventListener("execution_success", event => {
            if (matchesRun(event)) activeRun = null;
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["R2T2Subtitle", "R2T2SaveTranscript"].includes(nodeData.name)) return;
        const formats = nodeData.name === "R2T2Subtitle" ? ["srt", "vtt"] : ["txt", "json", "srt", "vtt"];
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info, ...args) {
            const values = info?.widgets_values;
            const count = (this.widgets || []).filter(widget =>
                widget.name !== PREVIEW_WIDGET && widget.serialize !== false).length;
            // 0.1.3/0.1.4 appended a serialized empty preview widget. ComfyUI's
            // forceInput migration mistakes that extra value for a leading
            // result_json placeholder and removes format. Strip only our
            // recognizable trailing placeholder BEFORE that migration runs.
            if (Array.isArray(values) && values.length === count + 1 && formats.includes(values[0])
                    && (values.at(-1) === "" || values.at(-1) == null)) {
                info = {...info, widgets_values: values.slice(0, -1)};
                if (info.widgets_values_named) {
                    info.widgets_values_named = {...info.widgets_values_named};
                    delete info.widgets_values_named[PREVIEW_WIDGET];
                }
            }
            return configure?.call(this, info, ...args);
        };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            created?.apply(this, args);
            const root = document.createElement("div");
            Object.assign(root.style, {width: "100%", padding: "8px", boxSizing: "border-box",
                color: "var(--input-text, #ddd)", fontSize: "12px"});
            const status = document.createElement("div");
            const download = document.createElement("a");
            download.textContent = "Download / 下载";
            download.style.display = "none";
            download.style.color = "#80caff";
            const preview = document.createElement("pre");
            Object.assign(preview.style, {whiteSpace: "pre-wrap", maxHeight: "220px", overflow: "auto",
                userSelect: "text", margin: "6px 0"});
            root.append(status, download, preview);
            const widget = this.addDOMWidget(PREVIEW_WIDGET, "r2t2-subtitle", root, {serialize: false});
            // Current LiteGraph serializes by widget.serialize, whereas API
            // prompt construction also consults options.serialize.
            widget.serialize = false;
            this.r2t2SubtitlePanel = {status, download, preview, run: null, completed: false};
            const height = nodeData.name === "R2T2Subtitle" ? 600 : 450;
            const width = nodeData.name === "R2T2Subtitle" ? 430 : 300;
            this.size = [Math.max(this.size[0], width), Math.max(this.size[1], height)];
        };
        const clear = clearPanel;
        const executing = nodeType.prototype.onExecutionStart;
        nodeType.prototype.onExecutionStart = function (...args) {
            pendingPanel(this);
            return executing?.apply(this, args);
        };
        const failed = nodeType.prototype.onExecutionError;
        nodeType.prototype.onExecutionError = function (...args) {
            const panel = this.r2t2SubtitlePanel;
            if (panel && !panel.completed) {
                clear(this);
                panel.status.textContent = "Export failed / 导出失败";
            }
            return failed?.apply(this, args);
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message, ...args) {
            executed?.call(this, message, ...args);
            clear(this);
            const panel = this.r2t2SubtitlePanel;
            if (!panel) return;
            // ComfyUI replays onExecuted for cached UI outputs as well. Empty
            // exports are completed results too, even though they have no file.
            panel.run = activeRun;
            panel.completed = true;
            // Old workflows can restore a small node size after onNodeCreated.
            const height = nodeData.name === "R2T2Subtitle" ? 600 : 450;
            const width = nodeData.name === "R2T2Subtitle" ? 430 : 300;
            const size = [Math.max(this.size[0], width), Math.max(this.size[1], height)];
            if (this.setSize) this.setSize(size);
            else this.size = size;
            const count = message.cue_count?.[0];
            const status = message.subtitle_status?.[0];
            panel.status.textContent = status === "empty" ? "没有可导出的字幕 / No subtitles to export"
                : count != null ? `${count} cues · ${status}${status === "requires_review" ? " · 分段时间估算，请校对" : ""}`
                : "Saved / 已保存";
            if (message.warnings?.length) panel.status.textContent += "\n" + message.warnings.join(", ");
            panel.preview.textContent = message.subtitle_preview?.[0] || message.text?.[0] || "";
            const file = message.files?.[0];
            if (file) {
                const query = new URLSearchParams({filename: file.filename, subfolder: file.subfolder || "", type: "output"});
                panel.download.href = api.apiURL("/view?" + query.toString());
                panel.download.download = file.filename;
                panel.download.style.display = "block";
            }
            this.setDirtyCanvas?.(true, true);
        };
    },
});
