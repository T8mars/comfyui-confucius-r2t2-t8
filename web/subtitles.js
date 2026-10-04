import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function clearPanel(node) {
    const panel = node.r2t2SubtitlePanel;
    if (!panel) return;
    panel.status.textContent = "";
    panel.preview.textContent = "";
    panel.download.style.display = "none";
    panel.download.removeAttribute("href");
}

app.registerExtension({
    name: "confucius4.r2t2.subtitles",
    setup() {
        api.addEventListener("execution_start", () => {
            for (const node of app.graph?._nodes || []) clearPanel(node);
        });
        api.addEventListener("execution_error", () => {
            for (const node of app.graph?._nodes || []) {
                clearPanel(node);
                if (node.r2t2SubtitlePanel) node.r2t2SubtitlePanel.status.textContent = "Export failed / 导出失败";
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["R2T2Subtitle", "R2T2SaveTranscript"].includes(nodeData.name)) return;
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
            this.addDOMWidget("R2T2 subtitle preview", "r2t2-subtitle", root, {serialize: false});
            this.r2t2SubtitlePanel = {status, download, preview};
            const height = nodeData.name === "R2T2Subtitle" ? 600 : 450;
            const width = nodeData.name === "R2T2Subtitle" ? 430 : 300;
            this.size = [Math.max(this.size[0], width), Math.max(this.size[1], height)];
        };
        const clear = clearPanel;
        const executing = nodeType.prototype.onExecutionStart;
        nodeType.prototype.onExecutionStart = function (...args) {
            clear(this);
            return executing?.apply(this, args);
        };
        const failed = nodeType.prototype.onExecutionError;
        nodeType.prototype.onExecutionError = function (...args) {
            clear(this);
            if (this.r2t2SubtitlePanel) this.r2t2SubtitlePanel.status.textContent = "Export failed / 导出失败";
            return failed?.apply(this, args);
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message, ...args) {
            executed?.call(this, message, ...args);
            clear(this);
            const panel = this.r2t2SubtitlePanel;
            if (!panel) return;
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
