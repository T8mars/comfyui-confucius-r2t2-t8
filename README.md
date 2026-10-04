# Confucius4-R2T2 Q8 for ComfyUI

[简体中文](README.md) · [English](README_EN.md)

基于网易有道 [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) 的 Windows / NVIDIA CUDA 语音转文字节点。使用官方 **GGUF Q8 主模型 + Q8 音频 projector**；推理运行在独立的 Python 3.12 worker 中，不向现有 ComfyUI Python 安装上游 vLLM 依赖。

支持文件离线转写、文件流式转写、浏览器麦克风实时字幕和保存转写。长音频分段使用 [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) ONNX。模型文件及许可说明见 [Hugging Face 模型仓库](https://huggingface.co/t8star/Confucius-R2t2-Comfy)。

## 安装

1. 将本仓库克隆到 `ComfyUI/custom_nodes/comfyui-confucius-r2t2-t8`。已接入 Comfy Registry 发布流程；Manager 搜索安装需要官方版本审核通过。
2. 安装 Python 3.12、Git、CMake、Visual Studio 2022 C++ 构建工具和 CUDA Toolkit 12.8。在节点目录打开 PowerShell，运行：

   ```powershell
   .\scripts\setup_worker_windows.ps1
   ```

3. 重启 ComfyUI。首次设置会下载并校验约 2.19 GB 的 Q8 模型文件，并从固定版本的 llama.cpp 构建 CUDA 扩展；需要预留编译时间。默认编译目标为 RTX 50 系列（CUDA 架构 120）。其他 NVIDIA GPU 请先运行 `setup_worker_windows.ps1 -SkipNativeBuild`，再运行 `build_native_windows.ps1 -CudaArch <架构>`。

安装脚本将 worker 与模型放在节点目录下的 `.runtime/`、`models/`，两者均被 Git 忽略。手动接入已有 ComfyUI 源码目录可执行 `scripts/install_comfy_windows.ps1 -ComfyDir 'C:\path\to\ComfyUI'`。

## 使用

将 [文件工作流](workflows/confucius4_q8_file.json) **拖入 ComfyUI 前端画布**，在 `Load Audio` 重新选择自己的音频，点击 **运行**。将 [麦克风工作流](workflows/confucius4_q8_live.json) 拖入画布后，在 Live 节点点击 **Start microphone**，授权麦克风；点击 **Stop and finalize** 后，再点击 **运行** 保存最终文字。浏览器麦克风需要 localhost 或 HTTPS。

节点包括 `Confucius4 Q8 Loader`、`Confucius4 Transcribe`、`Confucius4 Hotwords`、`Confucius4 Live Microphone`、`Confucius4 Save Transcript`、`Confucius4 Save Subtitle` 和 `Confucius4 Unload Q8`。单次文件输入仅支持一段音频（batch=1）。中文为主、夹少量英文的文件流式音频可尝试 `Chinese` 语言提示。

### 视频字幕 / SRT（0.1.3）

将 [字幕工作流](workflows/confucius4_q8_subtitles.json) 拖入前端，选择音频或视频，点击运行。视频可放进 `ComfyUI/input` 后在 `Load Audio` 选择；读取第 1 条音轨，无需解码视频帧。

- Transcribe 开启 `subtitle_timings`，短文件也会收集完整段文本与时间。新增 `srt` 输出；Live 结束后同样输出 SRT。原 `text/language/result_json` 输出顺序保持兼容。
- `Save Transcript` 的 `format` 可选 **txt/json/srt/vtt**。`Save Subtitle` 额外提供中英文行宽、毫秒偏移、字幕文字/JSON/条数输出，以及前端预览与下载；文件保存在 `ComfyUI/output`。
- 时间来自 VAD/识别分段，**属于近似字幕，需要校对**；没有字/词对齐。长句只换行并提示过长，不按平均时间制造逐句时标。视频有音轨延迟或截取偏移时，可用 `offset_ms` 校正。
- 旧 JSON 缺少段文字或时间会要求重跑；`whole_audio_draft` 是显式的整段粗字幕回退。截断结果须开启 `allow_partial`，静音不生成空文件。不同偏移/内容生成不同文件，不覆盖已有字幕。

也可离线转换已保存的、有段文本和时间的 JSON：

```powershell
python scripts/json_to_subtitle.py transcript.json --format srt --offset-ms 0
```

字/词精确对齐、音轨选择节点和字幕烧录属于后续功能；各剪辑软件的字幕导入兼容性尚未逐一验证。
已在 ComfyUI 前端验证短 MP4 与 120 秒重复语音 MP4 的导出、预览、下载、时间偏移及旧工作流兼容性。

**批量热词（0.1.2）**：

- 直接在 Transcribe 的 `hotwords` 粘贴整份词表，用换行、逗号、分号或顿号分隔；保留英文词组中的空格。
- 使用 [共享热词工作流](workflows/confucius4_q8_hotwords.json)，集中修改 `Confucius4 Hotwords` 的 `words`，输出可连接多个 Transcribe 节点。
- 可选 `hotword_file` 读取 `ComfyUI/input` 下的 UTF-8 TXT 相对路径，例如 `r2t2_hotwords/brands.txt`。文件与粘贴内容合并去重，编辑文件后下次运行自动读取新内容。

TXT 文件上限 64 KiB；热词与 `context` 合计上限 8192 字符（含提示标签），并需留足模型的 token 上下文。热词用于辅助识别，不是强制纠错；优先放相关人名、品牌和术语。麦克风热词写入 Live 的 `context`（如 `Hotwords: 品牌甲, 品牌乙`），开始新会话后生效。

从 **0.1.1** 起，`offline` 模式处理超过 30 秒的文件时，会用 VAD 分段并对每段做一次完整离线识别，每段最多 20 秒；结果中的 `mode_executed` 为 `segmented_offline`。此前版本会改走流式解码，长文件的重复计算和字幕事件较多。`n_ctx` 是单段推理的上下文大小，增大它不会取消分段，也不决定文件能处理多少分钟。文件传输上限为 512 MiB 的原始 float32 PCM，实际可用时长取决于采样率和声道数；`stream_chunk_ms` 只影响流式模式。

本机 Windows / RTX 5090 Laptop / CUDA 12.8 已通过官方 Q8 文件的离线与流式推理、真实 ComfyUI 前端拖入工作流并点击运行，以及注入音频的 Live Start/Stop 前端流程。物理麦克风、自然连续长语音的质量及所有 GPU 架构尚未全面验收；强制分段结果会标记 `requires_review`。默认不保存原始音频。

## 许可与来源

节点代码按 [Apache 2.0](LICENSE) 发布；原生适配代码来源与修改见 [NOTICE](vendor/r2t2_native/NOTICE.md)。GGUF 模型及 projector 属网易有道，受其 [Model Use License Agreement](https://huggingface.co/t8star/Confucius-R2t2-Comfy/blob/main/MODEL_LICENSE) 约束；FireRedVAD 资产受 Apache 2.0 约束。使用或再分发模型时请阅读相应许可。本项目与模型原作者无官方隶属关系。

## 更多链接

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
