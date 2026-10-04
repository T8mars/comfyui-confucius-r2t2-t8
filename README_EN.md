# Confucius4-R2T2 Q8 for ComfyUI

[简体中文](README.md) · [English](README_EN.md)

Windows / NVIDIA CUDA speech-to-text nodes based on NetEase Youdao's [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2). Inference uses the official **GGUF Q8 decoder and Q8 audio projector** in an isolated Python 3.12 worker. The upstream vLLM dependencies are not installed into your existing ComfyUI environment.

The nodes provide offline and streaming file transcription, live browser-microphone captions, and transcript saving. Long-audio segmentation uses [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) ONNX. See the [Hugging Face model repository](https://huggingface.co/t8star/Confucius-R2t2-Comfy) for assets and license details.

## Install

1. Clone this repository into `ComfyUI/custom_nodes/comfyui-confucius-r2t2-t8`. Comfy Registry publishing is configured; installation through Manager search requires version approval.
2. Install Python 3.12, Git, CMake, Visual Studio 2022 C++ Build Tools, and CUDA Toolkit 12.8. Open PowerShell in the node directory and run:

   ```powershell
   .\scripts\setup_worker_windows.ps1
   ```

3. Restart ComfyUI. Setup downloads and verifies about 2.19 GB of Q8 model files and builds a CUDA extension from a pinned llama.cpp revision. The default CUDA architecture is 120 (RTX 50 series). For another NVIDIA GPU, run `setup_worker_windows.ps1 -SkipNativeBuild` first, then `build_native_windows.ps1 -CudaArch <architecture>`.

The worker and weights live under Git-ignored `.runtime/` and `models/` in the node directory. For a manual junction into an existing ComfyUI checkout, run `scripts/install_comfy_windows.ps1 -ComfyDir 'C:\path\to\ComfyUI'`.

## Use

**Drag the [file workflow](workflows/confucius4_q8_file.json) onto the ComfyUI frontend canvas.** Select your own audio in `Load Audio`, then click **Run**. Drag in the [microphone workflow](workflows/confucius4_q8_live.json), click **Start microphone**, grant browser access, then click **Stop and finalize** and **Run** to save the final transcript. Browser microphone access requires localhost or HTTPS.

Nodes: `Confucius4 Q8 Loader`, `Confucius4 Transcribe`, `Confucius4 Hotwords`, `Confucius4 Live Microphone`, `Confucius4 Save Transcript`, `Confucius4 Save Subtitle`, and `Confucius4 Unload Q8`. File input supports one audio item at a time (batch=1). For mostly Chinese speech with some English, try the `Chinese` language hint in file-streaming mode.

### Video subtitles / SRT (0.1.6)

Drag the [subtitle workflow](workflows/confucius4_q8_subtitles.json) into the frontend, choose audio or video, and click Run. Place a video in `ComfyUI/input` to select it in `Load Audio`; the first audio track is decoded without loading video frames.

- Enable Transcribe's `subtitle_timings` to collect complete segment text and timings, including short files. Transcribe and finalized Live sessions now have an `srt` output; the original `text/language/result_json` output order is preserved.
- `Save Transcript` accepts **txt/json/srt/vtt**. `Save Subtitle` also offers Chinese/English line widths, a millisecond offset, subtitle text/JSON/count outputs, and frontend preview/download. Files go into `ComfyUI/output`.
- These are **approximate VAD/segment timings requiring review**, not word alignment. Long cues wrap and carry a warning instead of inventing evenly spaced sentence timestamps. Use `offset_ms` for audio-track delays or trimmed-video offsets.
- Old JSON without segment text/timings requires re-transcription. `whole_audio_draft` explicitly permits a coarse whole-file cue. Truncated results require `allow_partial`; silence produces no empty file. Changed content/offsets get separate filenames, without overwriting subtitles.

0.1.6 fixes stale Live snapshots, duplicate starts and cancellation, and excludes temporary recording controls from saved workflows. Unload runs on every execution; interrupted worker connections use transport recovery. Subtitle migration also handles `srt/json` filename prefixes. Restart ComfyUI and refresh its frontend after updating. If an old file has already lost its `format`, re-import an example and set its parameters again.

0.1.4 fixes export caching, error handling, and special characters. Re-running restores moved output files while upstream recognition can remain cached. A later node failure preserves subtitles already saved in that run. Windows saving no longer requires hard links. If the convenience SRT output fails, transcription text and JSON remain available with `subtitle_export_error`; explicit subtitle export still validates strictly. SRT preserves ordinary `&` characters; players may interpret `<...>` as formatting tags. VTT uses standard character references.

Convert an existing JSON containing timed segment text without model dependencies:

```powershell
python scripts/json_to_subtitle.py transcript.json --format srt --offset-ms 0
```

Word alignment, track-selection nodes, and burned-in captions are future features; individual editors' subtitle import compatibility is not yet verified.
Frontend checks cover short MP4 and 120-second repeated-speech MP4 export, preview, download, timing offsets, and legacy workflow compatibility.

**Bulk hotwords (0.1.2):**

- Paste a whole list into Transcribe's `hotwords`, separated by lines, commas, semicolons or Chinese equivalents. Spaces inside phrases are preserved.
- Use the [shared hotword workflow](workflows/confucius4_q8_hotwords.json), edit `words` in `Confucius4 Hotwords`, and connect its output to multiple Transcribe nodes.
- The optional `hotword_file` reads a UTF-8 TXT path relative to `ComfyUI/input`, such as `r2t2_hotwords/brands.txt`. File and pasted entries are merged and deduplicated in first-seen order. File edits take effect on the next run.

TXT files are limited to 64 KiB. Context and hotwords together must fit 8192 characters including the prompt label, plus the model's token budget. Hotwords guide recognition rather than force corrections; use relevant names, brands and terms. For microphone sessions, place hotwords in Live's `context` (e.g. `Hotwords: brand A, brand B`) before starting a new session.

Since **0.1.1**, `offline` files longer than 30 seconds are split with VAD and decoded once per segment, with a maximum segment length of 20 seconds. Results report `mode_executed=segmented_offline`. Earlier versions switched these files to streaming decoding, with repeated inference and accumulated caption events. `n_ctx` controls the inference context for one segment; increasing it does not remove segmentation or define the maximum file duration. File transport accepts up to 512 MiB of raw float32 PCM, so available duration depends on sample rate and channel count. `stream_chunk_ms` affects streaming mode only.

The official Q8 files have been exercised on Windows / RTX 5090 Laptop / CUDA 12.8 for offline and streaming inference, frontend workflow drag-and-run, and a live frontend Start/Stop path with injected audio. Physical-microphone capture, accuracy on uninterrupted natural long speech, and other GPU architectures remain unverified. Forced segmentation is marked `requires_review`. Raw audio is not saved by default.

## Licenses and provenance

Node code is released under [Apache 2.0](LICENSE); native adapter provenance and changes are documented in [NOTICE](vendor/r2t2_native/NOTICE.md). The GGUF model and projector belong to NetEase Youdao and are governed by its [Model Use License Agreement](https://huggingface.co/t8star/Confucius-R2t2-Comfy/blob/main/MODEL_LICENSE). FireRedVAD assets are Apache 2.0. Read the applicable licenses before using or redistributing weights. This project is not officially affiliated with the model creators.

## Links

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI package](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
