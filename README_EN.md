# Confucius4-R2T2 Q8 for ComfyUI

[简体中文](README.md) · [English](README_EN.md)

Windows / NVIDIA CUDA speech-to-text nodes based on NetEase Youdao's [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2). Inference uses the official **GGUF Q8 decoder and Q8 audio projector** in an isolated Python 3.12 worker. The upstream vLLM dependencies are not installed into your existing ComfyUI environment.

The nodes provide offline and streaming file transcription, live browser-microphone captions, and transcript saving. Long-audio segmentation uses [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) ONNX. See the [Hugging Face model repository](https://huggingface.co/t8star/Confucius-R2t2-Comfy) for assets and license details.

## Install

1. Install **Confucius4-R2T2 Q8 ASR** with ComfyUI Manager, or clone this repository into `ComfyUI/custom_nodes/comfyui-confucius-r2t2-t8`.
2. Install Python 3.12, Git, CMake, Visual Studio 2022 C++ Build Tools, and CUDA Toolkit 12.8. Open PowerShell in the node directory and run:

   ```powershell
   .\scripts\setup_worker_windows.ps1
   ```

3. Restart ComfyUI. Setup downloads and verifies about 2.19 GB of Q8 model files and builds a CUDA extension from a pinned llama.cpp revision. The default CUDA architecture is 120 (RTX 50 series). For another NVIDIA GPU, run `setup_worker_windows.ps1 -SkipNativeBuild` first, then `build_native_windows.ps1 -CudaArch <architecture>`.

The worker and weights live under Git-ignored `.runtime/` and `models/` in the node directory. For a manual junction into an existing ComfyUI checkout, run `scripts/install_comfy_windows.ps1 -ComfyDir 'C:\path\to\ComfyUI'`.

## Use

**Drag the [file workflow](workflows/confucius4_q8_file.json) onto the ComfyUI frontend canvas.** Select your own audio in `Load Audio`, then click **Run**. Drag in the [microphone workflow](workflows/confucius4_q8_live.json), click **Start microphone**, grant browser access, then click **Stop and finalize** and **Run** to save the final transcript. Browser microphone access requires localhost or HTTPS.

Nodes: `Confucius4 Q8 Loader`, `Confucius4 Transcribe`, `Confucius4 Live Microphone`, `Confucius4 Save Transcript`, and `Confucius4 Unload Q8`. File input supports one audio item at a time (batch=1). For mostly Chinese speech with some English, try the `Chinese` language hint in file-streaming mode.

The official Q8 files have been exercised on Windows / RTX 5090 Laptop / CUDA 12.8 for offline and streaming inference, frontend workflow drag-and-run, and a live frontend Start/Stop path with injected audio. Physical-microphone capture, accuracy on uninterrupted natural long speech, and other GPU architectures remain unverified. Forced segmentation is marked `requires_review`. Raw audio is not saved by default.

## Licenses and provenance

Node code is released under [Apache 2.0](LICENSE); native adapter provenance and changes are documented in [NOTICE](vendor/r2t2_native/NOTICE.md). The GGUF model and projector belong to NetEase Youdao and are governed by its [Model Use License Agreement](https://huggingface.co/t8star/Confucius-R2t2-Comfy/blob/main/MODEL_LICENSE). FireRedVAD assets are Apache 2.0. Read the applicable licenses before using or redistributing weights. This project is not officially affiliated with the model creators.

## Links

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI package](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
