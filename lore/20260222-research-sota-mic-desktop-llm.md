# 2026-02-22 — Research: State of the Art — Mic-to-Desktop-Control via LLMs

> Landscape survey of tools, libraries, and frameworks that enable voice-controlled
> desktop interaction powered by LLMs. Compiled to inform the voxmic roadmap.

---

## 1. Local / Offline Speech Recognition

The STT landscape has matured dramatically. Whisper variants dominate, but
specialized models now offer better speed/accuracy tradeoffs for specific use cases.

### 1.1 Whisper Family

| Variant | Params | WER (en) | Speed | License | Best For |
|---|---|---|---|---|---|
| **OpenAI Whisper large-v3-turbo** | 809M | ~3-5% | ~216x RT (GPU) | MIT | Multilingual batch |
| **whisper.cpp** (v1.8.3) | same | same | Real-time on edge | MIT | C/C++ cross-platform |
| **faster-whisper** (CTranslate2) | same | same | 4x faster than OAI | MIT | Fast Python GPU |
| **Distil-Whisper v3.5** | ~756M | within 1% of large-v3 | 6.3x faster | MIT | English-only speed |
| **MLX Whisper** | same | same | 30-40% faster on Apple Si | MIT | macOS native |
| **WhisperX** | same | same | 70x RT (batch) | BSD-4 | Alignment + diarization |

- **whisper.cpp v1.8.3** (Jan 2026): 12x iGPU boost via Vulkan; Metal improvements; XCFramework for iOS/macOS. Quantization: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.
- **faster-whisper**: Built-in Silero VAD; batched + quantized inference. Requires CUDA 12 + cuDNN 9 for latest.
- **Distil-Whisper v3.5**: Knowledge-distilled from large-v3. English-only, 49% smaller, 6x faster, within 1% WER.
- **WhisperX**: Adds forced alignment (wav2vec2) and speaker diarization (pyannote-audio) on top of faster-whisper. Batch-oriented.

### 1.2 Non-Whisper Models

| Model | Params | Speed | Streaming | License | Differentiator |
|---|---|---|---|---|---|
| **Vosk** | varies | 10-15% WER | Native | Apache 2.0 | Ultra-lightweight (50MB models), 20+ languages |
| **Moonshine v2** | 27-62M | Real-time on edge | Native | MIT (en) / Community (others) | Tiny models, 48% lower error than Whisper Tiny |
| **Silero VAD** | tiny | Sub-ms/chunk | Native | MIT | Industry-standard VAD, 6000+ language coverage |
| **Parakeet TDT v3** | 600M | RTFx >2000 | TDT | CC-BY-4.0 | Fastest GPU throughput, 25 European languages |
| **Voxtral Mini 4B RT** | 4B | <500ms stream | Native | Apache 2.0 | Sub-500ms streaming, 13 languages |
| **SenseVoice-Small** | ~234M | 70ms/10s audio | Via FunASR | Apache 2.0 | 15x faster than Whisper-Large, emotion detection |
| **Sherpa-ONNX** | varies | varies | Native | Apache 2.0 | Universal runtime, 12 language bindings, NPU support |

- **Moonshine v2** (Feb 2026): Ergodic streaming encoder with sliding-window attention. Three sizes: tiny, small, medium. ONNX format with memory-mapped loading.
- **Parakeet TDT v3**: FastConformer encoder + Token-and-Duration Transducer. 6.5x faster than NVIDIA Canary Qwen while maintaining competitive accuracy.
- **Voxtral Mini 4B Realtime** (Feb 2026): Configurable delay (240ms-2.4s). Only vLLM has native support; llama.cpp/llama-server does NOT yet support it (community Rust port exists).
- **SenseVoice**: Encoder-only model that also does language ID, emotion recognition, and audio event detection.
- **Sherpa-ONNX** (v1.12.25): Acts as universal deployment runtime for models from Whisper, NeMo, SenseVoice, Zipformer, etc. Runs on ESP32, RPi, phones, NPUs, and servers.

### 1.3 Streaming Wrappers

- **SimulStreaming** (2025, UFAL): 5x faster than its predecessor WhisperStreaming. Best-performing at IWSLT 2025. Uses AlignAtt policy for 2-4 second latency.
- **WhisperFlow**: Reduces per-word latency to ~0.5 seconds with negligible accuracy loss.

### 1.4 Leaderboard (Feb 2026)

| Model | WER | Speed | Note |
|---|---|---|---|
| NVIDIA Canary Qwen 2.5B | 5.63% | Moderate | #1 on HF Open ASR Leaderboard |
| IBM Granite Speech 3.3 8B | 5.85% | Moderate | Apache 2.0 |
| NVIDIA Parakeet TDT 0.6B | ~6% | RTFx >2000 | Fastest on the board |
| OpenAI Whisper large-v3 | 2.8% (clean) | Slow | Degrades to 15-25% with noise |

---

## 2. Voice-to-Desktop-Control Pipelines

### 2.1 Desktop Dictation Tools

| Tool | License | Platform | Processing | Differentiator |
|---|---|---|---|---|
| **Talon Voice** | Proprietary (free + paid) | Win/Mac/Linux | Local (Conformer) | Full hands-free: voice, eye tracking, noise triggers |
| **Superwhisper** | $8.49/mo | macOS/iOS | Local Whisper | AI formatting, 100+ languages, offline-first |
| **Wispr Flow** | $15/mo ($81M funded) | Win/Mac/iOS | Cloud only | Context-aware: screenshots + audio to cloud |
| **Buzz** | MIT | Win/Mac/Linux | Local (multiple) | GUI transcription app, Vulkan GPU support |
| **OpenWhispr** | MIT | Win/Mac/Linux | Local + cloud | BYOK cloud models, Parakeet/Whisper local |
| **Nerd Dictation** | GPL-2.0+ | Linux | Local (Vosk) | Minimal single-file Python, Wayland support |
| **RealtimeSTT** | MIT | Cross-platform | Local (faster-whisper) | Python library with VAD + wake words |

- **Talon Voice**: Core in Rust, scripted via embedded Python 3. Conformer model (paid beta) far better than free wav2letter. English-only for main model; multi-language via Vosk in beta.
- **Wispr Flow**: Screenshots the active window every few seconds and sends both audio + screen context to cloud. Powerful context awareness, but all data leaves the device.
- **RealtimeSTT**: Library (not standalone app). WebRTCVAD + SileroVAD + faster-whisper. Companion RealtimeTTS for audio I/O wrapping of LLMs.

### 2.2 Voice Coding

| Tool | License | Platform | Approach |
|---|---|---|---|
| **Talon + Cursorless** | MIT (Cursorless) | VS Code + Talon | AST-based structural editing via tree-sitter |
| **Serenade** | Apache 2.0 | Win/Mac/Linux | Natural language → code engine |
| **Continue.dev + Voice** | Apache 2.0 | VS Code/JetBrains | Voice as input to AI coding assistant |
| **VS Code Speech + Copilot** | Free (MS) | Win/Mac/Linux | Local STT → Copilot Chat |

- **Talon + Cursorless**: Most sophisticated voice coding. Uses colored "hat" decorations on tokens; composable marks + modifiers + actions. Steep learning curve but competitive editing speed.
- **Serenade**: Open-sourced Apache 2.0 but low community activity. Custom speech-to-code engine.
- **VS Code Speech**: Local STT → Copilot Chat prompt. The standalone Copilot extension is being deprecated and integrated directly into VS Code.

---

## 3. LLM-Powered Desktop Agents

| Agent | License | Approach | Key Strength |
|---|---|---|---|
| **Anthropic Computer Use** | Proprietary API | Screenshot → Claude → actions | Best accuracy; Claude Cowork for non-devs |
| **Open Interpreter** | AGPL-3.0 | NL → local code execution | 50K+ stars; multi-model via LiteLLM |
| **UFO3 Galaxy** (MS) | MIT | Dual GUI+API agent | Cross-device (Win/Linux/Android); Win32/COM APIs |
| **OmniParser V2** (MS) | AGPL (detect) / MIT (caption) | Screen → structured elements | Pairs any LLM into computer-use agent |
| **Self-Operating Computer** | MIT | Screenshot → mouse/keyboard | Multi-model: GPT-4o, Claude, Gemini, Qwen-VL |
| **ScreenAgent** | MIT (code) | Plan→Act→Reflect loop | Structured multi-step with reflection |
| **OS-Copilot / FRIDAY** | MIT | Self-improving agent | Curriculum-based self-learning; GAIA benchmark |

Key findings:
- **Anthropic Computer Use** scores nearly quintupled in 16 months. Claude Cowork (Jan 2026) provides a desktop app running in an isolated VM.
- **UFO3** (Nov 2025): 73K+ lines; combines visual GUI with direct Win32/COM API calls. Cross-device orchestration.
- **OmniParser V2**: 60% faster than V1; 39.6% on ScreenSpot Pro. Modular — turns any LLM into a computer-use agent.
- **OSWorld benchmark**: Humans 72.4% vs best agent ~29.9%. Large gap remains for end-to-end desktop task completion.

---

## 4. Multimodal (Voice + Vision) GUI Agents

| Model | Params | License | Differentiator |
|---|---|---|---|
| **CogAgent-9B** | 9B | Apache-2.0 (code) / CogVLM (model) | 1120x1120 resolution; screenshot-only |
| **ShowUI-2B** | 2B | Apache-2.0 | 75.1% zero-shot grounding; CVPR 2025 |
| **ShowUI-pi** | 450M | Apache-2.0 | Continuous drag trajectories in pixel space |
| **SeeClick** | — | Qwen-VL license | GUI grounding pre-training; ScreenSpot benchmark |
| **Ferret-UI Lite** | 3B | CC-BY-NC | On-device; matches models 24x larger |
| **WebVoyager** | — | Unspecified | Set-of-Mark web navigation; ACL 2024 |

- **ShowUI family** (Apache-2.0): Most permissively-licensed lightweight option. ShowUI-pi generates continuous cursor trajectories (drags/draws), not just clicks.
- **Ferret-UI Lite** (Feb 2026): Apple's 3B on-device agent. Matches much larger models but CC-BY-NC limits commercial use.

---

## 5. Voice Assistant Frameworks

| Framework | License | Platform | Key Feature |
|---|---|---|---|
| **Home Assistant Voice** | Apache 2.0 | Python/ESP32 | $13 hardware, Wyoming protocol, largest ecosystem |
| **OVOS** (OpenVoiceOS) | Apache 2.0 | Python | Mycroft successor, HiveMind distributed arch |
| **Rhasspy** | MIT | Python/Linux | Wyoming protocol creator, HA tech incubator |
| **Willow** | Apache 2.0 | ESP-IDF (C) | <500ms end-to-end on $50 ESP32-S3-BOX |
| **Leon AI** | MIT | Node.js/Python | Agentic rewrite with ReAct architecture underway |

- **Wyoming protocol** is becoming a de facto standard for voice component communication (STT ↔ intent ↔ TTS).
- **Home Assistant Voice Chapter 11** (Oct 2025): Multilingual assistants, dual wake words, Speech-to-Phrase for constrained grammar.

---

## 6. Voice Agent Orchestration

| Framework | License | Transport | Key Feature |
|---|---|---|---|
| **Pipecat** (Daily.co) | BSD 2-Clause | WebRTC | Frame-based pipeline, 40+ AI plugins |
| **LiveKit Agents** | Apache 2.0 | WebRTC | Room-based, sub-1s global latency, K8s native |
| **TEN Framework** (Agora) | Apache 2.0 | Agora RTC | Full-duplex, ms-level latency, extension graphs |
| **Vocode** | MIT | WebSocket/telephony | Low-level modular toolkit |
| **Retell AI** | Proprietary | Cloud | $40M+ ARR, 40M calls/month, drag-and-drop |

These orchestrate the STT → LLM → TTS pipeline for real-time conversational AI. They converge on WebRTC transport and composable pipeline architectures.

---

## 7. Speech-to-Speech Models

Native audio models that bypass the traditional STT → text LLM → TTS cascade.

| Model | License | Latency | Differentiator |
|---|---|---|---|
| **GPT-4o Realtime** | Proprietary | ~200ms | Native audio tokens, SIP/WebRTC, MCP support |
| **Gemini 2.5 Flash Native Audio** | Proprietary | Real-time | 70+ language translation, preserves intonation |
| **Moshi** (Kyutai, 7.6B) | Apache 2.0 / CC-BY-4.0 | ~200ms | Full-duplex, runs on consumer GPU (4-bit: 4GB) |
| **Ultravox** (Fixie.ai) | MIT | Fast | Audio → LLM projector, 42 languages, multi-base |
| **Qwen3-Omni** (Alibaba) | Permissive | Real-time | Thinker-Talker MoE, 30B-A3B |
| **Hume EVI 4** | Proprietary | ~600ms | Emotional intelligence, vocal modulation analysis |

- **Moshi**: 7.6B params, full-duplex (listens while speaking). Implementations in Python, Rust (candle), and MLX. 4-bit quantization fits in 4GB VRAM.
- **Ultravox v0.6**: Multimodal projector maps audio directly into LLM embedding space. Available on Llama 3.3 (8B/70B), Gemma3, Qwen3 bases.
- **Qwen-Audio family** iterated 3 major releases in 13 months: Qwen2-Audio → Qwen2.5-Omni → Qwen3-Omni.

---

## 8. Rust-Specific Options

### 8.1 Speech Recognition in Rust

| Crate | Version | License | Downloads/mo | Pure Rust? | Notes |
|---|---|---|---|---|---|
| **whisper-rs** | 0.15.1 | Unlicense | ~27.5K | No (whisper.cpp) | Most mature Whisper path |
| **vosk-rs** | 0.3.1 | MIT | ~2.2K | No (Vosk) | Offline, low resource |
| **sherpa-rs** | 0.6.8 | MIT | ~1.4K | No (sherpa-onnx) | STT+TTS+VAD+diarize; streaming |

### 8.2 ML Inference in Rust

| Crate | Version | License | Downloads/mo | Pure Rust? | Key Feature |
|---|---|---|---|---|---|
| **ort** | 2.0.0-rc.11 | MIT/Apache | ~822K | No (ONNX RT) | CUDA/TensorRT/DirectML/CoreML; fastest |
| **candle** | 0.9.2 | MIT/Apache | ~466K | Yes | Whisper/LLaMA/BERT; CUDA+Metal+WASM |
| **burn** | 0.21-pre | MIT/Apache | ~70K | Yes | Training+inference; CubeCL multi-backend |
| **tract** | 0.23-dev | MIT/Apache | ~38K | Yes | ONNX/NNEF; CPU-only; embedded-friendly |
| **llama-cpp-2** | 0.1.135 | MIT/Apache | ~27K | No (llama.cpp) | Tracks upstream daily; GGUF quantized |

### 8.3 Audio & Desktop Crates

| Crate | Version | License | Downloads/mo | Purpose |
|---|---|---|---|---|
| **cpal** | 0.17.3 | Apache-2.0 | ~666K | Audio I/O (WASAPI/CoreAudio/ALSA) |
| **rubato** | 1.0.1 | MIT | ~444K | Sample rate conversion |
| **rodio** | 0.22.1 | MIT/Apache | ~406K | Audio playback |
| **symphonia** | 0.5.5 | MPL-2.0 | ~376K | Audio decoding (pure Rust) |
| **tray-icon** | 0.21.3 | MIT/Apache | ~1.08M | System tray (Tauri ecosystem) |
| **global-hotkey** | 0.7.0 | MIT/Apache | ~239K | Global hotkeys (X11 only on Linux) |
| **enigo** | 0.6.1 | MIT | ~62K | Input simulation (Win/Mac/Linux) |
| **nnnoiseless** | 0.5.2 | BSD-3 | ~17K | RNNoise port for noise suppression |
| **rdev** | 0.5.3 | MIT | ~22K | Input listening (unmaintained; use rdevin) |

### 8.4 Native Rust Whisper (No C++ Dependencies)

Three paths exist:

1. **candle Whisper** (Hugging Face) — **Recommended**. Full Whisper implementation in pure Rust. All models (tiny→large-v3) + Distil-Whisper. ~3x real-time on CPU, faster on CUDA. 22MB binary. Actively maintained.

2. **whisper-burn** — Functional but less maintained (342 stars, 49 commits). wgpu backend unstable for large models. Model conversion requires Python.

3. **rusty-whisper** (tract + ONNX) — Proof-of-concept only. No KV-caching (4x slower). tract operator gaps may block newer models.

**Verdict**: candle is the viable pure-Rust path today. For production quality, whisper-rs (C++ FFI) is more battle-tested.

---

## 9. Architectural Patterns

### 9.1 Streaming Pipelines

The dominant architecture for voice-controlled desktop apps:

```
Mic → VAD → STT (streaming) → LLM (function calling) → Action/TTS
         ↑                            ↓
    Silero VAD              enigo (typing) / OS APIs
```

Key patterns:
- **VAD-gated recording**: Silero VAD detects speech start/stop, avoiding processing silence
- **Chunked streaming**: Feed audio in 200-500ms chunks to streaming STT
- **Function calling**: LLM emits structured tool calls (type text, click, press key) rather than freeform text

### 9.2 Edge Inference

For single-binary desktop apps like voxmic:
- **Model-as-resource**: Embed quantized model in binary or download-on-first-run
- **GGUF quantization**: Q4/Q5 models reduce Whisper large-v3 from 3GB to ~1.5GB
- **iGPU acceleration**: whisper.cpp Vulkan backend now 12x faster on integrated GPUs
- **Separation of concerns**: Thin local client + local inference server (llama-server pattern)

### 9.3 Privacy-First Design

- All audio processed locally; never leaves the device
- Temp files deleted immediately after transcription
- No persistent audio storage
- Loopback-only network access (127.0.0.1)

---

## 10. Recommendations for voxmic

### 10.1 Current Architecture (Keep)

voxmic's existing design — Ctrl+Win hotkey → cpal audio → HTTP POST to llama-server → enigo typing — is sound and matches the "thin client + local server" pattern used by the most successful tools. **Keep this architecture.**

### 10.2 STT Backend Priority

| Priority | Backend | Rationale |
|---|---|---|
| **P0 (current)** | Voxtral Mini via llama-server | Already working; good accuracy |
| **P1 (next)** | whisper-rs behind `TranscriptionBackend` trait | Most mature Rust FFI; all Whisper models; CUDA/ROCm |
| **P2 (future)** | candle native Whisper | Pure Rust; no cmake; 22MB binary; eliminates whisper.cpp build step |
| **P3 (watch)** | sherpa-rs | Streaming STT + VAD + TTS in one package; rapidly improving |

### 10.3 Streaming Path

When voxmic adds streaming transcription:
1. **Silero VAD** (via sherpa-rs or ONNX) for speech detection
2. **Moonshine v2** or **Voxtral Mini 4B Realtime** for low-latency streaming STT
3. Feed partial transcripts to user as they speak (like Superwhisper/Wispr Flow)

### 10.4 LLM-Powered Commands (Future)

To evolve from dictation to voice-controlled desktop:
1. **Detect intent**: Is this dictation text or a command? (lightweight classifier or LLM)
2. **Function calling**: Route commands to structured actions via LLM tool use
3. **OmniParser** (if screen context needed): Screen → structured elements → LLM reasoning
4. **Keep it local**: Moshi (4GB, full-duplex) or Ultravox (MIT, multi-base) for on-device speech-to-speech

### 10.5 Noise Suppression

Add **nnnoiseless** (pure Rust RNNoise port) before WAV encoding. Input must be 48kHz mono — add rubato for sample rate conversion if cpal captures at a different rate.

### 10.6 Crate Versions to Target

```toml
# Current plan (from Rust design lore) — validated, versions updated
cpal          = "0.17"    # Audio I/O
tray-icon     = "0.21"    # System tray
global-hotkey = "0.7"     # Hotkey
enigo         = "0.6"     # Text injection
ureq          = "2"       # HTTP to llama-server

# Phase 2 additions
whisper-rs    = "0.15"    # Local Whisper fallback (feature-gated)
nnnoiseless   = "0.5"     # Noise suppression
rubato        = "1.0"     # Sample rate conversion

# Phase 3 (pure Rust path)
candle-core           = "0.9"     # Native Whisper inference
candle-transformers   = "0.9"     # Whisper model implementation
```

### 10.7 What NOT to Build

- **Don't build a voice coding grammar** — Talon + Cursorless is years ahead and niche
- **Don't build a voice assistant framework** — OVOS/HA Voice own that space
- **Don't build a real-time orchestration layer** — Pipecat/LiveKit are battle-tested
- **Do focus on**: single-binary dictation → gradually add command detection → LLM tool use

---

## Sources

### Speech Recognition
- [OpenAI Whisper](https://github.com/openai/whisper) — [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Distil-Whisper](https://github.com/huggingface/distil-whisper) — [MLX Whisper](https://pypi.org/project/mlx-whisper/) — [WhisperX](https://github.com/m-bain/whisperX)
- [Vosk](https://alphacephei.com/vosk/) — [Moonshine](https://github.com/moonshine-ai/moonshine) — [Silero VAD](https://github.com/snakers4/silero-vad)
- [NVIDIA Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — [Voxtral Mini Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)
- [SimulStreaming](https://github.com/ufal/SimulStreaming) — [Northflank STT Benchmarks 2026](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)

### Desktop Control & Coding
- [Talon Voice](https://talonvoice.com/) — [Cursorless](https://github.com/cursorless-dev/cursorless) — [Serenade](https://serenade.ai/)
- [Superwhisper](https://superwhisper.com/) — [Wispr Flow](https://wisprflow.ai/) — [Buzz](https://github.com/chidiwilliams/buzz)
- [Nerd Dictation](https://github.com/ideasman42/nerd-dictation) — [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) — [OpenWhispr](https://github.com/OpenWhispr/openwhispr)
- [Continue.dev](https://github.com/continuedev/continue) — [VS Code Speech](https://github.com/microsoft/vscode/wiki/VS-Code-Speech)

### LLM Desktop Agents
- [Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) — [Open Interpreter](https://github.com/openinterpreter/open-interpreter)
- [UFO3](https://github.com/microsoft/UFO) — [OmniParser V2](https://github.com/microsoft/OmniParser)
- [Self-Operating Computer](https://github.com/OthersideAI/self-operating-computer) — [ScreenAgent](https://github.com/niuzaisheng/ScreenAgent)
- [OS-Copilot](https://github.com/OS-Copilot/OS-Copilot) — [ShowUI](https://github.com/showlab/ShowUI) — [Ferret-UI](https://github.com/apple/ml-ferret)

### Voice Frameworks & Orchestration
- [Home Assistant Voice](https://www.home-assistant.io/voice_control/) — [OVOS](https://openvoiceos.org/) — [Rhasspy](https://github.com/rhasspy)
- [Willow](https://github.com/HeyWillow/willow) — [Leon AI](https://github.com/leon-ai/leon)
- [Pipecat](https://github.com/pipecat-ai/pipecat) — [LiveKit Agents](https://github.com/livekit/agents) — [TEN Framework](https://github.com/TEN-framework/ten-framework)

### Speech-to-Speech Models
- [GPT-4o Realtime](https://platform.openai.com/docs/models/gpt-realtime) — [Gemini Native Audio](https://ai.google.dev/gemini-api/docs/live)
- [Moshi](https://github.com/kyutai-labs/moshi) — [Ultravox](https://github.com/fixie-ai/ultravox)
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) — [Hume EVI](https://www.hume.ai/empathic-voice-interface)

### Rust Ecosystem
- [whisper-rs](https://github.com/tazz4843/whisper-rs) — [sherpa-rs](https://github.com/thewh1teagle/sherpa-rs) — [vosk-rs](https://github.com/Bear-03/vosk-rs)
- [candle](https://github.com/huggingface/candle) — [burn](https://github.com/tracel-ai/burn) — [tract](https://github.com/sonos/tract) — [ort](https://github.com/pykeio/ort)
- [cpal](https://github.com/RustAudio/cpal) — [rodio](https://github.com/RustAudio/rodio) — [rubato](https://github.com/HEnquist/rubato) — [symphonia](https://github.com/pdeljanov/Symphonia)
- [nnnoiseless](https://github.com/jneem/nnnoiseless) — [enigo](https://github.com/enigo-rs/enigo) — [tray-icon](https://github.com/tauri-apps/tray-icon) — [global-hotkey](https://github.com/tauri-apps/global-hotkey)
- [llama-cpp-2](https://crates.io/crates/llama-cpp-2) — [whisper-burn](https://github.com/Gadersd/whisper-burn)
