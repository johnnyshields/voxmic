# 2026-02-22 — voxctrl: Pluggable Voice-to-Action Pipeline (Design)

## Overview

Architecture design for **voxctrl** — extending the Rust tray dictation app into a
pluggable voice-to-action pipeline. Each stage of the pipeline is defined by a trait
with multiple implementations selectable at compile-time (feature flags) and runtime
(config.json).

This supersedes the earlier design lore (20260222-1555) for file layout and crate
stack while preserving its core decisions (Rust, Tauri ecosystem, cpal, enigo,
winit main thread).

---

## Pipeline Architecture

```
Mic (cpal) → [VAD] → [STT] → [Router] → [Action]
               ↑        ↑        ↑          ↑
           pluggable  pluggable  pluggable  pluggable
```

Each stage is a Rust trait. Implementations are gated behind Cargo feature flags.
At runtime, `config.json` selects which compiled-in backend to use.

### Stage Traits

```rust
// vad/mod.rs
pub trait VoiceDetector: Send + Sync {
    /// Returns true if the audio chunk contains speech.
    fn is_speech(&mut self, samples: &[f32], sample_rate: u32) -> bool;
    fn name(&self) -> &str;
}

// stt/mod.rs
pub trait Transcriber: Send + Sync {
    /// Transcribe a WAV file, return text.
    fn transcribe(&self, wav_path: &std::path::Path) -> anyhow::Result<String>;
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
}

// router/mod.rs
pub enum Intent {
    Dictate(String),              // Type this text
    Command { action: String, args: serde_json::Value },  // Execute action
}

pub trait IntentRouter: Send + Sync {
    fn route(&self, text: &str) -> anyhow::Result<Intent>;
    fn name(&self) -> &str;
}

// action/mod.rs
pub trait ActionExecutor: Send + Sync {
    fn execute(&self, intent: &Intent) -> anyhow::Result<()>;
    fn name(&self) -> &str;
}
```

---

## Feature Flags

```toml
[features]
default = ["stt-voxtral-http", "vad-energy"]

# STT backends
stt-voxtral-http   = ["ureq"]                              # HTTP to llama-server
stt-whisper-cpp    = ["whisper-rs"]                         # FFI, needs cmake
stt-whisper-native = ["candle-core", "candle-transformers", "candle-nn", "hf-hub", "tokenizers"]

# VAD
vad-energy = []                                             # Built-in threshold
vad-silero = ["ort"]                                        # ONNX Runtime

# Router
router-llm = ["stt-voxtral-http"]                           # LLM-based intent classification

# Action
action-computer-use = []                                     # Screen → LLM → OS actions (Phase 3)

# GPU acceleration (opt-in, cross-cutting)
cuda = ["whisper-rs?/cuda", "ort?/cuda", "candle-core?/cuda"]
```

### Default Build (Phase 1)

```bash
cargo build                    # ~5MB binary, zero C deps
```

Includes: `stt-voxtral-http` + `vad-energy` + `PassthroughRouter` + `TypeText`

### Full Build (Phase 2+)

```bash
cargo build --features stt-whisper-cpp,vad-silero,cuda   # ~50MB, needs cmake + CUDA
```

---

## File Layout

```
voxctrl/
├── Cargo.toml
├── src/
│   ├── main.rs                # Entry point, winit event loop, state machine
│   ├── config.rs              # Serde config, runtime backend selection
│   ├── pipeline.rs            # Pipeline struct wiring stages together
│   ├── audio.rs               # cpal InputStream, chunk buffer, noise suppression
│   ├── vad/
│   │   ├── mod.rs             # VoiceDetector trait + create_vad() factory
│   │   └── energy.rs          # EnergyVAD (threshold-based, always available)
│   ├── stt/
│   │   ├── mod.rs             # Transcriber trait + create_transcriber() factory
│   │   └── voxtral_http.rs    # HTTP POST to llama-server (default)
│   ├── router/
│   │   ├── mod.rs             # IntentRouter trait + Intent enum + factory
│   │   └── passthrough.rs     # PassthroughRouter (everything is dictation)
│   ├── action/
│   │   ├── mod.rs             # ActionExecutor trait + factory
│   │   └── type_text.rs       # Enigo typing + window re-focus
│   ├── tray.rs                # tray-icon + muda menu, icon generation
│   └── hotkey.rs              # global-hotkey Ctrl+Win detection
```

Phase 2 additions (feature-gated, not in default build):

```
│   ├── vad/
│   │   └── silero.rs          # SileroVAD via ort (feature: vad-silero)
│   ├── stt/
│   │   ├── whisper_cpp.rs     # whisper-rs FFI (feature: stt-whisper-cpp)
│   │   └── whisper_native.rs  # candle pure Rust (feature: stt-whisper-native)
│   ├── router/
│   │   └── llm.rs             # LLM classifier (feature: router-llm)
│   └── action/
│       └── computer_use.rs    # Screen parse → LLM → actions (feature: action-computer-use)
```

---

## Config Schema

```json
{
  "stt": {
    "backend": "voxtral-http",
    "voxtral_url": "http://127.0.0.1:5200",
    "whisper_model": "small",
    "whisper_device": "cpu",
    "whisper_compute_type": "int8"
  },
  "vad": {
    "backend": "energy",
    "energy_threshold": 0.015,
    "silero_threshold": 0.5
  },
  "router": {
    "backend": "passthrough"
  },
  "action": {
    "backend": "type-text"
  },
  "audio": {
    "device_pattern": "DJI",
    "sample_rate": 16000,
    "chunk_duration_ms": 100
  },
  "hotkey": {
    "modifier": "ctrl",
    "trigger": "win"
  }
}
```

Backwards-compatible with existing `config.json` — the config loader falls back
to flat keys (`"backend"`, `"device_pattern"`, etc.) if nested keys are absent.

---

## Threading Model (unchanged from prior design)

```
main thread        winit EventLoop
                   ├── tray-icon events  (menu open/click)
                   └── global-hotkey events
                         ├── Ctrl+Win DOWN → set state = RECORDING
                         └── Win UP        → set state = TRANSCRIBING
                                             spawn transcription thread

audio thread       cpal InputStream (always open, low latency)
                   └── if state == RECORDING → push chunk to buffer
                       if VAD enabled → also check is_speech()

transcription      spawned per utterance
thread             ├── drain chunk buffer → WAV tempfile
                   ├── pipeline.transcribe(wav_path)
                   ├── pipeline.route(text)
                   └── pipeline.execute(intent)
                         → set state = IDLE, update tray icon
```

State machine: `IDLE → RECORDING → TRANSCRIBING → IDLE`

The Pipeline struct holds `Arc<dyn Transcriber>`, `Arc<dyn IntentRouter>`,
`Arc<dyn ActionExecutor>`. VAD runs inline in the audio callback.

---

## Crate Stack

```toml
[dependencies]
# Core (always included)
anyhow        = "1"
log           = "0.4"
env_logger    = "0.11"
serde         = { version = "1", features = ["derive"] }
serde_json    = "1"
dirs          = "6"

# Tray + hotkey (Tauri ecosystem)
tray-icon     = "0.21"
global-hotkey = "0.7"
winit         = { version = "0.30", default-features = false, features = ["rwh_06"] }
muda          = "0.15"

# Audio
cpal          = "0.17"
hound         = "3.5"

# Text injection
enigo         = "0.6"

# STT: voxtral-http (default)
ureq          = { version = "2", features = ["json"], optional = true }

# STT: whisper-cpp (optional)
whisper-rs    = { version = "0.15", optional = true }

# STT: candle native (optional)
candle-core          = { version = "0.9", optional = true }
candle-nn            = { version = "0.9", optional = true }
candle-transformers  = { version = "0.9", optional = true }
hf-hub               = { version = "0.4", optional = true }
tokenizers           = { version = "0.21", optional = true }

# VAD: silero (optional)
ort           = { version = "2.0.0-rc.11", optional = true }

# Utility
tempfile      = "3"
```

---

## Factory Pattern for Runtime Selection

Each module has a `create_*()` factory function that reads config and returns
a `Box<dyn Trait>`:

```rust
// stt/mod.rs
pub fn create_transcriber(cfg: &config::SttConfig) -> anyhow::Result<Box<dyn Transcriber>> {
    match cfg.backend.as_str() {
        "voxtral-http" => {
            #[cfg(feature = "stt-voxtral-http")]
            return Ok(Box::new(voxtral_http::VoxtralHttpTranscriber::new(cfg)));
            #[cfg(not(feature = "stt-voxtral-http"))]
            anyhow::bail!("stt-voxtral-http feature not compiled in");
        }
        "whisper-cpp" => {
            #[cfg(feature = "stt-whisper-cpp")]
            return Ok(Box::new(whisper_cpp::WhisperCppTranscriber::new(cfg)?));
            #[cfg(not(feature = "stt-whisper-cpp"))]
            anyhow::bail!("stt-whisper-cpp feature not compiled in");
        }
        "whisper-native" => {
            #[cfg(feature = "stt-whisper-native")]
            return Ok(Box::new(whisper_native::WhisperNativeTranscriber::new(cfg)?));
            #[cfg(not(feature = "stt-whisper-native"))]
            anyhow::bail!("stt-whisper-native feature not compiled in");
        }
        other => anyhow::bail!("Unknown STT backend: {other}"),
    }
}
```

Same pattern for VAD, router, and action modules.

---

## Phase Plan

### Phase 1 — Parity with Python app (this implementation)

Build the full pipeline scaffold with default backends:
- `EnergyVAD` — threshold-based speech detection
- `VoxtralHttpTranscriber` — HTTP POST to llama-server
- `PassthroughRouter` — all text is dictation
- `TypeTextAction` — enigo types at cursor

Same UX as `tray_app.py`: hold Ctrl+Win, speak, release, text appears.
Single binary, no cmake, no C dependencies.

### Phase 2 — Better STT + noise suppression

- Add `SileroVAD` via ort (ONNX model, ~2MB)
- Add `nnnoiseless` noise suppression in audio pipeline
- Add `WhisperCppTranscriber` (feature-gated, needs cmake)
- Add `WhisperNativeTranscriber` via candle (pure Rust)
- CUDA feature flag for GPU acceleration

### Phase 3 — Voice commands

- `LlmRouter` — LLM function-calling to classify dictation vs commands
- `ComputerUseAction` — screenshot → OmniParser → LLM → enigo/OS actions
- Additional action types: open URL, run command, press hotkey

---

## Testing Strategy

### Unit Tests
- `config.rs`: load/save round-trip, backwards compatibility with flat config
- `vad/energy.rs`: silence vs speech detection with synthetic audio
- `router/passthrough.rs`: always returns `Intent::Dictate`
- `stt/voxtral_http.rs`: mock HTTP server response parsing

### Integration Tests
- `pipeline.rs`: mock all stages, verify wiring
- Full pipeline with `EnergyVAD` + mock `Transcriber` + `PassthroughRouter` + mock `ActionExecutor`

### Manual Verification (same as prior design)
1. Binary starts → green tray icon visible
2. Hold Ctrl+Win → icon turns red
3. Speak, release Win → icon turns amber, then green
4. Text appears in focused window
5. Menu: backend switch warns if server not running

---

## Security (unchanged)

- Loopback-only network access (127.0.0.1)
- Audio temp files deleted immediately after transcription
- No persistent audio storage
- enigo uses OS-level SendInput/CGEvent — same trust level as user session
