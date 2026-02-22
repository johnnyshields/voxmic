# 2026-02-22 — voxctrl: Cross-Platform Rust Tray Dictation App (Design)

## Overview

Design discussion for **voxctrl** — a rewrite of the Python `tray_app.py` dictation
app in Rust, targeting a single self-contained binary (`voxctrl.exe` on Windows,
`voxctrl` on macOS/Linux) with no runtime dependencies.

The app listens for a global **Ctrl+Win** hotkey (hold to record, release to
transcribe), sends recorded audio to a running **llama-server** instance running
**Voxtral Mini** via HTTP, and types the transcribed text at the cursor in whatever
window was active when the hotkey fired.

---

## Key Architectural Decisions

### 1. Rust over Python
| Factor | Python (`tray_app.py`) | Rust (`voxctrl`) |
|---|---|---|
| Distribution | Needs Python + 6 pip packages | Single binary, zero runtime |
| Startup time | ~2–3 s (VM + imports) | < 50 ms |
| Memory overhead | ~150 MB | ~30 MB |
| Windows API access | ctypes hacks | First-class via `windows` crate |
| Scheduled task | Must locate `python.exe` | Just point at `voxctrl.exe` |

**Rationale:** The end goal is a clean MSI/pkg installer with no external
prerequisites. A single Rust binary makes that trivial.

### 2. Voxtral-only backend (no whisper-rs / whisper.cpp)
`whisper-rs` is bindings to **whisper.cpp** and has no relation to Voxtral Mini.
Voxtral is served by **llama-server** and consumed over HTTP
(`POST /v1/audio/transcriptions`). This means:

- No C++ toolchain or cmake required
- No whisper.cpp build step (~5 min on first compile)
- Backend is a single `ureq::post(...)` call
- llama-server is already managed by the `VoxtralLLM` scheduled task

`whisper-rs` can be added later as a local/offline fallback if needed, behind a
feature flag.

### 3. Tauri ecosystem crates for tray + hotkey
`tray-icon`, `global-hotkey`, and `muda` are all maintained by the Tauri project
and designed to work together. They are the closest thing Rust has to a
"blessed" cross-platform tray stack, without pulling in a full webview framework.

### 4. winit event loop on main thread
`winit` (required by `tray-icon`) must run on the main thread on macOS. All
platform-specific tray-icon event processing also benefits from this. Audio,
hotkey callbacks, and transcription all run on background threads and communicate
back via `Arc<Mutex<AppState>>`.

### 5. enigo for cross-platform text injection
`enigo` supports Windows `SendInput`, macOS CGEvent, and Linux X11/Wayland. It
is preferred over platform-specific ctypes hacks used in the Python version.

---

## Planned Crate Stack

```toml
[dependencies]
# Tray + menu + hotkey (Tauri ecosystem — designed to work together)
tray-icon     = "0.19"
global-hotkey = "0.6"
winit         = "0.30"
muda          = "0.15"

# Audio capture
cpal          = "0.15"   # WASAPI (Win) / CoreAudio (Mac) / ALSA (Linux)
hound         = "3.5"    # WAV encode/decode

# Voxtral backend
ureq          = { version = "2", features = ["json"] }

# Text injection
enigo         = "0.2"

# Config
serde         = { version = "1", features = ["derive"] }
serde_json    = "1"
```

All pure Rust. No C/C++ build steps. Compiles on all three platforms.

---

## Threading Model

```
main thread        winit EventLoop
                   ├── tray-icon events  (menu open/click)
                   └── global-hotkey events
                         ├── Ctrl+Win DOWN → set state = RECORDING
                         └── Win UP        → set state = TRANSCRIBING
                                             spawn transcription thread

audio thread       cpal InputStream (always open, low latency)
                   └── if state == RECORDING → push chunk to
                         Arc<Mutex<Vec<f32>>>

transcription      spawned per utterance
thread             ├── drain chunk buffer
                   ├── hound::write WAV to tempfile
                   ├── ureq POST to llama-server
                   └── enigo::Keyboard::text(result)
                         → set state = IDLE
                         → update tray icon colour
```

State transitions: `IDLE → RECORDING → TRANSCRIBING → IDLE`

---

## Platform-Specific Considerations

### Windows
- Everything works out of the box
- Target: `x86_64-pc-windows-msvc`
- Output: `voxctrl.exe` (single file, no DLLs needed for pure-Rust build)

### macOS
- `winit` event loop **must** run on the main thread (enforced by AppKit)
- Global hotkey + `enigo` require **Accessibility permission** — standard
  prompt on first launch (same as Superwhisper, Whisper for Mac, etc.)
- Distribution: `.app` bundle via `cargo-bundle`; code signing needed for
  Gatekeeper

### Linux
- `tray-icon` requires **`libappindicator3`** at runtime
  (`sudo apt install libappindicator3-1`)
- Global hotkeys work on **X11** only; **Wayland** blocks them by design
  (security model). Mitigation: offer a tray-menu "Start recording" button
  as a Wayland-safe alternative to the hotkey
- `enigo` supports X11 and partial Wayland

---

## Files to Create (future implementation)

| File | Purpose |
|---|---|
| `voxctrl/src/main.rs` | Entry point, winit event loop, state machine |
| `voxctrl/src/config.rs` | Load/save `config.json` (serde) |
| `voxctrl/src/audio.rs` | cpal InputStream, always-open stream, chunk buffer |
| `voxctrl/src/hotkey.rs` | global-hotkey setup, Ctrl+Win detection |
| `voxctrl/src/backend/mod.rs` | `TranscriptionBackend` trait |
| `voxctrl/src/backend/voxtral.rs` | ureq HTTP POST to llama-server |
| `voxctrl/src/tray.rs` | tray-icon + muda menu, icon image generation |
| `voxctrl/src/typing.rs` | enigo text injection, window re-focus |
| `voxctrl/Cargo.toml` | Workspace manifest |

---

## Relationship to Existing Python App

`tray_app.py` remains functional and is the reference implementation.
`voxctrl` is a ground-up Rust rewrite with the same UX contract:

- Same `config.json` schema (adds no new fields)
- Same Voxtral HTTP endpoint (`127.0.0.1:5200`)
- Same hotkey (Ctrl+Win)
- Same tray icon colours (green/red/amber)
- `setup.ps1` `VoxtralMic` task can point at `voxctrl.exe` once built

---

## Security Considerations

- No network access except `127.0.0.1:5200` (loopback only, llama-server)
- Audio is written to a temp file, transcribed, then immediately deleted
- No audio is stored persistently
- `enigo` uses OS-level `SendInput` / CGEvent — same trust level as the
  active user session; no privilege escalation

---

## Testing Approach

No automated tests defined yet. Planned:

- Unit test `config.rs` load/save round-trip
- Mock `TranscriptionBackend` trait for hotkey → type integration test
- Manual verification checklist (same as `tray_app.py` plan):
  1. Binary starts → green tray icon visible
  2. Hold Ctrl+Win → icon turns red
  3. Speak, release Win → icon turns amber, then green
  4. Text appears in focused window
  5. Menu: backend switch warns if llama-server not running
  6. macOS/Linux: Accessibility/X11 permission flow works

---

## Future Enhancements

- **whisper-rs feature flag** — local offline fallback (requires cmake/MSVC,
  disabled by default)
- **MSI installer (Windows)** — `cargo-wix` or WiX toolset; `voxctrl.exe` +
  scheduled task registration
- **macOS pkg** — `cargo-bundle` + `pkgbuild`
- **Wayland hotkey workaround** — tray menu "Hold to dictate" button using
  `global-hotkey`'s portal-based backend when available
- **Voice activity detection** — run lightweight VAD in the audio thread to
  auto-stop recording on silence (mirrors `transcribe.py` behaviour)
- **Noise suppression** — RNNoise via `nnnoiseless` crate before WAV write
- **Streaming transcription** — chunked POST to llama-server as audio arrives
  (requires server-sent events support)
