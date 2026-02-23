# voxctrl

Real-time voice-to-action pipeline for Windows. Captures mic audio, detects speech, transcribes locally, and routes text to actions (type-to-screen, LLM commands, computer use).

## Pipeline

```
Mic → VAD → STT → Router → Action
```

| Stage | Backends |
|-------|----------|
| **Audio** | Any WASAPI input device (configurable pattern match) |
| **VAD** | Energy (RMS threshold), Silero ONNX |
| **STT** | Whisper (pure Rust/candle), Voxtral (HTTP or native), Whisper.cpp |
| **Router** | Passthrough, LLM |
| **Action** | Type text (enigo), Computer use |

## Architecture

Cargo workspace with three crates:

| Crate | Purpose |
|-------|---------|
| `voxctrl-core` | Config, pipeline, audio capture, VAD, lightweight STT backends, model management |
| `voxctrl-stt` | Heavy ML inference backends (whisper-native via candle, voxtral-native) |
| `voxctrl` | Binary — GUI (system tray + egui Settings window) and TUI modes |

The Settings window runs as a subprocess. It communicates with the main tray app via a Windows named pipe (`voxctrl-stt`) for STT test requests.

## Requirements

- Windows 10/11
- [Rust toolchain](https://rustup.rs/) (for building from source)

## Install

Download the MSI installer from Releases, or build from source:

```bash
cargo build --release --target x86_64-pc-windows-gnu
```

The MSI can be built with [msitools](https://wiki.gnome.org/msitools):

```bash
wixl --ext ui -o target/voxctrl-0.2.0-x86_64.msi wix/main.wxs
```

## Usage

Run `voxctrl.exe`. It starts in **GUI mode** by default:

- System tray icon with menu (toggle listening, open Settings, quit)
- Global hotkey `Ctrl+Win+Space` to toggle mic on/off
- Settings window for configuring all pipeline stages and managing models

For terminal mode:

```
voxctrl.exe --tui
```

## Config

`config.json` lives next to the executable. Nested format with per-stage sections:

```json
{
  "stt": {
    "backend": "voxtral-http",
    "voxtral_url": "http://127.0.0.1:5200",
    "whisper_model": "small",
    "whisper_device": "cpu"
  },
  "vad": {
    "backend": "energy",
    "energy_threshold": 0.015
  },
  "audio": {
    "device_pattern": "DJI",
    "sample_rate": 16000
  },
  "hotkey": {
    "shortcut": "Ctrl+Super+Space"
  },
  "models": {
    "models_directory": null
  }
}
```

All fields are optional — missing values use defaults. Legacy flat configs are auto-migrated.

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `gui` | yes | System tray + egui Settings window |
| `tui` | yes | Terminal UI (ratatui) |
| `stt-voxtral-http` | yes | Voxtral HTTP backend |
| `stt-whisper-native` | yes | Pure Rust Whisper (candle) |
| `stt-voxtral-native` | yes | Native Voxtral inference |
| `stt-whisper-cpp` | no | Whisper.cpp via whisper-rs bindings |
| `vad-energy` | yes | RMS energy VAD |
| `vad-silero` | no | Silero ONNX VAD |
| `cuda` | no | GPU acceleration |

## Logs

GUI mode writes to `voxctrl.log` next to the executable (no console window attached).
