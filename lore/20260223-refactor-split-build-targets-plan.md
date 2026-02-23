# Refactor: Split Build Targets into Cargo Workspace

**Date:** 2026-02-23
**Branch:** refactor-split-build-targets/supervisor

## Goal

Split the monolithic voxctrl package into a Cargo workspace so GUI changes don't trigger recompilation of heavy ML inference dependencies (candle, burn, whisper-rs).

## Architecture

3-crate workspace:
- **voxctrl-core** (lib) — config, pipeline, audio, recording, VAD, router, action, models, IPC, lightweight STT (voxtral-http only)
- **voxctrl-stt** (lib) — heavy STT backends: whisper-native (candle), whisper-cpp (FFI), voxtral-native (burn)
- **voxctrl** (bin) — main.rs, GUI (tray, hotkey, egui settings), TUI

## Key Design Decision: Extra Factory Pattern

The `voxctrl-stt` crate injects its heavy backends into `voxctrl-core`'s STT factory via a callback parameter:

```rust
// voxctrl-core::stt::create_transcriber() accepts an optional extra_factory
// voxctrl-stt::stt_factory() implements it for heavy backends
// main.rs passes Some(&voxctrl_stt::stt_factory) when building the pipeline
```

This avoids circular dependencies: core defines the `Transcriber` trait, stt implements it, binary wires them together.

## Build Speed Impact

- GUI change → recompiles binary crate only (seconds)
- Core logic change → recompiles core + binary (no ML deps)
- STT change → recompiles stt + binary (core stays cached)
