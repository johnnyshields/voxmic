# 2026-02-22 — voxctrl Rust Implementation Summary

## What was built

Rewrote the Python `tray_app.py` dictation app as a Rust binary in `voxctrl/`.
9 files, ~500 lines of Rust, zero C/C++ build dependencies.

## Files created

| File | Purpose | Lines |
|------|---------|-------|
| `voxctrl/Cargo.toml` | Manifest with all dependencies | 30 |
| `voxctrl/src/main.rs` | Entry point, winit event loop, shared types | 143 |
| `voxctrl/src/config.rs` | Config struct + serde load/save from config.json | 125 |
| `voxctrl/src/backend/mod.rs` | `TranscriptionBackend` trait | 11 |
| `voxctrl/src/backend/voxtral.rs` | Voxtral HTTP multipart POST via ureq | 78 |
| `voxctrl/src/audio.rs` | cpal always-open audio stream | 92 |
| `voxctrl/src/typing.rs` | enigo text injection | 17 |
| `voxctrl/src/hotkey.rs` | global-hotkey Ctrl+Super+Space toggle | 117 |
| `voxctrl/src/tray.rs` | tray-icon + muda menu + generated circle icons | 64 |

## Key decisions made during implementation

1. **Ctrl+Win+Space toggle** instead of Ctrl+Win hold-to-record — `global-hotkey` crate
   only supports key-down events, not key-up. Toggle: first press starts recording,
   second press stops and transcribes.

2. **Manual multipart/form-data** for ureq v2 — ureq doesn't have built-in multipart
   support, so the boundary/Content-Disposition is constructed manually in voxtral.rs.

3. **Cross-compile target**: `x86_64-pc-windows-gnu` — compiles in WSL2 with mingw
   linker. Native Linux check fails because we lack `libasound2-dev`, `libgtk-3-dev`,
   `libxdo-dev` system packages (no sudo access).

4. **Dead code warnings** are expected — `save_config()`, `update_tray_icon()`,
   `is_available()`, `name()`, `with_url()` are public API for future use.

## Verification

- `cargo check --target x86_64-pc-windows-gnu` passes cleanly (4 dead_code warnings only)
- Config schema matches existing `config.json`
- State machine: Idle → Recording → Transcribing → Idle

## What's next

- Build and test on actual Windows (`cargo build --release`)
- Wire `update_tray_icon()` calls into hotkey state transitions (currently TODO comments)
- Add Quit menu item handler (MenuEvent processing in the winit loop)
- Test end-to-end with llama-server running Voxtral
