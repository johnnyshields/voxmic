# Test Windows Build — Plan

**Date:** 2026-02-22
**Branch:** `feat-test-build-windows/supervisor`

## Goal

Verify voxctrl cross-compiles for Windows from WSL2 and clean up compiler warnings.

## Findings

- **Windows cross-build passes** with `cargo build --target x86_64-pc-windows-gnu --release`
- Produces valid 11MB PE32+ executable at `target/x86_64-pc-windows-gnu/release/voxctrl.exe`
- All dependencies (cpal/WASAPI, tray-icon, global-hotkey, enigo, muda, winit) link successfully
- 16 dead_code warnings — all from public API not yet wired (VAD pipeline, model UI, save_config, update_tray_icon)
- Native Linux tests fail only due to missing `libasound2-dev`

## Plan

1. Install `libasound2-dev` to unblock native tests
2. Add `#[allow(dead_code)]` annotations on intentional future-API items (9 files)
3. Verify: 0 warnings on Windows target, all tests pass natively
4. Commit clean build
