# feat: Settings Auto-Save + Hot-Reload

**Date:** 2026-02-23
**Branch:** feat-settings-auto-save/supervisor

## Problem

Settings changes require clicking "Save" then restarting the app. Users want instant feedback and instant apply.

## Design

### SharedPipeline (lock-free swap)
`Mutex<Arc<Pipeline>>` wrapper in `pipeline.rs`. `.get()` returns cheap Arc clone (nanosecond lock). `.swap()` atomically replaces inner Arc. In-flight operations keep old pipeline alive.

### Config file watching
Main process polls `config.json` mtime every 500ms in `about_to_wait()`. On change, loads new config, diffs against stored config, applies changes.

### Hot-reload in main process
- **Audio changed**: drop stream, `start_capture()` with new config
- **STT/VAD/Router/Action/GPU changed**: rebuild pipeline via `Pipeline::from_config()`, swap into SharedPipeline
- **Hotkeys changed**: unregister old, register new (deferred while settings subprocess open)

### Settings UI auto-save
- Remove Save button. Detect field changes per-frame, auto-save on change.
- Per-section green "✓" with 1.5s fade using `HashMap<String, Instant>`.

## Files touched
- `voxctrl-core/src/config.rs` — PartialEq derives, `config_mtime()`, public `config_path()`
- `voxctrl-core/src/pipeline.rs` — SharedPipeline type
- `voxctrl-core/src/stt_server.rs` — use SharedPipeline
- `voxctrl-core/src/recording.rs` — use SharedPipeline
- `voxctrl/src/main.rs` — config watching, `apply_config_changes()`, `rebuild_pipeline()`
- `voxctrl/src/tui.rs` — use SharedPipeline
- `voxctrl/src/ui/model_table.rs` — auto-save, per-section green check, remove Save button

## Key safety properties
- Recording in progress: safe — thread holds its own Arc<Pipeline> snapshot
- Config write race: `load_config()` handles parse failures gracefully; 500ms poll retries
- Hotkey conflict: deferred while settings subprocess open
