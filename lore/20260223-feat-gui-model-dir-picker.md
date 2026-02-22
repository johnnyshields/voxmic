# feat-gui-model-dir-picker

**Date**: 2026-02-23
**Branch**: `feat-gui-model-dir-picker/supervisor`

## Summary

Add a Models Directory picker and per-model Local Path column to the Settings window.

## Changes

1. **Cargo.toml** — Add `"dep:rfd"` to the `gui` feature so native folder dialogs work
2. **src/config.rs** — New `ModelsConfig` struct with `models_directory: Option<PathBuf>` and `model_paths: HashMap<String, PathBuf>` for per-model overrides
3. **src/models/cache_scanner.rs** — `scan_models_directory()` for LM Studio-style `publisher/model/` layout; `apply_model_paths()` for per-model overrides
4. **src/models/mod.rs** — `scan_cache()` now takes `&ModelsConfig` and applies all three scan sources
5. **src/ui/model_table.rs** — "Models Directory" row in Settings tab with Browse/Reset; "Local Path" column (rightmost) in Models tab with per-row folder pickers; window widened to 650px

## Design Decisions

- LM Studio-style `publisher/model-name/` directory layout for custom models directory
- `rfd::FileDialog::pick_folder()` sync API (blocks UI briefly while OS dialog open)
- Per-model path overrides persisted in `config.json` under `models.model_paths`
- Backwards-compatible: `ModelsConfig` uses `serde(default)` so old configs load fine
