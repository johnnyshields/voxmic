# Wire up Download/Delete buttons in Settings > Models tab

**Date**: 2026-02-23

## Summary
Connected the Download and Delete buttons in the Models tab so they actually
perform HuggingFace model downloads and local cache deletion.

## Changes
1. **Cargo.toml** — added `dep:ureq` to the `gui` feature so HTTP downloads work
   when the UI is compiled.
2. **src/ui/model_table.rs** — rewired `draw_models_tab` to collect a
   `pending_action` from button clicks, process it after dropping the registry
   lock:
   - **Download**: spawns a background thread that creates the HF cache directory
     structure, downloads each file via `ureq`, updates progress between files,
     and calls `scan_cache()` on completion.
   - **Delete**: synchronously removes the model's cache directory and rescans.
3. **src/models/downloader.rs** — left as-is (stub becomes unused; can clean up later).
