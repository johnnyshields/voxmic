# Settings UI Sections Restructure Plan

**Date:** 2026-02-23
**Branch:** feat-settings-ui-sections/supervisor

## Context

The Settings tab is a flat 2-column grid with all fields mixed together. The user wants:
1. Both Dictation and Computer Use hotkeys in a "Hotkeys" section
2. Settings UI broken into sections: Input, Hotkeys, Speech-to-Text, Voice Activity Detection, Computer Use
3. Computer Use support for both remote (Anthropic) and local LLM
4. Models Directory + HF Token moved to top of Models tab
5. A 3rd "CU Models" sub-tab in the Models tab

## Files Modified

- `crates/voxctrl-core/src/config.rs` — Add `cu_provider_type` to `ActionConfig`
- `crates/voxctrl-core/src/models/catalog.rs` — Add `ModelCategory::ComputerUse` variant
- `crates/voxctrl/src/ui/model_table.rs` — Main UI restructure:
  - `draw_settings_tab()`: flat grid → grouped sections (Input, Hotkeys, STT, VAD, CU)
  - `draw_models_tab()`: HF Token + Models Dir at top, add CU Models sub-tab
  - `SettingsApp`: new CU fields, save/load logic
  - Window size: 520 → 620 height

## Key Decisions

- Sections use `ui.group()` with `ui.strong()` heading (matches Test tab pattern)
- CU provider: "anthropic" (remote) vs "local" ComboBox — just changes base URL
- CU models tab: empty state for now ("No CU models in catalog yet")
- CU settings fields feature-gated behind `cu-*` features (existing pattern)
- Entire settings tab wrapped in `ScrollArea::vertical()` for overflow
- Solo implementation (all changes in 3 files, tightly coupled)
