# Computer Use Test Tab + Dual Hotkeys

**Date**: 2026-02-23
**Branch**: refactor-split-build-targets/supervisor

## Summary

Add a Computer Use step to the GUI Test tab and support dual hotkeys (dictation + computer-use).

## Phases

- **A**: Agent streaming — `AgentEvent` enum + `run_agent_streaming()` in `voxctrl-cu/src/agent.rs`
- **B**: Mock provider — `MockProvider` implementing `AccessibilityProvider` for dry-run testing
- **C**: Config — add `cu_shortcut: Option<String>` to `HotkeyConfig`
- **D**: Main process — register both hotkeys, route CU hotkey events
- **E**: Settings tab — second hotkey capture input for CU hotkey
- **F**: Test tab — Step 5 (Computer Use) with goal input, agent log, mock toggle
- **G**: Feature gates — wrap CU test step behind `cu-*` feature flags

## Key Files

| File | Changes |
|------|---------|
| `crates/voxctrl-cu/src/agent.rs` | `AgentEvent`, `run_agent_streaming()` |
| `crates/voxctrl-cu/src/mock_provider.rs` | New: `MockProvider` |
| `crates/voxctrl-cu/src/lib.rs` | Export `MockProvider`, `AgentEvent` |
| `crates/voxctrl-core/src/config.rs` | `cu_shortcut` field in `HotkeyConfig` |
| `crates/voxctrl/src/hotkey.rs` | Dual hotkey registration + routing |
| `crates/voxctrl/src/main.rs` | Store both hotkey IDs |
| `crates/voxctrl/src/ui/model_table.rs` | CU test step UI, dual hotkey test, settings input |
| `crates/voxctrl/Cargo.toml` | No changes needed (cu-* features already gate deps) |
