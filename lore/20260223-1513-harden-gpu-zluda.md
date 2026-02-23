# Harden GPU/ZLUDA Module

Post-implementation hardening of Phase 1 GPU infrastructure (detection, ZLUDA DLL management, config, UI).

## Effort/Impact Assessment Table

| # | Opportunity | Effort | Impact | Action |
|---|-------------|--------|--------|--------|
| 1 | Extract repeated `exe_dir` resolution in main.rs (3 identical calls) | Quick | Medium | Auto-fix |
| 2 | Consolidate duplicate `install_zluda_dlls` calls in main.rs | Quick | Medium | Auto-fix |
| 3 | Silent error on ZLUDA DLL cleanup (line 333: `let _ =`) | Quick | Medium | Auto-fix |
| 4 | Use `SYSTEMROOT` env var instead of hardcoded `C:\Windows\System32` | Quick | Medium | Auto-fix |
| 5 | Add `#[cfg(windows)]` to `detect_gpus()` body + empty vec fallback | Easy | Medium | Auto-fix |
| 6 | Add missing tests: unknown backend, Intel GPU, partial install | Easy | Medium | Auto-fix |
| 7 | Move `GPU_BACKENDS` list to `gpu/mod.rs` as shared source of truth | Easy | Medium | Ask first |
| 8 | Make `GpuConfig.backend` a typed enum instead of String | Moderate | High | Ask first |

## Opportunity Details

### #1 — Extract repeated exe_dir resolution
- **What**: `std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()))` appears 3 times in the ZLUDA block. Extract to a single `let exe_dir = ...` before the match.
- **Where**: `crates/voxctrl/src/main.rs` lines 304, 316, 332
- **Why**: DRY, easier to reason about one resolution point

### #2 — Consolidate duplicate install_zluda_dlls calls
- **What**: After both "found installed" and "just downloaded" paths, the same `install_zluda_dlls(zluda_dir, exe_dir)` call is made. Extract post-match.
- **Where**: `crates/voxctrl/src/main.rs` lines 305, 317
- **Why**: Less code, single responsibility for the match arms

### #3 — Silent error on ZLUDA cleanup
- **What**: `let _ = uninstall_zluda_dlls(...)` silently swallows errors. Other error paths use `log::error!`. Change to `log::warn!`.
- **Where**: `crates/voxctrl/src/main.rs` line 333
- **Why**: Consistency; cleanup failures are worth knowing about

### #4 — Use SYSTEMROOT env var
- **What**: Replace `Path::new("C:\\Windows\\System32")` with `std::env::var("SYSTEMROOT")` + fallback. More robust on non-standard Windows installs.
- **Where**: `crates/voxctrl-core/src/gpu/mod.rs` line 42
- **Why**: Portability; respects Windows config

### #5 — Platform-gate detect_gpus()
- **What**: Wrap the DLL-probing body of `detect_gpus()` in `#[cfg(windows)]` and return empty vec on other platforms. Currently returns empty on non-Windows anyway (paths don't exist), but this makes intent explicit.
- **Where**: `crates/voxctrl-core/src/gpu/mod.rs`
- **Why**: Clarity; prevents future confusion

### #6 — Add missing test cases
- **What**: Add tests for:
  - `resolve_gpu_mode` with unknown/invalid backend string (should fall through to auto)
  - Intel-only GPU scenario
  - `is_zluda_active()` returning false when not all DLLs are present
- **Where**: `crates/voxctrl-core/src/gpu/mod.rs`, `crates/voxctrl-core/src/gpu/zluda.rs`
- **Why**: Edge case coverage

### #7 — Shared GPU_BACKENDS list
- **What**: Move `GPU_BACKENDS: &[(&str, &str)]` from `model_table.rs` to `gpu/mod.rs` and reuse in both UI and config validation.
- **Where**: `crates/voxctrl-core/src/gpu/mod.rs`, `crates/voxctrl/src/ui/model_table.rs`
- **Why**: Single source of truth; enables config validation
- **Trade-offs**: Couples UI labels to core crate (minor)

### #8 — GpuConfig.backend as typed enum
- **What**: Replace `backend: String` with `backend: GpuBackend` enum (Auto, Cuda, Zluda, DirectMl, Wgpu, Cpu). Add serde rename for JSON compat. Eliminates `resolve_gpu_mode`'s string matching.
- **Where**: `config.rs` (GpuConfig), `gpu/mod.rs` (resolve_gpu_mode, GpuBackend enum), `model_table.rs` (combo box)
- **Why**: Type safety; invalid values caught at parse time. This is a fixed set unlike STT/VAD which have extensible factories.
- **Trade-offs**: Breaking change to `config.json` format if anyone has one (unlikely since this was just added). Slightly more complex serde. Eliminates the need for #7 (shared backend list) since the enum IS the source of truth.

## Verification

1. `cargo check` — compiles without `zluda`
2. `cargo check --features zluda` — compiles with `zluda`
3. `cargo test -p voxctrl-core` — all existing tests pass
4. `cargo test -p voxctrl-core --features zluda` — all tests including ZLUDA tests pass

## Execution Protocol
**DO NOT implement any changes without user approval.**
For EACH opportunity, use `AskUserQuestion`.
Options: "Implement" / "Skip (add to TODO.md)" / "Do not implement"
Ask ALL questions before beginning any implementation work.
(do NOT do alternating ask then implement, ask then implement, etc.)
Quick items may be batched into one multi-select AskUserQuestion.
After all items resolved, run the project's test suite.
