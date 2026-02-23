# Harden: Post-Workspace-Split Code Quality Review

## Context

The voxctrl monolith was just split into a 3-crate Cargo workspace (`voxctrl-core`, `voxctrl-stt`, `voxctrl`). All 18 tests pass, clippy is clean (no errors, pre-existing warnings only), incremental builds work correctly. This harden pass reviews the split holistically for code quality, duplication, and missing test coverage.

## Effort/Impact Assessment Table

| # | Opportunity | Effort | Impact | Action |
|---|-------------|--------|--------|--------|
| 1 | Deduplicate PIPE_NAME constant across stt_client.rs and stt_server.rs | Quick | Medium | Auto-fix |
| 2 | Deduplicate `required_stt_model_id` / `required_vad_model_id` logic between `catalog.rs` and `model_table.rs` | Easy | Medium | Ask first |
| 3 | Add `Default` impl for `SharedState` (clippy warning) | Quick | Low | Auto-fix |
| 4 | Add warn log for LLM router JSON parse failures | Quick | Medium | Auto-fix |
| 5 | Add unit tests for `stt_factory()` dispatch and `PendingTranscriber` fallback | Moderate | High | Ask first |
| 6 | Duplicate deps in binary Cargo.toml that are already transitive via voxctrl-core | Easy | Low | Ask first |

## Opportunity Details

### #1 — Deduplicate PIPE_NAME constant
- **What**: Extract `PIPE_NAME` into a shared constant in `voxctrl-core` (e.g. in `lib.rs` or a shared `ipc` module) instead of defining it separately in both `stt_client.rs:11` and `stt_server.rs:18`.
- **Where**: `crates/voxctrl-core/src/stt_client.rs`, `crates/voxctrl-core/src/stt_server.rs`
- **Why**: Single source of truth — if the pipe name ever changes, only one place to update.

### #2 — Deduplicate model ID resolution logic
- **What**: The `required_stt_model_id()` logic in `model_table.rs:597-608` duplicates `catalog::required_model_id()` at `catalog.rs:103-121`. Similarly `required_vad_model_id()` at `model_table.rs:611-615`. Extract `required_stt_model_id(backend, whisper_model)` and `required_vad_model_id(backend)` as standalone functions in `catalog.rs` and call them from both places.
- **Where**: `crates/voxctrl-core/src/models/catalog.rs`, `crates/voxctrl/src/ui/model_table.rs`
- **Why**: Model name mappings are a data concern that belongs in catalog. The UI currently maintains its own copy — if model names change, both must be updated.

### #3 — Add Default impl for SharedState
- **What**: Derive or implement `Default` for `SharedState` to satisfy clippy `new_without_default` warning.
- **Where**: `crates/voxctrl-core/src/lib.rs:30-42`
- **Why**: Clippy best practice; makes the type usable in generic contexts.

### #4 — Add warn log for LLM router parse failures
- **What**: When `serde_json::from_str(content)` fails in the LLM router, log the failure and the raw content for debugging.
- **Where**: `crates/voxctrl-core/src/router/llm.rs:48`
- **Why**: Currently silently falls back to dictation, making LLM integration failures invisible.

### #5 — Add unit tests for STT factory and PendingTranscriber
- **What**: Test that `stt_factory()` returns `None` for unknown backends, and `Some(Err(..))` for known-but-disabled backends. Test that `PendingTranscriber` returns the expected error message and `is_available() == false`. Test that `create_transcriber()` falls back to `PendingTranscriber` when the factory returns an error.
- **Where**: New `#[cfg(test)]` blocks in `crates/voxctrl-stt/src/lib.rs` and `crates/voxctrl-core/src/stt/mod.rs`
- **Why**: The SttFactory pattern is the key architectural change of this refactor and has zero test coverage. Error paths should be verified.

### #6 — Remove duplicate dependencies from binary Cargo.toml
- **What**: The binary crate directly depends on `anyhow`, `log`, `env_logger`, `serde_json`, `tempfile`, `hound`, `cpal`, `dirs` — all of which are already dependencies of `voxctrl-core`. Some are used directly in `main.rs` (e.g. `log`, `anyhow`, `cpal::Stream`, `env_logger`) and must stay. Others like `tempfile`, `hound` are only used transitively and can be removed.
- **Where**: `crates/voxctrl/Cargo.toml`
- **Why**: Cleaner dependency graph, less maintenance burden.
- **Trade-offs**: Removing a direct dep means if core ever stops using it, the binary silently breaks. Keeping explicit deps is safer but redundant.

## Execution Protocol
**DO NOT implement any changes without user approval.**
For EACH opportunity, use `AskUserQuestion`.
Options: "Implement" / "Skip (add to TODO.md)" / "Do not implement"
Ask ALL questions before beginning any implementation work.
Quick items may be batched into one multi-select AskUserQuestion.
After all items resolved, run the project's test suite.
