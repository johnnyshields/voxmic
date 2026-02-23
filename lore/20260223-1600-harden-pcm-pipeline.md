# Harden PCM Pipeline — Post-Implementation Cleanup

**Date:** 2026-02-23
**Scope:** voxctrl-core, voxctrl-stt, voxctrl (UI)

## Context
After implementing the "feed raw PCM" changes (lore/20260223-feed-raw-pcm-eliminate-temp-wav.md), there are cleanup opportunities: dead code, a trivial wrapper, duplicated WAV-loading logic, a silent-error bug, and a missing `transcribe_pcm` override for whisper-cpp.

## Effort/Impact Assessment Table

| # | Opportunity | Effort | Impact | Action |
|---|-------------|--------|--------|--------|
| 1 | Remove dead `Pipeline::process()` (WAV-path version) | Quick | Medium | Auto-fix |
| 2 | Inline trivial `transcribe_via_pipeline()` wrapper | Quick | Low | Auto-fix |
| 3 | Fix silent error swallowing in `model_table.rs` WAV loading (`filter_map(\|s\| s.ok())` → proper error propagation) | Quick | High | Auto-fix |
| 4 | Add `transcribe_pcm` override to `WhisperCppTranscriber` (already extracts PCM from WAV) | Easy | Medium | Ask first |
| 5 | Extract shared WAV→PCM loading helper to deduplicate across whisper_native.rs, whisper_cpp.rs, model_table.rs | Moderate | Medium | Ask first |
| 6 | Add tests for `transcribe_pcm` default impl, PCM wire protocol, `process_pcm` | Moderate | High | Ask first |

## Opportunity Details

### #1 — Remove dead `Pipeline::process()`
- **What**: `process(&self, wav_path: &Path)` has zero callers (confirmed by grep). Remove it and the unused `Path` import.
- **Where**: `crates/voxctrl-core/src/pipeline.rs:49-57`, line 2 (`use std::path::Path`)
- **Why**: Dead code after the PCM migration — confusing to maintain alongside `process_pcm`.

### #2 — Inline `transcribe_via_pipeline()`
- **What**: The function is `fn transcribe_via_pipeline(chunks, sample_rate, pipeline) { pipeline.process_pcm(chunks, sample_rate) }` — a one-liner with no added logic. Inline the call at the only call site.
- **Where**: `crates/voxctrl-core/src/recording.rs:53-59` (definition), line 38 (call site)
- **Why**: Eliminates an indirection that obscures what's happening. The function added value when it managed temp files; now it's just forwarding.

### #3 — Fix silent error swallowing in WAV loading
- **What**: `model_table.rs:1387-1395` uses `filter_map(|s| s.ok())` which silently drops corrupt/truncated WAV samples. The same logic in `whisper_native.rs:311-317` and `whisper_cpp.rs:78-85` correctly uses `.collect::<Result<_, _>>()?` to propagate errors.
- **Where**: `crates/voxctrl/src/ui/model_table.rs:1385-1397`
- **Why**: Bug — a corrupted WAV file will silently produce a shorter (or empty) audio buffer, leading to mysterious transcription failures instead of a clear error message.

### #4 — Add `transcribe_pcm` override to WhisperCppTranscriber
- **What**: `whisper_cpp.rs::transcribe()` already reads WAV → extracts f32 PCM → feeds `state.full(params, &samples)`. Override `transcribe_pcm` to skip the WAV read and go straight to `state.full()`.
- **Where**: `crates/voxctrl-stt/src/whisper_cpp.rs` — extract inference into a helper, add `transcribe_pcm` impl
- **Why**: Same zero-copy benefit whisper-native got. When transcribing via the pipe or pipeline, whisper-cpp currently goes through the default trait impl which writes a temp WAV, then `transcribe()` reads it back.
- **Trade-offs**: whisper-cpp is a secondary backend; low risk but moderate effort to test manually.

### #5 — Extract shared WAV→PCM loading helper
- **What**: WAV→f32 PCM loading (open, check 16-bit vs float, collect with error propagation) is duplicated in 3 files. Extract to a `load_wav_pcm(path) -> Result<(Vec<f32>, u32)>` function.
- **Where**: New function in `voxctrl-core/src/stt/mod.rs` or a new `audio.rs` module. Used by `whisper_native.rs:303-318`, `whisper_cpp.rs:74-85`, `model_table.rs:1376-1397`.
- **Why**: DRY — three copies means three places to update and three chances for inconsistency (as #3 demonstrates).
- **Trade-offs**: Adds a cross-crate dependency if placed in `voxctrl-core` (both `voxctrl-stt` and `voxctrl` would call it). Already the case — both depend on `voxctrl-core`.

### #6 — Add tests for new PCM code paths
- **What**: The new `transcribe_pcm` default impl, `process_pcm`, and PCM wire protocol have no dedicated tests. Add unit tests.
- **Where**:
  - `voxctrl-core/src/stt/mod.rs` — test default `transcribe_pcm` round-trips correctly (write WAV → read back should match)
  - `voxctrl-core/src/stt_server.rs` + `stt_client.rs` — test wire format encoding/decoding
  - `voxctrl-core/src/pipeline.rs` — test `process_pcm` with a mock transcriber
- **Why**: These are new code paths exercised only by integration (manual testing). Unit tests catch regressions.
- **Trade-offs**: Wire protocol tests need mock streams (not a real named pipe). Pipeline tests need mock Transcriber/Router/Action impls.

## Execution Protocol
**DO NOT implement any changes without user approval.**
For EACH opportunity, use `AskUserQuestion`.
Options: "Implement" / "Skip (add to TODO.md)" / "Do not implement"
Ask ALL questions before beginning any implementation work.
(do NOT do alternating ask then implement, ask then implement, etc.)
Quick items may be batched into one multi-select AskUserQuestion.
After all items resolved, run the project's test suite.
