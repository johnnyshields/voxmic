# Harden: Whisper-Native Transcription Fix

## Context

The recent commit (`6df2b73`) added `model.reset_kv_cache()` before each inference, an inference counter, encoder output diagnostics, and a stability test to fix transcription degradation in `WhisperNativeTranscriber`. This plan assesses hardening opportunities in that code.

**File**: `crates/voxctrl-stt/src/whisper_native.rs`

## Effort/Impact Assessment Table

| # | Opportunity | Effort | Impact | Action |
|---|-------------|--------|--------|--------|
| 1 | Remove redundant sort in `build_token_mask()` | Quick | Low | Auto-fix |
| 2 | Include `call_num` in completion log for end-to-end correlation | Quick | Medium | Auto-fix |
| 3 | Extract hardcoded fallback token IDs to named constants | Quick | Low | Auto-fix |
| 4 | Log warnings on silent token ID fallback | Easy | Medium | Auto-fix |
| 5 | Add test: `build_token_mask` with `HashSet` equivalence (validates sort removal) | Easy | Low | Auto-fix |
| 6 | Add test: begin_suppress_mask applied only on step 0 | Moderate | Medium | Ask first |

## Opportunity Details

### #1 — Remove redundant sort in `build_token_mask()`
- **What**: `build_token_mask()` sorts its input despite the comment saying it's pre-sorted. Replace with `HashSet` lookup — clearer intent, no sort needed.
- **Where**: `whisper_native.rs:353-366`

### #2 — Include `call_num` in completion log
- **What**: The `log::info!("[whisper-dbg] Final text: ...")` doesn't include the inference number. Add it for correlation.
- **Where**: `whisper_native.rs:273`

### #3 — Extract hardcoded fallback token IDs to named constants
- **What**: Magic numbers `50258`, `50257`, `50359`, `50363` → module-level constants.
- **Where**: `whisper_native.rs:103-108`

### #4 — Log warnings on silent token ID fallback
- **What**: Add `log::warn!` when tokenizer lookup fails and fallback is used.
- **Where**: `whisper_native.rs:103-108`

### #5 — Add test: `build_token_mask` HashSet equivalence
- **What**: Confirm refactored HashSet version produces identical output.
- **Where**: `whisper_native.rs` tests

### #6 — Add test: begin_suppress_mask applied only on step 0
- **What**: Test mask arithmetic on synthetic tensors to verify step-0-only application.
- **Where**: `whisper_native.rs` tests
