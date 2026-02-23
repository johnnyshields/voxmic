# Harden: Whisper Degradation R2

**Date**: 2026-02-23
**Scope**: `crates/voxctrl-stt/src/whisper_native.rs`
**Reviewing**: Commit `a83ed1f` (per-inference model rebuild + hallucination guards)

## Findings

| # | Opportunity | Effort | Impact |
|---|-------------|--------|--------|
| 1 | Remove redundant `drop(_verify)` — use `let _ =` idiom | Quick | Low |
| 2 | Fix repetition detector off-by-one (triggers at 4th token, not 3rd) | Quick | High |
| 3 | Add unit tests for hallucination guard logic (duration limit, repetition) | Easy | High |
| 4 | Add unit test for `model_to_repo` branching logic | Quick | Low |

## Key Issue: Repetition Detector Off-by-One (#2)

`consecutive_repeats` starts at 0 and increments on seeing a duplicate. With `MAX_TOKEN_REPEATS = 3`, the guard fires after the 4th consecutive same-token, not the 3rd as intended. Fix: rename to `MAX_CONSECUTIVE_DUPLICATES = 2` (allow 2 dupes = 3 total).

## Implementation (all 4 implemented)

1. **`drop(_verify)` → `let _ =`** — single-line cleanup
2. **Off-by-one fix** — renamed `MAX_TOKEN_REPEATS` → `MAX_CONSECUTIVE_DUPLICATES = 2`, fixed log message to report actual total (`consecutive_repeats + 1`)
3. **Hallucination guard tests** — `duration_token_limit` (0s, 0.5s, 2s, 15s, 30s) + repetition detector (no repeats, 1 dup continues, 2 dups halt, reset between runs, empty/single input)
4. **`model_to_repo` tests** — short name → `openai/whisper-{name}`, full repo ID passthrough
