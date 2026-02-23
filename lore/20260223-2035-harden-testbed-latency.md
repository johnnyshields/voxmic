# Harden: Testbed Latency Measurements

**Date:** 2026-02-23
**Branch:** feat-testbed-latency/supervisor
**File:** `crates/voxctrl/src/ui/model_table.rs`

## Context
The testbed latency feature was just implemented and merged with master's PCM-based
transcription refactor. The merge introduced some dead code and inconsistencies that
need cleanup.

## Effort/Impact Assessment Table

| # | Opportunity | Effort | Impact | Action |
|---|-------------|--------|--------|--------|
| 1 | Remove 3 unused `*_start_time` fields | Quick | Medium | Auto-fix |
| 2 | Remove dead `wav_encode` field & UI code | Quick | Medium | Auto-fix |
| 3 | Remove misleading `#[allow(dead_code)]` on live CU fields | Quick | Low | Auto-fix |
| 4 | Unify timing: route `load_audio_file_and_transcribe` through `transcribe_chunks` | Easy | High | Auto-fix |

## Opportunity Details

### #1 — Remove unused `*_start_time` fields
- **What**: `mic_start_time`, `stt_start_time`, `cu_start_time` are set but never read.
  Timing is computed locally instead (via `mic_start` in `start_mic_test`, `total_start`
  in thread closures, `cu_start` in CU thread).
- **Where**: `TestState` struct (lines 323, 327, 334), `Default` impl (lines 366, 370, 376),
  `start_mic_test` (line 1450), `stop_recording_and_transcribe` (line 1498),
  `load_audio_file_and_transcribe` (line 1540), `start_cu_test` (line 1609)
- **Why**: Dead code clutter; misleads readers into thinking these fields drive the timing

### #2 — Remove dead `wav_encode` field & UI code
- **What**: `SttTiming::wav_encode_secs` is always `0.0` after master's PCM refactor
  removed WAV encoding. Remove the field, `stt_wav_encode_latency` from `TestState`,
  and the two UI conditionals that display it.
- **Where**: `SttTiming` (line 275), `TestState` (line 329), `Default` (line 372),
  polling (line 1027), inline UI (lines 1277-1281), grid UI (lines 1413-1418),
  both thread closures that construct `SttTiming`
- **Why**: Displaying a metric that's always zero is misleading; removes ~20 lines

### #3 — Remove misleading `#[allow(dead_code)]` on CU latency fields
- **What**: `cu_latency` and `cu_timing_slot` are used (polled, displayed) but carry
  `#[allow(dead_code)]`. Only `cu_start_time` is truly dead (covered by #1).
  Move the annotation to only the CU fields that genuinely need it (same as existing
  `cu_goal`, `cu_running`, etc.).
- **Where**: Lines 335-338
- **Why**: Annotations should accurately describe reality

### #4 — Unify timing: route `load_audio_file_and_transcribe` through `transcribe_chunks`
- **What**: Currently `stop_recording_and_transcribe` calls `transcribe_chunks()` (which
  wraps timing + logging) while `load_audio_file_and_transcribe` duplicates timing inline
  and calls `transcribe_pcm_via_server_or_direct` directly. Route both through
  `transcribe_chunks()` so timing semantics are consistent and the audio stats logging
  applies to both paths.
- **Where**: `load_audio_file_and_transcribe` thread closure (lines 1579-1596),
  `transcribe_chunks` (lines 1776-1797)
- **Why**: Eliminates duplicated timing logic, ensures consistent semantics (currently
  the file-load path sets `transcribe_secs = total_secs` which double-counts overhead)

## Execution Protocol
**DO NOT implement any changes without user approval.**
For EACH opportunity, use `AskUserQuestion`.
Options: "Implement" / "Skip (add to TODO.md)" / "Do not implement"
Ask ALL questions before beginning any implementation work.
(do NOT do alternating ask then implement, ask then implement, etc.)
Quick items may be batched into one multi-select AskUserQuestion.
After all items resolved, run the project's test suite.
