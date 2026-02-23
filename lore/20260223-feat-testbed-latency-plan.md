# Testbed Latency Measurements — Plan

**Date:** 2026-02-23
**Branch:** feat-testbed-latency/supervisor
**Status:** Implemented

## Goal
Add detailed, UI-visible latency measurements to the testbed (Settings > Test tab) for each pipeline action and between actions.

## Current State
- Testbed has 6 steps: Mic → Hotkey → VAD → STT → Computer Use → Output
- Only `log::info!` timing exists in `transcribe_chunks()` and `pipeline.rs`
- No latency data shown in the UI

## Approach
Solo implementation — all changes in `crates/voxctrl/src/ui/model_table.rs`.

1. Add latency fields to `TestState` struct
2. Capture `Instant::now()` at each action boundary
3. Pass timing data from background threads via `Arc<Mutex<Option<T>>>` slots (same pattern as existing result/status slots)
4. Display latency inline per step and as a summary in step 6

### Key measurements:
- Mic initialization time
- Recording duration
- STT breakdown: WAV encode + transcription (server vs direct)
- CU agent execution time
- Total pipeline time
