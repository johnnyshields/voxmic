# Feed Raw PCM Directly to Transcriber — Eliminate All Temp WAV Files

**Date:** 2026-02-23
**Scope:** voxctrl-core, voxctrl-stt, voxctrl (UI)

## Problem

Audio flowed through multiple temp WAV files before reaching the model:
- Main pipeline: chunks -> temp WAV -> `pipeline.process(path)` -> `stt.transcribe(path)` reads WAV back
- Testbed: chunks -> temp WAV -> read bytes -> pipe -> server writes temp WAV -> `stt.transcribe(path)` reads WAV back
- The f32 PCM samples were in memory at every step -- WAV serialization was pure overhead

## Changes

### 1. Transcriber trait (`crates/voxctrl-core/src/stt/mod.rs`)
- Added `transcribe_pcm(&self, samples: &[f32], sample_rate: u32) -> Result<String>` with default impl that writes temp WAV and delegates to `transcribe(path)` for backwards compat
- Existing backends (voxtral-http, whisper-cpp, voxtral-native, pending) get the default impl for free

### 2. WhisperNativeTranscriber (`crates/voxctrl-stt/src/whisper_native.rs`)
- Extracted core inference into `run_inference(&self, samples, sample_rate)` private method
- `transcribe_pcm()` overrides the default: calls `run_inference()` directly (zero-copy PCM path)
- `transcribe(path)` reads WAV then delegates to `run_inference()`

### 3. Wire protocol (`stt_server.rs` + `stt_client.rs`)
- Changed from WAV-based to PCM-based protocol
- New format: `[4B sample_rate BE] [4B num_samples BE] [N*4B f32 samples LE]`
- Server calls `transcribe_pcm()` directly, no temp file
- Client renamed: `transcribe_pcm_via_server(samples, sample_rate)`

### 4. Pipeline (`pipeline.rs`)
- Added `process_pcm(samples, sample_rate)` alongside existing `process(path)`
- Extracted shared route+execute logic into `route_and_execute()` helper

### 5. Recording (`recording.rs`)
- `transcribe_via_pipeline()` now just calls `pipeline.process_pcm()` directly
- Removed tempfile/hound dependencies from this file

### 6. Testbed (`model_table.rs`)
- `transcribe_chunks()` sends PCM directly (no temp WAV)
- `load_audio_file_and_transcribe()` reads WAV with hound, sends PCM
- Replaced `transcribe_via_server_or_direct()` with `transcribe_pcm_via_server_or_direct()`
- Removed `transcribe_wav_data()` entirely

## Verification
- `cargo check` -- compiles clean
- `cargo test -p voxctrl-stt -p voxctrl-core` -- 46 tests pass
- `cargo build --release --target x86_64-pc-windows-gnu` -- cross-compile succeeds
