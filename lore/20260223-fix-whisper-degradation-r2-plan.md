# Fix Whisper Degradation R2

**Date**: 2026-02-23
**Branch**: fix-whisper-degradation-r2/supervisor
**Status**: Plan

## Problem

Whisper-native (candle) transcription degrades across repeated calls. R1 added `model.reset_kv_cache()` but the problem persists.

## R2 Root Cause Analysis

Deep investigation of candle-transformers 0.8.4 source revealed:

1. **`reset_kv_cache()` is redundant** — the decode loop already passes `flush=true` on step 0, which clears and recomputes the cross-attention cache. R1's fix was a no-op.

2. **KV cache is the only mutable state** — exhaustive audit of the Whisper model struct tree confirms `kv_cache: Option<(Tensor, Tensor)>` in `MultiHeadAttention` is the sole mutable field. No Cell, RefCell, running stats, or other interior mutability exists.

3. **Since cache is already properly reset**, degradation likely comes from either:
   - A subtle candle tensor lifecycle issue (Arc references preventing cleanup between calls)
   - Accumulated computational graph fragments when reusing the same model struct
   - Both would be eliminated by creating a fresh model per inference

4. **VarBuilder is reusable** — `Whisper::load(&vb, config)` borrows the VarBuilder (not consumes). Weights are shared via Arc over mmapped safetensors. Per-call reconstruction costs ~microseconds (struct creation, no tensor computation).

## Fix

### 1. Per-inference model reconstruction (primary)
Replace `Mutex<Whisper>` with stored `VarBuilder` + `Config`. Create a fresh `Whisper` for each inference call. Zero memory overhead (weights shared via Arc).

### 2. Hallucination detection (safety net)
- Repetition detector: same token 3+ times consecutively → early EOT
- Duration-proportional decode limit: `min(224, max(10, duration_secs * 15))`
- Non-ASCII detector: CJK tokens when expecting English → early EOT

### 3. Cleanup
Remove redundant `reset_kv_cache()`, simplify debug logging.

## Files

- `crates/voxctrl-stt/src/whisper_native.rs` — all changes
