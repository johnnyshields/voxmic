# ZLUDA GPU Acceleration Plan

**Date**: 2026-02-23
**Branch**: feat-zluda-acceleration/supervisor
**Status**: Planning

## Goal

Add ZLUDA support to voxctrl so the app can use GPU acceleration on AMD GPUs (via CUDA emulation) and NVIDIA GPUs (native CUDA).

## Key Findings

1. **cudarc dynamic loading**: candle's CUDA backend (cudarc) loads `nvcuda.dll` dynamically at runtime. No link-time dependency. ZLUDA's `nvcuda.dll` can be loaded transparently.

2. **candle-kernels requires nvcc**: The `candle-core/cuda` feature requires `candle-kernels`, which compiles CUDA kernels via `nvcc` at build time. This means full CUDA support requires the CUDA toolkit in the build environment.

3. **ZLUDA on Windows**: Place DLLs (nvcuda.dll, cublas.dll, etc.) next to the exe, or use `zluda.exe` launcher. Requires AMD drivers + HIP SDK.

4. **Current state**: The `cuda` feature flag exists but is a no-op marker — doesn't propagate to `candle-core/cuda`. The whisper-native backend already has `Device::new_cuda(0)` behind `#[cfg(feature = "cuda")]`.

## Architecture

Two-phase approach:

### Phase 1 (No CUDA SDK needed)
- GPU detection module (vendor detection via driver DLL probing)
- ZLUDA runtime management (download, extract, DLL placement)
- `GpuConfig` in config.rs (backend, device_id, zluda settings)
- Settings UI GPU section
- GPU init in main.rs before pipeline startup

### Phase 2 (CUDA SDK required)
- Wire `cuda` feature: `voxctrl-stt/cuda` → `candle-core/cuda`
- Wire `whisper-rs/cuda` for whisper.cpp backend
- Build + test with CUDA on WSL2

## New Files
- `crates/voxctrl-core/src/gpu/mod.rs` — GPU detection, mode resolution
- `crates/voxctrl-core/src/gpu/zluda.rs` — ZLUDA DLL management

## Modified Files
- `crates/voxctrl-core/src/config.rs` — GpuConfig
- `crates/voxctrl-core/src/lib.rs` — export gpu module
- `crates/voxctrl/src/main.rs` — GPU init
- `crates/voxctrl/src/ui/model_table.rs` — Settings GPU section
- `crates/voxctrl-stt/Cargo.toml` — cuda feature propagation
- Various Cargo.toml — new deps (zip, ureq)
