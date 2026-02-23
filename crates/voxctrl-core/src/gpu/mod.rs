//! GPU detection and backend resolution.

#[cfg(feature = "zluda")]
pub mod zluda;

use crate::config::GpuConfig;
use std::path::Path;

// ── GPU hardware detection ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Unknown,
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuVendor::Nvidia => write!(f, "NVIDIA"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    pub device_id: u32,
}

/// Detect GPUs by probing for vendor driver DLLs in System32.
///
/// This is a lightweight check — no GPU API initialization required.
pub fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    let sys32 = Path::new("C:\\Windows\\System32");

    // NVIDIA: nvcuda.dll (CUDA driver) or nvapi64.dll
    if sys32.join("nvcuda.dll").exists() || sys32.join("nvapi64.dll").exists() {
        gpus.push(GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "NVIDIA GPU".into(),
            device_id: 0,
        });
    }

    // AMD: amdxc64.dll (DirectX driver) or atioglxx.dll (OpenGL)
    if sys32.join("amdxc64.dll").exists() || sys32.join("atioglxx.dll").exists() {
        gpus.push(GpuInfo {
            vendor: GpuVendor::Amd,
            name: "AMD GPU".into(),
            device_id: if gpus.is_empty() { 0 } else { 1 },
        });
    }

    // Intel: igdumdim64.dll (user-mode driver) or similar
    if sys32.join("igdumdim64.dll").exists() || sys32.join("igd10iumd64.dll").exists() {
        gpus.push(GpuInfo {
            vendor: GpuVendor::Intel,
            name: "Intel GPU".into(),
            device_id: if gpus.is_empty() { 0 } else { gpus.len() as u32 },
        });
    }

    gpus
}

// ── GPU mode resolution ──────────────────────────────────────────────────

/// Resolved GPU execution mode for ML inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMode {
    /// NVIDIA native CUDA.
    Cuda,
    /// AMD via ZLUDA (CUDA emulation layer).
    Zluda,
    /// DirectML (Windows ML runtime).
    DirectMl,
    /// WebGPU via wgpu (cross-platform fallback).
    Wgpu,
    /// CPU-only inference.
    Cpu,
}

impl std::fmt::Display for GpuMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuMode::Cuda => write!(f, "CUDA"),
            GpuMode::Zluda => write!(f, "ZLUDA"),
            GpuMode::DirectMl => write!(f, "DirectML"),
            GpuMode::Wgpu => write!(f, "wgpu"),
            GpuMode::Cpu => write!(f, "CPU"),
        }
    }
}

/// Resolve the GPU mode from config + detected hardware.
///
/// When `backend` is `"auto"`:
/// - NVIDIA detected → Cuda
/// - AMD detected → Zluda
/// - Otherwise → Cpu
pub fn resolve_gpu_mode(cfg: &GpuConfig, gpus: &[GpuInfo]) -> GpuMode {
    match cfg.backend.as_str() {
        "cuda" => GpuMode::Cuda,
        "zluda" => GpuMode::Zluda,
        "directml" => GpuMode::DirectMl,
        "wgpu" => GpuMode::Wgpu,
        "cpu" => GpuMode::Cpu,
        "auto" | _ => auto_resolve(gpus),
    }
}

fn auto_resolve(gpus: &[GpuInfo]) -> GpuMode {
    // Prefer NVIDIA (native CUDA) over AMD (ZLUDA)
    for gpu in gpus {
        match gpu.vendor {
            GpuVendor::Nvidia => return GpuMode::Cuda,
            _ => {}
        }
    }
    for gpu in gpus {
        match gpu.vendor {
            GpuVendor::Amd => return GpuMode::Zluda,
            _ => {}
        }
    }
    GpuMode::Cpu
}

/// Map resolved GPU mode to the `whisper_device` config string expected by
/// candle-core and whisper.cpp backends.
pub fn gpu_mode_to_whisper_device(mode: GpuMode) -> &'static str {
    match mode {
        GpuMode::Cuda | GpuMode::Zluda => "cuda",
        _ => "cpu",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_nvidia() {
        let gpus = vec![GpuInfo { vendor: GpuVendor::Nvidia, name: "RTX 4090".into(), device_id: 0 }];
        let cfg = GpuConfig { backend: "auto".into(), device_id: 0, zluda_dir: None, zluda_auto_download: true };
        assert_eq!(resolve_gpu_mode(&cfg, &gpus), GpuMode::Cuda);
    }

    #[test]
    fn test_auto_amd() {
        let gpus = vec![GpuInfo { vendor: GpuVendor::Amd, name: "RX 7900".into(), device_id: 0 }];
        let cfg = GpuConfig { backend: "auto".into(), device_id: 0, zluda_dir: None, zluda_auto_download: true };
        assert_eq!(resolve_gpu_mode(&cfg, &gpus), GpuMode::Zluda);
    }

    #[test]
    fn test_auto_no_gpu() {
        let cfg = GpuConfig { backend: "auto".into(), device_id: 0, zluda_dir: None, zluda_auto_download: true };
        assert_eq!(resolve_gpu_mode(&cfg, &[]), GpuMode::Cpu);
    }

    #[test]
    fn test_explicit_cuda() {
        let cfg = GpuConfig { backend: "cuda".into(), device_id: 0, zluda_dir: None, zluda_auto_download: true };
        assert_eq!(resolve_gpu_mode(&cfg, &[]), GpuMode::Cuda);
    }

    #[test]
    fn test_explicit_cpu() {
        let cfg = GpuConfig { backend: "cpu".into(), device_id: 0, zluda_dir: None, zluda_auto_download: true };
        let gpus = vec![GpuInfo { vendor: GpuVendor::Nvidia, name: "RTX 4090".into(), device_id: 0 }];
        assert_eq!(resolve_gpu_mode(&cfg, &gpus), GpuMode::Cpu);
    }

    #[test]
    fn test_nvidia_preferred_over_amd() {
        let gpus = vec![
            GpuInfo { vendor: GpuVendor::Amd, name: "RX 7900".into(), device_id: 0 },
            GpuInfo { vendor: GpuVendor::Nvidia, name: "RTX 4090".into(), device_id: 1 },
        ];
        let cfg = GpuConfig { backend: "auto".into(), device_id: 0, zluda_dir: None, zluda_auto_download: true };
        assert_eq!(resolve_gpu_mode(&cfg, &gpus), GpuMode::Cuda);
    }

    #[test]
    fn test_whisper_device_mapping() {
        assert_eq!(gpu_mode_to_whisper_device(GpuMode::Cuda), "cuda");
        assert_eq!(gpu_mode_to_whisper_device(GpuMode::Zluda), "cuda");
        assert_eq!(gpu_mode_to_whisper_device(GpuMode::Cpu), "cpu");
        assert_eq!(gpu_mode_to_whisper_device(GpuMode::Wgpu), "cpu");
    }
}
