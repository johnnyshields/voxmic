//! ZLUDA DLL management — download, install, uninstall, status check.
//!
//! ZLUDA (<https://github.com/vosen/ZLUDA>) is a CUDA emulation layer for AMD GPUs.
//! It works by providing drop-in replacement DLLs (nvcuda.dll, nvml.dll) that
//! translate CUDA calls to AMD's ROCm/HIP runtime.

use std::path::{Path, PathBuf};

/// ZLUDA release asset URL pattern on GitHub.
const ZLUDA_GITHUB_API: &str = "https://api.github.com/repos/vosen/ZLUDA/releases/latest";

/// DLLs that ZLUDA provides (placed next to exe to override system nvcuda.dll).
const ZLUDA_DLLS: &[&str] = &["nvcuda.dll", "nvml.dll"];

// ── Status ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZludaStatus {
    /// ZLUDA DLLs not found in the expected directory.
    NotInstalled,
    /// ZLUDA DLLs present and ready.
    Installed(PathBuf),
    /// Download in progress.
    Downloading,
    /// Error during check or download.
    Error(String),
}

impl std::fmt::Display for ZludaStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZludaStatus::NotInstalled => write!(f, "Not installed"),
            ZludaStatus::Installed(p) => write!(f, "Installed ({})", p.display()),
            ZludaStatus::Downloading => write!(f, "Downloading..."),
            ZludaStatus::Error(e) => write!(f, "Error: {e}"),
        }
    }
}

// ── Check ────────────────────────────────────────────────────────────────

/// Default ZLUDA directory: `zluda/` next to the current executable.
pub fn default_zluda_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("zluda")))
}

/// Check whether ZLUDA DLLs are present in `dir`.
pub fn check_zluda(dir: &Path) -> ZludaStatus {
    if !dir.is_dir() {
        return ZludaStatus::NotInstalled;
    }
    for dll in ZLUDA_DLLS {
        if !dir.join(dll).exists() {
            return ZludaStatus::NotInstalled;
        }
    }
    ZludaStatus::Installed(dir.to_path_buf())
}

// ── Download ─────────────────────────────────────────────────────────────

/// Download the latest ZLUDA release from GitHub and extract to `dir`.
///
/// Calls `progress_cb` with a percentage (0-100) during download.
/// Returns the path to the extracted ZLUDA directory.
pub fn download_zluda(
    dir: &Path,
    progress_cb: impl Fn(u8),
) -> anyhow::Result<PathBuf> {
    use std::io::Read;

    log::info!("Fetching latest ZLUDA release from GitHub...");
    progress_cb(0);

    // 1. Query GitHub API for latest release
    let resp: serde_json::Value = ureq::get(ZLUDA_GITHUB_API)
        .set("User-Agent", "voxctrl")
        .set("Accept", "application/vnd.github.v3+json")
        .call()
        .map_err(|e| anyhow::anyhow!("Failed to query ZLUDA releases: {e}"))?
        .into_json()?;

    // 2. Find the Windows zip asset
    let asset = resp["assets"]
        .as_array()
        .and_then(|assets| {
            assets.iter().find(|a| {
                let name = a["name"].as_str().unwrap_or("");
                name.contains("windows") && name.ends_with(".zip")
            })
        })
        .ok_or_else(|| anyhow::anyhow!("No Windows zip asset found in ZLUDA release"))?;

    let download_url = asset["browser_download_url"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing download URL in ZLUDA release asset"))?;
    let asset_size = asset["size"].as_u64().unwrap_or(0);

    let tag = resp["tag_name"].as_str().unwrap_or("unknown");
    log::info!("Downloading ZLUDA {} from {}", tag, download_url);
    progress_cb(5);

    // 3. Download the zip
    let resp = ureq::get(download_url)
        .set("User-Agent", "voxctrl")
        .call()
        .map_err(|e| anyhow::anyhow!("Failed to download ZLUDA: {e}"))?;

    let mut zip_data = Vec::new();
    let mut reader = resp.into_reader();
    let mut buf = [0u8; 65536];
    let mut downloaded: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 { break; }
        zip_data.extend_from_slice(&buf[..n]);
        downloaded += n as u64;
        if asset_size > 0 {
            let pct = ((downloaded as f64 / asset_size as f64) * 85.0) as u8 + 5;
            progress_cb(pct.min(90));
        }
    }

    log::info!("Downloaded {} bytes, extracting...", zip_data.len());
    progress_cb(90);

    // 4. Extract zip to target directory
    std::fs::create_dir_all(dir)?;
    let cursor = std::io::Cursor::new(&zip_data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| anyhow::anyhow!("Failed to open ZLUDA zip: {e}"))?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let Some(name) = file.enclosed_name().map(|n| n.to_path_buf()) else {
            continue;
        };

        // Only extract DLL files to the target dir (flatten structure)
        if let Some(filename) = name.file_name() {
            let filename_str = filename.to_string_lossy();
            if filename_str.ends_with(".dll") {
                let dest = dir.join(filename);
                let mut out = std::fs::File::create(&dest)?;
                std::io::copy(&mut file, &mut out)?;
                log::info!("Extracted: {}", dest.display());
            }
        }
    }

    progress_cb(100);
    log::info!("ZLUDA extracted to {}", dir.display());
    Ok(dir.to_path_buf())
}

// ── Install / Uninstall ──────────────────────────────────────────────────

/// Copy ZLUDA DLLs from `zluda_dir` to `exe_dir` so they shadow the system ones.
///
/// The Windows DLL search order checks the exe directory first, so placing
/// ZLUDA's `nvcuda.dll` next to the binary makes CUDA calls go through ZLUDA.
pub fn install_zluda_dlls(zluda_dir: &Path, exe_dir: &Path) -> anyhow::Result<()> {
    for dll in ZLUDA_DLLS {
        let src = zluda_dir.join(dll);
        let dst = exe_dir.join(dll);
        if src.exists() {
            std::fs::copy(&src, &dst).map_err(|e| {
                anyhow::anyhow!("Failed to copy {} → {}: {e}", src.display(), dst.display())
            })?;
            log::info!("Installed ZLUDA DLL: {}", dst.display());
        } else {
            log::warn!("ZLUDA DLL not found: {}", src.display());
        }
    }
    Ok(())
}

/// Remove ZLUDA DLLs from `exe_dir`.
pub fn uninstall_zluda_dlls(exe_dir: &Path) -> anyhow::Result<()> {
    for dll in ZLUDA_DLLS {
        let path = exe_dir.join(dll);
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| {
                anyhow::anyhow!("Failed to remove {}: {e}", path.display())
            })?;
            log::info!("Removed ZLUDA DLL: {}", path.display());
        }
    }
    Ok(())
}

/// Check if ZLUDA DLLs are installed next to the executable.
pub fn is_zluda_active() -> bool {
    let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) else {
        return false;
    };
    ZLUDA_DLLS.iter().all(|dll| exe_dir.join(dll).exists())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_check_zluda_not_installed() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(check_zluda(dir.path()), ZludaStatus::NotInstalled);
    }

    #[test]
    fn test_check_zluda_installed() {
        let dir = tempfile::tempdir().unwrap();
        for dll in ZLUDA_DLLS {
            fs::write(dir.path().join(dll), b"fake").unwrap();
        }
        assert!(matches!(check_zluda(dir.path()), ZludaStatus::Installed(_)));
    }

    #[test]
    fn test_check_zluda_partial() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("nvcuda.dll"), b"fake").unwrap();
        // Missing nvml.dll
        assert_eq!(check_zluda(dir.path()), ZludaStatus::NotInstalled);
    }

    #[test]
    fn test_install_uninstall_zluda() {
        let zluda_dir = tempfile::tempdir().unwrap();
        let exe_dir = tempfile::tempdir().unwrap();

        // Create fake ZLUDA DLLs
        for dll in ZLUDA_DLLS {
            fs::write(zluda_dir.path().join(dll), b"fake-zluda-dll").unwrap();
        }

        // Install
        install_zluda_dlls(zluda_dir.path(), exe_dir.path()).unwrap();
        for dll in ZLUDA_DLLS {
            assert!(exe_dir.path().join(dll).exists());
        }

        // Uninstall
        uninstall_zluda_dlls(exe_dir.path()).unwrap();
        for dll in ZLUDA_DLLS {
            assert!(!exe_dir.path().join(dll).exists());
        }
    }

    #[test]
    fn test_zluda_status_display() {
        assert_eq!(ZludaStatus::NotInstalled.to_string(), "Not installed");
        assert_eq!(ZludaStatus::Downloading.to_string(), "Downloading...");
    }
}
