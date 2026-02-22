use std::path::PathBuf;
use crate::models::{DownloadStatus, ModelEntry};

/// Scan the HuggingFace Hub cache to determine which models are already downloaded.
///
/// HF Hub cache layout: `~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{hash}/`
pub fn scan_hf_cache(entries: &mut [ModelEntry]) {
    let cache_dir = match hf_cache_dir() {
        Some(d) => d,
        None => {
            log::debug!("Could not determine HF cache directory");
            return;
        }
    };

    for entry in entries.iter_mut() {
        if let Some(ref repo) = entry.info.hf_repo {
            let model_dir_name = format!("models--{}", repo.replace('/', "--"));
            let model_dir = cache_dir.join(&model_dir_name);
            let snapshots_dir = model_dir.join("snapshots");

            if snapshots_dir.is_dir() {
                // Find the latest snapshot (any subdirectory)
                if let Some((path, size)) = find_snapshot(&snapshots_dir) {
                    log::info!(
                        "Found cached model '{}' at {:?} ({} bytes)",
                        entry.info.id,
                        path,
                        size
                    );
                    entry.status = DownloadStatus::Downloaded {
                        path,
                        size_bytes: size,
                    };
                }
            }
        }
    }
}

/// Return the HF Hub cache directory path.
pub fn hf_cache_dir() -> Option<PathBuf> {
    // Respect HF_HOME / HF_HUB_CACHE env vars
    if let Ok(cache) = std::env::var("HF_HUB_CACHE") {
        return Some(PathBuf::from(cache));
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(home).join("hub"));
    }
    // Default: ~/.cache/huggingface/hub/
    dirs::cache_dir().map(|d| d.join("huggingface").join("hub"))
}

/// Find a snapshot directory and compute total file size.
fn find_snapshot(snapshots_dir: &std::path::Path) -> Option<(PathBuf, u64)> {
    let read_dir = std::fs::read_dir(snapshots_dir).ok()?;

    for dir_entry in read_dir.flatten() {
        let path = dir_entry.path();
        if path.is_dir() {
            let size = dir_size(&path);
            if size > 0 {
                return Some((path, size));
            }
        }
    }
    None
}

/// Recursively compute directory size in bytes.
fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(path) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_file() {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            } else if p.is_dir() {
                total += dir_size(&p);
            }
        }
    }
    total
}
