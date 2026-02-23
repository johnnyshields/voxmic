use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::models::{DownloadStatus, ModelEntry};

/// Scan the HuggingFace Hub cache to determine which models are already downloaded.
///
/// HF Hub cache layout: `~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{hash}/`
/// Only updates entries that are still `NotDownloaded`.
pub fn scan_hf_cache(entries: &mut [ModelEntry]) {
    let cache_dir = match hf_cache_dir() {
        Some(d) => d,
        None => {
            log::debug!("Could not determine HF cache directory");
            return;
        }
    };

    for entry in entries.iter_mut() {
        if !matches!(entry.status, DownloadStatus::NotDownloaded) {
            continue;
        }

        if let Some(ref repo) = entry.info.hf_repo {
            let model_dir_name = format!("models--{}", repo.replace('/', "--"));
            let model_dir = cache_dir.join(&model_dir_name);
            let snapshots_dir = model_dir.join("snapshots");

            if snapshots_dir.is_dir() {
                if let Some((path, size)) = find_snapshot(&snapshots_dir, &entry.info.hf_files) {
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

/// Return the effective cache directory for downloads.
pub fn effective_cache_dir() -> Option<PathBuf> {
    hf_cache_dir()
}

/// Find a snapshot directory that contains all required files.
fn find_snapshot(snapshots_dir: &Path, required_files: &[String]) -> Option<(PathBuf, u64)> {
    let read_dir = std::fs::read_dir(snapshots_dir).ok()?;

    for dir_entry in read_dir.flatten() {
        let path = dir_entry.path();
        if path.is_dir() && has_all_files(&path, required_files) {
            let size = dir_size(&path);
            if size > 0 {
                return Some((path, size));
            }
        }
    }
    None
}

/// Scan a user-chosen models directory using LM Studio-style layout: `publisher/model-name/`.
///
/// For each model entry with `hf_repo = "org/name"`, looks for `dir/org/name/`.
/// Only updates entries that are still `NotDownloaded`.
pub fn scan_models_directory(entries: &mut [ModelEntry], dir: &Path) {
    if !dir.is_dir() {
        log::debug!("Models directory {:?} does not exist", dir);
        return;
    }

    for entry in entries.iter_mut() {
        if !matches!(entry.status, DownloadStatus::NotDownloaded) {
            continue;
        }
        if let Some(ref repo) = entry.info.hf_repo {
            // repo is "org/name" — look for dir/org/name/
            let model_dir = dir.join(repo);
            if model_dir.is_dir() && has_all_files(&model_dir, &entry.info.hf_files) {
                let size = dir_size(&model_dir);
                if size > 0 {
                    log::info!(
                        "Found model '{}' in models directory at {:?} ({} bytes)",
                        entry.info.id,
                        model_dir,
                        size,
                    );
                    entry.status = DownloadStatus::Downloaded {
                        path: model_dir,
                        size_bytes: size,
                    };
                }
            }
        }
    }
}

/// Apply per-model path overrides from config.
///
/// Each key in `paths` is a model ID, and the value is the local directory.
/// Overrides any existing status (including Downloaded from other scanners).
pub fn apply_model_paths(entries: &mut [ModelEntry], paths: &HashMap<String, PathBuf>) {
    for entry in entries.iter_mut() {
        if let Some(override_path) = paths.get(&entry.info.id) {
            if override_path.is_dir() {
                let size = dir_size(override_path);
                log::info!(
                    "Per-model override for '{}' at {:?} ({} bytes)",
                    entry.info.id,
                    override_path,
                    size,
                );
                entry.status = DownloadStatus::Downloaded {
                    path: override_path.clone(),
                    size_bytes: size,
                };
            } else {
                log::warn!(
                    "Per-model override for '{}' points to non-existent directory {:?}",
                    entry.info.id,
                    override_path,
                );
            }
        }
    }
}

/// Check that all required files exist in a directory.
/// If `required_files` is empty, any non-empty directory is accepted.
fn has_all_files(dir: &Path, required_files: &[String]) -> bool {
    if required_files.is_empty() {
        return true;
    }
    required_files.iter().all(|f| dir.join(f).exists())
}

/// Recursively compute directory size in bytes.
fn dir_size(path: &Path) -> u64 {
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
