use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::models::{DownloadStatus, ModelEntry};

/// Scan the HuggingFace Hub cache to determine which models are already downloaded.
///
/// HF Hub cache layout: `~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{hash}/`
/// Only updates entries that are still `NotDownloaded`.
///
/// On WSL2, also checks Windows AppData HF cache as a fallback for entries
/// not found in the Linux cache.
pub fn scan_hf_cache(entries: &mut [ModelEntry]) {
    // Collect all cache directories to try: standard Linux path first, then WSL2 Windows paths.
    let mut cache_dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = hf_cache_dir() {
        cache_dirs.push(d);
    }
    cache_dirs.extend(wsl2_hf_cache_dirs());

    scan_hf_cache_in_dirs(entries, &cache_dirs);
}

/// Scan specific HF Hub cache directories. Used by `scan_hf_cache` and tests.
fn scan_hf_cache_in_dirs(entries: &mut [ModelEntry], cache_dirs: &[PathBuf]) {
    if cache_dirs.is_empty() {
        log::debug!("Could not determine any HF cache directory");
        return;
    }

    for entry in entries.iter_mut() {
        if !matches!(entry.status, DownloadStatus::NotDownloaded) {
            continue;
        }

        if let Some(ref repo) = entry.info.hf_repo {
            let model_dir_name = format!("models--{}", repo.replace('/', "--"));

            for cache_dir in cache_dirs {
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
                        break;
                    }
                }
            }
        }
    }
}

/// Return HF Hub cache directories from the Windows AppData path under WSL2.
///
/// Checks `/mnt/c/Users/*/AppData/Local/huggingface/hub/` for each Windows user profile.
/// Returns an empty vec if not running under WSL2 or no such paths exist.
pub fn wsl2_hf_cache_dirs() -> Vec<PathBuf> {
    let wsl_users = Path::new("/mnt/c/Users");
    if !wsl_users.is_dir() {
        return Vec::new();
    }

    let mut dirs = Vec::new();
    if let Ok(rd) = std::fs::read_dir(wsl_users) {
        for entry in rd.flatten() {
            let hub = entry
                .path()
                .join("AppData")
                .join("Local")
                .join("huggingface")
                .join("hub");
            if hub.is_dir() {
                log::debug!("WSL2: found Windows HF cache at {:?}", hub);
                dirs.push(hub);
            }
        }
    }
    dirs
}

/// Find a cached HF model by repo name across all known cache directories.
///
/// Searches the standard Linux HF cache first, then any WSL2 Windows AppData caches.
/// Returns the snapshot path if a snapshot containing all `required_files` is found.
pub fn find_hf_model(repo: &str, required_files: &[&str]) -> Option<PathBuf> {
    let mut cache_dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = hf_cache_dir() {
        cache_dirs.push(d);
    }
    cache_dirs.extend(wsl2_hf_cache_dirs());

    let model_dir_name = format!("models--{}", repo.replace('/', "--"));
    let owned_files: Vec<String> = required_files.iter().map(|s| s.to_string()).collect();

    for cache_dir in &cache_dirs {
        let snapshots_dir = cache_dir.join(&model_dir_name).join("snapshots");
        if snapshots_dir.is_dir() {
            if let Some((path, _size)) = find_snapshot(&snapshots_dir, &owned_files) {
                return Some(path);
            }
        }
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{DownloadStatus, ModelEntry};
    use crate::models::catalog::{ModelBackend, ModelCategory, ModelInfo};
    use std::fs;

    fn make_entry(id: &str, repo: &str, files: &[&str]) -> ModelEntry {
        ModelEntry {
            info: ModelInfo {
                id: id.into(),
                display_name: id.into(),
                backend: ModelBackend::Voxtral,
                category: ModelCategory::Stt,
                hf_repo: Some(repo.into()),
                hf_files: files.iter().map(|s| s.to_string()).collect(),
                approx_size_bytes: 0,
            },
            status: DownloadStatus::NotDownloaded,
            in_use: false,
        }
    }

    #[test]
    fn scan_hf_cache_finds_model_in_tmpdir() {
        let tmp = tempfile::tempdir().unwrap();
        // Build HF-style layout: models--org--name/snapshots/abc123/
        let snap = tmp.path()
            .join("models--test--model")
            .join("snapshots")
            .join("abc123");
        fs::create_dir_all(&snap).unwrap();
        fs::write(snap.join("weights.bin"), b"fake-weights").unwrap();

        let mut entries = vec![make_entry("test/model", "test/model", &["weights.bin"])];
        let dirs = vec![tmp.path().to_path_buf()];
        scan_hf_cache_in_dirs(&mut entries, &dirs);

        match &entries[0].status {
            DownloadStatus::Downloaded { path, size_bytes } => {
                assert_eq!(path, &snap);
                assert!(*size_bytes > 0);
            }
            other => panic!("Expected Downloaded, got {:?}", other),
        }
    }

    #[test]
    fn scan_hf_cache_skips_incomplete_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let snap = tmp.path()
            .join("models--test--partial")
            .join("snapshots")
            .join("abc");
        fs::create_dir_all(&snap).unwrap();
        // Only write one of two required files
        fs::write(snap.join("a.bin"), b"data").unwrap();

        let mut entries = vec![make_entry("test/partial", "test/partial", &["a.bin", "b.bin"])];
        let dirs = vec![tmp.path().to_path_buf()];
        scan_hf_cache_in_dirs(&mut entries, &dirs);

        assert!(matches!(entries[0].status, DownloadStatus::NotDownloaded));
    }

    #[test]
    fn scan_hf_cache_tries_multiple_dirs() {
        let empty = tempfile::tempdir().unwrap();
        let real = tempfile::tempdir().unwrap();
        let snap = real.path()
            .join("models--org--name")
            .join("snapshots")
            .join("v1");
        fs::create_dir_all(&snap).unwrap();
        fs::write(snap.join("model.bin"), b"data").unwrap();

        let mut entries = vec![make_entry("org/name", "org/name", &["model.bin"])];
        // First dir is empty (miss), second has the model (hit)
        let dirs = vec![empty.path().to_path_buf(), real.path().to_path_buf()];
        scan_hf_cache_in_dirs(&mut entries, &dirs);

        assert!(matches!(entries[0].status, DownloadStatus::Downloaded { .. }));
    }

    #[test]
    fn wsl2_hf_cache_dirs_returns_empty_when_no_mnt_c() {
        // On non-WSL systems (or if /mnt/c/Users doesn't exist), should return empty.
        // On WSL2 this still exercises the real path; either way the function shouldn't panic.
        let dirs = wsl2_hf_cache_dirs();
        // We can't assert much about the contents since it depends on the environment,
        // but the function must not panic and must return a Vec.
        assert!(dirs.iter().all(|d| d.is_dir()));
    }

    #[test]
    fn has_all_files_empty_required() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(has_all_files(tmp.path(), &[]));
    }

    #[test]
    fn has_all_files_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!has_all_files(tmp.path(), &["missing.txt".into()]));
    }

    #[test]
    fn has_all_files_present() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("a.txt"), b"x").unwrap();
        fs::write(tmp.path().join("b.txt"), b"y").unwrap();
        assert!(has_all_files(tmp.path(), &["a.txt".into(), "b.txt".into()]));
    }

    // ── scan_models_directory tests ──────────────────────────────────────

    #[test]
    fn scan_models_dir_finds_model() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("acme").join("great-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("weights.bin"), b"data").unwrap();

        let mut entries = vec![make_entry("acme/great-model", "acme/great-model", &["weights.bin"])];
        scan_models_directory(&mut entries, tmp.path());

        match &entries[0].status {
            DownloadStatus::Downloaded { path, size_bytes } => {
                assert_eq!(path, &model_dir);
                assert!(*size_bytes > 0);
            }
            other => panic!("Expected Downloaded, got {:?}", other),
        }
    }

    #[test]
    fn scan_models_dir_skips_incomplete() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("acme").join("partial");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("a.bin"), b"data").unwrap();
        // Missing b.bin

        let mut entries = vec![make_entry("acme/partial", "acme/partial", &["a.bin", "b.bin"])];
        scan_models_directory(&mut entries, tmp.path());

        assert!(matches!(entries[0].status, DownloadStatus::NotDownloaded));
    }

    #[test]
    fn scan_models_dir_ignores_nonexistent() {
        let mut entries = vec![make_entry("x/y", "x/y", &["f.bin"])];
        scan_models_directory(&mut entries, Path::new("/nonexistent/path/12345"));

        assert!(matches!(entries[0].status, DownloadStatus::NotDownloaded));
    }

    // ── apply_model_paths tests ──────────────────────────────────────────

    #[test]
    fn apply_model_paths_overrides_status() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("model.bin"), b"data").unwrap();

        let mut entries = vec![make_entry("my-model", "org/repo", &[])];
        let mut paths = HashMap::new();
        paths.insert("my-model".to_string(), tmp.path().to_path_buf());

        apply_model_paths(&mut entries, &paths);

        match &entries[0].status {
            DownloadStatus::Downloaded { path, size_bytes } => {
                assert_eq!(path, tmp.path());
                assert!(*size_bytes > 0);
            }
            other => panic!("Expected Downloaded, got {:?}", other),
        }
    }

    // ── .partial file tests ────────────────────────────────────────────

    #[test]
    fn partial_file_does_not_satisfy_has_all_files() {
        let tmp = tempfile::tempdir().unwrap();
        // Only a .partial file exists — the real file is missing
        fs::write(tmp.path().join("model.safetensors.partial"), b"incomplete").unwrap();

        assert!(!has_all_files(tmp.path(), &["model.safetensors".into()]));
    }

    #[test]
    fn partial_file_does_not_count_as_downloaded() {
        let tmp = tempfile::tempdir().unwrap();
        let snap = tmp.path()
            .join("models--test--partial-trap")
            .join("snapshots")
            .join("main");
        fs::create_dir_all(&snap).unwrap();
        // Write only a .partial file — the real required file is absent
        fs::write(snap.join("model.safetensors.partial"), b"incomplete-data").unwrap();

        let mut entries = vec![make_entry(
            "test/partial-trap",
            "test/partial-trap",
            &["model.safetensors"],
        )];
        let dirs = vec![tmp.path().to_path_buf()];
        scan_hf_cache_in_dirs(&mut entries, &dirs);

        assert!(
            matches!(entries[0].status, DownloadStatus::NotDownloaded),
            "A .partial file must not trick the cache scanner into marking a model as Downloaded"
        );
    }

    #[test]
    fn apply_model_paths_skips_nonexistent() {
        let mut entries = vec![make_entry("my-model", "org/repo", &[])];
        let mut paths = HashMap::new();
        paths.insert("my-model".to_string(), PathBuf::from("/nonexistent/12345"));

        apply_model_paths(&mut entries, &paths);

        assert!(matches!(entries[0].status, DownloadStatus::NotDownloaded));
    }
}
