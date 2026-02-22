pub mod catalog;
pub mod cache_scanner;
pub mod consent;
pub mod downloader;

#[cfg(feature = "gui")]
pub use catalog::ModelCategory;
pub use catalog::ModelInfo;

use crate::config::ModelsConfig;
use std::path::PathBuf;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum DownloadStatus {
    NotDownloaded,
    Downloading { progress_pct: u8 },
    Downloaded { path: PathBuf, size_bytes: u64 },
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub info: ModelInfo,
    pub status: DownloadStatus,
    pub in_use: bool,
}

pub struct ModelRegistry {
    entries: Vec<ModelEntry>,
}

impl ModelRegistry {
    pub fn new(models: Vec<ModelInfo>) -> Self {
        let entries = models
            .into_iter()
            .map(|info| ModelEntry {
                info,
                status: DownloadStatus::NotDownloaded,
                in_use: false,
            })
            .collect();
        Self { entries }
    }

    pub fn scan_cache(&mut self, models_cfg: &ModelsConfig) {
        // 1. Custom models directory (LM Studio-style layout)
        if let Some(ref dir) = models_cfg.models_directory {
            cache_scanner::scan_models_directory(&mut self.entries, dir);
        }
        // 2. HuggingFace cache (standard HF layout)
        cache_scanner::scan_hf_cache(&mut self.entries);
        // 3. Per-model path overrides (highest priority)
        if !models_cfg.model_paths.is_empty() {
            cache_scanner::apply_model_paths(&mut self.entries, &models_cfg.model_paths);
        }
    }

    /// Return the local path for a downloaded model, if available.
    pub fn model_path(&self, id: &str) -> Option<PathBuf> {
        self.get(id).and_then(|e| match &e.status {
            DownloadStatus::Downloaded { path, .. } => Some(path.clone()),
            _ => None,
        })
    }

    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.iter().find(|e| e.info.id == id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut ModelEntry> {
        self.entries.iter_mut().find(|e| e.info.id == id)
    }

    pub fn set_in_use(&mut self, id: &str) {
        // Clear previous in_use
        for entry in &mut self.entries {
            entry.in_use = false;
        }
        if let Some(entry) = self.get_mut(id) {
            entry.in_use = true;
        }
    }

    #[allow(dead_code)]
    pub fn entries(&self) -> &[ModelEntry] {
        &self.entries
    }
}
