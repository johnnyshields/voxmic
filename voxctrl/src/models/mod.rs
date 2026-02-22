pub mod catalog;
pub mod cache_scanner;
pub mod consent;
pub mod downloader;

pub use catalog::ModelInfo;

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

    pub fn scan_cache(&mut self) {
        cache_scanner::scan_hf_cache(&mut self.entries);
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
