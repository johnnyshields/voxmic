use anyhow::{bail, Result};
use crate::models::{ModelEntry, DownloadStatus};

/// Download a model's files from HuggingFace Hub.
///
/// TODO: Implement actual download using hf_hub crate or manual HTTP.
/// For now this is a stub that logs the repo URL for manual download.
pub fn download_model(entry: &mut ModelEntry) -> Result<()> {
    let repo = match &entry.info.hf_repo {
        Some(r) => r.clone(),
        None => {
            bail!("Model '{}' has no HuggingFace repo configured", entry.info.id);
        }
    };

    log::info!(
        "Download requested for '{}' from https://huggingface.co/{}",
        entry.info.display_name,
        repo
    );

    entry.status = DownloadStatus::Error(
        format!(
            "Auto-download not yet implemented. Please download manually from: \
             https://huggingface.co/{}",
            repo
        )
    );

    bail!(
        "Auto-download not yet implemented for '{}'. \
         Please download manually from: https://huggingface.co/{}",
        entry.info.display_name,
        repo
    )
}
