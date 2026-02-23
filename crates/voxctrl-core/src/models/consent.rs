use anyhow::{bail, Result};
use crate::config::Config;
use crate::models::ModelRegistry;
use crate::models::catalog::required_model_id;
use crate::models::DownloadStatus;

/// Ensure the model required by the config is available (downloaded).
/// Shows a consent dialog if the model needs to be downloaded.
pub fn ensure_model_available(cfg: &Config, registry: &ModelRegistry) -> Result<()> {
    let model_id = match required_model_id(cfg) {
        Some(id) => id,
        None => return Ok(()), // No local model required (e.g. HTTP backend)
    };

    let needs_download = match registry.get(&model_id) {
        Some(entry) => matches!(entry.status, DownloadStatus::NotDownloaded | DownloadStatus::Error(_)),
        None => {
            log::warn!("Model '{model_id}' not in catalog, skipping consent check");
            return Ok(());
        }
    };

    if !needs_download {
        return Ok(());
    }

    let display_name = registry.get(&model_id).unwrap().info.display_name.clone();
    let size = format_size(registry.get(&model_id).unwrap().info.approx_size_bytes);

    let approved = show_consent_dialog(&display_name, &size);

    if approved {
        bail!(
            "Model '{display_name}' is not downloaded. \
             Use the Settings panel to download models."
        )
    } else {
        bail!("User declined download of '{display_name}'")
    }
}

#[cfg(feature = "consent-dialog")]
fn show_consent_dialog(display_name: &str, size: &str) -> bool {
    use rfd::MessageDialogResult;

    let result = rfd::MessageDialog::new()
        .set_title("voxctrl — Model Download")
        .set_description(&format!(
            "voxctrl needs to download the {display_name} model ({size}).\n\n\
             This is a one-time download from HuggingFace.\n\n\
             Download now?"
        ))
        .set_buttons(rfd::MessageButtons::YesNo)
        .show();

    matches!(result, MessageDialogResult::Yes)
}

#[cfg(not(feature = "consent-dialog"))]
fn show_consent_dialog(display_name: &str, size: &str) -> bool {
    log::warn!(
        "Model '{display_name}' ({size}) needs download but consent-dialog feature is not enabled. \
         Proceeding without consent prompt."
    );
    true // Auto-approve when no dialog available
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.0} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{bytes} B")
    }
}
