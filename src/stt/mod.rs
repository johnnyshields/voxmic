//! Speech-to-Text — pluggable trait + factory.

#[cfg(feature = "stt-voxtral-http")]
pub mod voxtral_http;
#[cfg(feature = "stt-whisper-cpp")]
pub mod whisper_cpp;
#[cfg(feature = "stt-whisper-native")]
pub mod whisper_native;
#[cfg(feature = "stt-voxtral-native")]
pub mod voxtral_native;

use std::path::{Path, PathBuf};

use crate::config::SttConfig;

/// Trait for speech-to-text backends.
pub trait Transcriber: Send + Sync {
    /// Transcribe audio from a WAV file path.
    fn transcribe(&self, wav_path: &Path) -> anyhow::Result<String>;
    /// Human-readable name for logs and UI.
    fn name(&self) -> &str;
    /// Check if the backend is reachable / functional.
    #[allow(dead_code)]
    fn is_available(&self) -> bool;
}

// ── Pending placeholder ─────────────────────────────────────────────────

/// Placeholder transcriber returned when a backend can't be initialised
/// (missing feature, failed download, missing model, etc.).
///
/// The app stays alive; transcription attempts return a clear error.
struct PendingTranscriber {
    reason: String,
    backend: String,
}

impl Transcriber for PendingTranscriber {
    fn transcribe(&self, _wav_path: &Path) -> anyhow::Result<String> {
        anyhow::bail!("{} — {}", self.backend, self.reason)
    }
    fn name(&self) -> &str {
        "pending"
    }
    fn is_available(&self) -> bool {
        false
    }
}

/// Create an STT backend based on config.
///
/// `model_dir` is the resolved local path for backends that need local model files
/// (e.g. voxtral-native). Other backends ignore it.
///
/// Never fails fatally — returns a `PendingTranscriber` placeholder when the
/// requested backend can't be initialised (feature not compiled, model missing,
/// download failure, etc.).
pub fn create_transcriber(cfg: &SttConfig, model_dir: Option<PathBuf>) -> anyhow::Result<Box<dyn Transcriber>> {
    // Note: no `?` inside arms — errors must be captured in `result`
    // so the PendingTranscriber fallback below can handle them.
    let result: anyhow::Result<Box<dyn Transcriber>> = match cfg.backend.as_str() {
        "voxtral-http" => {
            #[cfg(feature = "stt-voxtral-http")]
            { Ok(Box::new(voxtral_http::VoxtralHttpTranscriber::new(cfg))) }
            #[cfg(not(feature = "stt-voxtral-http"))]
            { Err(anyhow::anyhow!("stt-voxtral-http feature not compiled in")) }
        }
        "whisper-cpp" => {
            #[cfg(feature = "stt-whisper-cpp")]
            { whisper_cpp::WhisperCppTranscriber::new(cfg).map(|t| Box::new(t) as _) }
            #[cfg(not(feature = "stt-whisper-cpp"))]
            { Err(anyhow::anyhow!("stt-whisper-cpp feature not compiled in")) }
        }
        "whisper-native" => {
            #[cfg(feature = "stt-whisper-native")]
            { whisper_native::WhisperNativeTranscriber::new(cfg, model_dir).map(|t| Box::new(t) as _) }
            #[cfg(not(feature = "stt-whisper-native"))]
            { Err(anyhow::anyhow!("stt-whisper-native feature not compiled in")) }
        }
        "voxtral-native" => {
            #[cfg(feature = "stt-voxtral-native")]
            { voxtral_native::VoxtralNativeTranscriber::new(model_dir).map(|t| Box::new(t) as _) }
            #[cfg(not(feature = "stt-voxtral-native"))]
            { Err(anyhow::anyhow!("stt-voxtral-native feature not compiled in")) }
        }
        other => Err(anyhow::anyhow!("Unknown STT backend: {other}")),
    };

    match result {
        Ok(t) => Ok(t),
        Err(e) => {
            let reason = format!("{e:#}");
            log::warn!("STT backend '{}' unavailable: {reason} — using pending placeholder", cfg.backend);
            Ok(Box::new(PendingTranscriber {
                backend: cfg.backend.clone(),
                reason,
            }))
        }
    }
}
