//! Speech-to-Text — pluggable trait + factory.

#[cfg(feature = "stt-voxtral-http")]
pub mod voxtral_http;
#[cfg(feature = "stt-whisper-cpp")]
pub mod whisper_cpp;
#[cfg(feature = "stt-whisper-native")]
pub mod whisper_native;
#[cfg(feature = "stt-voxtral-native")]
pub mod voxtral_native;

use std::path::Path;

use crate::config::SttConfig;

/// Trait for speech-to-text backends.
pub trait Transcriber: Send + Sync {
    /// Transcribe audio from a WAV file path.
    fn transcribe(&self, wav_path: &Path) -> anyhow::Result<String>;
    /// Human-readable name for logs and UI.
    fn name(&self) -> &str;
    /// Check if the backend is reachable / functional.
    fn is_available(&self) -> bool;
}

/// Create an STT backend based on config.
pub fn create_transcriber(cfg: &SttConfig) -> anyhow::Result<Box<dyn Transcriber>> {
    match cfg.backend.as_str() {
        "voxtral-http" => {
            #[cfg(feature = "stt-voxtral-http")]
            return Ok(Box::new(voxtral_http::VoxtralHttpTranscriber::new(cfg)));
            #[cfg(not(feature = "stt-voxtral-http"))]
            anyhow::bail!("stt-voxtral-http feature not compiled in");
        }
        "whisper-cpp" => {
            #[cfg(feature = "stt-whisper-cpp")]
            return Ok(Box::new(whisper_cpp::WhisperCppTranscriber::new(cfg)?));
            #[cfg(not(feature = "stt-whisper-cpp"))]
            anyhow::bail!("stt-whisper-cpp feature not compiled in");
        }
        "whisper-native" => {
            #[cfg(feature = "stt-whisper-native")]
            return Ok(Box::new(whisper_native::WhisperNativeTranscriber::new(cfg)?));
            #[cfg(not(feature = "stt-whisper-native"))]
            anyhow::bail!("stt-whisper-native feature not compiled in");
        }
        "voxtral-native" => {
            #[cfg(feature = "stt-voxtral-native")]
            return Ok(Box::new(voxtral_native::VoxtralNativeTranscriber::new(cfg)?));
            #[cfg(not(feature = "stt-voxtral-native"))]
            anyhow::bail!("stt-voxtral-native feature not compiled in");
        }
        other => anyhow::bail!("Unknown STT backend: {other}"),
    }
}
