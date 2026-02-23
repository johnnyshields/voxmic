//! voxctrl-stt — Heavy ML inference backends for speech-to-text.
//!
//! Provides whisper-native (candle), whisper-cpp (whisper-rs), and
//! voxtral-native (burn) backends. These are split from voxctrl-core
//! to avoid recompiling heavy ML dependencies when GUI code changes.

#[cfg(feature = "stt-whisper-cpp")]
pub mod whisper_cpp;
#[cfg(feature = "stt-whisper-native")]
pub mod whisper_native;
#[cfg(feature = "stt-voxtral-native")]
pub mod voxtral_native;

use std::path::PathBuf;
use voxctrl_core::config::SttConfig;
use voxctrl_core::stt::Transcriber;

/// Factory function for heavy STT backends.
///
/// Returns `Some(Ok(transcriber))` if this crate handles the backend,
/// `Some(Err(..))` if it handles the backend but init failed,
/// or `None` for unknown backends (lets core handle them).
pub fn stt_factory(
    cfg: &SttConfig,
    model_dir: Option<PathBuf>,
) -> Option<anyhow::Result<Box<dyn Transcriber>>> {
    match cfg.backend.as_str() {
        "whisper-cpp" => {
            #[cfg(feature = "stt-whisper-cpp")]
            { Some(whisper_cpp::WhisperCppTranscriber::new(cfg).map(|t| Box::new(t) as _)) }
            #[cfg(not(feature = "stt-whisper-cpp"))]
            { Some(Err(anyhow::anyhow!("stt-whisper-cpp feature not compiled in"))) }
        }
        "whisper-native" => {
            #[cfg(feature = "stt-whisper-native")]
            { Some(whisper_native::WhisperNativeTranscriber::new(cfg, model_dir).map(|t| Box::new(t) as _)) }
            #[cfg(not(feature = "stt-whisper-native"))]
            { Some(Err(anyhow::anyhow!("stt-whisper-native feature not compiled in"))) }
        }
        "voxtral-native" => {
            #[cfg(feature = "stt-voxtral-native")]
            { Some(voxtral_native::VoxtralNativeTranscriber::new(model_dir).map(|t| Box::new(t) as _)) }
            #[cfg(not(feature = "stt-voxtral-native"))]
            { let _ = model_dir; Some(Err(anyhow::anyhow!("stt-voxtral-native feature not compiled in"))) }
        }
        _ => None, // Unknown — let core handle it
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxctrl_core::config::SttConfig;

    fn make_cfg(backend: &str) -> SttConfig {
        SttConfig {
            backend: backend.into(),
            whisper_model: "tiny".into(),
            ..Default::default()
        }
    }

    #[test]
    fn unknown_backend_returns_none() {
        assert!(stt_factory(&make_cfg("unknown-backend"), None).is_none());
    }

    #[test]
    fn voxtral_http_returns_none() {
        // voxtral-http is handled by core, not stt crate
        assert!(stt_factory(&make_cfg("voxtral-http"), None).is_none());
    }

    #[test]
    fn known_backends_return_some() {
        // These return Some regardless of feature flags (Ok if compiled, Err if not)
        for backend in &["whisper-cpp", "whisper-native", "voxtral-native"] {
            let result = stt_factory(&make_cfg(backend), None);
            assert!(result.is_some(), "stt_factory should handle '{backend}'");
        }
    }
}
