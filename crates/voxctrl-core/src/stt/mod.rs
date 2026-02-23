//! Speech-to-Text — pluggable trait + factory.

#[cfg(feature = "stt-voxtral-http")]
pub mod voxtral_http;

use std::path::{Path, PathBuf};

use crate::config::SttConfig;

/// Trait for speech-to-text backends.
pub trait Transcriber: Send + Sync {
    /// Transcribe audio from a WAV file path.
    fn transcribe(&self, wav_path: &Path) -> anyhow::Result<String>;

    /// Transcribe from raw f32 PCM samples at the given sample rate.
    ///
    /// Default implementation writes a temp WAV and delegates to `transcribe(path)`.
    /// Backends that can work directly with PCM (e.g. whisper-native) should
    /// override this to skip the WAV round-trip.
    fn transcribe_pcm(&self, samples: &[f32], sample_rate: u32) -> anyhow::Result<String> {
        let tmp = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .map_err(|e| anyhow::anyhow!("create temp WAV: {e}"))?;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(tmp.path(), spec)
            .map_err(|e| anyhow::anyhow!("create WAV writer: {e}"))?;
        for &s in samples {
            let s16 = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(s16)?;
        }
        writer.finalize()?;
        self.transcribe(tmp.path())
    }

    /// Human-readable name for logs and UI.
    fn name(&self) -> &str;
    /// Check if the backend is reachable / functional.
    #[allow(dead_code)]
    fn is_available(&self) -> bool;
}

/// Function signature for an external factory that can create heavy STT backends.
///
/// Called by `create_transcriber()` for backend names it doesn't know.
/// Returns `Some(transcriber)` if the factory handles this backend,
/// or `None` to fall through to the "unknown backend" error.
pub type SttFactory = dyn Fn(&SttConfig, Option<PathBuf>) -> Option<anyhow::Result<Box<dyn Transcriber>>> + Send + Sync;

// ── Pending placeholder ─────────────────────────────────────────────────

/// Placeholder transcriber returned when a backend can't be initialised
/// (missing feature, failed download, missing model, etc.).
///
/// The app stays alive; transcription attempts return a clear error.
pub struct PendingTranscriber {
    reason: String,
    backend: String,
}

impl PendingTranscriber {
    pub fn new(backend: String, reason: String) -> Self {
        Self { reason, backend }
    }
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
/// `extra_factory` allows external crates (e.g. voxctrl-stt) to inject heavy
/// backends without this crate needing to depend on their ML libraries.
///
/// Never fails fatally — returns a `PendingTranscriber` placeholder when the
/// requested backend can't be initialised (feature not compiled, model missing,
/// download failure, etc.).
pub fn create_transcriber(
    cfg: &SttConfig,
    model_dir: Option<PathBuf>,
    extra_factory: Option<&SttFactory>,
) -> anyhow::Result<Box<dyn Transcriber>> {
    // Note: no `?` inside arms — errors must be captured in `result`
    // so the PendingTranscriber fallback below can handle them.
    let result: anyhow::Result<Box<dyn Transcriber>> = match cfg.backend.as_str() {
        "voxtral-http" => {
            #[cfg(feature = "stt-voxtral-http")]
            { Ok(Box::new(voxtral_http::VoxtralHttpTranscriber::new(cfg))) }
            #[cfg(not(feature = "stt-voxtral-http"))]
            { Err(anyhow::anyhow!("stt-voxtral-http feature not compiled in")) }
        }
        other => {
            // Try the external factory first (for heavy ML backends)
            if let Some(factory) = extra_factory {
                if let Some(factory_result) = factory(cfg, model_dir) {
                    factory_result
                } else {
                    Err(anyhow::anyhow!("Unknown STT backend: {other}"))
                }
            } else {
                Err(anyhow::anyhow!("Unknown STT backend: {other}"))
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn pending_transcriber_is_not_available() {
        let pt = PendingTranscriber::new("test-backend".into(), "not installed".into());
        assert!(!pt.is_available());
        assert_eq!(pt.name(), "pending");
    }

    #[test]
    fn pending_transcriber_returns_error_with_reason() {
        let pt = PendingTranscriber::new("whisper-cpp".into(), "feature not compiled".into());
        let result = pt.transcribe(Path::new("/tmp/test.wav"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("whisper-cpp"), "error should name the backend: {msg}");
        assert!(msg.contains("feature not compiled"), "error should include reason: {msg}");
    }

    #[test]
    fn create_transcriber_falls_back_to_pending_on_unknown_backend() {
        let cfg = crate::config::SttConfig {
            backend: "nonexistent-backend".into(),
            ..Default::default()
        };
        let t = create_transcriber(&cfg, None, None).expect("should not fail fatally");
        assert_eq!(t.name(), "pending");
        assert!(!t.is_available());
    }

    #[test]
    fn create_transcriber_falls_back_when_factory_returns_error() {
        let cfg = crate::config::SttConfig {
            backend: "broken-backend".into(),
            ..Default::default()
        };
        let factory: Box<SttFactory> = Box::new(|_cfg, _dir| {
            Some(Err(anyhow::anyhow!("init failed")))
        });
        let t = create_transcriber(&cfg, None, Some(&*factory)).expect("should not fail fatally");
        assert_eq!(t.name(), "pending");
        assert!(!t.is_available());
    }
}
