//! Speech-to-Text — pluggable trait + factory.

#[cfg(feature = "stt-voxtral-http")]
pub mod voxtral_http;

use std::path::{Path, PathBuf};

use crate::config::SttConfig;

/// Load a WAV file and return its f32 PCM samples and sample rate.
///
/// Handles both 16-bit integer and 32-bit float WAV formats.
pub fn load_wav_pcm(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            reader.into_samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>()
                .collect::<Result<_, _>>()?
        }
    };
    Ok((samples, spec.sample_rate))
}

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

    // ── load_wav_pcm tests ──────────────────────────────────────────────

    fn write_wav_i16(path: &Path, samples: &[i16], sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for &s in samples {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
    }

    fn write_wav_f32(path: &Path, samples: &[f32], sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for &s in samples {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
    }

    #[test]
    fn load_wav_pcm_i16_roundtrip() {
        let tmp = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        let i16_samples: Vec<i16> = vec![0, 16384, -16384, 32767, -32768];
        write_wav_i16(tmp.path(), &i16_samples, 16000);

        let (pcm, rate) = load_wav_pcm(tmp.path()).unwrap();
        assert_eq!(rate, 16000);
        assert_eq!(pcm.len(), i16_samples.len());
        for (got, &orig) in pcm.iter().zip(&i16_samples) {
            let expected = orig as f32 / 32768.0;
            assert!((got - expected).abs() < 1e-5, "expected {expected}, got {got}");
        }
    }

    #[test]
    fn load_wav_pcm_f32_roundtrip() {
        let tmp = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        let f32_samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        write_wav_f32(tmp.path(), &f32_samples, 44100);

        let (pcm, rate) = load_wav_pcm(tmp.path()).unwrap();
        assert_eq!(rate, 44100);
        assert_eq!(pcm.len(), f32_samples.len());
        for (got, &expected) in pcm.iter().zip(&f32_samples) {
            assert!((got - expected).abs() < 1e-6, "expected {expected}, got {got}");
        }
    }

    #[test]
    fn load_wav_pcm_nonexistent_file_returns_error() {
        let result = load_wav_pcm(Path::new("/tmp/nonexistent_wav_file_12345.wav"));
        assert!(result.is_err());
    }

    // ── transcribe_pcm default round-trip ───────────────────────────────

    struct MockWavTranscriber;

    impl Transcriber for MockWavTranscriber {
        fn transcribe(&self, wav_path: &Path) -> anyhow::Result<String> {
            let (samples, rate) = load_wav_pcm(wav_path)?;
            Ok(format!("{}@{}", samples.len(), rate))
        }
        fn name(&self) -> &str { "mock" }
        fn is_available(&self) -> bool { true }
    }

    #[test]
    fn transcribe_pcm_default_round_trip() {
        let t = MockWavTranscriber;
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = t.transcribe_pcm(&samples, 16000).unwrap();
        assert_eq!(result, "5@16000");
    }

    #[test]
    fn transcribe_pcm_default_preserves_sample_values() {
        use std::sync::Mutex;

        struct CaptureTranscriber {
            captured: Mutex<Vec<f32>>,
        }
        impl Transcriber for CaptureTranscriber {
            fn transcribe(&self, wav_path: &Path) -> anyhow::Result<String> {
                let (samples, _) = load_wav_pcm(wav_path)?;
                *self.captured.lock().unwrap() = samples;
                Ok("ok".into())
            }
            fn name(&self) -> &str { "capture-mock" }
            fn is_available(&self) -> bool { true }
        }

        let t = CaptureTranscriber { captured: Mutex::new(vec![]) };
        // Use values that survive i16 quantization cleanly
        let pcm = vec![0.0f32, 0.5, -0.5];
        t.transcribe_pcm(&pcm, 16000).unwrap();

        let captured = t.captured.lock().unwrap();
        assert_eq!(captured.len(), pcm.len());
        for (got, &orig) in captured.iter().zip(&pcm) {
            // i16 quantization introduces ~1/32768 error
            assert!((got - orig).abs() < 0.001, "expected ~{orig}, got {got}");
        }
    }
}
