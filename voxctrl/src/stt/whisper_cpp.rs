//! whisper.cpp backend via the whisper-rs crate.

use std::path::{Path, PathBuf};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::Transcriber;
use crate::config::SttConfig;

/// Transcribes audio using the whisper.cpp library (via whisper-rs bindings).
pub struct WhisperCppTranscriber {
    ctx: WhisperContext,
    language: Option<String>,
}

impl WhisperCppTranscriber {
    pub fn new(cfg: &SttConfig) -> anyhow::Result<Self> {
        let model_path = resolve_model_path(&cfg.whisper_model)?;
        log::info!(
            "WhisperCppTranscriber: loading model from {}",
            model_path.display()
        );
        let ctx = WhisperContext::new_with_params(
            model_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF-8 model path"))?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| anyhow::anyhow!("failed to load whisper.cpp model: {e}"))?;

        log::info!("WhisperCppTranscriber: model loaded");
        Ok(Self {
            ctx,
            language: cfg.whisper_language.clone(),
        })
    }
}

/// Resolve a model name like `"small"` to a ggml model file path.
///
/// If `model` is already a path to an existing file, it is returned as-is.
/// Otherwise we probe common download locations.
fn resolve_model_path(model: &str) -> anyhow::Result<PathBuf> {
    let path = PathBuf::from(model);
    if path.is_file() {
        return Ok(path);
    }

    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let candidates = [
        PathBuf::from(format!("models/ggml-{model}.bin")),
        home.join(format!(".cache/whisper/ggml-{model}.bin")),
        home.join(format!(".local/share/whisper/ggml-{model}.bin")),
    ];
    for c in &candidates {
        if c.is_file() {
            return Ok(c.clone());
        }
    }
    anyhow::bail!(
        "whisper.cpp model not found for '{model}'. Searched: {}",
        candidates
            .iter()
            .map(|c| c.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

impl Transcriber for WhisperCppTranscriber {
    fn transcribe(&self, wav_path: &Path) -> anyhow::Result<String> {
        // Load WAV as f32 PCM.
        let reader = hound::WavReader::open(wav_path)?;
        let spec = reader.spec();
        let samples: Vec<f32> = if spec.bits_per_sample == 16 {
            reader
                .into_samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<Result<_, _>>()?
        } else {
            reader.into_samples::<f32>().collect::<Result<_, _>>()?
        };

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        if let Some(lang) = &self.language {
            params.set_language(Some(lang));
        }
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state
            .full(params, &samples)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let n = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut text = String::new();
        for i in 0..n {
            let seg = state
                .full_get_segment_text(i)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            text.push_str(&seg);
        }
        let text = text.trim().to_string();
        log::debug!("WhisperCpp transcription: {text:?}");
        Ok(text)
    }

    fn name(&self) -> &str {
        "whisper.cpp"
    }

    fn is_available(&self) -> bool {
        true // If we loaded the model, we're available.
    }
}
