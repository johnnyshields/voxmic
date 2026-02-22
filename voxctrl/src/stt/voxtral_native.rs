//! Pure-Rust Voxtral backend — Voxtral Mini 4B Realtime via Burn ML framework.
//!
//! Runs the full Voxtral encoder-decoder locally with GPU acceleration (wgpu).
//! Model is downloaded from HuggingFace Hub on first use (~8.9GB BF16).
//!
//! Based on <https://github.com/TrevorS/voxtral-mini-realtime-rs>.

use std::path::Path;
use std::sync::Mutex;

use burn::backend::Wgpu;
use voxtral_mini_realtime::audio::{load_wav, resample_to_16k, MelConfig, MelSpectrogram};
use voxtral_mini_realtime::hub::{self, ModelPaths};
use voxtral_mini_realtime::models::config::VoxtralModelConfig;
use voxtral_mini_realtime::models::loader::VoxtralModelLoader;
use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;
use voxtral_mini_realtime::models::voxtral::VoxtralModel;
use voxtral_mini_realtime::tokenizer;

use super::Transcriber;
use crate::config::SttConfig;

type Backend = Wgpu;

/// Minimum token ID to keep in output (filters special/control tokens).
const MIN_TEXT_TOKEN: u32 = 1000;

pub struct VoxtralNativeTranscriber {
    model: Mutex<VoxtralModel<Backend>>,
    mel: MelSpectrogram,
    time_embedding: TimeEmbedding<Backend>,
    tokenizer: tokenizers::Tokenizer,
    device: burn::prelude::Device<Backend>,
    delay: usize,
}

impl VoxtralNativeTranscriber {
    pub fn new(cfg: &SttConfig) -> anyhow::Result<Self> {
        let device = burn::prelude::Default::default();

        // Download model from HuggingFace (cached after first run).
        let model_id = cfg
            .voxtral_url
            .strip_prefix("hf://")
            .unwrap_or(hub::VOXTRAL_MINI_4B_REALTIME);
        log::info!("VoxtralNativeTranscriber: downloading model {model_id}");
        let paths = hub::download(model_id, None)?;

        // Load model weights.
        log::info!("VoxtralNativeTranscriber: loading weights from {:?}", paths.weights);
        let loader = VoxtralModelLoader::from_file(&paths.weights)?;
        let model = loader.load(&device)?;

        // Mel spectrogram extractor.
        let mel = MelSpectrogram::new(MelConfig::default());

        // Time embeddings for the decoder.
        let config = VoxtralModelConfig::voxtral();
        let time_embedding = TimeEmbedding::new(&config, &device);

        // Tokenizer.
        let tok = tokenizers::Tokenizer::from_file(&paths.tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        // Configurable delay: maps to number of mel frames.
        // Default 6 = ~480ms (recommended for best quality/latency tradeoff).
        let delay = 6;

        log::info!("VoxtralNativeTranscriber: ready (delay={})", delay);
        Ok(Self {
            model: Mutex::new(model),
            mel,
            time_embedding,
            tokenizer: tok,
            device,
            delay,
        })
    }
}

impl Transcriber for VoxtralNativeTranscriber {
    fn transcribe(&self, wav_path: &Path) -> anyhow::Result<String> {
        // Load and resample audio to 16kHz mono.
        let audio = load_wav(wav_path)?;
        let audio = resample_to_16k(audio)?;

        // Compute mel spectrogram.
        let mel_data = self.mel.compute(&audio.samples);

        // Convert to burn tensor: [1, n_mels, n_frames].
        let n_mels = mel_data.len() / (audio.samples.len() / 160); // hop_length = 160
        let n_frames = mel_data.len() / n_mels;
        let mel_tensor = burn::prelude::Tensor::<Backend, 3>::from_floats(
            burn::prelude::TensorData::new(mel_data, [1, n_mels, n_frames]),
            &self.device,
        );

        // Compute time embeddings for decoder.
        let t_embed = self.time_embedding.forward(n_frames, self.delay, &self.device);

        // Run model inference.
        let mut model = self
            .model
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;

        let token_ids = model.transcribe_streaming(mel_tensor, t_embed);

        drop(model);

        // Filter control tokens and decode to text.
        let text_tokens: Vec<u32> = token_ids
            .into_iter()
            .filter(|&t| t >= MIN_TEXT_TOKEN)
            .collect();

        let text = self
            .tokenizer
            .decode(&text_tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;

        let text = text.trim().to_string();
        log::debug!("VoxtralNative transcription: {text:?}");
        Ok(text)
    }

    fn name(&self) -> &str {
        "Voxtral Native"
    }

    fn is_available(&self) -> bool {
        true
    }
}
