//! Pure-Rust Voxtral backend — Voxtral Mini 4B Realtime via Burn ML framework.
//!
//! Runs the full Voxtral encoder-decoder locally with GPU acceleration (wgpu).
//! Model must be downloaded via the Settings UI before use.
//!
//! Based on <https://github.com/TrevorS/voxtral-mini-realtime-rs>.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use burn::backend::Wgpu;
use burn::prelude::{Device, Tensor};
use burn::tensor::TensorData;
use voxtral_mini_realtime::audio::{load_wav, resample_to_16k, MelConfig, MelSpectrogram};
use voxtral_mini_realtime::audio::{pad_audio, PadConfig};
use voxtral_mini_realtime::hub::ModelPaths;
use voxtral_mini_realtime::models::loader::VoxtralModelLoader;
use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;
use voxtral_mini_realtime::models::voxtral::VoxtralModel;
use voxtral_mini_realtime::tokenizer::VoxtralTokenizer;

use super::Transcriber;

type Backend = Wgpu;

/// Decoder model dimension (d_model for Voxtral Mini 4B).
const DECODER_DIM: usize = 3072;

struct Inner {
    model: Mutex<VoxtralModel<Backend>>,
    mel: MelSpectrogram,
    time_embedding: TimeEmbedding,
    tokenizer: VoxtralTokenizer,
    device: Device<Backend>,
    delay: usize,
}

pub struct VoxtralNativeTranscriber {
    inner: Option<Inner>,
}

impl VoxtralNativeTranscriber {
    /// Create a new transcriber.
    ///
    /// - `model_dir = Some(path)`: load model from the given directory.
    /// - `model_dir = None`: construct in pending state (no auto-download).
    pub fn new(model_dir: Option<PathBuf>) -> anyhow::Result<Self> {
        let inner = match model_dir {
            Some(dir) => match Self::load_from_dir(&dir) {
                Ok(inner) => Some(inner),
                Err(e) => {
                    log::warn!("VoxtralNativeTranscriber: failed to load from {:?}: {e} — pending state", dir);
                    None
                }
            },
            None => {
                log::info!("VoxtralNativeTranscriber: no model directory — pending state");
                None
            }
        };
        Ok(Self { inner })
    }

    fn load_from_dir(dir: &Path) -> anyhow::Result<Inner> {
        let device: Device<Backend> = Default::default();

        log::info!("VoxtralNativeTranscriber: loading from {:?}", dir);
        let paths = ModelPaths::from_dir(dir);
        paths.validate()?;

        // Load model weights.
        log::info!("VoxtralNativeTranscriber: loading weights from {:?}", paths.weights);
        let loader = VoxtralModelLoader::from_file(&paths.weights)?;
        let model = loader.load(&device)?;

        // Mel spectrogram extractor (pre-configured for Voxtral).
        let mel = MelSpectrogram::new(MelConfig::voxtral());

        // Time embedding for decoder conditioning.
        let time_embedding = TimeEmbedding::new(DECODER_DIM);

        // Tokenizer.
        let tok = VoxtralTokenizer::from_file(&paths.tokenizer)?;

        let delay = 6;
        log::info!("VoxtralNativeTranscriber: ready (delay={})", delay);

        Ok(Inner {
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
        let inner = self.inner.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Model not downloaded — download from Settings")
        })?;

        // Load and resample audio to 16kHz mono.
        let audio = load_wav(wav_path)?;
        let audio = resample_to_16k(&audio)?;

        // Pad audio for streaming decode.
        let padded = pad_audio(&audio, &PadConfig::voxtral());

        // Compute log mel spectrogram → [n_frames][n_mels].
        let mel = inner.mel.compute_log(&padded.samples);
        let n_frames = mel.len();
        let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };

        // Transpose to [n_mels][n_frames] and flatten for the tensor.
        let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
        for (frame_idx, frame) in mel.iter().enumerate() {
            for (mel_idx, &val) in frame.iter().enumerate() {
                mel_transposed[mel_idx][frame_idx] = val;
            }
        }
        let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
        let mel_tensor: Tensor<Backend, 3> =
            Tensor::from_data(TensorData::new(mel_flat, [1, n_mels, n_frames]), &inner.device);

        // Compute time embedding for decoder conditioning.
        let t_embed = inner
            .time_embedding
            .embed::<Backend>(inner.delay as f32, &inner.device);

        // Run streaming transcription.
        let model = inner
            .model
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;

        let token_ids = model.transcribe_streaming(mel_tensor, t_embed);

        drop(model);

        // Cast i32 → u32 for tokenizer (tokenizer skips control tokens < 1000 internally).
        let token_ids: Vec<u32> = token_ids
            .into_iter()
            .filter_map(|t| u32::try_from(t).ok())
            .collect();

        let text = inner.tokenizer.decode(&token_ids)?;

        let text = text.trim().to_string();
        log::debug!("VoxtralNative transcription: {text:?}");
        Ok(text)
    }

    fn name(&self) -> &str {
        if self.inner.is_some() {
            "Voxtral Native"
        } else {
            "Voxtral Native (pending)"
        }
    }

    fn is_available(&self) -> bool {
        self.inner.is_some()
    }
}
