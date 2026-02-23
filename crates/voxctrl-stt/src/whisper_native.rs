//! Pure-Rust Whisper backend using candle-core + candle-transformers + hf-hub.
//!
//! Uses the candle-transformers reference mel spectrogram implementation
//! with pre-computed mel filter banks from the OpenAI whisper repo.

use std::path::Path;
use std::sync::Mutex;

use byteorder::{ByteOrder, LittleEndian};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper as m;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use voxctrl_core::stt::Transcriber;
use voxctrl_core::config::SttConfig;

const MAX_DECODE_TOKENS: usize = 224;

// ── Transcriber ─────────────────────────────────────────────────────────────

/// Pure-Rust Whisper transcriber backed by the candle framework.
pub struct WhisperNativeTranscriber {
    model: Mutex<m::model::Whisper>,
    config: m::Config,
    tokenizer: Tokenizer,
    device: Device,
    mel_filters: Vec<f32>,
    language_token: Option<u32>,
    sot_token: u32,
    eot_token: u32,
    transcribe_token: u32,
    no_timestamps_token: u32,
}

impl WhisperNativeTranscriber {
    pub fn new(cfg: &SttConfig, model_dir: Option<std::path::PathBuf>) -> anyhow::Result<Self> {
        let device = match cfg.whisper_device.as_str() {
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0)?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    log::warn!("CUDA requested but not compiled in; falling back to CPU");
                    Device::Cpu
                }
            }
            _ => Device::Cpu,
        };

        // Resolve model files: prefer local model_dir, fall back to hf_hub download.
        let (config_path, model_path, tokenizer_path) = if let Some(ref dir) = model_dir {
            let cp = dir.join("config.json");
            let mp = dir.join("model.safetensors");
            let tp = dir.join("tokenizer.json");
            if cp.exists() && mp.exists() && tp.exists() {
                log::info!("WhisperNativeTranscriber: loading from local dir {:?}", dir);
                (cp, mp, tp)
            } else {
                log::warn!("WhisperNativeTranscriber: local dir {:?} missing files, trying hf_hub", dir);
                Self::resolve_via_hub(cfg)?
            }
        } else {
            Self::resolve_via_hub(cfg)?
        };

        let config: m::Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

        // Load pre-computed mel filters (from OpenAI whisper assets, embedded at compile time).
        let mel_filters = load_mel_filters(config.num_mel_bins)?;
        log::info!(
            "WhisperNativeTranscriber: loaded {} mel filters ({} bins)",
            mel_filters.len(),
            config.num_mel_bins
        );

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        log::info!(
            "WhisperNativeTranscriber: loading weights ({} mel bins, device={device:?})",
            config.num_mel_bins
        );
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
        let model = m::model::Whisper::load(&vb, config.clone())?;

        // Look up special tokens from the tokenizer.
        let sot_token = tokenizer
            .token_to_id("<|startoftranscript|>")
            .unwrap_or(50258);
        let eot_token = tokenizer.token_to_id("<|endoftext|>").unwrap_or(50257);
        let transcribe_token = tokenizer.token_to_id("<|transcribe|>").unwrap_or(50359);
        let no_timestamps_token = tokenizer.token_to_id("<|notimestamps|>").unwrap_or(50363);

        let language_token = cfg.whisper_language.as_ref().and_then(|lang| {
            let tag = format!("<|{lang}|>");
            tokenizer.token_to_id(&tag)
        });

        log::info!("WhisperNativeTranscriber: ready");
        Ok(Self {
            model: Mutex::new(model),
            config,
            tokenizer,
            device,
            mel_filters,
            language_token,
            sot_token,
            eot_token,
            transcribe_token,
            no_timestamps_token,
        })
    }

    /// Download model files via hf_hub API.
    fn resolve_via_hub(cfg: &SttConfig) -> anyhow::Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
        let repo_id = model_to_repo(&cfg.whisper_model);
        log::info!("WhisperNativeTranscriber: downloading model {repo_id} via hf_hub");
        let api = Api::new()?;
        let repo = api.model(repo_id);
        let config_path = repo.get("config.json")?;
        let model_path = repo.get("model.safetensors")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        Ok((config_path, model_path, tokenizer_path))
    }
}

/// Map a short model name to a Hugging Face repo ID.
fn model_to_repo(model: &str) -> String {
    if model.contains('/') {
        return model.to_string();
    }
    format!("openai/whisper-{model}")
}

/// Load pre-computed mel filter bank from embedded binary data.
///
/// These are the exact same filters used by OpenAI's whisper and by
/// the candle-transformers reference implementation.
fn load_mel_filters(num_mel_bins: usize) -> anyhow::Result<Vec<f32>> {
    let mel_bytes: &[u8] = match num_mel_bins {
        80 => include_bytes!("melfilters.bytes"),
        128 => include_bytes!("melfilters128.bytes"),
        n => anyhow::bail!("Unsupported num_mel_bins={n}; expected 80 or 128"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);
    Ok(mel_filters)
}

impl Transcriber for WhisperNativeTranscriber {
    fn transcribe(&self, wav_path: &Path) -> anyhow::Result<String> {
        // ── Load & normalise audio ──────────────────────────────────────
        let reader = hound::WavReader::open(wav_path)?;
        let spec = reader.spec();
        let raw: Vec<f32> = if spec.bits_per_sample == 16 {
            reader
                .into_samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<Result<_, _>>()?
        } else {
            reader.into_samples::<f32>().collect::<Result<_, _>>()?
        };

        // ── Mel spectrogram (candle reference implementation) ─────────
        let mel = m::audio::pcm_to_mel(&self.config, &raw, &self.mel_filters);
        let mel_len = mel.len();
        let n_mel = self.config.num_mel_bins;
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, n_mel, mel_len / n_mel),
            &self.device,
        )?;

        // ── Encode ──────────────────────────────────────────────────────
        let mut model = self
            .model
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let encoder_output = model.encoder.forward(&mel_tensor, true)?;

        // ── Greedy decode ───────────────────────────────────────────────
        let mut tokens: Vec<u32> = vec![self.sot_token];
        if let Some(lang) = self.language_token {
            tokens.push(lang);
        }
        tokens.push(self.transcribe_token);
        tokens.push(self.no_timestamps_token);
        let prompt_len = tokens.len();

        for step in 0..MAX_DECODE_TOKENS {
            let flush = step == 0;
            let input: &[u32] = if flush {
                &tokens
            } else {
                std::slice::from_ref(tokens.last().unwrap())
            };

            let token_t = Tensor::new(input, &self.device)?.unsqueeze(0)?;
            let logits = model.decoder.forward(&token_t, &encoder_output, flush)?;

            let seq_len = logits.dims()[1];
            let last_logits = logits.i((0, seq_len - 1))?;
            let next_token = last_logits
                .argmax(0)?
                .to_dtype(DType::U32)?
                .to_scalar::<u32>()?;

            if next_token == self.eot_token {
                break;
            }
            tokens.push(next_token);
        }

        drop(model); // release lock before tokenizer decode

        let output_tokens: Vec<u32> = tokens[prompt_len..].to_vec();
        let text = self
            .tokenizer
            .decode(&output_tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        let text = text.trim().to_string();

        log::debug!("WhisperNative transcription: {text:?}");
        Ok(text)
    }

    fn name(&self) -> &str {
        "Whisper (candle)"
    }

    fn is_available(&self) -> bool {
        true
    }
}
