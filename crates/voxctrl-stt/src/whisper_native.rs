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
    suppress_mask: Tensor,
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

        // Build suppress list: config.suppress_tokens + SOT + all timestamp tokens
        let mut suppress_tokens = config.suppress_tokens.clone();
        suppress_tokens.push(sot_token);
        // Suppress timestamp tokens (50364 and above, below vocab_size)
        for t in no_timestamps_token + 1..config.vocab_size as u32 {
            suppress_tokens.push(t);
        }
        suppress_tokens.sort_unstable();
        suppress_tokens.dedup();

        // Pre-compute suppress mask tensor (reused every decode step).
        let suppress_mask_vec: Vec<f32> = (0..config.vocab_size)
            .map(|i| {
                if suppress_tokens.binary_search(&(i as u32)).is_ok() {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let suppress_mask = Tensor::from_vec(suppress_mask_vec, config.vocab_size, &device)?;

        log::info!("WhisperNativeTranscriber: ready ({} suppress tokens)", suppress_tokens.len());
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
            suppress_mask,
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
        log::debug!(
            "[whisper-dbg] WAV: {:?}, channels={}, sample_rate={}, bits={}, format={:?}",
            wav_path, spec.channels, spec.sample_rate, spec.bits_per_sample, spec.sample_format
        );
        let raw: Vec<f32> = if spec.bits_per_sample == 16 {
            reader
                .into_samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<Result<_, _>>()?
        } else {
            reader.into_samples::<f32>().collect::<Result<_, _>>()?
        };

        let duration_secs = raw.len() as f64 / spec.sample_rate as f64;
        let (amin, amax, amean) = audio_stats(&raw);
        log::debug!(
            "[whisper-dbg] PCM: {} samples, {:.2}s, min={:.4}, max={:.4}, mean={:.6}",
            raw.len(), duration_secs, amin, amax, amean
        );
        if amax - amin < 1e-6 {
            log::warn!("[whisper-dbg] Audio appears to be silence/constant!");
        }

        // ── Resample to 16 kHz if needed ─────────────────────────────
        let raw = if spec.sample_rate != m::SAMPLE_RATE as u32 {
            let raw = resample(&raw, spec.sample_rate, m::SAMPLE_RATE as u32);
            log::info!(
                "[whisper-dbg] Resampled {}Hz -> {}Hz: {} samples ({:.2}s)",
                spec.sample_rate, m::SAMPLE_RATE, raw.len(),
                raw.len() as f64 / m::SAMPLE_RATE as f64
            );
            raw
        } else {
            raw
        };

        // ── Mel spectrogram (candle reference implementation) ─────────
        // pcm_to_mel pads the audio to produce exactly N_FRAMES (3000) frames,
        // matching whisper's 30-second chunk design.  The encoder's conv2 (stride 2)
        // halves this to max_source_positions (1500) before the positional embedding.
        let mel = m::audio::pcm_to_mel(&self.config, &raw, &self.mel_filters);
        let n_mel = self.config.num_mel_bins;
        let n_frames = mel.len() / n_mel;

        let (mmin, mmax, mmean) = audio_stats(&mel);
        log::debug!(
            "[whisper-dbg] Mel: {} bins x {} frames, mel min={:.4}, max={:.4}, mean={:.4}",
            n_mel, n_frames, mmin, mmax, mmean
        );

        let mel_tensor = Tensor::from_vec(mel, (1, n_mel, n_frames), &self.device)?;
        log::info!("[whisper-dbg] Mel tensor shape: {:?}", mel_tensor.dims());

        // ── Encode ──────────────────────────────────────────────────────
        let mut model = self
            .model
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let encoder_output = model.encoder.forward(&mel_tensor, true)?;
        log::info!("[whisper-dbg] Encoder output shape: {:?}", encoder_output.dims());

        // ── Greedy decode ───────────────────────────────────────────────
        let mut tokens: Vec<u32> = vec![self.sot_token];
        if let Some(lang) = self.language_token {
            tokens.push(lang);
        }
        tokens.push(self.transcribe_token);
        tokens.push(self.no_timestamps_token);
        let prompt_len = tokens.len();
        log::debug!(
            "[whisper-dbg] Prompt tokens: {:?} (sot={}, transcribe={}, no_ts={})",
            tokens, self.sot_token, self.transcribe_token, self.no_timestamps_token
        );

        for step in 0..MAX_DECODE_TOKENS {
            // Pass the full token sequence every step — the candle decoder's
            // self-attention doesn't use KV caching, so it needs all tokens.
            // Cross-attention KV cache is reused after the first forward pass.
            let flush = step == 0;

            let token_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let hidden = model.decoder.forward(&token_t, &encoder_output, flush)?;
            let logits = model.decoder.final_linear(&hidden)?;

            let seq_len = logits.dims()[1];
            let last_logits = logits.i((0, seq_len - 1))?;

            // Suppress special/timestamp tokens by setting their logits to -inf.
            let last_logits = (last_logits + &self.suppress_mask)?;

            let next_token = last_logits
                .argmax(0)?
                .to_dtype(DType::U32)?
                .to_scalar::<u32>()?;

            // Log first 10 tokens and every 50th after that
            if step < 10 || step % 50 == 0 {
                let token_text = self.tokenizer.decode(&[next_token], false).unwrap_or_default();
                let top_logit = last_logits.max(0)?.to_scalar::<f32>()?;
                log::info!(
                    "[whisper-dbg] Step {}: token={} {:?}, logit={:.2}",
                    step, next_token, token_text, top_logit
                );
            }

            if next_token == self.eot_token {
                log::info!("[whisper-dbg] EOT at step {}", step);
                break;
            }
            tokens.push(next_token);
        }

        drop(model); // release lock before tokenizer decode

        let output_tokens: Vec<u32> = tokens[prompt_len..].to_vec();
        log::debug!(
            "[whisper-dbg] Output tokens ({}): {:?}",
            output_tokens.len(),
            &output_tokens[..output_tokens.len().min(30)]
        );

        let text = self
            .tokenizer
            .decode(&output_tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        let text = text.trim().to_string();

        log::info!("[whisper-dbg] Final text: {:?}", text);
        Ok(text)
    }

    fn name(&self) -> &str {
        "Whisper (candle)"
    }

    fn is_available(&self) -> bool {
        true
    }
}

/// Resample audio from `from_rate` to `to_rate` using linear interpolation.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (samples.len() as f64 / ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx as usize;
        let frac = (src_idx - idx0 as f64) as f32;
        let s0 = samples[idx0];
        let s1 = if idx0 + 1 < samples.len() { samples[idx0 + 1] } else { s0 };
        out.push(s0 + frac * (s1 - s0));
    }
    out
}

fn audio_stats(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0f64;
    for &v in data {
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v as f64;
    }
    (min, max, (sum / data.len() as f64) as f32)
}
