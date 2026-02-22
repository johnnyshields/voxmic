//! Pure-Rust Whisper backend using candle-core + candle-transformers + hf-hub.
//!
//! Downloads the model from Hugging Face Hub on first use and runs inference
//! entirely in Rust — no C/C++ dependencies required.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper as m;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use super::Transcriber;
use crate::config::SttConfig;

// ── Audio constants (match OpenAI whisper) ──────────────────────────────────

const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const CHUNK_SECONDS: usize = 30;
const CHUNK_SAMPLES: usize = CHUNK_SECONDS * SAMPLE_RATE; // 480 000
const MAX_DECODE_TOKENS: usize = 224;

// ── Transcriber ─────────────────────────────────────────────────────────────

/// Pure-Rust Whisper transcriber backed by the candle framework.
pub struct WhisperNativeTranscriber {
    model: Mutex<m::model::Whisper>,
    tokenizer: Tokenizer,
    device: Device,
    mel_filters: Vec<f32>,
    n_mels: usize,
    language_token: Option<u32>,
    sot_token: u32,
    eot_token: u32,
    transcribe_token: u32,
    no_timestamps_token: u32,
}

impl WhisperNativeTranscriber {
    pub fn new(cfg: &SttConfig) -> anyhow::Result<Self> {
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

        let repo_id = model_to_repo(&cfg.whisper_model);
        log::info!("WhisperNativeTranscriber: downloading model {repo_id}");

        let api = Api::new()?;
        let repo = api.model(repo_id);
        let config_path = repo.get("config.json")?;
        let model_path = repo.get("model.safetensors")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        let config: m::Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let n_mels = config.num_mel_bins;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        log::info!(
            "WhisperNativeTranscriber: loading weights ({n_mels} mel bins, device={device:?})"
        );
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
        let model = m::model::Whisper::load(&vb, config)?;

        let mel_filters = compute_mel_filters(n_mels, N_FFT, SAMPLE_RATE as u32);

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
            tokenizer,
            device,
            mel_filters,
            n_mels,
            language_token,
            sot_token,
            eot_token,
            transcribe_token,
            no_timestamps_token,
        })
    }
}

/// Map a short model name to a Hugging Face repo ID.
fn model_to_repo(model: &str) -> String {
    if model.contains('/') {
        return model.to_string();
    }
    format!("openai/whisper-{model}")
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

        // Pad or truncate to exactly 30 s (what the model expects).
        let mut audio = vec![0.0f32; CHUNK_SAMPLES];
        let len = raw.len().min(CHUNK_SAMPLES);
        audio[..len].copy_from_slice(&raw[..len]);

        // ── Mel spectrogram ─────────────────────────────────────────────
        let mel = log_mel_spectrogram(&audio, &self.mel_filters, self.n_mels);
        let n_frames = CHUNK_SAMPLES / HOP_LENGTH;
        let mel_tensor = Tensor::from_vec(mel, &[1, self.n_mels, n_frames], &self.device)?;

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

// ── Audio processing (matches OpenAI whisper preprocessing) ─────────────────

/// Compute a log-mel spectrogram identical to OpenAI's whisper `log_mel_spectrogram`.
fn log_mel_spectrogram(samples: &[f32], mel_filters: &[f32], n_mels: usize) -> Vec<f32> {
    let fft_size = N_FFT.next_power_of_two(); // 400 → 512
    let n_freqs = N_FFT / 2 + 1; // 201
    let n_frames = samples.len() / HOP_LENGTH; // 3000 for 30 s

    // Periodic Hann window (PyTorch convention).
    let window: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos()))
        .collect();

    let mut mel = vec![0.0f32; n_mels * n_frames];

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP_LENGTH;

        // Window the frame and zero-pad to fft_size.
        let mut buf = vec![(0.0f32, 0.0f32); fft_size];
        for i in 0..N_FFT {
            let s = if start + i < samples.len() {
                samples[start + i]
            } else {
                0.0
            };
            buf[i] = (s * window[i], 0.0);
        }

        fft(&mut buf);

        // Power spectrum (magnitude²) of first n_freqs bins.
        // Apply mel filters and write to output.
        for mel_idx in 0..n_mels {
            let mut sum = 0.0f32;
            let filter_row = mel_idx * n_freqs;
            for freq_idx in 0..n_freqs {
                let (re, im) = buf[freq_idx];
                let power = re * re + im * im;
                sum += mel_filters[filter_row + freq_idx] * power;
            }
            mel[mel_idx * n_frames + frame_idx] = sum.max(1e-10).log10();
        }
    }

    // Clamp & normalise (whisper convention).
    let max_val = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut mel {
        *v = ((*v).max(max_val - 8.0) + 4.0) / 4.0;
    }

    mel
}

/// In-place radix-2 Cooley–Tukey FFT. `buf.len()` must be a power of two.
fn fft(buf: &mut [(f32, f32)]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation.
    let mut j: usize = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    // Butterfly passes.
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_step = -2.0 * std::f32::consts::PI / len as f32;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let angle = angle_step * k as f32;
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                let (re, im) = buf[start + k + half];
                let t_re = cos_a * re - sin_a * im;
                let t_im = cos_a * im + sin_a * re;
                let (u_re, u_im) = buf[start + k];
                buf[start + k] = (u_re + t_re, u_im + t_im);
                buf[start + k + half] = (u_re - t_re, u_im - t_im);
            }
        }
        len <<= 1;
    }
}

/// Compute an HTK mel filter bank matching `librosa.filters.mel` with Slaney normalisation.
fn compute_mel_filters(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let fmax = sample_rate as f64 / 2.0;

    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(fmax);

    // n_mels + 2 points equally spaced in mel space.
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (n_mels + 1) as f64)
        .collect();

    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Continuous FFT-bin indices.
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&hz| hz * n_fft as f64 / sample_rate as f64)
        .collect();

    let mut filters = vec![0.0f32; n_mels * n_freqs];

    for i in 0..n_mels {
        let f_left = bin_points[i];
        let f_center = bin_points[i + 1];
        let f_right = bin_points[i + 2];

        for j in 0..n_freqs {
            let freq = j as f64;
            let val = if freq >= f_left && freq < f_center {
                (freq - f_left) / (f_center - f_left)
            } else if freq >= f_center && freq <= f_right {
                (f_right - freq) / (f_right - f_center)
            } else {
                0.0
            };
            filters[i * n_freqs + j] = val as f32;
        }

        // Slaney normalisation: scale by 2 / bandwidth_hz.
        let enorm = 2.0 / (hz_points[i + 2] - hz_points[i]) as f32;
        for j in 0..n_freqs {
            filters[i * n_freqs + j] *= enorm;
        }
    }

    filters
}
