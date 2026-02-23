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
    begin_suppress_mask: Tensor,
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

        let config_text = std::fs::read_to_string(&config_path)?;
        let config: m::Config = serde_json::from_str(&config_text)?;

        // Parse begin_suppress_tokens from raw JSON (not in candle's Config struct).
        let begin_suppress_tokens: Vec<u32> = serde_json::from_str::<serde_json::Value>(&config_text)?
            .get("begin_suppress_tokens")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

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
        let suppress_tokens = build_suppress_token_list(
            &config.suppress_tokens, sot_token, no_timestamps_token, config.vocab_size,
        );

        // Pre-compute suppress mask tensor (reused every decode step).
        let suppress_mask_vec = build_token_mask(&suppress_tokens, config.vocab_size);
        let suppress_mask = Tensor::from_vec(suppress_mask_vec, config.vocab_size, &device)?;

        // Pre-compute begin_suppress mask (applied only on the first output token).
        // Whisper's begin_suppress_tokens typically includes EOT (50257) and space (220)
        // to prevent the model from immediately predicting "no speech".
        let begin_suppress_mask_vec = build_token_mask(&begin_suppress_tokens, config.vocab_size);
        let begin_suppress_mask = Tensor::from_vec(begin_suppress_mask_vec, config.vocab_size, &device)?;
        log::info!(
            "WhisperNativeTranscriber: begin_suppress_tokens={:?}",
            begin_suppress_tokens
        );

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
            begin_suppress_mask,
        })
    }

    /// Core inference: takes raw f32 PCM samples at any sample rate, resamples to 16 kHz,
    /// runs mel spectrogram + encoder + greedy decode, and returns the transcribed text.
    fn run_inference(&self, samples: &[f32], sample_rate: u32) -> anyhow::Result<String> {
        let duration_secs = samples.len() as f64 / sample_rate as f64;
        let (amin, amax, amean) = audio_stats(samples);
        log::debug!(
            "[whisper-dbg] PCM: {} samples, {:.2}s, min={:.4}, max={:.4}, mean={:.6}",
            samples.len(), duration_secs, amin, amax, amean
        );
        if amax - amin < 1e-6 {
            log::warn!("[whisper-dbg] Audio appears to be silence/constant!");
        }

        // ── Resample to 16 kHz if needed ─────────────────────────────
        let samples = if sample_rate != m::SAMPLE_RATE as u32 {
            let resampled = resample(samples, sample_rate, m::SAMPLE_RATE as u32);
            log::debug!(
                "[whisper-dbg] Resampled {}Hz -> {}Hz: {} samples ({:.2}s)",
                sample_rate, m::SAMPLE_RATE, resampled.len(),
                resampled.len() as f64 / m::SAMPLE_RATE as f64
            );
            resampled
        } else {
            samples.to_vec()
        };

        // ── Mel spectrogram (candle reference implementation) ─────────
        let mel = m::audio::pcm_to_mel(&self.config, &samples, &self.mel_filters);
        let n_mel = self.config.num_mel_bins;
        let n_frames = mel.len() / n_mel;

        let (mmin, mmax, mmean) = audio_stats(&mel);
        log::debug!(
            "[whisper-dbg] Mel: {} bins x {} frames, mel min={:.4}, max={:.4}, mean={:.4}",
            n_mel, n_frames, mmin, mmax, mmean
        );

        let mel_tensor = Tensor::from_vec(mel, (1, n_mel, n_frames), &self.device)?;
        log::debug!("[whisper-dbg] Mel tensor shape: {:?}", mel_tensor.dims());

        // ── Encode ──────────────────────────────────────────────────────
        let mut model = self
            .model
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let encoder_output = model.encoder.forward(&mel_tensor, true)?;
        log::debug!("[whisper-dbg] Encoder output shape: {:?}", encoder_output.dims());

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
            let flush = step == 0;

            let token_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let hidden = model.decoder.forward(&token_t, &encoder_output, flush)?;
            let logits = model.decoder.final_linear(&hidden)?;

            let seq_len = logits.dims()[1];
            let last_logits = logits.i((0, seq_len - 1))?;

            let mut last_logits = (last_logits + &self.suppress_mask)?;

            if step == 0 {
                last_logits = (last_logits + &self.begin_suppress_mask)?;
            }

            let next_token = last_logits
                .argmax(0)?
                .to_dtype(DType::U32)?
                .to_scalar::<u32>()?;

            if step < 10 || step % 50 == 0 {
                let token_text = self.tokenizer.decode(&[next_token], false).unwrap_or_default();
                let top_logit = last_logits.max(0)?.to_scalar::<f32>()?;
                log::debug!(
                    "[whisper-dbg] Step {}: token={} {:?}, logit={:.2}",
                    step, next_token, token_text, top_logit
                );
            }

            if next_token == self.eot_token {
                log::debug!("[whisper-dbg] EOT at step {}", step);
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
        let (samples, sample_rate) = voxctrl_core::stt::load_wav_pcm(wav_path)?;
        self.run_inference(&samples, sample_rate)
    }

    fn transcribe_pcm(&self, samples: &[f32], sample_rate: u32) -> anyhow::Result<String> {
        self.run_inference(samples, sample_rate)
    }

    fn name(&self) -> &str {
        "Whisper (candle)"
    }

    fn is_available(&self) -> bool {
        true
    }
}

/// Build the sorted, deduplicated list of tokens to suppress during decoding.
///
/// Includes the config's `suppress_tokens`, the SOT token, and all timestamp
/// tokens (from `no_timestamps_token + 1` up to `vocab_size`).
fn build_suppress_token_list(
    config_suppress_tokens: &[u32],
    sot_token: u32,
    no_timestamps_token: u32,
    vocab_size: usize,
) -> Vec<u32> {
    let mut tokens = config_suppress_tokens.to_vec();
    tokens.push(sot_token);
    for t in no_timestamps_token + 1..vocab_size as u32 {
        tokens.push(t);
    }
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// Build a float mask: `-inf` at each position in `suppressed`, `0.0` elsewhere.
fn build_token_mask(suppressed: &[u32], vocab_size: usize) -> Vec<f32> {
    // `suppressed` is expected to be sorted (from build_suppress_token_list or small list).
    let mut sorted = suppressed.to_vec();
    sorted.sort_unstable();
    (0..vocab_size)
        .map(|i| {
            if sorted.binary_search(&(i as u32)).is_ok() {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect()
}

/// Resample audio from `from_rate` to `to_rate` using rubato's FFT resampler
/// with proper polyphase anti-aliasing.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    use audioadapter_buffers::owned::InterleavedOwned;
    use rubato::{Fft, FixedSync, Resampler};

    let mut resampler = Fft::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        1024, // chunk size
        2,    // sub-chunks
        1,    // channels (mono)
        FixedSync::Input,
    )
    .expect("failed to create resampler");

    let output_len = resampler.process_all_needed_output_len(samples.len());
    let input_buf = InterleavedOwned::new_from(samples.to_vec(), 1, samples.len())
        .expect("failed to create input buffer");
    let mut output_buf = InterleavedOwned::new(0.0f32, 1, output_len);

    let (_, actual_output_len) = resampler
        .process_all_into_buffer(&input_buf, &mut output_buf, samples.len(), None)
        .expect("resampler failed");

    let output = output_buf.take_data();
    output[..actual_output_len].to_vec()
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── resample tests ───────────────────────────────────────────────────

    #[test]
    fn resample_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = resample(&input, 16000, 16000);
        assert_eq!(out, input);
    }

    #[test]
    fn resample_empty() {
        let out = resample(&[], 44100, 16000);
        assert!(out.is_empty());
    }

    #[test]
    fn resample_downsample_2x() {
        // 32 kHz -> 16 kHz: output should be approximately half the length.
        // Use a larger input so the FFT resampler has enough data to process.
        let input: Vec<f32> = (0..32000).map(|i| (i as f32 / 32000.0).sin()).collect();
        let out = resample(&input, 32000, 16000);
        let expected_len = 16000i64;
        assert!(
            (out.len() as i64 - expected_len).abs() < 100,
            "Expected ~{} samples, got {}",
            expected_len,
            out.len()
        );
    }

    #[test]
    fn resample_upsample_2x() {
        // 8 kHz -> 16 kHz: output should be approximately double the length.
        let input: Vec<f32> = (0..8000).map(|i| (i as f32 / 8000.0).sin()).collect();
        let out = resample(&input, 8000, 16000);
        let expected_len = 16000i64;
        assert!(
            (out.len() as i64 - expected_len).abs() < 100,
            "Expected ~{} samples, got {}",
            expected_len,
            out.len()
        );
    }

    // ── audio_stats tests ────────────────────────────────────────────────

    #[test]
    fn audio_stats_empty() {
        assert_eq!(audio_stats(&[]), (0.0, 0.0, 0.0));
    }

    #[test]
    fn audio_stats_single_value() {
        let (min, max, mean) = audio_stats(&[0.5]);
        assert!((min - 0.5).abs() < 1e-6);
        assert!((max - 0.5).abs() < 1e-6);
        assert!((mean - 0.5).abs() < 1e-6);
    }

    #[test]
    fn audio_stats_range() {
        let data = vec![-1.0, 0.0, 1.0];
        let (min, max, mean) = audio_stats(&data);
        assert!((min - (-1.0)).abs() < 1e-6);
        assert!((max - 1.0).abs() < 1e-6);
        assert!(mean.abs() < 1e-6);
    }

    // ── suppress mask tests ──────────────────────────────────────────────

    #[test]
    fn build_token_mask_basic() {
        let mask = build_token_mask(&[1, 3], 5);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask[0], 0.0);
        assert_eq!(mask[1], f32::NEG_INFINITY);
        assert_eq!(mask[2], 0.0);
        assert_eq!(mask[3], f32::NEG_INFINITY);
        assert_eq!(mask[4], 0.0);
    }

    #[test]
    fn build_token_mask_empty_suppressed() {
        let mask = build_token_mask(&[], 4);
        assert_eq!(mask.len(), 4);
        assert!(mask.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn build_token_mask_unsorted_input() {
        // build_token_mask should handle unsorted input (it sorts internally).
        let mask = build_token_mask(&[3, 1], 5);
        assert_eq!(mask[1], f32::NEG_INFINITY);
        assert_eq!(mask[3], f32::NEG_INFINITY);
        assert_eq!(mask[0], 0.0);
    }

    #[test]
    fn build_suppress_token_list_includes_sot_and_timestamps() {
        // vocab_size=20, sot=10, no_timestamps=15 -> timestamps 16..20 suppressed
        let config_tokens = vec![2, 5];
        let list = build_suppress_token_list(&config_tokens, 10, 15, 20);

        // Must contain config tokens, SOT, and timestamp tokens 16,17,18,19
        for &t in &[2, 5, 10, 16, 17, 18, 19] {
            assert!(list.contains(&t), "expected {t} in suppress list");
        }
        // Must be sorted and deduplicated
        for w in list.windows(2) {
            assert!(w[0] < w[1], "suppress list not sorted: {:?}", list);
        }
        // no_timestamps_token (15) itself should NOT be in the list
        // (unless it was in config_tokens, which it isn't here)
        assert!(!list.contains(&15));
    }

    #[test]
    fn suppress_mask_has_correct_dimensions() {
        // End-to-end: build list -> build mask -> check dimensions match vocab_size
        let vocab_size = 100;
        let list = build_suppress_token_list(&[1, 50], 50, 90, vocab_size);
        let mask = build_token_mask(&list, vocab_size);
        assert_eq!(mask.len(), vocab_size);

        // Count suppressed positions
        let n_suppressed = mask.iter().filter(|&&v| v == f32::NEG_INFINITY).count();
        assert_eq!(n_suppressed, list.len());
    }
}
