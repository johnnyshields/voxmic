//! Pure-Rust Whisper backend using candle-core + candle-transformers + hf-hub.
//!
//! Uses the candle-transformers reference mel spectrogram implementation
//! with pre-computed mel filter banks from the OpenAI whisper repo.

use std::collections::HashSet;
use std::path::Path;

use byteorder::{ByteOrder, LittleEndian};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper as m;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use voxctrl_core::stt::Transcriber;
use voxctrl_core::config::SttConfig;

const MAX_DECODE_TOKENS: usize = 224;

/// Maximum consecutive duplicates of the same token before forcing EOT.
/// A value of 2 means: allow the original + 2 duplicates (3 total), then halt.
const MAX_CONSECUTIVE_DUPLICATES: usize = 2;

// Fallback token IDs for the standard Whisper tokenizer. Used when
// `tokenizer.token_to_id()` returns `None` (e.g. a stripped or
// incompatible tokenizer file).
const FALLBACK_SOT_TOKEN: u32 = 50258;
const FALLBACK_EOT_TOKEN: u32 = 50257;
const FALLBACK_TRANSCRIBE_TOKEN: u32 = 50359;
const FALLBACK_NO_TIMESTAMPS_TOKEN: u32 = 50363;

// ── Transcriber ─────────────────────────────────────────────────────────────

/// Pure-Rust Whisper transcriber backed by the candle framework.
///
/// Stores the `VarBuilder` (mmapped weights via Arc) and `Config` instead of a
/// persistent `Whisper` model. A fresh model is constructed for each inference
/// call, guaranteeing zero mutable-state leakage between calls. Weight tensors
/// are shared via Arc so reconstruction is ~O(1).
pub struct WhisperNativeTranscriber {
    vb: VarBuilder<'static>,
    config: m::Config,
    tokenizer: Tokenizer,
    device: Device,
    mel_filters: Vec<f32>,
    language_token: Option<u32>,
    /// Whether the language is English (enables non-Latin hallucination detection).
    language_is_english: bool,
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

        // Look up special tokens from the tokenizer.
        let sot_token = tokenizer
            .token_to_id("<|startoftranscript|>")
            .unwrap_or_else(|| {
                log::warn!("Tokenizer missing <|startoftranscript|>, using fallback {FALLBACK_SOT_TOKEN}");
                FALLBACK_SOT_TOKEN
            });
        let eot_token = tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or_else(|| {
                log::warn!("Tokenizer missing <|endoftext|>, using fallback {FALLBACK_EOT_TOKEN}");
                FALLBACK_EOT_TOKEN
            });
        let transcribe_token = tokenizer
            .token_to_id("<|transcribe|>")
            .unwrap_or_else(|| {
                log::warn!("Tokenizer missing <|transcribe|>, using fallback {FALLBACK_TRANSCRIBE_TOKEN}");
                FALLBACK_TRANSCRIBE_TOKEN
            });
        let no_timestamps_token = tokenizer
            .token_to_id("<|notimestamps|>")
            .unwrap_or_else(|| {
                log::warn!("Tokenizer missing <|notimestamps|>, using fallback {FALLBACK_NO_TIMESTAMPS_TOKEN}");
                FALLBACK_NO_TIMESTAMPS_TOKEN
            });

        let language_is_english = cfg.whisper_language.as_deref() == Some("en");

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

        // Verify model loads correctly before committing to this VarBuilder.
        let _ = m::model::Whisper::load(&vb, config.clone())?;

        log::info!("WhisperNativeTranscriber: ready ({} suppress tokens)", suppress_tokens.len());
        Ok(Self {
            vb,
            config,
            tokenizer,
            device,
            mel_filters,
            language_token,
            language_is_english,
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
    ///
    /// A fresh `Whisper` model is constructed from the shared `VarBuilder` on each
    /// call, guaranteeing no mutable state carries over between inferences.
    fn run_inference(&self, samples: &[f32], sample_rate: u32) -> anyhow::Result<String> {
        let duration_secs = samples.len() as f64 / sample_rate as f64;
        log::info!(
            "[whisper] inference: {} samples, {:.2}s",
            samples.len(), duration_secs
        );

        if samples.is_empty() {
            return Ok(String::new());
        }

        let (amin, amax, _amean) = audio_stats(samples);
        if amax - amin < 1e-6 {
            log::warn!("[whisper] audio appears to be silence/constant");
        }

        // ── Resample to 16 kHz if needed ─────────────────────────────
        let samples = if sample_rate != m::SAMPLE_RATE as u32 {
            resample(samples, sample_rate, m::SAMPLE_RATE as u32)
        } else {
            samples.to_vec()
        };

        // ── Mel spectrogram (candle reference implementation) ─────────
        let mel = m::audio::pcm_to_mel(&self.config, &samples, &self.mel_filters);
        let n_mel = self.config.num_mel_bins;
        let n_frames = mel.len() / n_mel;

        let mel_tensor = Tensor::from_vec(mel, (1, n_mel, n_frames), &self.device)?;

        // ── Fresh model per inference (zero state leakage) ───────────
        let mut model = m::model::Whisper::load(&self.vb, self.config.clone())?;

        // ── Encode ──────────────────────────────────────────────────────
        let encoder_output = model.encoder.forward(&mel_tensor, true)?;

        // ── Greedy decode with hallucination guards ─────────────────────
        let mut tokens: Vec<u32> = vec![self.sot_token];
        if let Some(lang) = self.language_token {
            tokens.push(lang);
        }
        tokens.push(self.transcribe_token);
        tokens.push(self.no_timestamps_token);
        let prompt_len = tokens.len();

        // Duration-proportional token limit: short audio can't produce many tokens.
        let duration_token_limit = (duration_secs * 15.0).max(10.0) as usize;
        let token_limit = MAX_DECODE_TOKENS.min(duration_token_limit);

        let mut consecutive_repeats: usize = 0;
        let mut last_token: Option<u32> = None;

        for step in 0..token_limit {
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

            if next_token == self.eot_token {
                log::debug!("[whisper] EOT at step {}", step);
                break;
            }

            // ── Hallucination guard: repetition detector ─────────────
            if last_token == Some(next_token) {
                consecutive_repeats += 1;
                if consecutive_repeats >= MAX_CONSECUTIVE_DUPLICATES {
                    log::warn!(
                        "[whisper] halting: token {} seen {} times consecutively at step {}",
                        next_token, consecutive_repeats + 1, step
                    );
                    break;
                }
            } else {
                consecutive_repeats = 0;
            }
            last_token = Some(next_token);

            // ── Hallucination guard: non-Latin for English ───────────
            if self.language_is_english {
                if let Ok(text) = self.tokenizer.decode(&[next_token], false) {
                    if contains_non_latin(&text) {
                        log::warn!(
                            "[whisper] halting: non-Latin token {:?} at step {} (lang=en)",
                            text, step
                        );
                        break;
                    }
                }
            }

            tokens.push(next_token);
        }

        let output_tokens: Vec<u32> = tokens[prompt_len..].to_vec();

        let text = self
            .tokenizer
            .decode(&output_tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        let text = text.trim().to_string();

        log::info!("[whisper] result: {:?}", text);
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
    let suppressed_set: HashSet<u32> = suppressed.iter().copied().collect();
    (0..vocab_size)
        .map(|i| {
            if suppressed_set.contains(&(i as u32)) {
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

/// Returns `true` if `text` contains CJK, Hangul, or other non-Latin script
/// characters that indicate hallucination when the language is English.
fn contains_non_latin(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c,
            '\u{2E80}'..='\u{9FFF}'  // CJK Radicals Supplement through CJK Unified Ideographs
            | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
            | '\u{3040}'..='\u{30FF}' // Hiragana + Katakana
            | '\u{1100}'..='\u{11FF}' // Hangul Jamo
        )
    })
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
        // build_token_mask handles unsorted input via HashSet.
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

    // ── repeated inference stability test ────────────────────────────────

    /// Verify that repeated transcriptions of the same audio produce
    /// consistent results. This catches KV-cache state leaking between
    /// calls. Requires model files on disk — skip in CI.
    #[test]
    #[ignore] // requires model files; run manually with `cargo test -- --ignored`
    fn repeated_inference_stability() {
        let cfg = SttConfig::default();
        let transcriber = WhisperNativeTranscriber::new(&cfg, None)
            .expect("failed to load model (is it downloaded?)");

        // Generate a simple tone as repeatable test input (440 Hz, 2 seconds, 16 kHz).
        let sample_rate = 16000u32;
        let duration_secs = 2.0f32;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        let mut results = Vec::new();
        for i in 0..5 {
            let text = transcriber
                .run_inference(&samples, sample_rate)
                .unwrap_or_else(|e| panic!("inference #{i} failed: {e}"));
            eprintln!("  inference #{i}: {:?}", text);
            results.push(text);
        }

        // All results should be identical.
        for (i, result) in results.iter().enumerate().skip(1) {
            assert_eq!(
                &results[0], result,
                "inference #{i} differs from #0: {:?} vs {:?}",
                results[0], result
            );
        }
    }

    // ── HashSet equivalence test (#5) ───────────────────────────────────

    #[test]
    fn build_token_mask_hashset_equivalence() {
        // Reference: the old sorted + binary_search implementation.
        fn build_token_mask_sorted(suppressed: &[u32], vocab_size: usize) -> Vec<f32> {
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

        let vocab_size = 15;

        // Normal unsorted input
        let tokens = vec![3, 1, 4, 1, 5, 9, 2, 6];
        assert_eq!(
            build_token_mask(&tokens, vocab_size),
            build_token_mask_sorted(&tokens, vocab_size),
        );

        // Input with duplicates
        let tokens_dup = vec![5, 5, 5, 3, 3];
        assert_eq!(
            build_token_mask(&tokens_dup, vocab_size),
            build_token_mask_sorted(&tokens_dup, vocab_size),
        );

        // Empty input
        assert_eq!(
            build_token_mask(&[], vocab_size),
            build_token_mask_sorted(&[], vocab_size),
        );

        // All tokens suppressed
        let all: Vec<u32> = (0..vocab_size as u32).collect();
        assert_eq!(
            build_token_mask(&all, vocab_size),
            build_token_mask_sorted(&all, vocab_size),
        );
    }

    // ── begin_suppress_mask step-0-only test (#6) ───────────────────────

    #[test]
    fn begin_suppress_mask_applied_only_on_step_0() {
        // Simulate the decode loop mask logic with synthetic tensors.
        let vocab_size = 10;

        // suppress_mask: suppress tokens 2, 5
        let suppress_mask_vec = build_token_mask(&[2, 5], vocab_size);
        let suppress_mask =
            Tensor::from_vec(suppress_mask_vec, vocab_size, &Device::Cpu).unwrap();

        // begin_suppress_mask: additionally suppress tokens 0, 7
        let begin_suppress_mask_vec = build_token_mask(&[0, 7], vocab_size);
        let begin_suppress_mask =
            Tensor::from_vec(begin_suppress_mask_vec, vocab_size, &Device::Cpu).unwrap();

        // Uniform logits (all 1.0)
        let logits = Tensor::ones(vocab_size, DType::F32, &Device::Cpu).unwrap();

        // Step 0: both masks applied (mirrors the decode loop)
        let step0_logits = logits.broadcast_add(&suppress_mask).unwrap();
        let step0_logits = step0_logits.broadcast_add(&begin_suppress_mask).unwrap();
        let step0: Vec<f32> = step0_logits.to_vec1().unwrap();

        // Tokens 0, 2, 5, 7 should be suppressed
        for idx in [0, 2, 5, 7] {
            assert!(
                step0[idx].is_infinite() && step0[idx].is_sign_negative(),
                "step 0: token {idx} should be -inf, got {}",
                step0[idx],
            );
        }
        // Tokens 1, 3, 4, 6, 8, 9 should be untouched
        for idx in [1, 3, 4, 6, 8, 9] {
            assert!(
                (step0[idx] - 1.0).abs() < 1e-6,
                "step 0: token {idx} should be 1.0, got {}",
                step0[idx],
            );
        }

        // Step 1+: only suppress_mask applied
        let step1_logits = logits.broadcast_add(&suppress_mask).unwrap();
        let step1: Vec<f32> = step1_logits.to_vec1().unwrap();

        // Tokens 2, 5 should be suppressed
        for idx in [2, 5] {
            assert!(
                step1[idx].is_infinite() && step1[idx].is_sign_negative(),
                "step 1+: token {idx} should be -inf, got {}",
                step1[idx],
            );
        }
        // Tokens 0, 7 should NOT be suppressed (begin_suppress not applied)
        for idx in [0, 7] {
            assert!(
                (step1[idx] - 1.0).abs() < 1e-6,
                "step 1+: token {idx} should be 1.0, got {}",
                step1[idx],
            );
        }
    }

    // ── contains_non_latin tests ────────────────────────────────────────

    #[test]
    fn contains_non_latin_ascii() {
        assert!(!contains_non_latin("hello world"));
        assert!(!contains_non_latin("Hello, World! 123"));
        assert!(!contains_non_latin(""));
    }

    #[test]
    fn contains_non_latin_cjk() {
        assert!(contains_non_latin("\u{4e16}\u{754c}")); // 世界
        assert!(contains_non_latin("hello \u{4e16}\u{754c}"));
    }

    #[test]
    fn contains_non_latin_hangul() {
        assert!(contains_non_latin("\u{d55c}\u{ad6d}\u{c5b4}")); // 한국어
    }

    #[test]
    fn contains_non_latin_japanese() {
        assert!(contains_non_latin("\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}")); // こんにちは
        assert!(contains_non_latin("\u{30ab}\u{30bf}\u{30ab}\u{30ca}")); // カタカナ
    }

    #[test]
    fn contains_non_latin_accented_latin() {
        // Accented Latin characters should NOT trigger the detector.
        assert!(!contains_non_latin("caf\u{e9}")); // café
        assert!(!contains_non_latin("\u{fc}ber")); // über
    }

    // ── duration-proportional token limit tests ─────────────────────────

    /// Compute the effective token limit the same way the decode loop does.
    fn compute_token_limit(duration_secs: f64) -> usize {
        let duration_token_limit = (duration_secs * 15.0).max(10.0) as usize;
        MAX_DECODE_TOKENS.min(duration_token_limit)
    }

    #[test]
    fn duration_token_limit_short_audio() {
        // 0.5s: 0.5 * 15 = 7.5, clamped to min 10
        assert_eq!(compute_token_limit(0.5), 10);
    }

    #[test]
    fn duration_token_limit_two_seconds() {
        // 2s: 2 * 15 = 30
        assert_eq!(compute_token_limit(2.0), 30);
    }

    #[test]
    fn duration_token_limit_fifteen_seconds() {
        // 15s: 15 * 15 = 225, capped at MAX_DECODE_TOKENS (224)
        assert_eq!(compute_token_limit(15.0), MAX_DECODE_TOKENS);
    }

    #[test]
    fn duration_token_limit_thirty_seconds() {
        // 30s: 30 * 15 = 450, capped at MAX_DECODE_TOKENS (224)
        assert_eq!(compute_token_limit(30.0), MAX_DECODE_TOKENS);
    }

    #[test]
    fn duration_token_limit_zero() {
        // 0s: 0 * 15 = 0, clamped to min 10
        assert_eq!(compute_token_limit(0.0), 10);
    }

    // ── repetition detector tests ───────────────────────────────────────

    /// Simulate the repetition detection logic from the decode loop.
    /// Returns the number of tokens accepted before the detector halts.
    fn run_repetition_detector(token_sequence: &[u32]) -> usize {
        let mut consecutive_repeats: usize = 0;
        let mut last_token: Option<u32> = None;
        let mut accepted = 0;

        for &next_token in token_sequence {
            if last_token == Some(next_token) {
                consecutive_repeats += 1;
                if consecutive_repeats >= MAX_CONSECUTIVE_DUPLICATES {
                    break;
                }
            } else {
                consecutive_repeats = 0;
            }
            last_token = Some(next_token);
            accepted += 1;
        }
        accepted
    }

    #[test]
    fn repetition_detector_no_repeats() {
        // All distinct tokens: all accepted
        assert_eq!(run_repetition_detector(&[1, 2, 3, 4, 5]), 5);
    }

    #[test]
    fn repetition_detector_two_duplicates_halts() {
        // Token 7 appears 3 times: original + 2 duplicates → halt on 3rd
        assert_eq!(run_repetition_detector(&[7, 7, 7, 99]), 2);
    }

    #[test]
    fn repetition_detector_one_duplicate_continues() {
        // Token 7 appears twice (1 duplicate), then different token → no halt
        assert_eq!(run_repetition_detector(&[7, 7, 8]), 3);
    }

    #[test]
    fn repetition_detector_reset_between_runs() {
        // Two separate runs of 2: A A B B → no halt (each run only 1 duplicate)
        assert_eq!(run_repetition_detector(&[1, 1, 2, 2, 3]), 5);
    }

    #[test]
    fn repetition_detector_halts_exactly_at_threshold() {
        // Exactly MAX_CONSECUTIVE_DUPLICATES + 1 = 3 of the same token
        // Should accept 2 (original + 1 duplicate), halt before accepting 3rd
        assert_eq!(run_repetition_detector(&[5, 5, 5]), 2);
    }

    #[test]
    fn repetition_detector_empty_input() {
        assert_eq!(run_repetition_detector(&[]), 0);
    }

    #[test]
    fn repetition_detector_single_token() {
        assert_eq!(run_repetition_detector(&[42]), 1);
    }

    // ── model_to_repo tests ─────────────────────────────────────────────

    #[test]
    fn model_to_repo_short_name() {
        assert_eq!(model_to_repo("large-v3"), "openai/whisper-large-v3");
        assert_eq!(model_to_repo("tiny.en"), "openai/whisper-tiny.en");
        assert_eq!(model_to_repo("base"), "openai/whisper-base");
    }

    #[test]
    fn model_to_repo_full_repo_id() {
        // Already contains '/' → passthrough
        assert_eq!(model_to_repo("openai/whisper-large-v3"), "openai/whisper-large-v3");
        assert_eq!(model_to_repo("custom-org/my-model"), "custom-org/my-model");
    }
}
