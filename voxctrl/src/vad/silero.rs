//! Silero VAD — neural voice activity detection via ONNX Runtime.
//!
//! Uses the Silero VAD v4 ONNX model. Download from:
//! <https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx>
//!
//! Place the file at `silero_vad.onnx` next to the binary, or set the
//! `SILERO_VAD_MODEL` environment variable to the full path.

use super::VoiceDetector;
use ort::{session::Session, value::Tensor};

/// Chunk size expected by Silero VAD at 16 kHz.
const CHUNK_16K: usize = 512;
/// LSTM hidden/cell state dimensions: [2, batch=1, 64].
const STATE_SHAPE: [usize; 3] = [2, 1, 64];
const STATE_LEN: usize = STATE_SHAPE[0] * STATE_SHAPE[1] * STATE_SHAPE[2];

pub struct SileroVad {
    session: Session,
    threshold: f32,
    h: Vec<f32>,
    c: Vec<f32>,
}

impl SileroVad {
    /// Create a new Silero VAD instance.
    ///
    /// Loads the ONNX model from `$SILERO_VAD_MODEL` or `silero_vad.onnx`
    /// next to the executable.
    pub fn new(threshold: f32) -> anyhow::Result<Self> {
        let model_path = std::env::var("SILERO_VAD_MODEL").unwrap_or_else(|_| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("silero_vad.onnx")))
                .unwrap_or_else(|| "silero_vad.onnx".into())
                .to_string_lossy()
                .into_owned()
        });

        let session = Session::builder()?.commit_from_file(&model_path)?;

        Ok(Self {
            session,
            threshold,
            h: vec![0.0; STATE_LEN],
            c: vec![0.0; STATE_LEN],
        })
    }
}

impl VoiceDetector for SileroVad {
    fn is_speech(&mut self, samples: &[f32], _sample_rate: u32) -> bool {
        // Silero VAD expects exactly CHUNK_16K samples at 16 kHz.
        // Pad or truncate to fit.
        let mut chunk = vec![0.0f32; CHUNK_16K];
        let copy_len = samples.len().min(CHUNK_16K);
        chunk[..copy_len].copy_from_slice(&samples[..copy_len]);

        let input = match Tensor::from_array(([1usize, CHUNK_16K], chunk.into_boxed_slice())) {
            Ok(t) => t,
            Err(e) => {
                log::error!("silero: failed to create input tensor: {e}");
                return false;
            }
        };
        let sr = match Tensor::from_array(([1usize], vec![16000_i64].into_boxed_slice())) {
            Ok(t) => t,
            Err(e) => {
                log::error!("silero: failed to create sr tensor: {e}");
                return false;
            }
        };
        let h = match Tensor::from_array((STATE_SHAPE.to_vec(), self.h.clone().into_boxed_slice()))
        {
            Ok(t) => t,
            Err(e) => {
                log::error!("silero: failed to create h tensor: {e}");
                return false;
            }
        };
        let c = match Tensor::from_array((STATE_SHAPE.to_vec(), self.c.clone().into_boxed_slice()))
        {
            Ok(t) => t,
            Err(e) => {
                log::error!("silero: failed to create c tensor: {e}");
                return false;
            }
        };

        let outputs = match self.session.run(ort::inputs![input, sr, h, c]) {
            Ok(o) => o,
            Err(e) => {
                log::error!("silero: inference failed: {e}");
                return false;
            }
        };

        // Extract speech probability.
        let prob = outputs
            .get(0)
            .and_then(|v| v.try_extract_raw_tensor::<f32>().ok())
            .map(|(_, data)| data[0])
            .unwrap_or(0.0);

        // Update LSTM states for next call.
        if let Some(hn) = outputs.get(1) {
            if let Ok((_, data)) = hn.try_extract_raw_tensor::<f32>() {
                if data.len() == STATE_LEN {
                    self.h.copy_from_slice(data);
                }
            }
        }
        if let Some(cn) = outputs.get(2) {
            if let Ok((_, data)) = cn.try_extract_raw_tensor::<f32>() {
                if data.len() == STATE_LEN {
                    self.c.copy_from_slice(data);
                }
            }
        }

        prob > self.threshold
    }

    fn name(&self) -> &str {
        "silero"
    }
}
