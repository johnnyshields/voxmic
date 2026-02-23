//! Voice Activity Detection — pluggable trait + factory.

#[cfg(feature = "vad-energy")]
pub mod energy;
#[cfg(feature = "vad-silero")]
pub mod silero;

use crate::config::VadConfig;

/// Trait for voice activity detection backends.
#[allow(dead_code)]
pub trait VoiceDetector: Send {
    /// Returns true if the audio chunk likely contains speech.
    fn is_speech(&mut self, samples: &[f32], sample_rate: u32) -> bool;
    fn name(&self) -> &str;
}

/// Create a VAD backend based on config.
#[allow(dead_code)]
pub fn create_vad(cfg: &VadConfig) -> anyhow::Result<Box<dyn VoiceDetector>> {
    match cfg.backend.as_str() {
        "energy" => {
            #[cfg(feature = "vad-energy")]
            return Ok(Box::new(energy::EnergyVad::new(cfg.energy_threshold)));
            #[cfg(not(feature = "vad-energy"))]
            anyhow::bail!("vad-energy feature not compiled in");
        }
        "silero" => {
            #[cfg(feature = "vad-silero")]
            return Ok(Box::new(silero::SileroVad::new(cfg.silero_threshold)?));
            #[cfg(not(feature = "vad-silero"))]
            anyhow::bail!("vad-silero feature not compiled in");
        }
        "none" => Ok(Box::new(NullVad)),
        other => anyhow::bail!("Unknown VAD backend: {other}"),
    }
}

/// No-op VAD that considers everything as speech.
#[allow(dead_code)]
struct NullVad;
impl VoiceDetector for NullVad {
    fn is_speech(&mut self, _samples: &[f32], _sample_rate: u32) -> bool {
        true
    }
    fn name(&self) -> &str {
        "none"
    }
}
