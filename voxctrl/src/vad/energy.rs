//! Energy-based Voice Activity Detection — simple RMS threshold.

use super::VoiceDetector;

/// Simple energy-based VAD that computes the RMS of audio samples
/// and compares against a configurable threshold.
pub struct EnergyVad {
    threshold: f64,
}

impl EnergyVad {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl VoiceDetector for EnergyVad {
    fn is_speech(&mut self, samples: &[f32], _sample_rate: u32) -> bool {
        if samples.is_empty() {
            return false;
        }
        let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let rms = (sum_sq / samples.len() as f64).sqrt();
        rms > self.threshold
    }

    fn name(&self) -> &str {
        "energy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_not_speech() {
        let mut vad = EnergyVad::new(0.015);
        let silence = vec![0.0f32; 1600]; // 100ms at 16kHz
        assert!(!vad.is_speech(&silence, 16000));
    }

    #[test]
    fn loud_signal_is_speech() {
        let mut vad = EnergyVad::new(0.015);
        let loud = vec![0.5f32; 1600]; // constant 0.5 amplitude
        assert!(vad.is_speech(&loud, 16000));
    }

    #[test]
    fn empty_buffer_is_not_speech() {
        let mut vad = EnergyVad::new(0.015);
        assert!(!vad.is_speech(&[], 16000));
    }

    #[test]
    fn threshold_boundary() {
        // RMS of a constant signal of amplitude A is A itself.
        // With threshold 0.1, a signal at 0.1 should NOT trigger (strictly greater).
        let mut vad = EnergyVad::new(0.1);
        let at_threshold = vec![0.1f32; 1600];
        assert!(!vad.is_speech(&at_threshold, 16000));

        let above = vec![0.101f32; 1600];
        assert!(vad.is_speech(&above, 16000));
    }

    #[test]
    fn sine_wave_triggers() {
        let mut vad = EnergyVad::new(0.01);
        // 440 Hz sine wave at 16kHz sample rate, amplitude 0.3
        let samples: Vec<f32> = (0..1600)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        assert!(vad.is_speech(&samples, 16000));
    }
}
