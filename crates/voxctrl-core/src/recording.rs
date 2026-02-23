//! Recording state machine — shared between GUI and TUI.

use std::sync::Arc;

use crate::config::Config;
use crate::pipeline::Pipeline;
use crate::{AppStatus, SharedState};

/// Toggle the recording state: Idle → Recording → Transcribing → (back to Idle).
pub fn toggle_recording(
    state: &Arc<SharedState>,
    cfg: &Config,
    pipeline: Arc<Pipeline>,
) {
    let current = *state.status.lock().unwrap();
    match current {
        AppStatus::Idle => {
            state.chunks.lock().unwrap().clear();
            *state.status.lock().unwrap() = AppStatus::Recording;
            log::info!("Recording started");
        }
        AppStatus::Recording => {
            *state.status.lock().unwrap() = AppStatus::Transcribing;
            log::info!("Recording stopped, transcribing…");

            let chunks: Vec<f32> = state.chunks.lock().unwrap().drain(..).collect();
            if chunks.is_empty() {
                log::info!("No audio captured, returning to idle");
                *state.status.lock().unwrap() = AppStatus::Idle;
                return;
            }

            let state_clone = state.clone();
            let sample_rate = cfg.audio.sample_rate;
            std::thread::Builder::new()
                .name("transcription".into())
                .spawn(move || {
                    if let Err(e) = transcribe_via_pipeline(&chunks, sample_rate, &pipeline) {
                        log::error!("Pipeline error: {e}");
                    }
                    *state_clone.status.lock().unwrap() = AppStatus::Idle;
                    log::info!("Back to idle");
                })
                .expect("spawn transcription thread");
        }
        AppStatus::Transcribing => {
            log::debug!("Ignoring toggle — already transcribing");
        }
    }
}

/// Send PCM chunks through the pipeline directly (no temp WAV).
fn transcribe_via_pipeline(
    chunks: &[f32],
    sample_rate: u32,
    pipeline: &Pipeline,
) -> anyhow::Result<()> {
    pipeline.process_pcm(chunks, sample_rate)
}
