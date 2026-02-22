//! Global hotkey — Ctrl+Super+Space toggle.

use std::sync::Arc;

use anyhow::{Context, Result};
use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager};

use crate::config::Config;
use crate::pipeline::Pipeline;
use crate::{AppStatus, SharedState};

/// Register the Ctrl+Super+Space toggle hotkey.
pub fn setup_hotkeys() -> Result<(GlobalHotKeyManager, u32)> {
    let manager = GlobalHotKeyManager::new().context("create hotkey manager")?;
    let hotkey = HotKey::new(Some(Modifiers::CONTROL | Modifiers::SUPER), Code::Space);
    let id = hotkey.id();
    manager.register(hotkey).context("register Ctrl+Super+Space")?;
    Ok((manager, id))
}

/// Handle a hotkey event: toggle Idle → Recording → Transcribing.
pub fn handle_hotkey_event(
    event: &GlobalHotKeyEvent,
    hotkey_id: Option<u32>,
    state: &Arc<SharedState>,
    cfg: &Config,
    pipeline: Arc<Pipeline>,
) {
    if Some(event.id) != hotkey_id {
        return;
    }

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
            log::debug!("Ignoring hotkey — already transcribing");
        }
    }
}

/// Write chunks to WAV tempfile, run the pipeline.
fn transcribe_via_pipeline(
    chunks: &[f32],
    sample_rate: u32,
    pipeline: &Pipeline,
) -> anyhow::Result<()> {
    let tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .context("create temp WAV")?;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(tmp.path(), spec).context("create WAV writer")?;
    for &sample in chunks {
        let s16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(s16)?;
    }
    writer.finalize().context("finalize WAV")?;

    pipeline.process(tmp.path())
}
