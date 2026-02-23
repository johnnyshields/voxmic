//! Recording state machine — shared between GUI and TUI.

use std::sync::Arc;

use crate::config::Config;
use crate::pipeline::SharedPipeline;
use crate::{AppStatus, SharedState};

/// Toggle the recording state: Idle → Recording → Transcribing → (back to Idle).
///
/// - **Idle → Recording**: clears buffered chunks, sets status to Recording.
/// - **Recording → Transcribing → Idle**: drains chunks, spawns a transcription
///   thread (or returns to Idle immediately if no audio was captured).
/// - **Transcribing → (ignored)**: toggle is a no-op while a transcription is
///   already in flight.
pub fn toggle_recording(
    state: &Arc<SharedState>,
    cfg: &Config,
    pipeline: &Arc<SharedPipeline>,
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
            // Snapshot the current pipeline — in-flight transcription keeps it alive
            let snap = pipeline.get();
            std::thread::Builder::new()
                .name("transcription".into())
                .spawn(move || {
                    if let Err(e) = snap.process_pcm(&chunks, sample_rate) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use crate::action::ActionExecutor;
    use crate::router::{Intent, IntentRouter};
    use crate::stt::Transcriber;

    struct StubTranscriber;
    impl Transcriber for StubTranscriber {
        fn transcribe(&self, _: &std::path::Path) -> anyhow::Result<String> { Ok(String::new()) }
        fn transcribe_pcm(&self, _: &[f32], _: u32) -> anyhow::Result<String> { Ok("ok".into()) }
        fn name(&self) -> &str { "stub" }
        fn is_available(&self) -> bool { true }
    }

    struct StubRouter;
    impl IntentRouter for StubRouter {
        fn route(&self, text: &str) -> anyhow::Result<Intent> { Ok(Intent::Dictate(text.into())) }
        fn name(&self) -> &str { "stub" }
    }

    struct StubAction {
        executed: Arc<Mutex<Vec<String>>>,
    }
    impl ActionExecutor for StubAction {
        fn execute(&self, intent: &Intent) -> anyhow::Result<()> {
            if let Intent::Dictate(t) = intent {
                self.executed.lock().unwrap().push(t.clone());
            }
            Ok(())
        }
        fn name(&self) -> &str { "stub" }
    }

    use crate::pipeline::{Pipeline, SharedPipeline};

    fn make_pipeline() -> (Arc<SharedPipeline>, Arc<Mutex<Vec<String>>>) {
        let executed = Arc::new(Mutex::new(vec![]));
        let pipeline = Arc::new(SharedPipeline::new(Pipeline {
            stt: Box::new(StubTranscriber),
            router: Box::new(StubRouter),
            action: Box::new(StubAction { executed: executed.clone() }),
        }));
        (pipeline, executed)
    }

    #[test]
    fn idle_to_recording() {
        let state = Arc::new(SharedState::new());
        let (pipeline, _) = make_pipeline();
        let cfg = Config::default();

        toggle_recording(&state, &cfg, &pipeline);

        assert_eq!(*state.status.lock().unwrap(), AppStatus::Recording);
        assert!(state.chunks.lock().unwrap().is_empty());
    }

    #[test]
    fn recording_with_audio_transitions_to_transcribing_then_idle() {
        let state = Arc::new(SharedState::new());
        let (pipeline, executed) = make_pipeline();
        let cfg = Config::default();

        // Idle → Recording
        *state.status.lock().unwrap() = AppStatus::Recording;
        state.chunks.lock().unwrap().extend_from_slice(&[0.1, 0.2, 0.3]);

        // Recording → Transcribing (spawns thread)
        toggle_recording(&state, &cfg, &pipeline);

        // Wait for the transcription thread to finish
        for _ in 0..200 {
            if *state.status.lock().unwrap() == AppStatus::Idle {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert_eq!(*state.status.lock().unwrap(), AppStatus::Idle);
        assert_eq!(&*executed.lock().unwrap(), &["ok"]);
    }

    #[test]
    fn recording_empty_audio_returns_to_idle() {
        let state = Arc::new(SharedState::new());
        let (pipeline, _) = make_pipeline();
        let cfg = Config::default();

        *state.status.lock().unwrap() = AppStatus::Recording;
        // No chunks pushed — empty audio

        toggle_recording(&state, &cfg, &pipeline);

        // Should return to Idle synchronously (no thread spawned)
        assert_eq!(*state.status.lock().unwrap(), AppStatus::Idle);
    }

    #[test]
    fn transcribing_ignores_toggle() {
        let state = Arc::new(SharedState::new());
        let (pipeline, _) = make_pipeline();
        let cfg = Config::default();

        *state.status.lock().unwrap() = AppStatus::Transcribing;

        toggle_recording(&state, &cfg, &pipeline);

        assert_eq!(*state.status.lock().unwrap(), AppStatus::Transcribing);
    }
}

