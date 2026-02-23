//! Pipeline — wires together STT → Router → Action.

use std::path::{Path, PathBuf};

use crate::action::{ActionExecutor, ActionFactory};
use crate::config::Config;
use crate::router::{Intent, IntentRouter};
use crate::stt::{SttFactory, Transcriber};

pub struct Pipeline {
    pub stt: Box<dyn Transcriber>,
    pub router: Box<dyn IntentRouter>,
    pub action: Box<dyn ActionExecutor>,
}

impl Pipeline {
    /// Build a pipeline from config, creating all backends.
    ///
    /// `stt_model_dir` is the resolved local model path for backends that need
    /// local model files (e.g. voxtral-native). Other backends ignore it.
    ///
    /// `stt_factory` allows external crates to inject heavy STT backends.
    /// `action_factory` allows external crates to inject action backends (e.g. computer-use).
    pub fn from_config(
        cfg: &Config,
        stt_model_dir: Option<PathBuf>,
        stt_factory: Option<&SttFactory>,
        action_factory: Option<&ActionFactory>,
    ) -> anyhow::Result<Self> {
        let stt = crate::stt::create_transcriber(&cfg.stt, stt_model_dir, stt_factory)?;
        let router = crate::router::create_router(&cfg.router)?;
        let action = crate::action::create_action(&cfg.action, action_factory)?;

        log::info!(
            "Pipeline: STT={}, Router={}, Action={}",
            stt.name(),
            router.name(),
            action.name(),
        );

        Ok(Self {
            stt,
            router,
            action,
        })
    }

    /// Run the full pipeline: transcribe → route → execute.
    pub fn process(&self, wav_path: &Path) -> anyhow::Result<()> {
        let start = std::time::Instant::now();

        // STT
        let text = self.stt.transcribe(wav_path)?;
        let stt_elapsed = start.elapsed().as_secs_f64();

        self.route_and_execute(start, stt_elapsed, text)
    }

    /// Run the full pipeline from raw PCM: transcribe → route → execute.
    pub fn process_pcm(&self, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
        let start = std::time::Instant::now();

        // STT
        let text = self.stt.transcribe_pcm(samples, sample_rate)?;
        let stt_elapsed = start.elapsed().as_secs_f64();

        self.route_and_execute(start, stt_elapsed, text)
    }

    /// Shared tail of the pipeline: log STT result, route, execute.
    fn route_and_execute(
        &self,
        start: std::time::Instant,
        stt_elapsed: f64,
        text: String,
    ) -> anyhow::Result<()> {
        if text.is_empty() {
            log::info!("STT returned empty text ({:.1}s), skipping", stt_elapsed);
            return Ok(());
        }

        let preview = if text.len() > 80 { &text[..80] } else { &text };
        log::info!("STT ({:.1}s): {}", stt_elapsed, preview);

        // Route
        let intent = self.router.route(&text)?;
        match &intent {
            Intent::Dictate(t) => log::debug!("Router → Dictate({} chars)", t.len()),
            Intent::Command { action, .. } => log::info!("Router → Command({})", action),
        }

        // Execute
        self.action.execute(&intent)?;

        log::info!("Pipeline complete in {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }
}
