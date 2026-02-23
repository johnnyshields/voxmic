//! Pipeline — wires together STT → Router → Action.

use std::path::{Path, PathBuf};

use crate::action::ActionExecutor;
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
    pub fn from_config(
        cfg: &Config,
        stt_model_dir: Option<PathBuf>,
        stt_factory: Option<&SttFactory>,
    ) -> anyhow::Result<Self> {
        let stt = crate::stt::create_transcriber(&cfg.stt, stt_model_dir, stt_factory)?;
        let router = crate::router::create_router(&cfg.router)?;
        let action = crate::action::create_action(&cfg.action)?;

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
