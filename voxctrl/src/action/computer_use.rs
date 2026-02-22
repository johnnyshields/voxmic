//! Computer-use action — stub for future LLM-driven desktop automation.
//!
//! TODO: screenshot capture (win32 or X11 grab)
//! TODO: OmniParser for UI element detection
//! TODO: LLM reasoning loop (screenshot → parse → plan → execute)

use crate::config::ActionConfig;
use crate::router::Intent;
use super::ActionExecutor;

/// Stub executor for LLM-driven computer-use actions.
pub struct ComputerUseAction {
    _backend: String,
}

impl ComputerUseAction {
    pub fn new(cfg: &ActionConfig) -> anyhow::Result<Self> {
        log::info!("ComputerUseAction: initialized (stub)");
        Ok(Self {
            _backend: cfg.backend.clone(),
        })
    }
}

impl ActionExecutor for ComputerUseAction {
    fn execute(&self, intent: &Intent) -> anyhow::Result<()> {
        match intent {
            Intent::Dictate(text) => {
                log::info!("ComputerUseAction [stub]: would type text: {text:?}");
            }
            Intent::Command { action, args } => {
                log::info!("ComputerUseAction [stub]: would execute action={action:?} args={args}");
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "computer-use"
    }
}
