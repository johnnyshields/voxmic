//! Action Executor — pluggable trait + factory.

pub mod type_text;
#[cfg(feature = "action-computer-use")]
pub mod computer_use;

use crate::config::ActionConfig;
use crate::router::Intent;

/// Trait for action execution backends.
pub trait ActionExecutor: Send + Sync {
    /// Execute an intent (type text, run command, etc.).
    fn execute(&self, intent: &Intent) -> anyhow::Result<()>;
    fn name(&self) -> &str;
}

/// Create an action executor based on config.
pub fn create_action(cfg: &ActionConfig) -> anyhow::Result<Box<dyn ActionExecutor>> {
    match cfg.backend.as_str() {
        "type-text" => Ok(Box::new(type_text::TypeTextAction)),
        "computer-use" => {
            #[cfg(feature = "action-computer-use")]
            return Ok(Box::new(computer_use::ComputerUseAction::new(cfg)?));
            #[cfg(not(feature = "action-computer-use"))]
            anyhow::bail!("action-computer-use feature not compiled in");
        }
        other => anyhow::bail!("Unknown action backend: {other}"),
    }
}
