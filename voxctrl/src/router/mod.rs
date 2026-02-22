//! Intent Router — pluggable trait + factory.
//!
//! Routes transcribed text to an intent: dictation (type text) or command (execute action).

pub mod passthrough;
#[cfg(feature = "router-llm")]
pub mod llm;

use crate::config::RouterConfig;

/// What should happen with the transcribed text.
#[derive(Debug, Clone)]
pub enum Intent {
    /// Type this text at the cursor.
    Dictate(String),
    /// Execute a named action with arguments.
    #[allow(dead_code)]
    Command {
        action: String,
        args: serde_json::Value,
    },
}

/// Trait for intent routing backends.
pub trait IntentRouter: Send + Sync {
    /// Classify transcribed text into an intent.
    fn route(&self, text: &str) -> anyhow::Result<Intent>;
    fn name(&self) -> &str;
}

/// Create a router backend based on config.
pub fn create_router(cfg: &RouterConfig) -> anyhow::Result<Box<dyn IntentRouter>> {
    match cfg.backend.as_str() {
        "passthrough" => Ok(Box::new(passthrough::PassthroughRouter)),
        "llm" => {
            #[cfg(feature = "router-llm")]
            return Ok(Box::new(llm::LlmRouter::new(cfg)?));
            #[cfg(not(feature = "router-llm"))]
            anyhow::bail!("router-llm feature not compiled in");
        }
        other => anyhow::bail!("Unknown router backend: {other}"),
    }
}
