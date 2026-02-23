//! Action Executor — pluggable trait + factory.

pub mod type_text;

use crate::config::ActionConfig;
use crate::router::Intent;

/// Trait for action execution backends.
pub trait ActionExecutor: Send + Sync {
    /// Execute an intent (type text, run command, etc.).
    fn execute(&self, intent: &Intent) -> anyhow::Result<()>;
    fn name(&self) -> &str;
}

/// Function signature for an external factory that can create action backends.
///
/// Called by `create_action()` for backend names it doesn't know.
/// Returns `Some(executor)` if the factory handles this backend,
/// or `None` to fall through to the "unknown backend" error.
pub type ActionFactory =
    dyn Fn(&ActionConfig) -> Option<anyhow::Result<Box<dyn ActionExecutor>>> + Send + Sync;

/// Create an action executor based on config.
///
/// `extra_factory` allows external crates (e.g. voxctrl-cu) to inject backends
/// without this crate needing to depend on their libraries.
pub fn create_action(
    cfg: &ActionConfig,
    extra_factory: Option<&ActionFactory>,
) -> anyhow::Result<Box<dyn ActionExecutor>> {
    match cfg.backend.as_str() {
        "type-text" => Ok(Box::new(type_text::TypeTextAction)),
        other => {
            if let Some(factory) = extra_factory {
                if let Some(result) = factory(cfg) {
                    return result;
                }
            }
            anyhow::bail!("Unknown action backend: {other}");
        }
    }
}
