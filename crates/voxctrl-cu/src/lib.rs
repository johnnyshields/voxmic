//! voxctrl-cu — Cross-platform computer-use via accessibility APIs + LLM agent loop.
//!
//! Provides the `AccessibilityProvider` trait, unified UI tree types,
//! and an LLM agent loop that drives desktop automation through Claude API.

pub mod actions;
pub mod agent;
pub mod executor;
pub mod prompt;
pub mod mock_provider;
pub mod provider;
pub mod screenshot;
pub mod tree;

pub use actions::{UiAction, UiActionResult};
pub use agent::{AgentEvent, AgentConfig, AgentResult};
pub use mock_provider::MockProvider;
pub use provider::AccessibilityProvider;
pub use tree::{ElementId, UiNode, UiRect, UiRole, UiState, UiTree};

use voxctrl_core::action::ActionFactory;
use voxctrl_core::config::ActionConfig;

/// Factory function for the computer-use action backend.
///
/// Pass a `provider_factory` that creates the platform-specific
/// `AccessibilityProvider` (e.g. from voxctrl-cu-windows).
///
/// Returns `Some(Ok(executor))` if config backend is "computer-use",
/// `None` otherwise (letting the next factory or core handle it).
pub fn cu_factory(
    provider_factory: &(dyn Fn(&ActionConfig) -> Option<anyhow::Result<Box<dyn AccessibilityProvider>>> + Send + Sync),
    cfg: &ActionConfig,
) -> Option<anyhow::Result<Box<dyn voxctrl_core::action::ActionExecutor>>> {
    if cfg.backend != "computer-use" {
        return None;
    }

    let provider = match provider_factory(cfg) {
        Some(Ok(p)) => p,
        Some(Err(e)) => return Some(Err(e)),
        None => {
            return Some(Err(anyhow::anyhow!(
                "No accessibility provider available for this platform"
            )));
        }
    };

    Some(executor::ComputerUseExecutor::new(cfg, provider).map(|e| {
        Box::new(e) as Box<dyn voxctrl_core::action::ActionExecutor>
    }))
}

/// Build an `ActionFactory` closure that chains platform provider factories.
///
/// Usage in the binary crate:
/// ```ignore
/// let action_factory = voxctrl_cu::action_factory(provider_factory);
/// Pipeline::from_config(&cfg, model_dir, stt_factory, Some(&action_factory))
/// ```
pub fn action_factory(
    provider_factory: Box<dyn Fn(&ActionConfig) -> Option<anyhow::Result<Box<dyn AccessibilityProvider>>> + Send + Sync>,
) -> Box<ActionFactory> {
    Box::new(move |cfg: &ActionConfig| cu_factory(&*provider_factory, cfg))
}
