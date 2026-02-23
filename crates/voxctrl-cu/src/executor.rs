//! ComputerUseExecutor — implements ActionExecutor, bridges Intent → agent loop.

use voxctrl_core::action::ActionExecutor;
use voxctrl_core::config::ActionConfig;
use voxctrl_core::router::Intent;

use crate::agent::{self, AgentConfig};
use crate::provider::AccessibilityProvider;

/// ActionExecutor that runs an LLM agent loop to perform desktop automation.
pub struct ComputerUseExecutor {
    provider: Box<dyn AccessibilityProvider>,
    agent_config: AgentConfig,
}

impl ComputerUseExecutor {
    pub fn new(
        cfg: &ActionConfig,
        provider: Box<dyn AccessibilityProvider>,
    ) -> anyhow::Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            anyhow::anyhow!(
                "ANTHROPIC_API_KEY environment variable not set — required for computer-use backend"
            )
        })?;

        let agent_config = AgentConfig {
            model: cfg
                .cu_model
                .clone()
                .unwrap_or_else(|| "claude-sonnet-4-20250514".into()),
            api_base_url: cfg
                .cu_api_base_url
                .clone()
                .unwrap_or_else(|| "https://api.anthropic.com".into()),
            api_key: api_key.into(),
            max_iterations: cfg.cu_max_iterations.unwrap_or(10),
            max_tree_depth: cfg.cu_max_tree_depth.unwrap_or(8),
            include_screenshots: cfg.cu_include_screenshots.unwrap_or(false),
        };

        log::info!(
            "ComputerUseExecutor: model={}, provider={}, max_iter={}, screenshots={}",
            agent_config.model,
            provider.platform_name(),
            agent_config.max_iterations,
            agent_config.include_screenshots,
        );

        Ok(Self {
            provider,
            agent_config,
        })
    }
}

impl ActionExecutor for ComputerUseExecutor {
    fn execute(&self, intent: &Intent) -> anyhow::Result<()> {
        let goal = match intent {
            Intent::Dictate(text) => {
                // For dictation, the goal is to type the text at the cursor
                format!("Type the following text at the current cursor position: {text}")
            }
            Intent::Command { action, args } => {
                // For commands, the goal is the action description
                if args.is_null() || args.as_object().map_or(true, |m| m.is_empty()) {
                    action.clone()
                } else {
                    format!("{action} (parameters: {args})")
                }
            }
        };

        log::info!("ComputerUse: executing goal={:?}", goal);

        let result = agent::run_agent(&*self.provider, &self.agent_config, &goal)?;

        log::info!(
            "ComputerUse: completed in {} iterations, {} actions — {}",
            result.iterations,
            result.actions_performed.len(),
            if result.summary.len() > 100 {
                &result.summary[..100]
            } else {
                &result.summary
            }
        );

        Ok(())
    }

    fn name(&self) -> &str {
        "computer-use"
    }
}
