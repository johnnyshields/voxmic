//! Type-text action — types dictated text at the cursor using enigo.

use crate::router::Intent;
use super::ActionExecutor;
use enigo::{Enigo, Keyboard, Settings};

/// Types dictated text at the current cursor position.
pub struct TypeTextAction;

impl ActionExecutor for TypeTextAction {
    fn execute(&self, intent: &Intent) -> anyhow::Result<()> {
        match intent {
            Intent::Dictate(text) => {
                let mut enigo = Enigo::new(&Settings::default())
                    .map_err(|e| anyhow::anyhow!("failed to init enigo: {e}"))?;
                enigo
                    .text(text)
                    .map_err(|e| anyhow::anyhow!("failed to type text: {e}"))?;
                log::debug!("TypeTextAction: typed {} chars", text.len());
                Ok(())
            }
            Intent::Command { action, .. } => {
                log::warn!(
                    "TypeTextAction: commands not supported (got action={action:?}), ignoring"
                );
                Ok(())
            }
        }
    }

    fn name(&self) -> &str {
        "type-text"
    }
}
