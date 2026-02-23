//! UiAction types — LLM tool calls map to these.

use serde::{Deserialize, Serialize};

use crate::tree::ElementId;

/// Scroll direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScrollDirection {
    Up,
    Down,
    Left,
    Right,
}

/// Actions the LLM agent can request on UI elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UiAction {
    Click { element_id: ElementId },
    SetValue { element_id: ElementId, value: String },
    SendKeys { keys: String },
    Scroll { element_id: ElementId, direction: ScrollDirection, amount: u32 },
    Toggle { element_id: ElementId },
    Expand { element_id: ElementId },
    Collapse { element_id: ElementId },
    Select { element_id: ElementId },
    Focus { element_id: ElementId },
    Wait { ms: u64 },
}

impl std::fmt::Display for UiAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UiAction::Click { element_id } => write!(f, "click({element_id})"),
            UiAction::SetValue { element_id, value } => {
                write!(f, "set_value({element_id}, {:?})", value)
            }
            UiAction::SendKeys { keys } => write!(f, "send_keys({keys:?})"),
            UiAction::Scroll { element_id, direction, .. } => {
                write!(f, "scroll({element_id}, {direction:?})")
            }
            UiAction::Toggle { element_id } => write!(f, "toggle({element_id})"),
            UiAction::Expand { element_id } => write!(f, "expand({element_id})"),
            UiAction::Collapse { element_id } => write!(f, "collapse({element_id})"),
            UiAction::Select { element_id } => write!(f, "select({element_id})"),
            UiAction::Focus { element_id } => write!(f, "focus({element_id})"),
            UiAction::Wait { ms } => write!(f, "wait({ms}ms)"),
        }
    }
}

/// Result of performing a UiAction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiActionResult {
    /// Whether the action was performed successfully.
    pub success: bool,
    /// Human-readable description of what happened.
    pub message: String,
}

impl UiActionResult {
    pub fn ok(message: impl Into<String>) -> Self {
        Self { success: true, message: message.into() }
    }

    pub fn err(message: impl Into<String>) -> Self {
        Self { success: false, message: message.into() }
    }
}
