//! macOS AXAccessibility provider — stub implementation.

#[cfg(target_os = "macos")]
use voxctrl_cu::{AccessibilityProvider, UiAction, UiActionResult, UiNode, UiTree};

#[cfg(target_os = "macos")]
pub struct MacosAxProvider;

#[cfg(target_os = "macos")]
impl MacosAxProvider {
    pub fn new() -> anyhow::Result<Self> {
        Err(anyhow::anyhow!("macOS accessibility provider not yet implemented"))
    }
}

#[cfg(target_os = "macos")]
impl AccessibilityProvider for MacosAxProvider {
    fn get_focused_tree(&self) -> anyhow::Result<UiTree> {
        Err(anyhow::anyhow!("macOS accessibility provider not yet implemented"))
    }

    fn get_tree_for_pid(&self, _pid: u32) -> anyhow::Result<UiTree> {
        Err(anyhow::anyhow!("macOS accessibility provider not yet implemented"))
    }

    fn find_elements(&self, _query: &str) -> anyhow::Result<Vec<UiNode>> {
        Err(anyhow::anyhow!("macOS accessibility provider not yet implemented"))
    }

    fn perform_action(&self, _action: &UiAction) -> anyhow::Result<UiActionResult> {
        Err(anyhow::anyhow!("macOS accessibility provider not yet implemented"))
    }

    fn capture_screenshot(&self) -> anyhow::Result<Option<Vec<u8>>> {
        Err(anyhow::anyhow!("macOS accessibility provider not yet implemented"))
    }

    fn platform_name(&self) -> &str {
        "macos-ax"
    }
}
