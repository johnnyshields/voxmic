//! AccessibilityProvider trait — platform-specific implementations live in separate crates.

use crate::actions::{UiAction, UiActionResult};
use crate::tree::UiTree;
use crate::tree::UiNode;

/// Cross-platform accessibility provider trait.
///
/// Each platform crate (voxctrl-cu-windows, voxctrl-cu-macos, voxctrl-cu-linux)
/// provides an implementation that talks to the native accessibility API.
pub trait AccessibilityProvider: Send + Sync {
    /// Get the UI tree for the currently focused window.
    fn get_focused_tree(&self) -> anyhow::Result<UiTree>;

    /// Get the UI tree for a specific process.
    fn get_tree_for_pid(&self, pid: u32) -> anyhow::Result<UiTree>;

    /// Search for elements matching a text query (name, role, or description).
    fn find_elements(&self, query: &str) -> anyhow::Result<Vec<UiNode>>;

    /// Perform an action on a UI element.
    fn perform_action(&self, action: &UiAction) -> anyhow::Result<UiActionResult>;

    /// Capture a screenshot of the current screen (PNG bytes).
    /// Returns None if screenshots are not supported on this platform.
    fn capture_screenshot(&self) -> anyhow::Result<Option<Vec<u8>>>;

    /// Human-readable platform name (e.g. "windows-uia", "macos-ax", "linux-atspi").
    fn platform_name(&self) -> &str;
}
