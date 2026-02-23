//! LinuxAtspiProvider — stub implementation of AccessibilityProvider for Linux AT-SPI2.

#[cfg(target_os = "linux")]
use voxctrl_cu::actions::{UiAction, UiActionResult};
#[cfg(target_os = "linux")]
use voxctrl_cu::tree::{UiNode, UiTree};
#[cfg(target_os = "linux")]
use voxctrl_cu::AccessibilityProvider;

/// Linux AT-SPI2 accessibility provider.
///
/// Currently a stub — all methods return errors. The real implementation will
/// use the `atspi` crate with an embedded tokio runtime (see `runtime.rs`).
#[cfg(target_os = "linux")]
pub struct LinuxAtspiProvider;

#[cfg(target_os = "linux")]
impl LinuxAtspiProvider {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self)
    }
}

#[cfg(target_os = "linux")]
impl AccessibilityProvider for LinuxAtspiProvider {
    fn get_focused_tree(&self) -> anyhow::Result<UiTree> {
        Err(anyhow::anyhow!("Linux AT-SPI2 provider not yet implemented"))
    }

    fn get_tree_for_pid(&self, _pid: u32) -> anyhow::Result<UiTree> {
        Err(anyhow::anyhow!("Linux AT-SPI2 provider not yet implemented"))
    }

    fn find_elements(&self, _query: &str) -> anyhow::Result<Vec<UiNode>> {
        Err(anyhow::anyhow!("Linux AT-SPI2 provider not yet implemented"))
    }

    fn perform_action(&self, _action: &UiAction) -> anyhow::Result<UiActionResult> {
        Err(anyhow::anyhow!("Linux AT-SPI2 provider not yet implemented"))
    }

    fn capture_screenshot(&self) -> anyhow::Result<Option<Vec<u8>>> {
        Err(anyhow::anyhow!("Linux AT-SPI2 provider not yet implemented"))
    }

    fn platform_name(&self) -> &str {
        "linux-atspi"
    }
}
