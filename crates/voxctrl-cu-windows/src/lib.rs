//! voxctrl-cu-windows — Windows UI Automation provider for computer-use.
//!
//! Uses the `uiautomation` crate to interact with Windows accessibility APIs.
//! All Windows-specific code is gated behind `#[cfg(windows)]`.

#[cfg(windows)]
mod actions;
#[cfg(windows)]
mod provider;
#[cfg(windows)]
mod roles;
#[cfg(windows)]
mod screenshot;
#[cfg(windows)]
mod tree_walker;

/// Factory function that creates a Windows UIA provider.
///
/// Returns `Some(Ok(provider))` on Windows, `None` on other platforms.
pub fn windows_provider_factory(
    cfg: &voxctrl_core::config::ActionConfig,
) -> Option<anyhow::Result<Box<dyn voxctrl_cu::AccessibilityProvider>>> {
    #[cfg(windows)]
    {
        let _ = cfg;
        Some(
            provider::WindowsUiaProvider::new()
                .map(|p| Box::new(p) as Box<dyn voxctrl_cu::AccessibilityProvider>),
        )
    }
    #[cfg(not(windows))]
    {
        let _ = cfg;
        None
    }
}
