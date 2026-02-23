//! voxctrl-cu-linux — Linux AT-SPI2 accessibility provider (stub).

mod provider;

// Future modules:
// mod tree_walker;
// mod actions;
// mod roles;
// mod screenshot;
// mod runtime;

pub fn linux_provider_factory(
    cfg: &voxctrl_core::config::ActionConfig,
) -> Option<anyhow::Result<Box<dyn voxctrl_cu::AccessibilityProvider>>> {
    let _ = cfg;
    #[cfg(target_os = "linux")]
    {
        Some(
            provider::LinuxAtspiProvider::new()
                .map(|p| Box::new(p) as Box<dyn voxctrl_cu::AccessibilityProvider>),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}
