mod provider;

pub fn macos_provider_factory(cfg: &voxctrl_core::config::ActionConfig) -> Option<anyhow::Result<Box<dyn voxctrl_cu::AccessibilityProvider>>> {
    #[cfg(target_os = "macos")]
    {
        Some(provider::MacosAxProvider::new().map(|p| Box::new(p) as Box<dyn voxctrl_cu::AccessibilityProvider>))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = cfg;
        None
    }
}
