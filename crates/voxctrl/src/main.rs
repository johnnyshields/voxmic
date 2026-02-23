//! voxctrl — Pluggable voice-to-action pipeline
//!
//! Mic → VAD → STT → Router → Action
//!
//! GUI mode: Tray icon + global hotkey (default).
//! TUI mode: Terminal UI with Space to toggle (`--tui`).

#[cfg(feature = "gui")]
mod hotkey;
#[cfg(feature = "gui")]
mod tray;
#[cfg(feature = "tui")]
mod tui;
#[cfg(feature = "gui")]
mod ui;

use std::sync::{Arc, Mutex};

use anyhow::Result;

use voxctrl_core::SharedState;
use voxctrl_core::config;
use voxctrl_core::models;
use voxctrl_core::pipeline;
use voxctrl_core::audio;
use voxctrl_core::stt_server;

// Compile-time check: at least one UI feature must be enabled.
#[cfg(not(any(feature = "gui", feature = "tui")))]
compile_error!("At least one of the `gui` or `tui` features must be enabled.");

// ── UI mode selection ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UiMode {
    #[cfg(feature = "gui")]
    Gui,
    #[cfg(feature = "tui")]
    Tui,
}

fn pick_ui_mode() -> UiMode {
    let wants_tui = std::env::args().any(|a| a == "--tui" || a == "-t");

    #[cfg(all(feature = "gui", feature = "tui"))]
    {
        if wants_tui {
            return UiMode::Tui;
        }
        return UiMode::Gui;
    }

    #[cfg(all(feature = "gui", not(feature = "tui")))]
    {
        if wants_tui {
            log::warn!("--tui requested but `tui` feature not compiled; falling back to GUI");
        }
        return UiMode::Gui;
    }

    #[cfg(all(feature = "tui", not(feature = "gui")))]
    {
        let _ = wants_tui;
        return UiMode::Tui;
    }
}

// ── GUI event loop ────────────────────────────────────────────────────────

#[cfg(feature = "gui")]
fn run_gui(
    state: Arc<SharedState>,
    cfg: config::Config,
    pipeline: Arc<pipeline::Pipeline>,
    audio_stream: cpal::Stream,
    registry: Arc<Mutex<models::ModelRegistry>>,
) -> Result<()> {
    use global_hotkey::GlobalHotKeyEvent;
    use muda::MenuEvent;
    use tray_icon::TrayIconEvent;
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};

    struct App {
        state: Arc<SharedState>,
        #[allow(dead_code)]
        tray: Option<tray_icon::TrayIcon>,
        #[allow(dead_code)]
        hotkey_manager: Option<global_hotkey::GlobalHotKeyManager>,
        hotkey_ids: hotkey::HotkeyIds,
        cfg: config::Config,
        pipeline: Arc<pipeline::Pipeline>,
        _audio_stream: Option<cpal::Stream>,
        #[allow(dead_code)]
        registry: Arc<Mutex<models::ModelRegistry>>,
        menu_ids: tray::TrayMenuIds,
        settings_child: Option<std::process::Child>,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

        fn window_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            _window_id: winit::window::WindowId,
            _event: WindowEvent,
        ) {
        }

        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            if let Ok(event) = TrayIconEvent::receiver().try_recv() {
                log::trace!("Tray event: {:?}", event);
            }

            // Process menu events (Quit, Manage Models...)
            if let Ok(event) = MenuEvent::receiver().try_recv() {
                if event.id == self.menu_ids.quit {
                    log::info!("Quit requested");
                    _event_loop.exit();
                } else if event.id == self.menu_ids.settings {
                    if self.settings_child.is_none() {
                        log::info!("Opening settings...");
                        if let Some(ref mgr) = self.hotkey_manager {
                            hotkey::unregister_hotkeys(mgr, &self.hotkey_ids);
                        }
                        self.settings_child = ui::open_settings(self.registry.clone());
                    } else {
                        log::info!("Settings already open");
                    }
                }
            }

            // Check if Settings subprocess has exited → re-register hotkeys
            let settings_exited = match self.settings_child {
                Some(ref mut child) => match child.try_wait() {
                    Ok(Some(status)) => {
                        log::info!("Settings subprocess exited: {status}");
                        true
                    }
                    Ok(None) => false,
                    Err(e) => {
                        log::warn!("Error polling settings subprocess: {e}");
                        true
                    }
                },
                None => false,
            };
            if settings_exited {
                self.settings_child = None;
                if let Some(ref mgr) = self.hotkey_manager {
                    hotkey::reregister_hotkeys(mgr, &self.hotkey_ids);
                }
            }

            if let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
                hotkey::handle_hotkey_event(
                    &event,
                    &self.hotkey_ids,
                    &self.state,
                    &self.cfg,
                    self.pipeline.clone(),
                );
            }
        }
    }

    let event_loop = EventLoop::new()?;

    let (tray, menu_ids) = tray::build_tray()?;
    log::info!("Tray icon created");

    let (hotkey_manager, hotkey_ids) = match hotkey::setup_hotkeys(&cfg.hotkey) {
        Ok(Some((mgr, ids))) => (Some(mgr), ids),
        Ok(None) => (None, hotkey::HotkeyIds { dictation: None, computer_use: None, dict_hotkey: None, cu_hotkey: None }),
        Err(e) => return Err(e),
    };

    // Update tray tooltip to reflect pending subsystems
    {
        let stt_pending = pipeline.stt.name().contains("pending");
        let hotkey_pending = hotkey_ids.dictation.is_none();
        if stt_pending || hotkey_pending {
            let mut parts = Vec::new();
            if hotkey_pending { parts.push("hotkey"); }
            if stt_pending { parts.push("STT model"); }
            let tooltip = format!("voxctrl — pending: {}", parts.join(", "));
            let _ = tray.set_tooltip(Some(&tooltip));
        }
    }

    let mut app = App {
        state,
        tray: Some(tray),
        hotkey_manager,
        hotkey_ids,
        cfg,
        pipeline,
        _audio_stream: Some(audio_stream),
        registry,
        menu_ids,
        settings_child: None,
    };

    if app.hotkey_ids.dictation.is_none() {
        log::warn!("Hotkey not active — configure in Settings");
    }
    log::info!("Ready — green=idle  red=recording  amber=transcribing");
    event_loop.run_app(&mut app)?;

    Ok(())
}

// ── Entry point ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // Log to both stderr and a file next to the exe for diagnostics.
    let log_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("voxctrl.log")));
    let log_file = log_path.as_ref().and_then(|p| {
        std::fs::File::create(p).ok()
    });

    let mut builder = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    );
    builder.format_timestamp_secs();
    if let Some(file) = log_file {
        use std::io::Write;
        let file = std::sync::Mutex::new(file);
        builder.format(move |buf, record| {
            let line = format!(
                "[{} {} {}] {}\n",
                buf.timestamp_seconds(),
                record.level(),
                record.module_path().unwrap_or(""),
                record.args(),
            );
            // Write to both stderr and file
            let _ = buf.write_all(line.as_bytes());
            if let Ok(mut f) = file.lock() {
                let _ = f.write_all(line.as_bytes());
                let _ = f.flush();
            }
            Ok(())
        });
    }
    builder.init();

    // Subprocess mode: settings window runs as its own eframe app (separate EventLoop).
    #[cfg(feature = "gui")]
    if std::env::args().any(|a| a == "--settings") {
        return ui::run_settings_standalone();
    }

    if let Err(e) = run() {
        log::error!("Fatal: {e:#}");
        eprintln!("Error: {e:#}");
        return Err(e);
    }
    Ok(())
}

/// Build the action factory that chains platform-specific computer-use providers.
///
/// Returns None if no cu-* features are enabled (action module falls back to core builtins).
#[cfg(any(feature = "cu-windows", feature = "cu-macos", feature = "cu-linux"))]
fn build_action_factory() -> Option<Box<voxctrl_core::action::ActionFactory>> {
    // Collect available platform provider factories
    let mut providers: Vec<Box<dyn Fn(&voxctrl_core::config::ActionConfig) -> Option<anyhow::Result<Box<dyn voxctrl_cu::AccessibilityProvider>>> + Send + Sync>> = Vec::new();

    #[cfg(feature = "cu-windows")]
    providers.push(Box::new(voxctrl_cu_windows::windows_provider_factory));

    #[cfg(feature = "cu-macos")]
    providers.push(Box::new(voxctrl_cu_macos::macos_provider_factory));

    #[cfg(feature = "cu-linux")]
    providers.push(Box::new(voxctrl_cu_linux::linux_provider_factory));

    if providers.is_empty() {
        return None;
    }

    // Build a combined provider factory that tries each platform in order
    let provider_factory = move |cfg: &voxctrl_core::config::ActionConfig| -> Option<anyhow::Result<Box<dyn voxctrl_cu::AccessibilityProvider>>> {
        for factory in &providers {
            if let Some(result) = factory(cfg) {
                return Some(result);
            }
        }
        None
    };

    Some(voxctrl_cu::action_factory(Box::new(provider_factory)))
}

#[cfg(not(any(feature = "cu-windows", feature = "cu-macos", feature = "cu-linux")))]
fn build_action_factory() -> Option<Box<voxctrl_core::action::ActionFactory>> {
    None
}

fn run() -> Result<()> {
    log::info!("─── voxctrl v{} starting ───", env!("CARGO_PKG_VERSION"));

    let ui_mode = pick_ui_mode();
    log::info!("UI mode: {:?}", ui_mode);

    let cfg = config::load_config();
    log::info!("Config: stt={}, vad={}, router={}, action={}, gpu={}",
        cfg.stt.backend, cfg.vad.backend, cfg.router.backend, cfg.action.backend, cfg.gpu.backend);

    // ── GPU detection & ZLUDA setup ──────────────────────────────────────
    let gpus = voxctrl_core::gpu::detect_gpus();
    for gpu in &gpus {
        log::info!("Detected GPU: {} ({}, device {})", gpu.name, gpu.vendor, gpu.device_id);
    }
    let gpu_mode = voxctrl_core::gpu::resolve_gpu_mode(&cfg.gpu, &gpus);
    log::info!("GPU mode: {}", gpu_mode);

    // ZLUDA DLL management (only when zluda feature compiled in)
    #[cfg(feature = "zluda")]
    {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()));

        if gpu_mode == voxctrl_core::gpu::GpuMode::Zluda {
            let zluda_dir = cfg.gpu.zluda_dir.clone()
                .or_else(voxctrl_core::gpu::zluda::default_zluda_dir)
                .unwrap_or_else(|| std::path::PathBuf::from("zluda"));

            let status = voxctrl_core::gpu::zluda::check_zluda(&zluda_dir);
            let zluda_available = match status {
                voxctrl_core::gpu::zluda::ZludaStatus::Installed(ref p) => {
                    log::info!("ZLUDA found at {}", p.display());
                    true
                }
                voxctrl_core::gpu::zluda::ZludaStatus::NotInstalled if cfg.gpu.zluda_auto_download => {
                    log::info!("ZLUDA not found, downloading...");
                    match voxctrl_core::gpu::zluda::download_zluda(&zluda_dir, |pct| {
                        log::info!("ZLUDA download: {}%", pct);
                    }) {
                        Ok(_) => true,
                        Err(e) => {
                            log::error!("Failed to download ZLUDA: {e}");
                            false
                        }
                    }
                }
                _ => {
                    log::info!("ZLUDA not available (status: {})", status);
                    false
                }
            };

            if zluda_available {
                if let Some(ref exe_dir) = exe_dir {
                    if let Err(e) = voxctrl_core::gpu::zluda::install_zluda_dlls(&zluda_dir, exe_dir) {
                        log::error!("Failed to install ZLUDA DLLs: {e}");
                    }
                }
            }
        } else if voxctrl_core::gpu::zluda::is_zluda_active() {
            // Clean up stale ZLUDA DLLs when not in ZLUDA mode
            log::info!("Removing stale ZLUDA DLLs (not in ZLUDA mode)");
            if let Some(ref exe_dir) = exe_dir {
                if let Err(e) = voxctrl_core::gpu::zluda::uninstall_zluda_dlls(exe_dir) {
                    log::warn!("Failed to uninstall ZLUDA DLLs: {e}");
                }
            }
        }
    }

    #[cfg(not(feature = "zluda"))]
    if gpu_mode == voxctrl_core::gpu::GpuMode::Zluda {
        log::warn!("ZLUDA mode resolved but `zluda` feature not compiled in; falling back to CPU");
    }

    // Override whisper_device based on resolved GPU mode
    let mut cfg = cfg;
    let whisper_device = voxctrl_core::gpu::gpu_mode_to_whisper_device(gpu_mode);
    if cfg.stt.whisper_device != whisper_device {
        log::info!("Overriding whisper_device: {} → {}", cfg.stt.whisper_device, whisper_device);
        cfg.stt.whisper_device = whisper_device.into();
    }

    // Build model registry and scan cache (respecting config paths / cache_dir)
    let mut registry = models::ModelRegistry::new(models::catalog::all_models());
    registry.scan_cache(&cfg.models);
    log::info!("Model registry scanned ({} entries)", registry.entries().len());

    // Resolve model path from registry (no auto-download; pending state if missing)
    let stt_model_dir = models::catalog::required_model_id(&cfg)
        .and_then(|id| registry.model_path(&id));
    log::info!("STT model dir: {:?}", stt_model_dir);

    // Mark in-use model
    if let Some(model_id) = models::catalog::required_model_id(&cfg) {
        registry.set_in_use(&model_id);
    }

    #[cfg(feature = "gui")]
    let registry = Arc::new(Mutex::new(registry));
    let state = Arc::new(SharedState::new());

    // Build action factory chain (computer-use providers, feature-gated)
    let action_factory = build_action_factory();

    log::info!("Creating pipeline...");
    let pipeline = Arc::new(pipeline::Pipeline::from_config(
        &cfg,
        stt_model_dir,
        Some(&voxctrl_stt::stt_factory),
        action_factory.as_deref(),
    )?);
    log::info!("Pipeline created");

    log::info!("Starting audio capture...");
    let audio_stream = audio::start_capture(state.clone(), &cfg)?;
    log::info!("Audio stream open (always-on)");

    log::info!("Starting STT server...");
    stt_server::start(pipeline.clone())?;
    log::info!("STT server started");

    match ui_mode {
        #[cfg(feature = "gui")]
        UiMode::Gui => run_gui(state, cfg, pipeline, audio_stream, registry)?,
        #[cfg(feature = "tui")]
        UiMode::Tui => tui::run_tui(state, cfg, pipeline, audio_stream)?,
    }

    log::info!("─── voxctrl stopped ───");
    Ok(())
}
