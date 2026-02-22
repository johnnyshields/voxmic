//! voxctrl — Pluggable voice-to-action pipeline
//!
//! Mic → VAD → STT → Router → Action
//!
//! GUI mode: Tray icon + global hotkey (default).
//! TUI mode: Terminal UI with Space to toggle (`--tui`).

mod action;
mod audio;
mod config;
#[cfg(feature = "gui")]
mod hotkey;
mod models;
mod pipeline;
mod recording;
mod router;
mod stt;
mod stt_client;
mod stt_server;
#[cfg(feature = "gui")]
mod tray;
#[cfg(feature = "tui")]
mod tui;
#[cfg(feature = "gui")]
mod ui;
mod vad;

use std::sync::{Arc, Mutex};

use anyhow::Result;

// Compile-time check: at least one UI feature must be enabled.
#[cfg(not(any(feature = "gui", feature = "tui")))]
compile_error!("At least one of the `gui` or `tui` features must be enabled.");

// ── Shared state ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppStatus {
    Idle,
    Recording,
    Transcribing,
}

pub struct SharedState {
    pub status: Mutex<AppStatus>,
    pub chunks: Mutex<Vec<f32>>,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            status: Mutex::new(AppStatus::Idle),
            chunks: Mutex::new(Vec::new()),
        }
    }
}

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
        hotkey_id: Option<u32>,
        cfg: config::Config,
        pipeline: Arc<pipeline::Pipeline>,
        _audio_stream: Option<cpal::Stream>,
        #[allow(dead_code)]
        registry: Arc<Mutex<models::ModelRegistry>>,
        menu_ids: tray::TrayMenuIds,
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
                    log::info!("Opening settings...");
                    ui::open_settings(self.registry.clone());
                }
            }

            if let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
                hotkey::handle_hotkey_event(
                    &event,
                    self.hotkey_id,
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

    let (hotkey_manager, hotkey_id) = match hotkey::setup_hotkeys(&cfg.hotkey) {
        Ok(Some((mgr, id))) => (Some(mgr), Some(id)),
        Ok(None) => (None, None),
        Err(e) => return Err(e),
    };

    // Update tray tooltip to reflect pending subsystems
    {
        let stt_pending = pipeline.stt.name().contains("pending");
        let hotkey_pending = hotkey_id.is_none();
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
        hotkey_id,
        cfg,
        pipeline,
        _audio_stream: Some(audio_stream),
        registry,
        menu_ids,
    };

    if hotkey_id.is_none() {
        log::warn!("Hotkey not active — configure in Settings");
    }
    log::info!("Ready — green=idle  red=recording  amber=transcribing");
    event_loop.run_app(&mut app)?;

    Ok(())
}

// ── Entry point ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    // Subprocess mode: settings window runs as its own eframe app (separate EventLoop).
    #[cfg(feature = "gui")]
    if std::env::args().any(|a| a == "--settings") {
        return ui::run_settings_standalone();
    }

    log::info!("─── voxctrl v{} starting ───", env!("CARGO_PKG_VERSION"));

    let ui_mode = pick_ui_mode();
    log::info!("UI mode: {:?}", ui_mode);

    let cfg = config::load_config();
    log::info!("Config: stt={}, vad={}, router={}, action={}",
        cfg.stt.backend, cfg.vad.backend, cfg.router.backend, cfg.action.backend);

    // Build model registry and scan cache (respecting config paths / cache_dir)
    let mut registry = models::ModelRegistry::new(models::catalog::all_models());
    registry.scan_cache(&cfg.models);

    // Resolve model path from registry (no auto-download; pending state if missing)
    let stt_model_dir = models::catalog::required_model_id(&cfg)
        .and_then(|id| registry.model_path(&id));

    // Mark in-use model
    if let Some(model_id) = models::catalog::required_model_id(&cfg) {
        registry.set_in_use(&model_id);
    }

    #[cfg(feature = "gui")]
    let registry = Arc::new(Mutex::new(registry));
    let state = Arc::new(SharedState::new());
    let pipeline = Arc::new(pipeline::Pipeline::from_config(&cfg, stt_model_dir)?);
    let audio_stream = audio::start_capture(state.clone(), &cfg)?;
    log::info!("Audio stream open (always-on)");

    stt_server::start(pipeline.clone(), cfg.stt.stt_server_port)?;

    match ui_mode {
        #[cfg(feature = "gui")]
        UiMode::Gui => run_gui(state, cfg, pipeline, audio_stream, registry)?,
        #[cfg(feature = "tui")]
        UiMode::Tui => tui::run_tui(state, cfg, pipeline, audio_stream)?,
    }

    log::info!("─── voxctrl stopped ───");
    Ok(())
}
