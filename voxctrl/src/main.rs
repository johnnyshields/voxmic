//! voxctrl — Pluggable voice-to-action pipeline
//!
//! Mic → VAD → STT → Router → Action
//!
//! Hold Ctrl+Win+Space to toggle recording. Tray icon: green=idle, red=recording, amber=transcribing.

mod action;
mod audio;
mod config;
mod hotkey;
mod models;
mod pipeline;
mod router;
mod stt;
mod tray;
mod vad;
#[cfg(feature = "ui-model-manager")]
mod ui;

use std::sync::{Arc, Mutex};

use anyhow::Result;
use global_hotkey::GlobalHotKeyEvent;
use muda::MenuEvent;
use tray_icon::TrayIconEvent;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};

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

// ── App handler for winit event loop ───────────────────────────────────────

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
            } else if event.id == self.menu_ids.manage_models {
                log::info!("Opening model manager...");
                #[cfg(feature = "ui-model-manager")]
                {
                    ui::open_model_manager(self.registry.clone());
                }
                #[cfg(not(feature = "ui-model-manager"))]
                {
                    log::warn!("Model manager UI not available (compile with --features ui-model-manager)");
                }
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

// ── Entry point ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    log::info!("─── voxctrl v{} starting ───", env!("CARGO_PKG_VERSION"));

    let cfg = config::load_config();
    log::info!("Config: stt={}, vad={}, router={}, action={}",
        cfg.stt.backend, cfg.vad.backend, cfg.router.backend, cfg.action.backend);

    // Build model registry and scan cache
    let mut registry = models::ModelRegistry::new(models::catalog::all_models());
    registry.scan_cache();

    // Consent check — ensure required model is available
    if let Err(e) = models::consent::ensure_model_available(&cfg, &mut registry) {
        log::error!("Model not available: {e}");
        // Continue anyway — user declined download, backend may still work (e.g. HTTP backends)
    }

    // Mark in-use model
    if let Some(model_id) = models::catalog::required_model_id(&cfg) {
        registry.set_in_use(&model_id);
    }

    let registry = Arc::new(Mutex::new(registry));

    let state = Arc::new(SharedState::new());

    // Build the pluggable pipeline
    let pipeline = Arc::new(pipeline::Pipeline::from_config(&cfg)?);

    // Event loop (must be on main thread for macOS)
    let event_loop = EventLoop::new()?;

    let (tray, menu_ids) = tray::build_tray()?;
    log::info!("Tray icon created");

    let (hotkey_manager, hotkey_id) = hotkey::setup_hotkeys()?;
    log::info!("Global hotkey registered");

    let audio_stream = audio::start_capture(state.clone(), &cfg)?;
    log::info!("Audio stream open (always-on)");

    let mut app = App {
        state,
        tray: Some(tray),
        hotkey_manager: Some(hotkey_manager),
        hotkey_id: Some(hotkey_id),
        cfg,
        pipeline,
        _audio_stream: Some(audio_stream),
        registry,
        menu_ids,
    };

    log::info!("Ready — green=idle  red=recording  amber=transcribing");
    event_loop.run_app(&mut app)?;

    log::info!("─── voxctrl stopped ───");
    Ok(())
}
