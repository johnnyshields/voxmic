use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

use voxctrl_core::config::{self, GpuBackend};
use voxctrl_core::models::{DownloadStatus, ModelCategory, ModelRegistry};
use voxctrl_core::models::catalog::ModelInfo;

// ── Option tables for combo boxes ─────────────────────────────────────────

const STT_BACKENDS: &[(&str, &str)] = &[
    ("voxtral-http", "Voxtral HTTP"),
    ("voxtral-native", "Voxtral Native"),
    ("whisper-cpp", "Whisper C++"),
    ("whisper-native", "Whisper Native"),
];

const WHISPER_MODELS: &[(&str, &str)] = &[
    ("tiny", "Tiny (75 MB)"),
    ("small", "Small (461 MB)"),
    ("medium", "Medium (1.5 GB)"),
    ("large-v3", "Large v3 (3.1 GB)"),
];

const VAD_BACKENDS: &[(&str, &str)] = &[
    ("energy", "Energy (built-in)"),
    ("silero", "Silero VAD v5"),
];

#[allow(dead_code)] // used only when cu-* features enabled
const CU_PROVIDER_TYPES: &[(&str, &str)] = &[
    ("anthropic", "Anthropic (Remote)"),
    ("local", "Local LLM"),
];

const GPU_BACKENDS: &[(GpuBackend, &str)] = &[
    (GpuBackend::Auto, "Auto-detect"),
    (GpuBackend::Cuda, "CUDA (NVIDIA)"),
    (GpuBackend::Zluda, "ZLUDA (AMD)"),
    (GpuBackend::DirectMl, "DirectML"),
    (GpuBackend::Wgpu, "WebGPU"),
    (GpuBackend::Cpu, "CPU only"),
];

fn lookup_label(options: &'static [(&str, &str)], value: &str) -> &'static str {
    options
        .iter()
        .find(|(v, _)| *v == value)
        .map(|(_, label)| *label)
        .unwrap_or("Unknown")
}

// ── Key capture helpers ───────────────────────────────────────────────────

/// Map an egui::Key to the token string expected by `parse_key_code()`.
fn egui_key_to_shortcut_token(key: egui::Key) -> Option<&'static str> {
    match key {
        egui::Key::A => Some("A"),
        egui::Key::B => Some("B"),
        egui::Key::C => Some("C"),
        egui::Key::D => Some("D"),
        egui::Key::E => Some("E"),
        egui::Key::F => Some("F"),
        egui::Key::G => Some("G"),
        egui::Key::H => Some("H"),
        egui::Key::I => Some("I"),
        egui::Key::J => Some("J"),
        egui::Key::K => Some("K"),
        egui::Key::L => Some("L"),
        egui::Key::M => Some("M"),
        egui::Key::N => Some("N"),
        egui::Key::O => Some("O"),
        egui::Key::P => Some("P"),
        egui::Key::Q => Some("Q"),
        egui::Key::R => Some("R"),
        egui::Key::S => Some("S"),
        egui::Key::T => Some("T"),
        egui::Key::U => Some("U"),
        egui::Key::V => Some("V"),
        egui::Key::W => Some("W"),
        egui::Key::X => Some("X"),
        egui::Key::Y => Some("Y"),
        egui::Key::Z => Some("Z"),
        egui::Key::Num0 => Some("0"),
        egui::Key::Num1 => Some("1"),
        egui::Key::Num2 => Some("2"),
        egui::Key::Num3 => Some("3"),
        egui::Key::Num4 => Some("4"),
        egui::Key::Num5 => Some("5"),
        egui::Key::Num6 => Some("6"),
        egui::Key::Num7 => Some("7"),
        egui::Key::Num8 => Some("8"),
        egui::Key::Num9 => Some("9"),
        egui::Key::Space => Some("Space"),
        egui::Key::Enter => Some("Enter"),
        egui::Key::Tab => Some("Tab"),
        egui::Key::Backspace => Some("Backspace"),
        egui::Key::Delete => Some("Delete"),
        egui::Key::Insert => Some("Insert"),
        egui::Key::Home => Some("Home"),
        egui::Key::End => Some("End"),
        egui::Key::PageUp => Some("PageUp"),
        egui::Key::PageDown => Some("PageDown"),
        egui::Key::ArrowUp => Some("Up"),
        egui::Key::ArrowDown => Some("Down"),
        egui::Key::ArrowLeft => Some("Left"),
        egui::Key::ArrowRight => Some("Right"),
        egui::Key::F1 => Some("F1"),
        egui::Key::F2 => Some("F2"),
        egui::Key::F3 => Some("F3"),
        egui::Key::F4 => Some("F4"),
        egui::Key::F5 => Some("F5"),
        egui::Key::F6 => Some("F6"),
        egui::Key::F7 => Some("F7"),
        egui::Key::F8 => Some("F8"),
        egui::Key::F9 => Some("F9"),
        egui::Key::F10 => Some("F10"),
        egui::Key::F11 => Some("F11"),
        egui::Key::F12 => Some("F12"),
        // Escape is handled separately (cancel capture)
        _ => None,
    }
}

/// Build a shortcut string like "Ctrl+Super+Space" from egui modifiers + key.
fn build_shortcut_string(
    modifiers: &egui::Modifiers,
    key: egui::Key,
    include_super: bool,
) -> Option<String> {
    let token = egui_key_to_shortcut_token(key)?;
    let mut parts = Vec::new();
    if modifiers.ctrl || modifiers.command {
        parts.push("Ctrl");
    }
    if modifiers.alt {
        parts.push("Alt");
    }
    if modifiers.shift {
        parts.push("Shift");
    }
    if include_super {
        parts.push("Super");
    }
    parts.push(token);
    Some(parts.join("+"))
}

/// Toggle "Super" in a shortcut string when the include-super checkbox changes.
fn toggle_super_in_shortcut(include_super: bool, shortcut: &mut String) {
    if include_super {
        // Insert Super before the final (key) token
        let parts: Vec<&str> = shortcut.split('+').collect();
        let mut new_parts = Vec::with_capacity(parts.len() + 1);
        for (i, part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                new_parts.push("Super");
            }
            new_parts.push(part);
        }
        *shortcut = new_parts.join("+");
    } else {
        // Remove Super/Win/Meta/Cmd tokens
        *shortcut = shortcut
            .split('+')
            .filter(|t| {
                !matches!(
                    t.trim().to_lowercase().as_str(),
                    "super" | "win" | "meta" | "cmd"
                )
            })
            .collect::<Vec<_>>()
            .join("+");
    }
}

/// Draw a hotkey capture widget within a 2-column grid (produces 2 rows).
fn draw_hotkey_capture(
    ui: &mut egui::Ui,
    capture_state: &mut CaptureState,
    capture_target: &mut CaptureTarget,
    target: CaptureTarget,
    shortcut: &mut String,
    include_super: &mut bool,
    label: &str,
) {
    ui.label(label);
    ui.horizontal(|ui| {
        if *capture_state == CaptureState::Listening && *capture_target == target {
            let mods = ui.ctx().input(|i| i.modifiers);
            let mut parts: Vec<&str> = Vec::new();
            if mods.ctrl || mods.command {
                parts.push("Ctrl");
            }
            if mods.alt {
                parts.push("Alt");
            }
            if mods.shift {
                parts.push("Shift");
            }
            if *include_super {
                parts.push("Super");
            }
            let text = if parts.is_empty() {
                "Press keys...".to_string()
            } else {
                format!("{}+...", parts.join("+"))
            };
            ui.add(egui::Button::new(
                egui::RichText::new(text).color(egui::Color32::YELLOW),
            ));
            if ui.button("Cancel").clicked() {
                *capture_state = CaptureState::Idle;
            }
        } else {
            let display = if shortcut.is_empty() {
                "Click to set hotkey..."
            } else {
                shortcut.as_str()
            };
            let enabled = *capture_state == CaptureState::Idle;
            if ui.add_enabled(enabled, egui::Button::new(display)).clicked() {
                *capture_state = CaptureState::Listening;
                *capture_target = target;
            }
            if !shortcut.is_empty() && ui.small_button("\u{2715}").clicked() {
                shortcut.clear();
            }
        }
    });
    ui.end_row();

    ui.label("");
    let old_super = *include_super;
    ui.checkbox(include_super, "Include Super/Win key");
    if *include_super != old_super && !shortcut.is_empty() {
        toggle_super_in_shortcut(*include_super, shortcut);
    }
    ui.end_row();
}

// ── App state ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum CaptureState {
    Idle,
    Listening,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // ComputerUse only used when cu-* features enabled
enum CaptureTarget {
    Dictation,
    ComputerUse,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Settings,
    Test,
    Models,
}

struct TestHotkeyState {
    _manager: GlobalHotKeyManager,
    hotkey_id: u32,
    pressed: bool,
}

struct TestState {
    // Step 1: Mic test
    mic_active: bool,
    mic_level: f32,
    mic_stream: Option<cpal::Stream>,
    mic_level_rx: Option<std::sync::mpsc::Receiver<f32>>,
    mic_sample_rate: u32,

    // Step 2: Hotkey test
    hotkey_bypass: bool,
    hotkey_detected: bool,

    // Step 3: VAD test
    vad_bypass: bool,
    vad_detecting: bool,

    // Step 4: STT test
    stt_status: String,
    stt_result: String,
    stt_result_slot: Option<Arc<std::sync::Mutex<Option<String>>>>,
    stt_status_slot: Option<Arc<std::sync::Mutex<Option<String>>>>,

    // Step 5: Computer Use test (fields used when cu-* features enabled)
    #[allow(dead_code)]
    cu_goal: String,
    #[allow(dead_code)]
    cu_use_mock: bool,
    #[allow(dead_code)]
    cu_running: bool,
    #[allow(dead_code)]
    cu_log: Vec<String>,
    #[allow(dead_code)]
    cu_summary: String,
    #[allow(dead_code)]
    cu_event_rx: Option<std::sync::mpsc::Receiver<String>>,
    #[allow(dead_code)]
    cu_done_rx: Option<std::sync::mpsc::Receiver<Result<String, String>>>,

    // Shared audio buffer for recording
    test_chunks: Arc<std::sync::Mutex<Vec<f32>>>,
    recording: Arc<std::sync::atomic::AtomicBool>,
}

impl Default for TestState {
    fn default() -> Self {
        Self {
            mic_active: false,
            mic_level: 0.0,
            mic_stream: None,
            mic_level_rx: None,
            mic_sample_rate: 16000,
            hotkey_bypass: false,
            hotkey_detected: false,
            vad_bypass: true,
            vad_detecting: false,
            stt_status: String::new(),
            stt_result: String::new(),
            stt_result_slot: None,
            stt_status_slot: None,
            cu_goal: String::new(),
            cu_use_mock: true,
            cu_running: false,
            cu_log: Vec::new(),
            cu_summary: String::new(),
            cu_event_rx: None,
            cu_done_rx: None,
            test_chunks: Arc::new(std::sync::Mutex::new(Vec::new())),
            recording: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
}

pub struct SettingsApp {
    registry: Arc<Mutex<ModelRegistry>>,
    tab: Tab,
    prev_tab: Tab,
    model_tab: ModelCategory,
    test: TestState,
    // Editable config fields
    available_devices: Vec<String>,
    selected_device: String,
    hotkey_dict_shortcut: String,
    hotkey_cu_shortcut: String,
    stt_backend: String,
    whisper_model: String,
    vad_backend: String,
    gpu_backend: GpuBackend,
    gpu_detected: String,
    gpu_mode: String,
    #[cfg(feature = "zluda")]
    zluda_status: String,
    #[cfg(feature = "zluda")]
    zluda_downloading: bool,
    #[cfg(feature = "zluda")]
    zluda_progress_rx: Option<std::sync::mpsc::Receiver<u8>>,
    #[cfg(feature = "zluda")]
    zluda_done_rx: Option<std::sync::mpsc::Receiver<Result<(), String>>>,
    hf_token: String,
    show_hf_token: bool,
    models_directory: Option<PathBuf>,
    model_paths: HashMap<String, PathBuf>,
    saved_flash: Option<std::time::Instant>,
    // Computer Use settings (fields used when cu-* features enabled)
    #[allow(dead_code)]
    cu_provider_type: String,
    #[allow(dead_code)]
    cu_model: String,
    #[allow(dead_code)]
    cu_api_base_url: String,
    #[allow(dead_code)]
    cu_max_iterations: String,
    #[allow(dead_code)]
    cu_max_tree_depth: String,
    #[allow(dead_code)]
    cu_include_screenshots: bool,
    // Hotkey capture
    capture_state: CaptureState,
    capture_target: CaptureTarget,
    hotkey_include_super: bool,
    hotkey_cu_include_super: bool,
    // Test tab
    test_hotkey: Option<TestHotkeyState>,
    test_hotkey_error: Option<String>,
    test_hotkey_cu: Option<TestHotkeyState>,
    test_hotkey_cu_error: Option<String>,
}

// ── Subprocess launcher ───────────────────────────────────────────────────

/// Open the settings window as a subprocess.
///
/// winit 0.30 forbids multiple EventLoops per process, so we can't call
/// `eframe::run_native` from a thread while the main GUI loop is running.
/// Instead, re-exec ourselves with `--settings`.
pub fn open_settings(_registry: Arc<Mutex<ModelRegistry>>) {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            log::error!("Cannot locate own exe for settings window: {e}");
            return;
        }
    };

    match std::process::Command::new(exe)
        .arg("--settings")
        .spawn()
    {
        Ok(_) => log::info!("Settings subprocess launched"),
        Err(e) => log::error!("Failed to spawn settings window: {e}"),
    }
}

/// Run the settings window as a standalone eframe app (called via `--settings`).
pub fn run_settings_standalone() -> anyhow::Result<()> {
    let cfg = config::load_config();

    let mut registry = voxctrl_core::models::ModelRegistry::new(voxctrl_core::models::catalog::all_models());
    registry.scan_cache(&cfg.models);

    if let Some(model_id) = voxctrl_core::models::catalog::required_model_id(&cfg) {
        registry.set_in_use(&model_id);
    }

    let registry = Arc::new(Mutex::new(registry));
    let include_super = cfg
        .hotkey
        .dict_shortcut
        .to_lowercase()
        .split('+')
        .any(|t| matches!(t.trim(), "super" | "win" | "meta" | "cmd"));

    let app = SettingsApp {
        registry,
        tab: Tab::Settings,
        prev_tab: Tab::Settings,
        model_tab: ModelCategory::Stt,
        test: TestState::default(),
        available_devices: voxctrl_core::audio::list_input_devices(),
        selected_device: cfg.audio.device_pattern.clone(),
        hotkey_dict_shortcut: cfg.hotkey.dict_shortcut.clone(),
        hotkey_cu_shortcut: cfg.hotkey.cu_shortcut.clone().unwrap_or_default(),
        stt_backend: cfg.stt.backend.clone(),
        whisper_model: cfg.stt.whisper_model.clone(),
        vad_backend: cfg.vad.backend.clone(),
        gpu_backend: cfg.gpu.backend,
        gpu_detected: {
            let gpus = voxctrl_core::gpu::detect_gpus();
            if gpus.is_empty() {
                "No GPU detected".into()
            } else {
                gpus.iter().map(|g| format!("{} ({})", g.name, g.vendor)).collect::<Vec<_>>().join(", ")
            }
        },
        gpu_mode: {
            let gpus = voxctrl_core::gpu::detect_gpus();
            voxctrl_core::gpu::resolve_gpu_mode(&cfg.gpu, &gpus).to_string()
        },
        #[cfg(feature = "zluda")]
        zluda_status: {
            let dir = cfg.gpu.zluda_dir.clone()
                .or_else(voxctrl_core::gpu::zluda::default_zluda_dir)
                .unwrap_or_else(|| std::path::PathBuf::from("zluda"));
            voxctrl_core::gpu::zluda::check_zluda(&dir).to_string()
        },
        #[cfg(feature = "zluda")]
        zluda_downloading: false,
        #[cfg(feature = "zluda")]
        zluda_progress_rx: None,
        #[cfg(feature = "zluda")]
        zluda_done_rx: None,
        hf_token: load_hf_token(),
        show_hf_token: false,
        models_directory: cfg.models.models_directory.clone(),
        model_paths: cfg.models.model_paths.clone(),
        saved_flash: None,
        cu_provider_type: cfg.action.cu_provider_type.clone(),
        cu_model: cfg.action.cu_model.clone().unwrap_or_default(),
        cu_api_base_url: cfg.action.cu_api_base_url.clone().unwrap_or_default(),
        cu_max_iterations: cfg.action.cu_max_iterations.map_or(String::new(), |v| v.to_string()),
        cu_max_tree_depth: cfg.action.cu_max_tree_depth.map_or(String::new(), |v| v.to_string()),
        cu_include_screenshots: cfg.action.cu_include_screenshots.unwrap_or(false),
        capture_state: CaptureState::Idle,
        capture_target: CaptureTarget::Dictation,
        hotkey_include_super: include_super,
        hotkey_cu_include_super: cfg.hotkey.cu_shortcut.as_deref().map_or(false, |s| {
            s.to_lowercase().split('+').any(|t| matches!(t.trim(), "super" | "win" | "meta" | "cmd"))
        }),
        test_hotkey: None,
        test_hotkey_error: None,
        test_hotkey_cu: None,
        test_hotkey_cu_error: None,
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([650.0, 620.0])
            .with_title("voxctrl — Settings"),
        ..Default::default()
    };

    eframe::run_native(
        "voxctrl — Settings",
        options,
        Box::new(move |_cc| Ok(Box::new(app))),
    )
    .map_err(|e| anyhow::anyhow!("Settings window error: {e}"))
}

// ── eframe App ────────────────────────────────────────────────────────────

impl eframe::App for SettingsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── Handle key capture ────────────────────────────────────────
        if self.capture_state == CaptureState::Listening {
            let include_super = match self.capture_target {
                CaptureTarget::Dictation => self.hotkey_include_super,
                CaptureTarget::ComputerUse => self.hotkey_cu_include_super,
            };
            let captured = ctx.input(|i| {
                for event in &i.events {
                    if let egui::Event::Key {
                        key,
                        pressed: true,
                        repeat: false,
                        modifiers,
                        ..
                    } = event
                    {
                        if *key == egui::Key::Escape {
                            return Some(None); // cancel
                        }
                        if let Some(shortcut) =
                            build_shortcut_string(modifiers, *key, include_super)
                        {
                            if crate::hotkey::parse_shortcut(&shortcut).is_ok() {
                                return Some(Some(shortcut));
                            }
                        }
                    }
                }
                None
            });

            match captured {
                Some(None) => self.capture_state = CaptureState::Idle,
                Some(Some(shortcut)) => {
                    match self.capture_target {
                        CaptureTarget::Dictation => self.hotkey_dict_shortcut = shortcut,
                        CaptureTarget::ComputerUse => self.hotkey_cu_shortcut = shortcut,
                    }
                    self.capture_state = CaptureState::Idle;
                }
                None => {}
            }
        }

        // ── Handle tab transitions ────────────────────────────────────
        if self.tab != self.prev_tab {
            if self.prev_tab == Tab::Test {
                self.unregister_test_hotkey();
            }
            if self.tab == Tab::Test {
                self.register_test_hotkey();
            }
            self.prev_tab = self.tab;
        }

        // ── Poll hotkey events on Test tab ────────────────────────────
        if self.tab == Tab::Test {
            while let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
                if let Some(ref mut state) = self.test_hotkey {
                    if event.id == state.hotkey_id {
                        state.pressed = match event.state {
                            HotKeyState::Pressed => true,
                            HotKeyState::Released => false,
                        };
                    }
                }
                if let Some(ref mut state) = self.test_hotkey_cu {
                    if event.id == state.hotkey_id {
                        state.pressed = match event.state {
                            HotKeyState::Pressed => true,
                            HotKeyState::Released => false,
                        };
                    }
                }
            }
        }

        // ── Poll ZLUDA download progress ─────────────────────────────
        #[cfg(feature = "zluda")]
        if self.zluda_downloading {
            if let Some(ref rx) = self.zluda_progress_rx {
                while let Ok(pct) = rx.try_recv() {
                    self.zluda_status = format!("Downloading... {}%", pct);
                }
            }
            if let Some(ref rx) = self.zluda_done_rx {
                if let Ok(result) = rx.try_recv() {
                    self.zluda_downloading = false;
                    self.zluda_progress_rx = None;
                    self.zluda_done_rx = None;
                    match result {
                        Ok(()) => {
                            self.zluda_status = "Downloaded successfully".into();
                        }
                        Err(e) => {
                            self.zluda_status = format!("Download failed: {e}");
                        }
                    }
                }
            }
        }

        // ── Draw UI ───────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Settings, "Settings");
                ui.selectable_value(&mut self.tab, Tab::Test, "Test");
                ui.selectable_value(&mut self.tab, Tab::Models, "Models");
            });
            ui.separator();

            match self.tab {
                Tab::Settings => self.draw_settings_tab(ui),
                Tab::Test => self.draw_test_tab(ui),
                Tab::Models => self.draw_models_tab(ui),
            }
        });

        let has_download = {
            let reg = self.registry.lock().unwrap();
            reg.entries()
                .iter()
                .any(|e| matches!(e.status, DownloadStatus::Downloading { .. }))
        };
        #[allow(unused_mut)]
        let mut has_bg_work = has_download;
        #[cfg(feature = "zluda")]
        { has_bg_work = has_bg_work || self.zluda_downloading; }
        let repaint_ms =
            if self.capture_state == CaptureState::Listening
                || self.tab == Tab::Test
            {
                50
            } else if has_bg_work {
                200
            } else {
                500
            };
        ctx.request_repaint_after(std::time::Duration::from_millis(repaint_ms));
    }
}

// ── Settings tab ──────────────────────────────────────────────────────────

impl SettingsApp {
    fn draw_settings_tab(&mut self, ui: &mut egui::Ui) {
        let required_stt = self.required_stt_model_id();
        let required_vad = self.required_vad_model_id();
        let stt_missing = required_stt.as_ref().map_or(false, |id| !self.is_model_downloaded(id));
        let vad_missing = required_vad.as_ref().map_or(false, |id| !self.is_model_downloaded(id));

        egui::ScrollArea::vertical().show(ui, |ui| {
            // ── Input section ──
            ui.group(|ui| {
                ui.strong("Input");
                egui::Grid::new("settings_input").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    ui.label("Mic Input");
                    egui::ComboBox::from_id_salt("mic_input")
                        .selected_text(if self.selected_device.is_empty() {
                            "Default"
                        } else {
                            &self.selected_device
                        })
                        .show_ui(ui, |ui| {
                            for name in &self.available_devices {
                                ui.selectable_value(
                                    &mut self.selected_device,
                                    name.clone(),
                                    name.as_str(),
                                );
                            }
                        });
                    ui.end_row();
                });
            });

            ui.add_space(4.0);

            // ── Hotkeys section ──
            ui.group(|ui| {
                ui.strong("Hotkeys");
                egui::Grid::new("settings_hotkeys").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    draw_hotkey_capture(
                        ui,
                        &mut self.capture_state,
                        &mut self.capture_target,
                        CaptureTarget::Dictation,
                        &mut self.hotkey_dict_shortcut,
                        &mut self.hotkey_include_super,
                        "Dictation",
                    );

                    #[cfg(any(feature = "cu-windows", feature = "cu-macos", feature = "cu-linux"))]
                    {
                        draw_hotkey_capture(
                            ui,
                            &mut self.capture_state,
                            &mut self.capture_target,
                            CaptureTarget::ComputerUse,
                            &mut self.hotkey_cu_shortcut,
                            &mut self.hotkey_cu_include_super,
                            "CU Hotkey",
                        );
                    }
                });
            });

            ui.add_space(4.0);

            // ── Speech-to-Text section ──
            ui.group(|ui| {
                ui.strong("Speech-to-Text");
                egui::Grid::new("settings_stt").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    ui.label("STT Backend");
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_id_salt("stt_backend")
                            .selected_text(lookup_label(STT_BACKENDS, &self.stt_backend))
                            .show_ui(ui, |ui| {
                                for &(value, label) in STT_BACKENDS {
                                    ui.selectable_value(&mut self.stt_backend, value.into(), label);
                                }
                            });
                        if stt_missing {
                            ui.colored_label(egui::Color32::RED, "model not downloaded");
                        }
                    });
                    ui.end_row();

                    if self.stt_backend.starts_with("whisper") {
                        ui.label("Whisper Model");
                        egui::ComboBox::from_id_salt("whisper_model")
                            .selected_text(lookup_label(WHISPER_MODELS, &self.whisper_model))
                            .show_ui(ui, |ui| {
                                for &(value, label) in WHISPER_MODELS {
                                    ui.selectable_value(
                                        &mut self.whisper_model,
                                        value.into(),
                                        label,
                                    );
                                }
                            });
                        ui.end_row();
                    }
                });
            });

            ui.add_space(4.0);

            // ── Voice Activity Detection section ──
            ui.group(|ui| {
                ui.strong("Voice Activity Detection");
                egui::Grid::new("settings_vad").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    ui.label("VAD Backend");
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_id_salt("vad_backend")
                            .selected_text(lookup_label(VAD_BACKENDS, &self.vad_backend))
                            .show_ui(ui, |ui| {
                                for &(value, label) in VAD_BACKENDS {
                                    ui.selectable_value(&mut self.vad_backend, value.into(), label);
                                }
                            });
                        if vad_missing {
                            ui.colored_label(egui::Color32::RED, "model not downloaded");
                        }
                    });
                    ui.end_row();
                });
            });

            ui.add_space(4.0);

            // ── GPU / Acceleration section ──
            ui.group(|ui| {
                ui.strong("GPU / Acceleration");
                egui::Grid::new("settings_gpu").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    ui.label("GPU Backend");
                    {
                        let selected_label = GPU_BACKENDS
                            .iter()
                            .find(|(v, _)| *v == self.gpu_backend)
                            .map(|(_, label)| *label)
                            .unwrap_or("Unknown");
                        egui::ComboBox::from_id_salt("gpu_backend")
                            .selected_text(selected_label)
                            .show_ui(ui, |ui| {
                                for &(value, label) in GPU_BACKENDS {
                                    ui.selectable_value(&mut self.gpu_backend, value, label);
                                }
                            });
                    }
                    ui.end_row();

                    ui.label("Detected GPU");
                    ui.label(&self.gpu_detected);
                    ui.end_row();

                    ui.label("GPU Mode");
                    ui.label(&self.gpu_mode);
                    ui.end_row();

                    #[cfg(feature = "zluda")]
                    {
                        ui.label("ZLUDA Status");
                        ui.horizontal(|ui| {
                            ui.label(&self.zluda_status);
                            if !self.zluda_downloading {
                                if ui.small_button("Download ZLUDA").clicked() {
                                    self.start_zluda_download();
                                }
                            } else {
                                ui.spinner();
                                ui.label("Downloading...");
                            }
                        });
                        ui.end_row();
                    }
                });
            });

            ui.add_space(4.0);

            // ── Computer Use section (feature-gated) ──
            #[cfg(any(feature = "cu-windows", feature = "cu-macos", feature = "cu-linux"))]
            {
                ui.group(|ui| {
                    ui.strong("Computer Use");
                    egui::Grid::new("settings_cu").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                        ui.label("Provider");
                        egui::ComboBox::from_id_salt("cu_provider")
                            .selected_text(lookup_label(CU_PROVIDER_TYPES, &self.cu_provider_type))
                            .show_ui(ui, |ui| {
                                for &(value, label) in CU_PROVIDER_TYPES {
                                    ui.selectable_value(&mut self.cu_provider_type, value.into(), label);
                                }
                            });
                        ui.end_row();

                        ui.label("Model");
                        ui.add(egui::TextEdit::singleline(&mut self.cu_model).desired_width(200.0).hint_text("claude-sonnet-4-20250514"));
                        ui.end_row();

                        ui.label("API Base URL");
                        ui.add(egui::TextEdit::singleline(&mut self.cu_api_base_url).desired_width(200.0).hint_text("https://api.anthropic.com"));
                        ui.end_row();

                        ui.label("Max Iterations");
                        ui.add(egui::TextEdit::singleline(&mut self.cu_max_iterations).desired_width(60.0).hint_text("10"));
                        ui.end_row();

                        ui.label("Max Tree Depth");
                        ui.add(egui::TextEdit::singleline(&mut self.cu_max_tree_depth).desired_width(60.0).hint_text("8"));
                        ui.end_row();

                        ui.label("Screenshots");
                        ui.checkbox(&mut self.cu_include_screenshots, "Include screenshots in agent context");
                        ui.end_row();
                    });
                });

                ui.add_space(4.0);
            }

            // Save button + flash
            if ui.button("  Save  ").clicked() {
                self.save_config();
            }

            if let Some(t) = self.saved_flash {
                if t.elapsed() < std::time::Duration::from_secs(3) {
                    ui.colored_label(egui::Color32::GREEN, "Saved!");
                } else {
                    self.saved_flash = None;
                }
            }
        });
    }

    fn required_stt_model_id(&self) -> Option<String> {
        voxctrl_core::models::catalog::required_stt_model_id(&self.stt_backend, &self.whisper_model)
    }

    fn required_vad_model_id(&self) -> Option<String> {
        voxctrl_core::models::catalog::required_vad_model_id(&self.vad_backend)
    }

    fn is_model_downloaded(&self, model_id: &str) -> bool {
        let registry = self.registry.lock().unwrap();
        registry.get(model_id).map_or(false, |e| {
            matches!(e.status, DownloadStatus::Downloaded { .. })
        })
    }

    #[cfg(feature = "zluda")]
    fn start_zluda_download(&mut self) {
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();

        let cfg = config::load_config();
        let zluda_dir = cfg.gpu.zluda_dir
            .or_else(voxctrl_core::gpu::zluda::default_zluda_dir)
            .unwrap_or_else(|| std::path::PathBuf::from("zluda"));

        self.zluda_downloading = true;
        self.zluda_progress_rx = Some(progress_rx);
        self.zluda_done_rx = Some(done_rx);

        std::thread::spawn(move || {
            match voxctrl_core::gpu::zluda::download_zluda(&zluda_dir, |pct| {
                let _ = progress_tx.send(pct);
            }) {
                Ok(_) => { let _ = done_tx.send(Ok(())); }
                Err(e) => { let _ = done_tx.send(Err(format!("{e:#}"))); }
            }
        });
    }

    fn save_config(&mut self) {
        let mut cfg = config::load_config();
        cfg.audio.device_pattern = self.selected_device.clone();
        cfg.hotkey.dict_shortcut = self.hotkey_dict_shortcut.clone();
        cfg.hotkey.cu_shortcut = if self.hotkey_cu_shortcut.is_empty() {
            None
        } else {
            Some(self.hotkey_cu_shortcut.clone())
        };
        cfg.stt.backend = self.stt_backend.clone();
        cfg.stt.whisper_model = self.whisper_model.clone();
        cfg.vad.backend = self.vad_backend.clone();
        cfg.action.cu_provider_type = self.cu_provider_type.clone();
        cfg.action.cu_model = if self.cu_model.is_empty() { None } else { Some(self.cu_model.clone()) };
        cfg.action.cu_api_base_url = if self.cu_api_base_url.is_empty() { None } else { Some(self.cu_api_base_url.clone()) };
        cfg.action.cu_max_iterations = self.cu_max_iterations.parse::<u32>().ok();
        cfg.action.cu_max_tree_depth = self.cu_max_tree_depth.parse::<usize>().ok();
        cfg.action.cu_include_screenshots = Some(self.cu_include_screenshots);
        cfg.gpu.backend = self.gpu_backend;
        cfg.models.models_directory = self.models_directory.clone();
        cfg.models.model_paths = self.model_paths.clone();
        config::save_config(&cfg);
        save_hf_token(&self.hf_token);
        // Rescan cache with updated model config
        let mut reg = self.registry.lock().unwrap();
        reg.scan_cache(&cfg.models);
        drop(reg);
        self.saved_flash = Some(std::time::Instant::now());
        log::info!("Config saved");
    }
}

// ── Test tab ──────────────────────────────────────────────────────────────

impl SettingsApp {
    fn draw_test_tab(&mut self, ui: &mut egui::Ui) {
        // Poll async STT result from background thread
        if let Some(ref slot) = self.test.stt_status_slot {
            if let Some(status) = slot.lock().unwrap().take() {
                self.test.stt_status = status;
            }
        }
        let stt_done = self.test.stt_result_slot.as_ref()
            .and_then(|slot| slot.lock().unwrap().take());
        if let Some(result) = stt_done {
            self.test.stt_result = result;
            self.test.stt_result_slot = None;
            self.test.stt_status_slot = None;
        }

        // Poll CU agent events
        if let Some(ref rx) = self.test.cu_event_rx {
            while let Ok(line) = rx.try_recv() {
                self.test.cu_log.push(line);
            }
        }
        if let Some(ref rx) = self.test.cu_done_rx {
            if let Ok(result) = rx.try_recv() {
                match result {
                    Ok(summary) => {
                        self.test.cu_summary = summary;
                        self.test.cu_log.push("Agent completed successfully.".into());
                    }
                    Err(e) => {
                        self.test.cu_log.push(format!("ERROR: {e}"));
                    }
                }
                self.test.cu_running = false;
                self.test.cu_event_rx = None;
                self.test.cu_done_rx = None;
            }
        }

        // Drain mic level readings
        if let Some(rx) = &self.test.mic_level_rx {
            while let Ok(level) = rx.try_recv() {
                self.test.mic_level = level;
            }
        }
        if !self.test.mic_active {
            self.test.mic_level *= 0.9;
        }

        let cfg = config::load_config();

        ui.heading("Pipeline Test");
        ui.label("Test each stage of the audio pipeline.");
        ui.add_space(8.0);

        // ── Step 1: Mic Test ──
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("1. Microphone");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.test.mic_active {
                        if ui.button("Stop").clicked() {
                            self.test.mic_stream = None;
                            self.test.mic_level_rx = None;
                            self.test.mic_active = false;
                            self.test.mic_level = 0.0;
                        }
                    } else if ui.button("Start").clicked() {
                        self.start_mic_test(&cfg);
                    }
                });
            });
            // Volume meter bar
            let level = (self.test.mic_level * 10.0).clamp(0.0, 1.0);
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), 14.0),
                egui::Sense::hover(),
            );
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 2.0, egui::Color32::from_gray(40));
            let filled = egui::Rect::from_min_size(
                rect.min,
                egui::vec2(rect.width() * level, rect.height()),
            );
            let color = if level > 0.7 {
                egui::Color32::RED
            } else if level > 0.3 {
                egui::Color32::YELLOW
            } else {
                egui::Color32::GREEN
            };
            painter.rect_filled(filled, 2.0, color);
            if self.test.mic_active {
                ui.label("Speak to see levels");
            } else {
                ui.label("Click Start to test microphone");
            }
        });

        ui.add_space(4.0);

        // ── Step 2: Hotkey Test (live detection) ──
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("2. Hotkey");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut self.test.hotkey_bypass, "Bypass");
                });
            });
            if self.test.hotkey_bypass {
                ui.colored_label(egui::Color32::YELLOW, "Bypassed — hotkey step skipped");
                self.test.hotkey_detected = true;
            } else if self.hotkey_dict_shortcut.is_empty() {
                ui.label("No hotkey configured. Set one in the Settings tab.");
            } else if let Some(ref error) = self.test_hotkey_error {
                ui.colored_label(egui::Color32::RED, error);
                if ui.small_button("Retry").clicked() {
                    self.test_hotkey_error = None;
                    self.register_test_hotkey();
                }
            } else if let Some(ref state) = self.test_hotkey {
                let (color, label) = if state.pressed {
                    self.test.hotkey_detected = true;
                    (egui::Color32::GREEN, "ACTIVE — hotkey detected!")
                } else {
                    (egui::Color32::YELLOW, "Ready — press your hotkey...")
                };
                ui.horizontal(|ui| {
                    let radius = 8.0;
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(radius * 2.0, radius * 2.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().circle_filled(rect.center(), radius, color);
                    ui.label(egui::RichText::new(label).color(color));
                });
            } else {
                ui.label("Registering hotkey...");
            }

            // CU hotkey indicator
            #[cfg(any(feature = "cu-windows", feature = "cu-macos", feature = "cu-linux"))]
            {
                if !self.hotkey_cu_shortcut.is_empty() {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("CU Hotkey:");
                        if let Some(ref error) = self.test_hotkey_cu_error {
                            ui.colored_label(egui::Color32::RED, error);
                        } else if let Some(ref state) = self.test_hotkey_cu {
                            let (color, label) = if state.pressed {
                                (egui::Color32::GREEN, "ACTIVE")
                            } else {
                                (egui::Color32::YELLOW, "Ready")
                            };
                            let radius = 6.0;
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(radius * 2.0, radius * 2.0),
                                egui::Sense::hover(),
                            );
                            ui.painter().circle_filled(rect.center(), radius, color);
                            ui.label(egui::RichText::new(label).color(color));
                        } else {
                            ui.label("Not registered");
                        }
                    });
                }
            }
        });

        ui.add_space(4.0);

        // ── Step 3: VAD Test ──
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("3. Voice Activity Detection");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut self.test.vad_bypass, "Bypass (always on)");
                });
            });
            if self.test.vad_bypass {
                ui.colored_label(egui::Color32::YELLOW, "Bypassed — all audio treated as speech");
            } else {
                ui.label(format!("Backend: {}", cfg.vad.backend));
                if self.test.mic_active {
                    if self.test.mic_level > 0.03 {
                        self.test.vad_detecting = true;
                        ui.colored_label(egui::Color32::GREEN, "Speech detected");
                    } else {
                        self.test.vad_detecting = false;
                        ui.label("Silence");
                    }
                } else {
                    ui.label("Start mic test first");
                }
            }
        });

        ui.add_space(4.0);

        // ── Step 4: STT Test ──
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("4. Speech-to-Text");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let is_recording = self.test.recording.load(std::sync::atomic::Ordering::Relaxed);
                    if is_recording {
                        if ui.button("Stop & Transcribe").clicked() {
                            self.stop_recording_and_transcribe(&cfg);
                        }
                    } else if self.test.stt_status == "Transcribing..." {
                        ui.add_enabled(false, egui::Button::new("Transcribing..."));
                    } else {
                        if ui.button("Record").clicked() {
                            self.start_test_recording(&cfg);
                        }
                        if ui.button("Load File...").clicked() {
                            self.load_audio_file_and_transcribe(&cfg);
                        }
                    }
                });
            });
            ui.label(format!("Backend: {}", cfg.stt.backend));
            if !self.test.stt_status.is_empty() {
                ui.label(&self.test.stt_status);
            }
            if !self.test.stt_result.is_empty() {
                ui.separator();
                ui.monospace(&self.test.stt_result);
            }
        });

        ui.add_space(4.0);

        // ── Step 5: Computer Use ──
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("5. Computer Use");
            });

            #[cfg(any(feature = "cu-windows", feature = "cu-macos", feature = "cu-linux"))]
            {
                ui.horizontal(|ui| {
                    ui.label("Goal:");
                    ui.add(egui::TextEdit::singleline(&mut self.test.cu_goal)
                        .desired_width(ui.available_width() - 60.0)
                        .hint_text("e.g. Type 'Hello' in the text editor"));
                });

                // Pre-fill from STT result if goal is empty and STT has a result
                if self.test.cu_goal.is_empty() && !self.test.stt_result.is_empty() {
                    self.test.cu_goal = self.test.stt_result.clone();
                }

                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.test.cu_use_mock, "Dry-run (mock provider)");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.test.cu_running {
                            ui.add_enabled(false, egui::Button::new("Running..."));
                        } else if ui.button("Run").clicked() {
                            self.start_cu_test();
                        }
                    });
                });

                if !self.test.cu_log.is_empty() {
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .max_height(150.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for line in &self.test.cu_log {
                                let color = if line.starts_with("ERROR") {
                                    egui::Color32::RED
                                } else if line.starts_with("Tool:") {
                                    egui::Color32::YELLOW
                                } else if line.contains("successfully") {
                                    egui::Color32::GREEN
                                } else {
                                    egui::Color32::GRAY
                                };
                                ui.colored_label(color, egui::RichText::new(line).monospace());
                            }
                        });
                }

                if !self.test.cu_summary.is_empty() {
                    ui.separator();
                    ui.label("Summary:");
                    ui.monospace(&self.test.cu_summary);
                }
            }

            #[cfg(not(any(feature = "cu-windows", feature = "cu-macos", feature = "cu-linux")))]
            {
                ui.colored_label(
                    egui::Color32::GRAY,
                    "Computer Use not available \u{2014} enable cu-windows feature",
                );
            }
        });

        ui.add_space(4.0);

        // ── Step 6: Final Output ──
        ui.group(|ui| {
            ui.strong("6. Output");
            if !self.test.cu_summary.is_empty() {
                ui.label("Computer Use result:");
                ui.monospace(&self.test.cu_summary);
                ui.colored_label(egui::Color32::GREEN, "Full pipeline test complete!");
            } else if !self.test.stt_result.is_empty() {
                ui.label("Transcription result:");
                ui.monospace(&self.test.stt_result);
                ui.colored_label(egui::Color32::GREEN, "Pipeline test complete!");
            } else {
                ui.label("Complete steps above to see final output");
            }
        });
    }

    fn start_mic_test(&mut self, cfg: &config::Config) {
        let (tx, rx) = std::sync::mpsc::channel();
        match voxctrl_core::audio::start_test_capture(
            &cfg.audio.device_pattern,
            cfg.audio.sample_rate,
            tx,
            Arc::clone(&self.test.test_chunks),
            Arc::clone(&self.test.recording),
        ) {
            Ok((stream, actual_rate)) => {
                self.test.mic_stream = Some(stream);
                self.test.mic_level_rx = Some(rx);
                self.test.mic_active = true;
                self.test.mic_sample_rate = actual_rate;
            }
            Err(e) => {
                log::error!("Failed to start mic test: {e}");
                self.test.stt_status = format!("Mic error: {e}");
            }
        }
    }

    fn start_test_recording(&mut self, cfg: &config::Config) {
        if !self.test.mic_active {
            self.start_mic_test(cfg);
        }
        self.test.test_chunks.lock().unwrap().clear();
        self.test.recording.store(true, std::sync::atomic::Ordering::Relaxed);
        self.test.stt_status = "Recording...".into();
        self.test.stt_result.clear();
    }

    fn stop_recording_and_transcribe(&mut self, cfg: &config::Config) {
        self.test.recording.store(false, std::sync::atomic::Ordering::Relaxed);
        self.test.stt_status = "Transcribing...".into();

        let chunks: Vec<f32> = self.test.test_chunks.lock().unwrap().drain(..).collect();
        if chunks.is_empty() {
            self.test.stt_status = "No audio recorded".into();
            return;
        }

        let sample_rate = self.test.mic_sample_rate;
        let stt_cfg = cfg.stt.clone();

        let result_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let status_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let result_writer = Arc::clone(&result_slot);
        let status_writer = Arc::clone(&status_slot);

        self.test.stt_result_slot = Some(result_slot);
        self.test.stt_status_slot = Some(status_slot);

        std::thread::spawn(move || {
            match transcribe_chunks(&chunks, sample_rate, &stt_cfg) {
                Ok(text) => {
                    *result_writer.lock().unwrap() = Some(text);
                    *status_writer.lock().unwrap() = Some("Done".into());
                }
                Err(e) => {
                    *result_writer.lock().unwrap() = Some(String::new());
                    *status_writer.lock().unwrap() = Some(format!("STT error: {e}"));
                }
            }
        });
    }

    fn load_audio_file_and_transcribe(&mut self, cfg: &config::Config) {
        let path = rfd::FileDialog::new()
            .add_filter("WAV audio", &["wav"])
            .pick_file();
        let path = match path {
            Some(p) => p,
            None => return,
        };

        self.test.stt_status = format!("Transcribing {}...", path.file_name().unwrap_or_default().to_string_lossy());
        self.test.stt_result.clear();

        let (samples, sample_rate) = match voxctrl_core::stt::load_wav_pcm(&path) {
            Ok(v) => v,
            Err(e) => {
                self.test.stt_status = format!("Load error: {e}");
                return;
            }
        };

        let stt_cfg = cfg.stt.clone();

        let result_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let status_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let result_writer = Arc::clone(&result_slot);
        let status_writer = Arc::clone(&status_slot);

        self.test.stt_result_slot = Some(result_slot);
        self.test.stt_status_slot = Some(status_slot);

        std::thread::spawn(move || {
            match transcribe_pcm_via_server_or_direct(&samples, sample_rate, &stt_cfg) {
                Ok(text) => {
                    *result_writer.lock().unwrap() = Some(text);
                    *status_writer.lock().unwrap() = Some("Done".into());
                }
                Err(e) => {
                    *result_writer.lock().unwrap() = Some(String::new());
                    *status_writer.lock().unwrap() = Some(format!("STT error: {e}"));
                }
            }
        });
    }

    #[cfg(any(feature = "cu-windows", feature = "cu-macos", feature = "cu-linux"))]
    fn start_cu_test(&mut self) {
        let (event_tx, event_rx) = std::sync::mpsc::channel::<String>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<String, String>>();

        let goal = self.test.cu_goal.clone();
        let use_mock = self.test.cu_use_mock;

        self.test.cu_log.clear();
        self.test.cu_running = true;
        self.test.cu_summary.clear();
        self.test.cu_event_rx = Some(event_rx);
        self.test.cu_done_rx = Some(done_rx);

        std::thread::spawn(move || {
            let provider: Box<dyn voxctrl_cu::AccessibilityProvider> = if use_mock {
                Box::new(voxctrl_cu::MockProvider::new())
            } else {
                // Fall back to mock if no real provider
                Box::new(voxctrl_cu::MockProvider::new())
            };

            let cfg_file = voxctrl_core::config::load_config();
            let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();

            if api_key.is_empty() {
                let _ = event_tx.send("ERROR: ANTHROPIC_API_KEY not set".into());
                let _ = done_tx.send(Err("ANTHROPIC_API_KEY environment variable not set".into()));
                return;
            }

            let agent_cfg = voxctrl_cu::AgentConfig {
                api_key: voxctrl_cu::agent::ApiKey::new(api_key),
                model: cfg_file.action.cu_model.unwrap_or_else(|| "claude-sonnet-4-20250514".into()),
                api_base_url: cfg_file.action.cu_api_base_url.unwrap_or_else(|| "https://api.anthropic.com".into()),
                max_iterations: cfg_file.action.cu_max_iterations.unwrap_or(10),
                max_tree_depth: cfg_file.action.cu_max_tree_depth.unwrap_or(8),
                include_screenshots: cfg_file.action.cu_include_screenshots.unwrap_or(false),
            };

            let (agent_event_tx, agent_event_rx) = std::sync::mpsc::channel();

            // Forward agent events as formatted log lines
            let event_tx_clone = event_tx.clone();
            std::thread::spawn(move || {
                while let Ok(event) = agent_event_rx.recv() {
                    let line = match &event {
                        voxctrl_cu::AgentEvent::IterationStart { iteration, max } => {
                            format!("--- Iteration {iteration}/{max} ---")
                        }
                        voxctrl_cu::AgentEvent::LlmResponse { text } => {
                            format!("LLM: {}", text.chars().take(200).collect::<String>())
                        }
                        voxctrl_cu::AgentEvent::ToolCall { name, input } => {
                            format!("Tool: {name}({input})")
                        }
                        voxctrl_cu::AgentEvent::ActionResult { action, success, message } => {
                            let status = if *success { "OK" } else { "ERROR" };
                            format!("  -> {status}: {action} — {message}")
                        }
                        voxctrl_cu::AgentEvent::TreeUpdate { element_count, window_title } => {
                            format!("Tree: {element_count} elements in \"{window_title}\"")
                        }
                    };
                    let _ = event_tx_clone.send(line);
                }
            });

            let _ = event_tx.send(format!("Starting agent with goal: {goal}"));

            match voxctrl_cu::agent::run_agent_streaming(
                &*provider,
                &agent_cfg,
                &goal,
                Some(agent_event_tx),
            ) {
                Ok(result) => {
                    let _ = done_tx.send(Ok(result.summary));
                }
                Err(e) => {
                    let _ = done_tx.send(Err(format!("{e:#}")));
                }
            }
        });
    }

    fn register_test_hotkey(&mut self) {
        self.test_hotkey = None;
        self.test_hotkey_error = None;

        if self.hotkey_dict_shortcut.is_empty() {
            return;
        }

        let hotkey: HotKey = match crate::hotkey::parse_shortcut(&self.hotkey_dict_shortcut) {
            Ok(hk) => hk,
            Err(e) => {
                self.test_hotkey_error = Some(format!("Invalid hotkey: {e}"));
                return;
            }
        };

        let manager = match GlobalHotKeyManager::new() {
            Ok(m) => m,
            Err(e) => {
                self.test_hotkey_error = Some(format!("Failed to create hotkey manager: {e}"));
                return;
            }
        };

        let id = hotkey.id();
        if let Err(e) = manager.register(hotkey) {
            self.test_hotkey_error = Some(format!(
                "Failed to register hotkey (is the main app running?): {e}"
            ));
            return;
        }

        self.test_hotkey = Some(TestHotkeyState {
            _manager: manager,
            hotkey_id: id,
            pressed: false,
        });

        // Also register CU hotkey for test tab
        self.test_hotkey_cu = None;
        self.test_hotkey_cu_error = None;

        if !self.hotkey_cu_shortcut.is_empty() {
            let cu_hotkey: HotKey = match crate::hotkey::parse_shortcut(&self.hotkey_cu_shortcut) {
                Ok(hk) => hk,
                Err(e) => {
                    self.test_hotkey_cu_error = Some(format!("Invalid CU hotkey: {e}"));
                    return;
                }
            };

            let cu_manager = match GlobalHotKeyManager::new() {
                Ok(m) => m,
                Err(e) => {
                    self.test_hotkey_cu_error = Some(format!("Failed to create CU hotkey manager: {e}"));
                    return;
                }
            };

            let cu_id = cu_hotkey.id();
            if let Err(e) = cu_manager.register(cu_hotkey) {
                self.test_hotkey_cu_error = Some(format!("Failed to register CU hotkey: {e}"));
                return;
            }

            self.test_hotkey_cu = Some(TestHotkeyState {
                _manager: cu_manager,
                hotkey_id: cu_id,
                pressed: false,
            });
        }
    }

    fn unregister_test_hotkey(&mut self) {
        self.test_hotkey = None;
        self.test_hotkey_error = None;
        self.test_hotkey_cu = None;
        self.test_hotkey_cu_error = None;
    }
}

fn transcribe_chunks(
    chunks: &[f32],
    sample_rate: u32,
    stt_cfg: &config::SttConfig,
) -> anyhow::Result<String> {
    let (cmin, cmax) = chunks.iter().fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let cmean: f64 = chunks.iter().map(|&v| v as f64).sum::<f64>() / chunks.len().max(1) as f64;
    let rms: f64 = (chunks.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / chunks.len().max(1) as f64).sqrt();
    log::info!(
        "[testbed] transcribe_chunks: {} samples, rate={}, duration={:.2}s, min={:.4}, max={:.4}, mean={:.6}, rms={:.6}",
        chunks.len(), sample_rate, chunks.len() as f64 / sample_rate as f64,
        cmin, cmax, cmean, rms
    );

    transcribe_pcm_via_server_or_direct(chunks, sample_rate, stt_cfg)
}

/// Try the named-pipe STT server first; fall back to a local transcriber.
fn transcribe_pcm_via_server_or_direct(
    samples: &[f32],
    sample_rate: u32,
    stt_cfg: &config::SttConfig,
) -> anyhow::Result<String> {
    log::info!("[testbed] Trying STT via named-pipe server...");
    match voxctrl_core::stt_client::transcribe_pcm_via_server(samples, sample_rate) {
        Ok(text) => {
            log::info!("[testbed] Server transcription OK: {:?}", text);
            Ok(text)
        }
        Err(server_err) => {
            log::warn!("[testbed] Server unavailable ({server_err:#}), trying direct transcriber...");
            let transcriber = voxctrl_core::stt::create_transcriber(stt_cfg, None, Some(&voxctrl_stt::stt_factory))?;
            let result = transcriber.transcribe_pcm(samples, sample_rate);
            match &result {
                Ok(text) => log::info!("[testbed] Direct transcription OK: {:?}", text),
                Err(e) => log::error!("[testbed] Direct transcription failed: {e:#}"),
            }
            result
        }
    }
}

// ── Models tab ────────────────────────────────────────────────────────────

impl SettingsApp {
    fn draw_models_tab(&mut self, ui: &mut egui::Ui) {
        // HF Token + Models Directory at top
        ui.group(|ui| {
            egui::Grid::new("models_config").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                ui.label("Hugging Face Token");
                ui.horizontal(|ui| {
                    let field = egui::TextEdit::singleline(&mut self.hf_token)
                        .password(!self.show_hf_token)
                        .desired_width(200.0);
                    ui.add(field);
                    if ui.selectable_label(self.show_hf_token, "Show").clicked() {
                        self.show_hf_token = !self.show_hf_token;
                    }
                });
                ui.end_row();

                ui.label("Models Directory");
                ui.horizontal(|ui| {
                    let display = match &self.models_directory {
                        Some(p) => p.display().to_string(),
                        None => "Default (HuggingFace cache)".into(),
                    };
                    ui.label(display);
                    if ui.small_button("Browse\u{2026}").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            self.models_directory = Some(path);
                        }
                    }
                    if self.models_directory.is_some() && ui.small_button("Reset").clicked() {
                        self.models_directory = None;
                    }
                });
                ui.end_row();
            });
        });
        ui.add_space(4.0);

        // Sub-tab bar with 3 tabs
        ui.horizontal(|ui| {
            if ui.selectable_label(self.model_tab == ModelCategory::Stt, "STT Models").clicked() {
                self.model_tab = ModelCategory::Stt;
            }
            if ui.selectable_label(self.model_tab == ModelCategory::Vad, "VAD Models").clicked() {
                self.model_tab = ModelCategory::Vad;
            }
            if ui.selectable_label(self.model_tab == ModelCategory::ComputerUse, "CU Models").clicked() {
                self.model_tab = ModelCategory::ComputerUse;
            }
        });
        ui.separator();

        // Snapshot data from registry (release lock before UI that may open dialogs)
        let (filtered, total_bytes) = {
            let registry = self.registry.lock().unwrap();
            let active_tab = self.model_tab;
            let filtered: Vec<_> = registry
                .entries()
                .iter()
                .filter(|e| e.info.category == active_tab)
                .map(|e| ModelRowSnapshot {
                    id: e.info.id.clone(),
                    display_name: e.info.display_name.clone(),
                    approx_size_bytes: e.info.approx_size_bytes,
                    status: e.status.clone(),
                    in_use: e.in_use,
                })
                .collect();
            let total_bytes: u64 = registry
                .entries()
                .iter()
                .filter_map(|e| {
                    if let DownloadStatus::Downloaded { size_bytes, .. } = &e.status {
                        Some(*size_bytes)
                    } else {
                        None
                    }
                })
                .sum();
            (filtered, total_bytes)
        };

        // Collect deferred actions (folder picker must happen outside grid)
        let mut browse_model_id: Option<String> = None;
        let mut pending_action: Option<Action> = None;

        // Table
        egui::Grid::new("model_table")
            .striped(true)
            .min_col_width(60.0)
            .show(ui, |ui| {
                // Header
                ui.strong("Model");
                ui.strong("Size");
                ui.strong("Status");
                ui.strong("Action");
                ui.strong("Local Path");
                ui.end_row();

                for entry in &filtered {
                    ui.label(&entry.display_name);
                    ui.label(format_size(entry.approx_size_bytes));

                    // Status column
                    match &entry.status {
                        DownloadStatus::NotDownloaded => {
                            ui.label("Not installed");
                        }
                        DownloadStatus::Downloading { progress_pct } => {
                            ui.add(
                                egui::ProgressBar::new(*progress_pct as f32 / 100.0)
                                    .text(format!("{}%", progress_pct))
                                    .desired_width(100.0),
                            );
                        }
                        DownloadStatus::Downloaded { .. } => {
                            if entry.in_use {
                                ui.colored_label(egui::Color32::GOLD, "\u{2605} In Use");
                            } else {
                                ui.label("Downloaded");
                            }
                        }
                        DownloadStatus::Error(msg) => {
                            let short = if msg.len() > 30 {
                                format!("{}...", &msg[..30])
                            } else {
                                msg.clone()
                            };
                            ui.colored_label(egui::Color32::RED, short);
                        }
                    }

                    // Action column
                    match &entry.status {
                        DownloadStatus::NotDownloaded | DownloadStatus::Error(_) => {
                            if ui.button("Download").clicked() {
                                pending_action = Some(Action::Download(entry.id.clone()));
                            }
                        }
                        DownloadStatus::Downloaded { .. } if entry.in_use => {
                            ui.add_enabled(false, egui::Button::new("In Use"));
                        }
                        DownloadStatus::Downloaded { .. } => {
                            if ui.button("Delete").clicked() {
                                pending_action = Some(Action::Delete(entry.id.clone()));
                            }
                        }
                        DownloadStatus::Downloading { .. } => {
                            ui.add_enabled(false, egui::Button::new("Downloading..."));
                        }
                    }

                    // Local Path column
                    ui.horizontal(|ui| {
                        if let Some(override_path) = self.model_paths.get(&entry.id) {
                            let display = abbreviate_path(override_path);
                            ui.label(display).on_hover_text(override_path.display().to_string());
                        } else if let DownloadStatus::Downloaded { ref path, .. } = entry.status {
                            let display = abbreviate_path(path);
                            ui.label(display).on_hover_text(path.display().to_string());
                        } else {
                            ui.label("\u{2014}"); // em-dash
                        }
                        if ui.small_button("\u{1F4C2}").on_hover_text("Set local path").clicked() {
                            browse_model_id = Some(entry.id.clone());
                        }
                    });

                    ui.end_row();
                }
            });

        if filtered.is_empty() {
            ui.colored_label(egui::Color32::GRAY, match self.model_tab {
                ModelCategory::Stt => "No STT models in catalog",
                ModelCategory::Vad => "No VAD models in catalog",
                ModelCategory::ComputerUse => "No CU models in catalog yet",
            });
        }

        // Handle deferred folder picker (outside grid / lock scope)
        if let Some(model_id) = browse_model_id {
            if let Some(path) = rfd::FileDialog::new().pick_folder() {
                self.model_paths.insert(model_id, path);
            }
        }

        // Footer
        ui.separator();
        if total_bytes > 0 {
            ui.label(format!("Total disk usage: {}", format_size(total_bytes)));
        }

        // Process action outside the lock
        if let Some(action) = pending_action {
            match action {
                Action::Download(model_id) => {
                    let info = {
                        let reg = self.registry.lock().unwrap();
                        reg.get(&model_id).map(|e| e.info.clone())
                    };
                    if let Some(info) = info {
                        spawn_download(info, Arc::clone(&self.registry));
                    }
                }
                Action::Delete(model_id) => do_delete(&model_id, &self.registry),
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Snapshot of a model row for rendering (avoids holding the registry lock during UI).
struct ModelRowSnapshot {
    id: String,
    display_name: String,
    approx_size_bytes: u64,
    status: DownloadStatus,
    in_use: bool,
}

enum Action {
    Download(String),
    Delete(String),
}

/// Spawn a background thread to download a model's files from HuggingFace.
fn spawn_download(info: ModelInfo, registry: Arc<Mutex<ModelRegistry>>) {
    let repo = match &info.hf_repo {
        Some(r) => r.clone(),
        None => {
            log::error!("Model '{}' has no HuggingFace repo configured", info.id);
            if let Some(entry) = registry.lock().unwrap().get_mut(&info.id) {
                entry.status = DownloadStatus::Error("No HF repo configured".into());
            }
            return;
        }
    };

    // Mark as downloading 0%
    if let Some(entry) = registry.lock().unwrap().get_mut(&info.id) {
        entry.status = DownloadStatus::Downloading { progress_pct: 0 };
    }

    std::thread::spawn(move || {
        if let Err(e) = download_model_files(&info, &repo, &registry) {
            log::error!("Download failed for '{}': {e}", info.id);
            if let Some(entry) = registry.lock().unwrap().get_mut(&info.id) {
                entry.status = DownloadStatus::Error(format!("{e}"));
            }
            return;
        }

        // Reset to NotDownloaded so the cache scanner picks it up
        if let Some(entry) = registry.lock().unwrap().get_mut(&info.id) {
            entry.status = DownloadStatus::NotDownloaded;
        }
        // Rescan cache so the entry picks up the downloaded path + size
        let cfg = config::load_config();
        let mut reg = registry.lock().unwrap();
        reg.scan_cache(&cfg.models);
        log::info!("Download complete for '{}' — restart app to use this model", info.id);
    });
}

/// Path to the HuggingFace token file.
fn hf_token_path() -> Option<std::path::PathBuf> {
    dirs::cache_dir().map(|d| d.join("huggingface").join("token"))
}

/// Load the HF token for the Settings UI field.
fn load_hf_token() -> String {
    hf_token().unwrap_or_default()
}

/// Save the HF token to `~/.cache/huggingface/token`.
fn save_hf_token(token: &str) {
    let path = match hf_token_path() {
        Some(p) => p,
        None => return,
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if token.is_empty() {
        let _ = std::fs::remove_file(&path);
    } else if let Err(e) = std::fs::write(&path, token.trim()) {
        log::error!("Failed to write HF token to {}: {e}", path.display());
    }
}

/// Read the HuggingFace API token from `HF_TOKEN` env var or `~/.cache/huggingface/token`.
fn hf_token() -> Option<String> {
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        if !tok.is_empty() {
            return Some(tok);
        }
    }
    // Also check legacy env var
    if let Ok(tok) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        if !tok.is_empty() {
            return Some(tok);
        }
    }
    // Read from token file
    let token_path = dirs::cache_dir()?.join("huggingface").join("token");
    std::fs::read_to_string(token_path).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Download all files for a model from HuggingFace Hub.
fn download_model_files(
    info: &ModelInfo,
    repo: &str,
    registry: &Arc<Mutex<ModelRegistry>>,
) -> anyhow::Result<()> {
    use std::io::{Read, Write};

    let token = hf_token();

    let cache_dir = voxctrl_core::models::cache_scanner::effective_cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine HF cache directory"))?;

    // HF Hub cache layout: models--{org}--{name}/snapshots/main/
    let model_dir_name = format!("models--{}", repo.replace('/', "--"));
    let snapshot_dir = cache_dir.join(&model_dir_name).join("snapshots").join("main");
    std::fs::create_dir_all(&snapshot_dir)?;

    let files = &info.hf_files;
    if files.is_empty() {
        anyhow::bail!("No files listed for model '{}'", info.id);
    }

    let total_expected = info.approx_size_bytes;
    let mut total_downloaded: u64 = 0;
    let mut last_update_bytes: u64 = 0;
    const UPDATE_INTERVAL: u64 = 256 * 1024; // update UI every 256 KB

    for (i, filename) in files.iter().enumerate() {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo, filename
        );
        log::info!("Downloading {} ({}/{})", url, i + 1, files.len());

        let mut req = ureq::get(&url);
        if let Some(ref tok) = token {
            req = req.set("Authorization", &format!("Bearer {tok}"));
        }
        let resp = req.call().map_err(|e| {
            let hint = if token.is_none() {
                " (no HF token found — set HF_TOKEN env var or run `huggingface-cli login`)"
            } else {
                ""
            };
            anyhow::anyhow!("HTTP request failed for {filename}: {e}{hint}")
        })?;

        // Check content type — HF returns text/html for auth errors / gated models
        let content_type = resp.header("content-type").unwrap_or("");
        if content_type.contains("text/html") {
            anyhow::bail!(
                "HuggingFace returned HTML instead of model file for {filename}. \
                 This usually means the model is gated or requires authentication. \
                 Set your HF token in Settings → Models tab."
            );
        }

        let dest = snapshot_dir.join(filename);

        // Create parent dirs for nested files (e.g. "subdir/file.bin")
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut out = std::fs::File::create(&dest)?;
        let mut reader = resp.into_reader();
        let mut buf = [0u8; 65536]; // 64 KB read chunks
        let mut file_bytes: u64 = 0;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            file_bytes += n as u64;
            total_downloaded += n as u64;

            if total_downloaded - last_update_bytes >= UPDATE_INTERVAL {
                last_update_bytes = total_downloaded;
                let pct = if total_expected > 0 {
                    (total_downloaded as f64 / total_expected as f64 * 100.0).min(99.0) as u8
                } else {
                    ((i as f64 / files.len() as f64) * 100.0) as u8
                };
                if let Some(entry) = registry.lock().unwrap().get_mut(&info.id) {
                    entry.status = DownloadStatus::Downloading { progress_pct: pct };
                }
            }
        }
        out.flush()?;

        // Validate: model files should be at least 1 KB (HTML error pages are small)
        if file_bytes < 1024 && filename.ends_with(".safetensors") {
            let _ = std::fs::remove_file(&dest);
            anyhow::bail!(
                "Downloaded {filename} is only {file_bytes} bytes — likely an error page, not model data. \
                 Check your HF token and network connection."
            );
        }
        log::info!("Downloaded {filename}: {} bytes", file_bytes);
    }

    Ok(())
}

/// Delete a model's HF cache directory and rescan.
fn do_delete(model_id: &str, registry: &Arc<Mutex<ModelRegistry>>) {
    let hf_repo = {
        let reg = registry.lock().unwrap();
        reg.get(model_id).and_then(|e| e.info.hf_repo.clone())
    };

    let repo = match hf_repo {
        Some(r) => r,
        None => {
            log::error!("Cannot delete '{}': no HF repo configured", model_id);
            return;
        }
    };

    let cache_dir = match voxctrl_core::models::cache_scanner::effective_cache_dir() {
        Some(d) => d,
        None => {
            log::error!("Cannot determine HF cache directory");
            return;
        }
    };

    let model_dir_name = format!("models--{}", repo.replace('/', "--"));
    let model_dir = cache_dir.join(&model_dir_name);

    if model_dir.is_dir() {
        if let Err(e) = std::fs::remove_dir_all(&model_dir) {
            log::error!("Failed to remove {}: {e}", model_dir.display());
            if let Some(entry) = registry.lock().unwrap().get_mut(model_id) {
                entry.status = DownloadStatus::Error(format!("Delete failed: {e}"));
            }
            return;
        }
        log::info!("Deleted cache dir: {}", model_dir.display());
    }

    // Rescan so status resets to NotDownloaded
    let cfg = config::load_config();
    let mut reg = registry.lock().unwrap();
    reg.scan_cache(&cfg.models);
    // If scan_cache didn't find it (expected), ensure it's marked NotDownloaded
    if let Some(entry) = reg.get_mut(model_id) {
        if matches!(entry.status, DownloadStatus::Downloaded { .. }) {
            entry.status = DownloadStatus::NotDownloaded;
        }
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.0} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{bytes} B")
    }
}

/// Shorten a path for display in the table (show last 2 components).
fn abbreviate_path(path: &std::path::Path) -> String {
    let components: Vec<_> = path.components().collect();
    if components.len() <= 2 {
        return path.display().to_string().replace('\\', "/");
    }
    let tail: PathBuf = components[components.len() - 2..].iter().collect();
    format!("\u{2026}/{}", tail.display().to_string().replace('\\', "/"))
}
