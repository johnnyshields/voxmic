use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use crate::config;
use crate::models::{ModelCategory, ModelRegistry, DownloadStatus};

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

fn lookup_label(options: &'static [(&str, &str)], value: &str) -> &'static str {
    options
        .iter()
        .find(|(v, _)| *v == value)
        .map(|(_, label)| *label)
        .unwrap_or("Unknown")
}

// ── App state ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Settings,
    Models,
}

pub struct SettingsApp {
    registry: Arc<Mutex<ModelRegistry>>,
    tab: Tab,
    model_tab: ModelCategory,
    // Editable config fields
    hotkey_shortcut: String,
    stt_backend: String,
    whisper_model: String,
    vad_backend: String,
    models_directory: Option<PathBuf>,
    model_paths: HashMap<String, PathBuf>,
    saved_flash: Option<std::time::Instant>,
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

    let mut registry = crate::models::ModelRegistry::new(crate::models::catalog::all_models());
    registry.scan_cache(&cfg.models);

    if let Some(model_id) = crate::models::catalog::required_model_id(&cfg) {
        registry.set_in_use(&model_id);
    }

    let registry = Arc::new(Mutex::new(registry));

    let app = SettingsApp {
        registry,
        tab: Tab::Settings,
        model_tab: ModelCategory::Stt,
        hotkey_shortcut: cfg.hotkey.shortcut.clone(),
        stt_backend: cfg.stt.backend.clone(),
        whisper_model: cfg.stt.whisper_model.clone(),
        vad_backend: cfg.vad.backend.clone(),
        models_directory: cfg.models.models_directory.clone(),
        model_paths: cfg.models.model_paths.clone(),
        saved_flash: None,
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([650.0, 400.0])
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
        egui::CentralPanel::default().show(ctx, |ui| {
            // Top-level tab bar
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Settings, "Settings");
                ui.selectable_value(&mut self.tab, Tab::Models, "Models");
            });
            ui.separator();

            match self.tab {
                Tab::Settings => self.draw_settings_tab(ui),
                Tab::Models => self.draw_models_tab(ui),
            }
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(500));
    }
}

// ── Settings tab ──────────────────────────────────────────────────────────

impl SettingsApp {
    fn draw_settings_tab(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("settings_grid")
            .num_columns(2)
            .spacing([12.0, 8.0])
            .show(ui, |ui| {
                ui.label("Hotkey");
                ui.text_edit_singleline(&mut self.hotkey_shortcut);
                ui.end_row();

                ui.label("STT Backend");
                egui::ComboBox::from_id_salt("stt_backend")
                    .selected_text(lookup_label(STT_BACKENDS, &self.stt_backend))
                    .show_ui(ui, |ui| {
                        for &(value, label) in STT_BACKENDS {
                            ui.selectable_value(&mut self.stt_backend, value.into(), label);
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

                ui.label("VAD Backend");
                egui::ComboBox::from_id_salt("vad_backend")
                    .selected_text(lookup_label(VAD_BACKENDS, &self.vad_backend))
                    .show_ui(ui, |ui| {
                        for &(value, label) in VAD_BACKENDS {
                            ui.selectable_value(&mut self.vad_backend, value.into(), label);
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

        ui.add_space(12.0);

        if ui.button("  Save  ").clicked() {
            self.save_config();
        }

        if let Some(t) = self.saved_flash {
            if t.elapsed() < std::time::Duration::from_secs(3) {
                ui.colored_label(egui::Color32::GREEN, "Saved! Restart voxctrl to apply.");
            } else {
                self.saved_flash = None;
            }
        }
    }

    fn save_config(&mut self) {
        let mut cfg = config::load_config();
        cfg.hotkey.shortcut = self.hotkey_shortcut.clone();
        cfg.stt.backend = self.stt_backend.clone();
        cfg.stt.whisper_model = self.whisper_model.clone();
        cfg.vad.backend = self.vad_backend.clone();
        cfg.models.models_directory = self.models_directory.clone();
        cfg.models.model_paths = self.model_paths.clone();
        config::save_config(&cfg);
        self.saved_flash = Some(std::time::Instant::now());
        log::info!("Config saved");
    }
}

// ── Models tab ────────────────────────────────────────────────────────────

impl SettingsApp {
    fn draw_models_tab(&mut self, ui: &mut egui::Ui) {
        // Sub-tab bar for model categories
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.model_tab == ModelCategory::Stt, "STT Models")
                .clicked()
            {
                self.model_tab = ModelCategory::Stt;
            }
            if ui
                .selectable_label(self.model_tab == ModelCategory::Vad, "VAD Models")
                .clicked()
            {
                self.model_tab = ModelCategory::Vad;
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
                            ui.label(format!("Downloading {progress_pct}%"));
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
                    let _action = match &entry.status {
                        DownloadStatus::NotDownloaded | DownloadStatus::Error(_) => {
                            if ui.button("Download").clicked() {
                                Some(Action::Download(entry.id.clone()))
                            } else {
                                None
                            }
                        }
                        DownloadStatus::Downloaded { .. } if entry.in_use => {
                            ui.add_enabled(false, egui::Button::new("In Use"));
                            None
                        }
                        DownloadStatus::Downloaded { .. } => {
                            if ui.button("Delete").clicked() {
                                Some(Action::Delete(entry.id.clone()))
                            } else {
                                None
                            }
                        }
                        DownloadStatus::Downloading { .. } => {
                            ui.add_enabled(false, egui::Button::new("..."));
                            None
                        }
                    };

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

        // Handle deferred folder picker (outside grid / lock scope)
        if let Some(model_id) = browse_model_id {
            if let Some(path) = rfd::FileDialog::new().pick_folder() {
                self.model_paths.insert(model_id, path);
            }
        }

        // Footer
        ui.separator();
        let dir_label = match &self.models_directory {
            Some(p) => format!("Models Directory: {}", p.display()),
            None => match crate::models::cache_scanner::hf_cache_dir() {
                Some(cache_dir) => format!("Models Directory: {}", cache_dir.display()),
                None => "Models Directory: (unknown)".into(),
            },
        };
        ui.label(dir_label);

        if total_bytes > 0 {
            ui.label(format!("Total disk usage: {}", format_size(total_bytes)));
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

#[allow(dead_code)]
enum Action {
    Download(String),
    Delete(String),
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
        return path.display().to_string();
    }
    let tail: PathBuf = components[components.len() - 2..].iter().collect();
    format!("\u{2026}/{}", tail.display())
}
