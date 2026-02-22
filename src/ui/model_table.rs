use std::sync::{Arc, Mutex, mpsc};
use crate::config;
use crate::models::{ModelCategory, ModelRegistry, DownloadStatus};
use crate::stt_client;

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
    Test,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TestStatus {
    Idle,
    Recording,
    Transcribing,
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
    saved_flash: Option<std::time::Instant>,
    // Test tab state
    stt_server_port: u16,
    test_status: TestStatus,
    test_audio_chunks: Arc<Mutex<Vec<f32>>>,
    test_audio_stream: Option<cpal::Stream>,
    test_sample_rate: u32,
    test_result_rx: Option<mpsc::Receiver<Result<String, String>>>,
    test_transcript: String,
    test_error: String,
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
    registry.scan_cache();

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
        saved_flash: None,
        stt_server_port: cfg.stt.stt_server_port,
        test_status: TestStatus::Idle,
        test_audio_chunks: Arc::new(Mutex::new(Vec::new())),
        test_audio_stream: None,
        test_sample_rate: 16000,
        test_result_rx: None,
        test_transcript: String::new(),
        test_error: String::new(),
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([500.0, 400.0])
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
                ui.selectable_value(&mut self.tab, Tab::Test, "Test");
            });
            ui.separator();

            match self.tab {
                Tab::Settings => self.draw_settings_tab(ui),
                Tab::Models => self.draw_models_tab(ui),
                Tab::Test => self.draw_test_tab(ui),
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

        // Table
        let registry = self.registry.lock().unwrap();
        let active_tab = self.model_tab;

        egui::Grid::new("model_table")
            .striped(true)
            .min_col_width(80.0)
            .show(ui, |ui| {
                // Header
                ui.strong("Model");
                ui.strong("Size");
                ui.strong("Status");
                ui.strong("Action");
                ui.end_row();

                let filtered: Vec<_> = registry
                    .entries()
                    .iter()
                    .filter(|e| e.info.category == active_tab)
                    .collect();

                for entry in &filtered {
                    ui.label(&entry.info.display_name);
                    ui.label(format_size(entry.info.approx_size_bytes));

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
                                Some(Action::Download(entry.info.id.clone()))
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
                                Some(Action::Delete(entry.info.id.clone()))
                            } else {
                                None
                            }
                        }
                        DownloadStatus::Downloading { .. } => {
                            ui.add_enabled(false, egui::Button::new("..."));
                            None
                        }
                    };

                    ui.end_row();
                }
            });

        // Footer
        ui.separator();
        if let Some(cache_dir) = crate::models::cache_scanner::hf_cache_dir() {
            ui.label(format!("Cache: {}", cache_dir.display()));
        }

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
        if total_bytes > 0 {
            ui.label(format!("Total disk usage: {}", format_size(total_bytes)));
        }
    }
}

// ── Test tab ──────────────────────────────────────────────────────────

impl SettingsApp {
    fn draw_test_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("STT Test");
        ui.label("Record audio and transcribe via the main voxctrl process.");
        ui.add_space(8.0);

        // Poll for transcription result
        if self.test_status == TestStatus::Transcribing {
            if let Some(rx) = &self.test_result_rx {
                if let Ok(result) = rx.try_recv() {
                    match result {
                        Ok(text) => {
                            self.test_transcript = text;
                            self.test_error.clear();
                        }
                        Err(e) => {
                            self.test_error = e;
                            self.test_transcript.clear();
                        }
                    }
                    self.test_status = TestStatus::Idle;
                    self.test_result_rx = None;
                }
            }
        }

        match self.test_status {
            TestStatus::Idle => {
                if ui.button("  Record  ").clicked() {
                    self.start_recording();
                }
            }
            TestStatus::Recording => {
                if ui.button("  Stop & Transcribe  ").clicked() {
                    self.stop_and_transcribe();
                }
                ui.colored_label(egui::Color32::RED, "Recording...");
            }
            TestStatus::Transcribing => {
                ui.add_enabled(false, egui::Button::new("  Transcribing...  "));
                ui.spinner();
            }
        }

        ui.add_space(12.0);

        if !self.test_transcript.is_empty() {
            ui.group(|ui| {
                ui.label("Transcript:");
                ui.monospace(&self.test_transcript);
            });
        }

        if !self.test_error.is_empty() {
            ui.colored_label(egui::Color32::RED, &self.test_error);
        }
    }

    fn start_recording(&mut self) {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        self.test_transcript.clear();
        self.test_error.clear();

        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => {
                self.test_error = "No audio input device found".into();
                return;
            }
        };

        let default_config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                self.test_error = format!("Cannot query audio config: {e}");
                return;
            }
        };

        let sample_rate = default_config.sample_rate().0;
        self.test_sample_rate = sample_rate;

        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let chunks = self.test_audio_chunks.clone();
        chunks.lock().unwrap().clear();

        let stream = match device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buf = chunks.lock().unwrap();
                buf.extend_from_slice(data);
            },
            |err| log::error!("Test audio capture error: {err}"),
            None,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.test_error = format!("Failed to open audio stream: {e}");
                return;
            }
        };

        if let Err(e) = stream.play() {
            self.test_error = format!("Failed to start audio stream: {e}");
            return;
        }

        self.test_audio_stream = Some(stream);
        self.test_status = TestStatus::Recording;
    }

    fn stop_and_transcribe(&mut self) {
        // Drop the stream to stop recording
        self.test_audio_stream = None;

        let samples: Vec<f32> = {
            let mut buf = self.test_audio_chunks.lock().unwrap();
            std::mem::take(&mut *buf)
        };

        if samples.is_empty() {
            self.test_error = "No audio captured".into();
            self.test_status = TestStatus::Idle;
            return;
        }

        let sample_rate = self.test_sample_rate;
        let wav_data = match encode_wav(&samples, sample_rate) {
            Ok(data) => data,
            Err(e) => {
                self.test_error = format!("Failed to encode WAV: {e}");
                self.test_status = TestStatus::Idle;
                return;
            }
        };

        let port = self.stt_server_port;
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let result = stt_client::transcribe_via_server(port, &wav_data)
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
        });

        self.test_result_rx = Some(rx);
        self.test_status = TestStatus::Transcribing;
    }
}

fn encode_wav(samples: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(sample_i16)?;
    }
    writer.finalize()?;
    Ok(cursor.into_inner())
}

// ── Helpers ───────────────────────────────────────────────────────────────

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
