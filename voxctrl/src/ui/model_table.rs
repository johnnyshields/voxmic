use std::sync::{Arc, Mutex};
use crate::models::{ModelCategory, ModelRegistry, DownloadStatus};

pub struct ModelManagerApp {
    registry: Arc<Mutex<ModelRegistry>>,
    active_tab: ModelCategory,
}

/// Open the model manager window on a background thread.
pub fn open_model_manager(registry: Arc<Mutex<ModelRegistry>>) {
    std::thread::Builder::new()
        .name("model-manager-ui".into())
        .spawn(move || {
            let options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_inner_size([600.0, 400.0])
                    .with_title("voxctrl — Model Manager"),
                ..Default::default()
            };

            if let Err(e) = eframe::run_native(
                "voxctrl — Model Manager",
                options,
                Box::new(move |_cc| Ok(Box::new(ModelManagerApp {
                    registry,
                    active_tab: ModelCategory::Stt,
                }))),
            ) {
                log::error!("Model manager window error: {e}");
            }
        })
        .expect("spawn model-manager-ui thread");
}

impl eframe::App for ModelManagerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Tab bar
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(self.active_tab == ModelCategory::Stt, "STT Models")
                    .clicked()
                {
                    self.active_tab = ModelCategory::Stt;
                }
                if ui
                    .selectable_label(self.active_tab == ModelCategory::Vad, "VAD Models")
                    .clicked()
                {
                    self.active_tab = ModelCategory::Vad;
                }
            });
            ui.separator();

            // Table
            let registry = self.registry.lock().unwrap();
            let active_tab = self.active_tab;

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

                        // Action column — collect button clicks but defer actual
                        // download/delete until the registry lock is released.
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

            // Total disk usage of downloaded models
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
        });

        // Request repaint for progress updates
        ctx.request_repaint_after(std::time::Duration::from_millis(500));
    }
}

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
