//! Config — nested sections for each pipeline stage, backwards-compatible with flat config.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ── Sub-configs for each pipeline stage ────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttConfig {
    #[serde(default = "default_stt_backend")]
    pub backend: String,
    #[serde(default = "default_voxtral_url")]
    pub voxtral_url: String,
    #[serde(default = "default_whisper_model")]
    pub whisper_model: String,
    #[serde(default = "default_whisper_device")]
    pub whisper_device: String,
    #[serde(default = "default_whisper_compute_type")]
    pub whisper_compute_type: String,
    #[serde(default)]
    pub whisper_language: Option<String>,
    #[serde(default = "default_stt_server_port")]
    pub stt_server_port: u16,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            backend: default_stt_backend(),
            voxtral_url: default_voxtral_url(),
            whisper_model: default_whisper_model(),
            whisper_device: default_whisper_device(),
            whisper_compute_type: default_whisper_compute_type(),
            whisper_language: None,
            stt_server_port: default_stt_server_port(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    #[serde(default = "default_vad_backend")]
    pub backend: String,
    #[serde(default = "default_energy_threshold")]
    pub energy_threshold: f64,
    #[serde(default = "default_silero_threshold")]
    pub silero_threshold: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            backend: default_vad_backend(),
            energy_threshold: default_energy_threshold(),
            silero_threshold: default_silero_threshold(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    #[serde(default = "default_router_backend")]
    pub backend: String,
    /// URL for LLM router (reuses voxtral URL by default).
    #[serde(default)]
    pub llm_url: Option<String>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            backend: default_router_backend(),
            llm_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionConfig {
    #[serde(default = "default_action_backend")]
    pub backend: String,
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            backend: default_action_backend(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyConfig {
    #[serde(default = "default_hotkey_shortcut")]
    pub shortcut: String,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            shortcut: default_hotkey_shortcut(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    #[serde(default = "default_device_pattern")]
    pub device_pattern: String,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_chunk_duration_ms")]
    pub chunk_duration_ms: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device_pattern: default_device_pattern(),
            sample_rate: default_sample_rate(),
            chunk_duration_ms: default_chunk_duration_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelsConfig {
    #[serde(default)]
    pub models_directory: Option<PathBuf>,
    #[serde(default)]
    pub model_paths: HashMap<String, PathBuf>,
}

// ── Top-level config ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub stt: SttConfig,
    #[serde(default)]
    pub vad: VadConfig,
    #[serde(default)]
    pub router: RouterConfig,
    #[serde(default)]
    pub action: ActionConfig,
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub hotkey: HotkeyConfig,
    #[serde(default)]
    pub models: ModelsConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            stt: SttConfig::default(),
            vad: VadConfig::default(),
            router: RouterConfig::default(),
            action: ActionConfig::default(),
            audio: AudioConfig::default(),
            hotkey: HotkeyConfig::default(),
            models: ModelsConfig::default(),
        }
    }
}

// ── Defaults ───────────────────────────────────────────────────────────────

fn default_stt_server_port() -> u16 { 5201 }
fn default_stt_backend() -> String { "voxtral-http".into() }
fn default_voxtral_url() -> String { "http://127.0.0.1:5200".into() }
fn default_whisper_model() -> String { "small".into() }
fn default_whisper_device() -> String { "cpu".into() }
fn default_whisper_compute_type() -> String { "int8".into() }
fn default_vad_backend() -> String { "energy".into() }
fn default_energy_threshold() -> f64 { 0.015 }
fn default_silero_threshold() -> f32 { 0.5 }
fn default_router_backend() -> String { "passthrough".into() }
fn default_action_backend() -> String { "type-text".into() }
fn default_hotkey_shortcut() -> String { "Ctrl+Super+Space".into() }
fn default_device_pattern() -> String { "DJI".into() }
fn default_sample_rate() -> u32 { 16000 }
fn default_chunk_duration_ms() -> u32 { 100 }

// ── Load / save ────────────────────────────────────────────────────────────

fn config_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("config.json")))
        .unwrap_or_else(|| PathBuf::from("config.json"))
}

/// Load config from config.json next to the binary.
///
/// Supports both the new nested format and the legacy flat format.
/// If the file contains flat keys (e.g. `"backend": "whisper"`), they are
/// mapped into the appropriate nested sections.
pub fn load_config() -> Config {
    let path = config_path();
    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => {
            log::info!("No config.json at {:?}, using defaults", path);
            return Config::default();
        }
    };

    // Try nested format first.
    if let Ok(cfg) = serde_json::from_str::<Config>(&contents) {
        // Check if it actually had nested keys (not just defaults).
        let raw: serde_json::Value = serde_json::from_str(&contents).unwrap_or_default();
        if raw.get("stt").is_some()
            || raw.get("vad").is_some()
            || raw.get("audio").is_some()
            || raw.get("hotkey").is_some()
            || raw.get("models").is_some()
        {
            return cfg;
        }
    }

    // Fall back to legacy flat format.
    load_legacy_config(&contents)
}

/// Map the old flat config.json into the new nested structure.
fn load_legacy_config(contents: &str) -> Config {
    #[derive(Deserialize)]
    struct FlatConfig {
        #[serde(default = "default_stt_backend")]
        backend: String,
        #[serde(default = "default_device_pattern")]
        device_pattern: String,
        #[serde(default = "default_sample_rate")]
        sample_rate: u32,
        #[serde(default)]
        chunk_duration: Option<f64>,
        #[serde(default = "default_energy_threshold")]
        silence_threshold: f64,
        #[serde(default = "default_whisper_model")]
        whisper_model: String,
        #[serde(default)]
        whisper_language: Option<String>,
        #[serde(default = "default_whisper_device")]
        whisper_device: String,
        #[serde(default = "default_whisper_compute_type")]
        whisper_compute_type: String,
    }

    let flat: FlatConfig = serde_json::from_str(contents).unwrap_or_else(|e| {
        log::warn!("Failed to parse legacy config: {e}. Using defaults.");
        serde_json::from_str("{}").unwrap()
    });

    let stt_backend = match flat.backend.as_str() {
        "voxtral" => "voxtral-http",
        "whisper" => "voxtral-http", // legacy "whisper" was faster-whisper; default to http now
        other => other,
    };

    Config {
        stt: SttConfig {
            backend: stt_backend.into(),
            voxtral_url: default_voxtral_url(),
            whisper_model: flat.whisper_model,
            whisper_device: flat.whisper_device,
            whisper_compute_type: flat.whisper_compute_type,
            whisper_language: flat.whisper_language,
            stt_server_port: default_stt_server_port(),
        },
        vad: VadConfig {
            backend: default_vad_backend(),
            energy_threshold: flat.silence_threshold,
            silero_threshold: default_silero_threshold(),
        },
        router: RouterConfig::default(),
        action: ActionConfig::default(),
        audio: AudioConfig {
            device_pattern: flat.device_pattern,
            sample_rate: flat.sample_rate,
            chunk_duration_ms: flat
                .chunk_duration
                .map(|d| (d * 1000.0) as u32)
                .unwrap_or(default_chunk_duration_ms()),
        },
        hotkey: HotkeyConfig::default(),
        models: ModelsConfig::default(),
    }
}

pub fn save_config(cfg: &Config) {
    let path = config_path();
    match serde_json::to_string_pretty(cfg) {
        Ok(contents) => {
            if let Err(e) = std::fs::write(&path, contents) {
                log::error!("Failed to write config.json: {e}");
            }
        }
        Err(e) => log::error!("Failed to serialize config: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = Config::default();
        assert_eq!(cfg.stt.backend, "voxtral-http");
        assert_eq!(cfg.stt.stt_server_port, 5201);
        assert_eq!(cfg.vad.backend, "energy");
        assert_eq!(cfg.router.backend, "passthrough");
        assert_eq!(cfg.action.backend, "type-text");
        assert_eq!(cfg.audio.sample_rate, 16000);
    }

    #[test]
    fn test_legacy_flat_config() {
        let json = r#"{
            "backend": "whisper",
            "device_pattern": "DJI",
            "sample_rate": 16000,
            "chunk_duration": 0.1,
            "silence_threshold": 0.015,
            "whisper_model": "small"
        }"#;
        let cfg = load_legacy_config(json);
        assert_eq!(cfg.stt.backend, "voxtral-http");
        assert_eq!(cfg.audio.device_pattern, "DJI");
        assert_eq!(cfg.audio.chunk_duration_ms, 100);
        assert_eq!(cfg.stt.whisper_model, "small");
        assert_eq!(cfg.vad.energy_threshold, 0.015);
    }

    #[test]
    fn test_nested_config_roundtrip() {
        let cfg = Config::default();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.stt.backend, cfg.stt.backend);
        assert_eq!(parsed.audio.sample_rate, cfg.audio.sample_rate);
    }
}
