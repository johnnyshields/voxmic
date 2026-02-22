use crate::config::Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelBackend {
    Voxtral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCategory {
    Stt,
    Vad,
}

impl std::fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCategory::Stt => write!(f, "STT"),
            ModelCategory::Vad => write!(f, "VAD"),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelInfo {
    pub id: String,
    pub display_name: String,
    pub backend: ModelBackend,
    pub category: ModelCategory,
    pub hf_repo: Option<String>,
    pub hf_files: Vec<String>,
    pub approx_size_bytes: u64,
}

const MB: u64 = 1_000_000;

pub fn all_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "openai/whisper-tiny".into(),
            display_name: "Whisper Tiny".into(),
            backend: ModelBackend::Voxtral,
            category: ModelCategory::Stt,
            hf_repo: Some("openai/whisper-tiny".into()),
            hf_files: vec!["model.safetensors".into(), "config.json".into(), "tokenizer.json".into()],
            approx_size_bytes: 75 * MB,
        },
        ModelInfo {
            id: "openai/whisper-small".into(),
            display_name: "Whisper Small".into(),
            backend: ModelBackend::Voxtral,
            category: ModelCategory::Stt,
            hf_repo: Some("openai/whisper-small".into()),
            hf_files: vec!["model.safetensors".into(), "config.json".into(), "tokenizer.json".into()],
            approx_size_bytes: 461 * MB,
        },
        ModelInfo {
            id: "openai/whisper-medium".into(),
            display_name: "Whisper Medium".into(),
            backend: ModelBackend::Voxtral,
            category: ModelCategory::Stt,
            hf_repo: Some("openai/whisper-medium".into()),
            hf_files: vec!["model.safetensors".into(), "config.json".into(), "tokenizer.json".into()],
            approx_size_bytes: 1_500 * MB,
        },
        ModelInfo {
            id: "openai/whisper-large-v3".into(),
            display_name: "Whisper Large v3".into(),
            backend: ModelBackend::Voxtral,
            category: ModelCategory::Stt,
            hf_repo: Some("openai/whisper-large-v3".into()),
            hf_files: vec!["model.safetensors".into(), "config.json".into(), "tokenizer.json".into()],
            approx_size_bytes: 3_100 * MB,
        },
        ModelInfo {
            id: "mistral/voxtral-mini".into(),
            display_name: "Voxtral Mini 4B".into(),
            backend: ModelBackend::Voxtral,
            category: ModelCategory::Stt,
            hf_repo: Some("mistralai/Voxtral-Mini-3B-2507".into()),
            hf_files: vec![
                "config.json".into(),
                "consolidated.safetensors".into(),
                "params.json".into(),
                "preprocessor_config.json".into(),
                "tekken.json".into(),
            ],
            approx_size_bytes: 9_400 * MB,
        },
        ModelInfo {
            id: "silero/vad-v5".into(),
            display_name: "Silero VAD v5".into(),
            backend: ModelBackend::Voxtral,
            category: ModelCategory::Vad,
            hf_repo: Some("snakers4/silero-vad".into()),
            hf_files: vec!["silero_vad.onnx".into()],
            approx_size_bytes: 2 * MB,
        },
    ]
}

/// Determine which model ID is required by the current config.
pub fn required_model_id(cfg: &Config) -> Option<String> {
    match cfg.stt.backend.as_str() {
        "whisper-native" | "whisper-cpp" => {
            let model = cfg.stt.whisper_model.as_str();
            match model {
                "tiny" => Some("openai/whisper-tiny".into()),
                "small" => Some("openai/whisper-small".into()),
                "medium" => Some("openai/whisper-medium".into()),
                "large-v3" | "large" => Some("openai/whisper-large-v3".into()),
                _ => {
                    log::warn!("Unknown whisper model '{model}', no catalog entry");
                    None
                }
            }
        }
        "voxtral-native" => Some("mistral/voxtral-mini".into()),
        _ => None,  // HTTP backends don't need local models
    }
}
