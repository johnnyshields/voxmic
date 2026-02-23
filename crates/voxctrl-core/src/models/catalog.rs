use crate::config::Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelBackend {
    Voxtral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCategory {
    Stt,
    Vad,
    ComputerUse,
}

impl std::fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCategory::Stt => write!(f, "STT"),
            ModelCategory::Vad => write!(f, "VAD"),
            ModelCategory::ComputerUse => write!(f, "CU"),
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
            hf_repo: Some("onnx-community/silero-vad".into()),
            hf_files: vec!["onnx/model.onnx".into()],
            approx_size_bytes: 2 * MB,
        },
    ]
}

/// Map an STT backend + whisper model size to the catalog model ID.
pub fn required_stt_model_id(backend: &str, whisper_model: &str) -> Option<String> {
    match backend {
        "whisper-native" | "whisper-cpp" => match whisper_model {
            "tiny" => Some("openai/whisper-tiny".into()),
            "small" => Some("openai/whisper-small".into()),
            "medium" => Some("openai/whisper-medium".into()),
            "large-v3" | "large" => Some("openai/whisper-large-v3".into()),
            _ => {
                log::warn!("Unknown whisper model '{whisper_model}', no catalog entry");
                None
            }
        },
        "voxtral-native" => Some("mistral/voxtral-mini".into()),
        _ => None, // HTTP backends don't need local models
    }
}

/// Map a VAD backend to the catalog model ID.
pub fn required_vad_model_id(vad_backend: &str) -> Option<String> {
    match vad_backend {
        "silero" => Some("silero/vad-v5".into()),
        _ => None,
    }
}

/// Determine which STT model ID is required by the current config.
pub fn required_model_id(cfg: &Config) -> Option<String> {
    required_stt_model_id(&cfg.stt.backend, &cfg.stt.whisper_model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_category_display() {
        assert_eq!(ModelCategory::Stt.to_string(), "STT");
        assert_eq!(ModelCategory::Vad.to_string(), "VAD");
        assert_eq!(ModelCategory::ComputerUse.to_string(), "CU");
    }

    #[test]
    fn test_silero_catalog_entry() {
        let models = all_models();
        let silero = models.iter().find(|m| m.id == "silero/vad-v5")
            .expect("silero/vad-v5 must be in the catalog");

        assert_eq!(silero.hf_repo.as_deref(), Some("onnx-community/silero-vad"));
        assert_eq!(silero.hf_files, vec!["onnx/model.onnx"]);
        assert_eq!(silero.category, ModelCategory::Vad);
    }

    #[test]
    fn test_required_vad_model_id_silero() {
        assert_eq!(required_vad_model_id("silero"), Some("silero/vad-v5".into()));
        assert_eq!(required_vad_model_id("energy"), None);
        assert_eq!(required_vad_model_id("none"), None);
    }
}
