//! Voxtral HTTP backend — multipart POST to llama-server /v1/audio/transcriptions.

use std::path::Path;

use super::Transcriber;
use crate::config::SttConfig;

/// Transcribes audio by sending WAV files to a Voxtral-compatible HTTP endpoint.
pub struct VoxtralHttpTranscriber {
    url: String,
}

impl VoxtralHttpTranscriber {
    pub fn new(cfg: &SttConfig) -> Self {
        let url = cfg.voxtral_url.trim_end_matches('/').to_string();
        log::info!("VoxtralHttpTranscriber: endpoint {url}");
        Self { url }
    }
}

impl Transcriber for VoxtralHttpTranscriber {
    fn transcribe(&self, wav_path: &Path) -> anyhow::Result<String> {
        let wav_bytes = std::fs::read(wav_path)?;
        let filename = wav_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("audio.wav");

        // ureq v2 has no built-in multipart — build the body manually.
        let boundary = format!(
            "----voxctrl{:016x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        let mut body = Vec::with_capacity(wav_bytes.len() + 256);
        write_multipart_file(
            &mut body,
            &boundary,
            "file",
            filename,
            "audio/wav",
            &wav_bytes,
        );
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let resp: serde_json::Value = ureq::post(&format!("{}/v1/audio/transcriptions", self.url))
            .set(
                "Content-Type",
                &format!("multipart/form-data; boundary={boundary}"),
            )
            .send_bytes(&body)?
            .into_json()?;

        let text = resp["text"].as_str().unwrap_or("").trim().to_string();
        log::debug!("VoxtralHttp transcription: {text:?}");
        Ok(text)
    }

    fn name(&self) -> &str {
        "Voxtral HTTP"
    }

    fn is_available(&self) -> bool {
        ureq::get(&format!("{}/health", self.url)).call().is_ok()
    }
}

/// Write a single file part into a multipart/form-data body.
fn write_multipart_file(
    body: &mut Vec<u8>,
    boundary: &str,
    field: &str,
    filename: &str,
    content_type: &str,
    data: &[u8],
) {
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n\
             Content-Type: {content_type}\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(data);
    body.extend_from_slice(b"\r\n");
}
