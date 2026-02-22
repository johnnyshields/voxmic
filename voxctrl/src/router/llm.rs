//! LLM-based intent router — classifies text as dictation or command via an LLM endpoint.

use super::{Intent, IntentRouter};
use crate::config::RouterConfig;

const SYSTEM_PROMPT: &str = r#"You are an intent classifier for a voice dictation system.
Given the user's transcribed speech, respond with EXACTLY one JSON object:
- If the text is ordinary dictation, respond: {"intent":"dictate"}
- If the text is a command (e.g. "open browser", "search for X"), respond: {"intent":"command","action":"<action_name>","args":{}}
Respond with only the JSON object, no other text."#;

/// Routes transcriptions through an LLM to classify intent.
pub struct LlmRouter {
    url: String,
}

impl LlmRouter {
    pub fn new(cfg: &RouterConfig) -> anyhow::Result<Self> {
        let url = cfg
            .llm_url
            .clone()
            .unwrap_or_else(|| "http://127.0.0.1:5200".into());
        log::info!("LlmRouter: using endpoint {url}");
        Ok(Self { url })
    }
}

impl IntentRouter for LlmRouter {
    fn route(&self, text: &str) -> anyhow::Result<Intent> {
        let body = serde_json::json!({
            "model": "mistral",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
        });

        let resp: serde_json::Value = ureq::post(&format!("{}/v1/chat/completions", self.url))
            .set("Content-Type", "application/json")
            .send_json(body)?
            .into_json()?;

        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");

        let parsed: serde_json::Value = serde_json::from_str(content).unwrap_or_default();

        match parsed["intent"].as_str() {
            Some("command") => Ok(Intent::Command {
                action: parsed["action"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string(),
                args: parsed["args"].clone(),
            }),
            _ => Ok(Intent::Dictate(text.to_string())),
        }
    }

    fn name(&self) -> &str {
        "llm"
    }
}
