//! LLM agent loop — drives desktop automation through Claude API.
//!
//! Flow:
//! 1. Capture UI tree (+ optional screenshot)
//! 2. Send system prompt + tool definitions + tree JSON + user goal → Claude API
//! 3. Parse response for tool_use blocks
//! 4. If no tool calls → agent is done
//! 5. Execute each UiAction via AccessibilityProvider
//! 6. Re-capture tree (observe changes)
//! 7. Send tool results + updated tree → Claude
//! 8. Repeat until done or max iterations

use serde_json::Value;

use crate::actions::{ScrollDirection, UiAction, UiActionResult};
use crate::prompt;
use crate::provider::AccessibilityProvider;
use crate::screenshot;
use crate::tree::ElementId;

/// A newtype wrapping an API key string that masks its value in Debug output.
#[derive(Clone)]
pub struct ApiKey {
    key: String,
    masked: bool,
}

impl ApiKey {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            masked: true,
        }
    }

    pub fn set_masked(&mut self, masked: bool) {
        self.masked = masked;
    }

    pub fn as_str(&self) -> &str {
        &self.key
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.masked {
            write!(f, "ApiKey(***)")
        } else {
            write!(f, "ApiKey({:?})", self.key)
        }
    }
}

impl From<String> for ApiKey {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for ApiKey {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Configuration for the agent loop.
pub struct AgentConfig {
    pub model: String,
    pub api_base_url: String,
    pub api_key: ApiKey,
    pub max_iterations: u32,
    pub max_tree_depth: usize,
    pub include_screenshots: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".into(),
            api_base_url: "https://api.anthropic.com".into(),
            api_key: ApiKey::new(String::new()),
            max_iterations: 10,
            max_tree_depth: 8,
            include_screenshots: false,
        }
    }
}

/// Result of running the agent loop.
pub struct AgentResult {
    /// Final text response from the agent.
    pub summary: String,
    /// Number of iterations the agent ran.
    pub iterations: u32,
    /// All actions performed.
    pub actions_performed: Vec<(UiAction, UiActionResult)>,
}

/// Run the LLM agent loop to achieve a user goal.
pub fn run_agent(
    provider: &dyn AccessibilityProvider,
    config: &AgentConfig,
    goal: &str,
) -> anyhow::Result<AgentResult> {
    let mut messages: Vec<Value> = Vec::new();
    let mut actions_performed = Vec::new();
    let mut iterations = 0u32;

    // Initial UI tree capture
    let tree = provider.get_focused_tree()?;
    let tree_json = tree.to_llm_json(config.max_tree_depth);

    // Build initial user message with goal + tree
    let mut content_parts: Vec<Value> = vec![serde_json::json!({
        "type": "text",
        "text": format!("Goal: {goal}\n\nCurrent UI tree:\n```json\n{tree_json}\n```")
    })];

    // Optionally add screenshot
    if config.include_screenshots {
        if let Ok(Some(png)) = provider.capture_screenshot() {
            content_parts.push(screenshot::screenshot_content_block(&png));
        }
    }

    messages.push(serde_json::json!({
        "role": "user",
        "content": content_parts
    }));

    loop {
        if iterations >= config.max_iterations {
            log::warn!("Agent loop hit max iterations ({}) — stopping", config.max_iterations);
            return Ok(AgentResult {
                summary: format!("Reached maximum iterations ({}) without completing goal", config.max_iterations),
                iterations,
                actions_performed,
            });
        }
        iterations += 1;

        log::info!("Agent iteration {}/{}", iterations, config.max_iterations);

        // Call Claude API
        let response = call_claude_api(config, &messages)?;

        // Extract content blocks from response
        let content = response["content"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        // Collect the assistant response for conversation history
        messages.push(serde_json::json!({
            "role": "assistant",
            "content": content
        }));

        // Find tool_use blocks
        let tool_uses: Vec<&Value> = content
            .iter()
            .filter(|b| b["type"].as_str() == Some("tool_use"))
            .collect();

        // If no tool calls, agent is done — extract final text
        if tool_uses.is_empty() {
            let summary = content
                .iter()
                .filter(|b| b["type"].as_str() == Some("text"))
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n");

            return Ok(AgentResult {
                summary,
                iterations,
                actions_performed,
            });
        }

        // Execute each tool call and collect results
        let mut tool_results: Vec<Value> = Vec::new();

        for tool_use in &tool_uses {
            let tool_name = tool_use["name"].as_str().unwrap_or("");
            let tool_id = tool_use["id"].as_str().unwrap_or("");
            let input = &tool_use["input"];

            log::info!("Agent tool call: {}({})", tool_name, input);

            let action = parse_tool_call(tool_name, input);
            let result = match &action {
                Ok(ui_action) => {
                    match ui_action {
                        UiAction::Wait { ms } => {
                            std::thread::sleep(std::time::Duration::from_millis(*ms));
                            UiActionResult::ok(format!("Waited {ms}ms"))
                        }
                        _ => provider.perform_action(ui_action)?,
                    }
                }
                Err(e) => UiActionResult::err(format!("Tool call '{tool_name}' failed: {e}")),
            };

            log::info!("  → {}: {}", if result.success { "ok" } else { "err" }, result.message);

            tool_results.push(serde_json::json!({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result.message,
                "is_error": !result.success,
            }));

            if let Ok(action) = action {
                actions_performed.push((action, result));
            }
        }

        // Re-capture UI tree after actions
        let updated_tree = provider.get_focused_tree()?;
        let updated_tree_json = updated_tree.to_llm_json(config.max_tree_depth);

        // Add updated tree observation
        tool_results.push(serde_json::json!({
            "type": "text",
            "text": format!("Updated UI tree after actions:\n```json\n{updated_tree_json}\n```")
        }));

        // Optionally add updated screenshot
        if config.include_screenshots {
            if let Ok(Some(png)) = provider.capture_screenshot() {
                tool_results.push(screenshot::screenshot_content_block(&png));
            }
        }

        messages.push(serde_json::json!({
            "role": "user",
            "content": tool_results
        }));
    }
}

/// Parse a tool call from the LLM into a UiAction.
fn parse_tool_call(tool_name: &str, input: &Value) -> anyhow::Result<UiAction> {
    let get_element_id = |input: &Value, tool_name: &str| -> anyhow::Result<ElementId> {
        let id = input["element_id"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("missing required field 'element_id'"))? as usize;
        if id > 100_000 {
            anyhow::bail!("{tool_name}: element_id {id} exceeds maximum (100000)");
        }
        Ok(ElementId {
            platform_handle: String::new(), // resolved by provider at action time
            index: id,
        })
    };

    match tool_name {
        "click" => Ok(UiAction::Click {
            element_id: get_element_id(input, tool_name)?,
        }),
        "set_value" => Ok(UiAction::SetValue {
            element_id: get_element_id(input, tool_name)?,
            value: input["value"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("set_value: missing required field 'value'"))?
                .into(),
        }),
        "send_keys" => Ok(UiAction::SendKeys {
            keys: input["keys"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("send_keys: missing required field 'keys'"))?
                .into(),
        }),
        "scroll" => {
            let element_id = get_element_id(input, tool_name)?;
            let direction = match input["direction"].as_str().unwrap_or("down") {
                "up" => ScrollDirection::Up,
                "down" => ScrollDirection::Down,
                "left" => ScrollDirection::Left,
                "right" => ScrollDirection::Right,
                other => anyhow::bail!("scroll: invalid direction '{other}' (expected: up, down, left, right)"),
            };
            let amount = input["amount"].as_u64().unwrap_or(3) as u32;
            if !(1..=100).contains(&amount) {
                anyhow::bail!("scroll: amount {amount} out of range (1-100)");
            }
            Ok(UiAction::Scroll {
                element_id,
                direction,
                amount,
            })
        }
        "toggle" => Ok(UiAction::Toggle {
            element_id: get_element_id(input, tool_name)?,
        }),
        "expand" => Ok(UiAction::Expand {
            element_id: get_element_id(input, tool_name)?,
        }),
        "collapse" => Ok(UiAction::Collapse {
            element_id: get_element_id(input, tool_name)?,
        }),
        "select" => Ok(UiAction::Select {
            element_id: get_element_id(input, tool_name)?,
        }),
        "focus" => Ok(UiAction::Focus {
            element_id: get_element_id(input, tool_name)?,
        }),
        "wait" => {
            let ms = input["ms"].as_u64().unwrap_or(1000);
            if !(1..=60_000).contains(&ms) {
                anyhow::bail!("wait: ms {ms} out of range (1-60000)");
            }
            Ok(UiAction::Wait { ms })
        }
        _ => anyhow::bail!("unknown tool '{tool_name}' — available: click, set_value, send_keys, scroll, toggle, expand, collapse, select, focus, wait"),
    }
}

/// Call the Claude Messages API.
fn call_claude_api(config: &AgentConfig, messages: &[Value]) -> anyhow::Result<Value> {
    let url = format!("{}/v1/messages", config.api_base_url);

    let body = serde_json::json!({
        "model": config.model,
        "max_tokens": 4096,
        "system": prompt::system_prompt(),
        "tools": prompt::tool_definitions(),
        "messages": messages,
    });

    log::debug!("Claude API request: {} messages", messages.len());

    let resp = ureq::post(&url)
        .set("x-api-key", config.api_key.as_str())
        .set("anthropic-version", "2023-06-01")
        .set("content-type", "application/json")
        .send_json(&body)
        .map_err(|e| anyhow::anyhow!("Claude API request failed: {e}"))?;

    let body: Value = resp.into_json()
        .map_err(|e| anyhow::anyhow!("Failed to parse Claude API response: {e}"))?;

    if let Some(err_type) = body["error"]["type"].as_str() {
        let msg = body["error"]["message"].as_str().unwrap_or("unknown error");
        anyhow::bail!("Claude API error ({err_type}): {msg}");
    }

    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_click_tool_call() {
        let input = serde_json::json!({"element_id": 5});
        let action = parse_tool_call("click", &input).unwrap();
        match action {
            UiAction::Click { element_id } => assert_eq!(element_id.index, 5),
            _ => panic!("expected Click"),
        }
    }

    #[test]
    fn parse_set_value_tool_call() {
        let input = serde_json::json!({"element_id": 2, "value": "hello"});
        let action = parse_tool_call("set_value", &input).unwrap();
        match action {
            UiAction::SetValue { element_id, value } => {
                assert_eq!(element_id.index, 2);
                assert_eq!(value, "hello");
            }
            _ => panic!("expected SetValue"),
        }
    }

    #[test]
    fn parse_send_keys_tool_call() {
        let input = serde_json::json!({"keys": "{Ctrl+S}"});
        let action = parse_tool_call("send_keys", &input).unwrap();
        match action {
            UiAction::SendKeys { keys } => assert_eq!(keys, "{Ctrl+S}"),
            _ => panic!("expected SendKeys"),
        }
    }

    #[test]
    fn parse_scroll_defaults() {
        let input = serde_json::json!({"element_id": 1, "direction": "down"});
        let action = parse_tool_call("scroll", &input).unwrap();
        match action {
            UiAction::Scroll { amount, .. } => assert_eq!(amount, 3),
            _ => panic!("expected Scroll"),
        }
    }

    #[test]
    fn parse_wait_tool_call() {
        let input = serde_json::json!({"ms": 500});
        let action = parse_tool_call("wait", &input).unwrap();
        match action {
            UiAction::Wait { ms } => assert_eq!(ms, 500),
            _ => panic!("expected Wait"),
        }
    }

    #[test]
    fn parse_unknown_tool_returns_error() {
        let input = serde_json::json!({});
        assert!(parse_tool_call("fly_to_moon", &input).is_err());
    }

    // Task #1: ApiKey tests

    #[test]
    fn api_key_masks_in_debug() {
        let key = ApiKey::new("sk-ant-secret123");
        assert!(!format!("{key:?}").contains("secret"));
        assert!(format!("{key:?}").contains("***"));
    }

    #[test]
    fn api_key_reveals_when_unmasked() {
        let mut key = ApiKey::new("sk-ant-secret123");
        key.set_masked(false);
        assert!(format!("{key:?}").contains("secret123"));
    }

    // Task #5: Boundary/edge-case tests

    #[test]
    fn parse_element_id_overflow() {
        let input = serde_json::json!({"element_id": 100_001});
        let err = parse_tool_call("click", &input).unwrap_err();
        assert!(err.to_string().contains("exceeds maximum"));
    }

    #[test]
    fn parse_wait_excessive_ms() {
        let input = serde_json::json!({"ms": 120_000});
        let err = parse_tool_call("wait", &input).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn parse_wait_zero_ms() {
        let input = serde_json::json!({"ms": 0});
        let err = parse_tool_call("wait", &input).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn parse_scroll_amount_overflow() {
        let input = serde_json::json!({"element_id": 1, "direction": "up", "amount": 999});
        let err = parse_tool_call("scroll", &input).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn parse_click_missing_element_id() {
        let input = serde_json::json!({});
        let err = parse_tool_call("click", &input).unwrap_err();
        assert!(err.to_string().contains("element_id"));
    }

    #[test]
    fn parse_set_value_missing_value() {
        let input = serde_json::json!({"element_id": 1});
        let err = parse_tool_call("set_value", &input).unwrap_err();
        assert!(err.to_string().contains("value"));
    }

    #[test]
    fn parse_send_keys_missing_keys() {
        let input = serde_json::json!({});
        let err = parse_tool_call("send_keys", &input).unwrap_err();
        assert!(err.to_string().contains("keys"));
    }
}
