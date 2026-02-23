//! Screenshot utilities — base64 encoding for LLM context.

/// Encode PNG screenshot bytes to a base64 string suitable for Claude API image content.
pub fn encode_screenshot_base64(png_bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(png_bytes)
}

/// Build a Claude API image content block from PNG bytes.
pub fn screenshot_content_block(png_bytes: &[u8]) -> serde_json::Value {
    serde_json::json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": encode_screenshot_base64(png_bytes)
        }
    })
}
