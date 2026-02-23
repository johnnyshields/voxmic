//! System prompt and tool definitions for the LLM agent.

/// System prompt for the computer-use agent.
pub fn system_prompt() -> String {
    r#"You are a desktop automation agent. You control a computer by interacting with UI elements through an accessibility API.

## How it works
- You receive a structured UI tree (JSON) representing the current state of the focused application
- You can perform actions on elements by referencing their numeric `id` field
- After each action, you receive an updated UI tree showing the result

## Rules
1. Always examine the UI tree carefully before acting
2. Reference elements by their numeric `id` (e.g. id=3 means element #3)
3. Use the most specific action available (e.g. `click` for buttons, `set_value` for text fields)
4. If a goal requires multiple steps, do them one at a time and verify each step
5. If an action fails, try alternative approaches (different element, different action)
6. When the goal is achieved, stop calling tools and explain what you did
7. If you cannot achieve the goal, explain why

## Available actions
- `click(element_id)` — Click/invoke a button, link, or menu item
- `set_value(element_id, value)` — Set text in an input field
- `send_keys(keys)` — Send keyboard input (e.g. "Hello" or "{Enter}" or "{Ctrl+S}")
- `scroll(element_id, direction, amount)` — Scroll up/down/left/right
- `toggle(element_id)` — Toggle a checkbox or switch
- `expand(element_id)` — Expand a tree node or combo box
- `collapse(element_id)` — Collapse a tree node or combo box
- `select(element_id)` — Select a list item or tab
- `focus(element_id)` — Set focus to an element
- `wait(ms)` — Wait for a specified number of milliseconds"#
        .into()
}

/// Tool definitions for Claude API tool_use format.
pub fn tool_definitions() -> Vec<serde_json::Value> {
    serde_json::from_str(TOOLS_JSON).expect("built-in tool definitions should be valid JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_match_known_actions() {
        let tools = tool_definitions();
        let tool_names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();

        // These must match the arms in agent::parse_tool_call()
        let expected = vec![
            "click", "set_value", "send_keys", "scroll", "toggle",
            "expand", "collapse", "select", "focus", "wait",
        ];

        assert_eq!(
            tool_names.len(),
            expected.len(),
            "tool count mismatch: tools JSON has {:?}, expected {:?}",
            tool_names,
            expected
        );

        for name in &expected {
            assert!(
                tool_names.contains(name),
                "missing tool definition for '{name}' — add it to TOOLS_JSON in prompt.rs"
            );
        }

        for name in &tool_names {
            assert!(
                expected.contains(name),
                "extra tool '{name}' in TOOLS_JSON — add a parse_tool_call arm in agent.rs"
            );
        }
    }
}

const TOOLS_JSON: &str = r#"[
  {
    "name": "click",
    "description": "Click/invoke a UI element (button, link, menu item, etc.)",
    "input_schema": {
      "type": "object",
      "properties": {
        "element_id": { "type": "integer", "description": "The numeric id of the element to click" }
      },
      "required": ["element_id"]
    }
  },
  {
    "name": "set_value",
    "description": "Set the text value of an input field",
    "input_schema": {
      "type": "object",
      "properties": {
        "element_id": { "type": "integer", "description": "The numeric id of the text input" },
        "value": { "type": "string", "description": "The text to set" }
      },
      "required": ["element_id", "value"]
    }
  },
  {
    "name": "send_keys",
    "description": "Send keyboard input. Use {Key} syntax for special keys: {Enter}, {Tab}, {Escape}, {Backspace}, {Delete}, {Up}, {Down}, {Left}, {Right}, {Home}, {End}, {Ctrl+KEY}, {Alt+KEY}, {Shift+KEY}",
    "input_schema": {
      "type": "object",
      "properties": {
        "keys": { "type": "string", "description": "The keys to send" }
      },
      "required": ["keys"]
    }
  },
  {
    "name": "scroll",
    "description": "Scroll a scrollable element",
    "input_schema": {
      "type": "object",
      "properties": {
        "element_id": { "type": "integer", "description": "The numeric id of the scrollable element" },
        "direction": { "type": "string", "enum": ["up", "down", "left", "right"] },
        "amount": { "type": "integer", "description": "Number of scroll units (default: 3)", "default": 3 }
      },
      "required": ["element_id", "direction"]
    }
  },
  {
    "name": "toggle",
    "description": "Toggle a checkbox or switch",
    "input_schema": {
      "type": "object",
      "properties": {
        "element_id": { "type": "integer", "description": "The numeric id of the checkbox/switch" }
      },
      "required": ["element_id"]
    }
  },
  {
    "name": "expand",
    "description": "Expand a tree node, combo box, or collapsible section",
    "input_schema": {
      "type": "object",
      "properties": {
        "element_id": { "type": "integer", "description": "The numeric id of the element to expand" }
      },
      "required": ["element_id"]
    }
  },
  {
    "name": "collapse",
    "description": "Collapse a tree node, combo box, or collapsible section",
    "input_schema": {
      "type": "object",
      "properties": {
        "element_id": { "type": "integer", "description": "The numeric id of the element to collapse" }
      },
      "required": ["element_id"]
    }
  },
  {
    "name": "select",
    "description": "Select a list item, tab, or selectable element",
    "input_schema": {
      "type": "object",
      "properties": {
        "element_id": { "type": "integer", "description": "The numeric id of the element to select" }
      },
      "required": ["element_id"]
    }
  },
  {
    "name": "focus",
    "description": "Set keyboard focus to an element",
    "input_schema": {
      "type": "object",
      "properties": {
        "element_id": { "type": "integer", "description": "The numeric id of the element to focus" }
      },
      "required": ["element_id"]
    }
  },
  {
    "name": "wait",
    "description": "Wait for a specified number of milliseconds (useful for animations or loading)",
    "input_schema": {
      "type": "object",
      "properties": {
        "ms": { "type": "integer", "description": "Milliseconds to wait", "default": 1000 }
      },
      "required": ["ms"]
    }
  }
]"#;
