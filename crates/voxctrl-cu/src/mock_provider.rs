//! Mock accessibility provider for testing the agent loop without a real desktop.

use crate::actions::{UiAction, UiActionResult};
use crate::provider::AccessibilityProvider;
use crate::tree::{ElementId, UiNode, UiRect, UiRole, UiState, UiTree};

/// A mock provider that returns hardcoded UI trees and logs actions.
///
/// Useful for testing the agent loop in the GUI test tab without requiring
/// a real accessibility provider or desktop environment.
pub struct MockProvider;

impl MockProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AccessibilityProvider for MockProvider {
    fn get_focused_tree(&self) -> anyhow::Result<UiTree> {
        Ok(UiTree {
            window_title: "Mock Notepad".into(),
            process_name: "notepad.exe".into(),
            element_count: 6,
            root: UiNode {
                id: ElementId { platform_handle: String::new(), index: 0 },
                role: UiRole::Window,
                name: "Mock Notepad".into(),
                description: None,
                bounds: Some(UiRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 }),
                states: vec![UiState::Enabled, UiState::Focused],
                value: None,
                available_actions: vec![],
                children: vec![
                    UiNode {
                        id: ElementId { platform_handle: String::new(), index: 1 },
                        role: UiRole::MenuBar,
                        name: "Menu Bar".into(),
                        description: None,
                        bounds: Some(UiRect { x: 0.0, y: 0.0, width: 800.0, height: 25.0 }),
                        states: vec![UiState::Enabled],
                        value: None,
                        available_actions: vec![],
                        children: vec![
                            UiNode {
                                id: ElementId { platform_handle: String::new(), index: 2 },
                                role: UiRole::MenuItem,
                                name: "File".into(),
                                description: None,
                                bounds: None,
                                states: vec![UiState::Enabled],
                                value: None,
                                available_actions: vec!["click".into()],
                                children: vec![],
                            },
                            UiNode {
                                id: ElementId { platform_handle: String::new(), index: 3 },
                                role: UiRole::MenuItem,
                                name: "Edit".into(),
                                description: None,
                                bounds: None,
                                states: vec![UiState::Enabled],
                                value: None,
                                available_actions: vec!["click".into()],
                                children: vec![],
                            },
                        ],
                    },
                    UiNode {
                        id: ElementId { platform_handle: String::new(), index: 4 },
                        role: UiRole::TextInput,
                        name: "Text Editor".into(),
                        description: None,
                        bounds: Some(UiRect { x: 0.0, y: 25.0, width: 800.0, height: 550.0 }),
                        states: vec![UiState::Enabled],
                        value: Some("Hello, World!".into()),
                        available_actions: vec!["set_value".into(), "focus".into()],
                        children: vec![],
                    },
                    UiNode {
                        id: ElementId { platform_handle: String::new(), index: 5 },
                        role: UiRole::StatusBar,
                        name: "Status Bar".into(),
                        description: None,
                        bounds: Some(UiRect { x: 0.0, y: 575.0, width: 800.0, height: 25.0 }),
                        states: vec![UiState::Enabled],
                        value: Some("Ln 1, Col 1".into()),
                        available_actions: vec![],
                        children: vec![],
                    },
                ],
            },
        })
    }

    fn get_tree_for_pid(&self, _pid: u32) -> anyhow::Result<UiTree> {
        self.get_focused_tree()
    }

    fn find_elements(&self, _query: &str) -> anyhow::Result<Vec<UiNode>> {
        Ok(vec![])
    }

    fn perform_action(&self, action: &UiAction) -> anyhow::Result<UiActionResult> {
        log::info!("MockProvider: {action}");
        Ok(UiActionResult::ok(format!("Mock: executed {action}")))
    }

    fn capture_screenshot(&self) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(None)
    }

    fn platform_name(&self) -> &str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_provider_returns_tree() {
        let provider = MockProvider::new();
        let tree = provider.get_focused_tree().unwrap();
        assert_eq!(tree.window_title, "Mock Notepad");
        assert_eq!(tree.element_count, 6);
    }

    #[test]
    fn mock_provider_performs_action() {
        let provider = MockProvider::new();
        let action = UiAction::Click {
            element_id: ElementId { platform_handle: String::new(), index: 2 },
        };
        let result = provider.perform_action(&action).unwrap();
        assert!(result.success);
        assert!(result.message.contains("Mock"));
    }

    #[test]
    fn mock_provider_no_screenshot() {
        let provider = MockProvider::new();
        assert!(provider.capture_screenshot().unwrap().is_none());
    }

    #[test]
    fn mock_provider_platform_name() {
        let provider = MockProvider::new();
        assert_eq!(provider.platform_name(), "mock");
    }
}
