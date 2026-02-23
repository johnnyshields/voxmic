//! Unified UI tree types — cross-platform representation of accessibility trees.

use serde::{Deserialize, Serialize};

/// Unique identifier for a UI element within a snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ElementId {
    /// Opaque platform-specific handle (e.g. UIA RuntimeId, AX element pointer).
    /// Used by the platform provider to relocate the element.
    pub platform_handle: String,
    /// Sequential index within this snapshot. The LLM references elements by this index.
    pub index: usize,
}

impl std::fmt::Display for ElementId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.index)
    }
}

/// Cross-platform UI element role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UiRole {
    Window,
    Button,
    TextInput,
    Label,
    CheckBox,
    RadioButton,
    ComboBox,
    List,
    ListItem,
    Menu,
    MenuItem,
    MenuBar,
    Tab,
    TabItem,
    Tree,
    TreeItem,
    Table,
    TableRow,
    TableCell,
    ScrollBar,
    Slider,
    ProgressBar,
    ToolBar,
    StatusBar,
    Dialog,
    Group,
    Pane,
    Link,
    Image,
    Document,
    Separator,
    TitleBar,
    Custom(String),
    Unknown,
}

impl std::fmt::Display for UiRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UiRole::Custom(s) => write!(f, "custom:{s}"),
            other => {
                let json = serde_json::to_string(other).unwrap_or_default();
                // Remove quotes from JSON string
                write!(f, "{}", json.trim_matches('"'))
            }
        }
    }
}

/// Bounding rectangle of a UI element (screen coordinates).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// UI element state flags.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UiState {
    Enabled,
    Disabled,
    Focused,
    Selected,
    Checked,
    Expanded,
    Collapsed,
    ReadOnly,
    Invisible,
    Offscreen,
}

/// A node in the UI tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiNode {
    pub id: ElementId,
    pub role: UiRole,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounds: Option<UiRect>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub states: Vec<UiState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub available_actions: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<UiNode>,
}

/// A complete UI tree snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiTree {
    /// Root window / application node.
    pub root: UiNode,
    /// Window title (for context).
    pub window_title: String,
    /// Process name (for context).
    pub process_name: String,
    /// Total number of elements in the tree.
    pub element_count: usize,
}

impl UiTree {
    /// Serialize the tree to a compact JSON string suitable for LLM context.
    ///
    /// Prunes nodes deeper than `max_depth` and strips invisible/offscreen nodes
    /// to reduce token usage.
    pub fn to_llm_json(&self, max_depth: usize) -> String {
        let pruned = PrunedTree {
            window_title: &self.window_title,
            process_name: &self.process_name,
            element_count: self.element_count,
            root: prune_node(&self.root, 0, max_depth),
        };
        serde_json::to_string(&pruned).unwrap_or_else(|_| "{}".into())
    }
}

#[derive(Serialize)]
struct PrunedTree<'a> {
    window_title: &'a str,
    process_name: &'a str,
    element_count: usize,
    root: Option<PrunedNode>,
}

#[derive(Serialize)]
struct PrunedNode {
    #[serde(rename = "id")]
    index: usize,
    role: String,
    #[serde(skip_serializing_if = "str::is_empty")]
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    states: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    actions: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<PrunedNode>,
}

fn prune_node(node: &UiNode, depth: usize, max_depth: usize) -> Option<PrunedNode> {
    // Skip invisible/offscreen nodes
    if node.states.contains(&UiState::Invisible) || node.states.contains(&UiState::Offscreen) {
        return None;
    }

    let children = if depth < max_depth {
        node.children
            .iter()
            .filter_map(|c| prune_node(c, depth + 1, max_depth))
            .collect()
    } else {
        Vec::new()
    };

    Some(PrunedNode {
        index: node.id.index,
        role: node.role.to_string(),
        name: node.name.clone(),
        value: node.value.clone(),
        states: node.states.iter().map(|s| format!("{s:?}").to_lowercase()).collect(),
        actions: node.available_actions.clone(),
        children,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> UiTree {
        UiTree {
            window_title: "Notepad".into(),
            process_name: "notepad.exe".into(),
            element_count: 3,
            root: UiNode {
                id: ElementId { platform_handle: "root".into(), index: 0 },
                role: UiRole::Window,
                name: "Notepad".into(),
                description: None,
                bounds: Some(UiRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 }),
                states: vec![UiState::Enabled, UiState::Focused],
                value: None,
                available_actions: vec![],
                children: vec![
                    UiNode {
                        id: ElementId { platform_handle: "edit".into(), index: 1 },
                        role: UiRole::TextInput,
                        name: "Text Editor".into(),
                        description: None,
                        bounds: Some(UiRect { x: 0.0, y: 30.0, width: 800.0, height: 550.0 }),
                        states: vec![UiState::Enabled],
                        value: Some("Hello".into()),
                        available_actions: vec!["set_value".into(), "focus".into()],
                        children: vec![],
                    },
                    UiNode {
                        id: ElementId { platform_handle: "hidden".into(), index: 2 },
                        role: UiRole::Pane,
                        name: "".into(),
                        description: None,
                        bounds: None,
                        states: vec![UiState::Invisible],
                        value: None,
                        available_actions: vec![],
                        children: vec![],
                    },
                ],
            },
        }
    }

    #[test]
    fn to_llm_json_prunes_invisible() {
        let tree = sample_tree();
        let json = tree.to_llm_json(8);
        assert!(!json.contains("hidden"), "invisible node should be pruned");
        assert!(json.contains("Text Editor"), "visible node should remain");
    }

    #[test]
    fn to_llm_json_respects_max_depth() {
        let tree = sample_tree();
        let json = tree.to_llm_json(0);
        // At depth 0, only root node, no children
        assert!(!json.contains("Text Editor"), "children should be pruned at depth 0");
    }

    #[test]
    fn element_id_display() {
        let id = ElementId { platform_handle: "abc".into(), index: 5 };
        assert_eq!(format!("{id}"), "#5");
    }

    #[test]
    fn all_invisible_tree() {
        let tree = UiTree {
            window_title: "Test".into(),
            process_name: "test".into(),
            element_count: 1,
            root: UiNode {
                id: ElementId { platform_handle: "r".into(), index: 0 },
                role: UiRole::Window,
                name: "Root".into(),
                description: None,
                bounds: None,
                states: vec![UiState::Invisible],
                value: None,
                available_actions: vec![],
                children: vec![],
            },
        };
        let json = tree.to_llm_json(8);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["root"].is_null(), "invisible root should be pruned");
    }

    #[test]
    fn mixed_visible_invisible_siblings() {
        let tree = UiTree {
            window_title: "Test".into(),
            process_name: "test".into(),
            element_count: 3,
            root: UiNode {
                id: ElementId { platform_handle: "r".into(), index: 0 },
                role: UiRole::Window,
                name: "Root".into(),
                description: None,
                bounds: None,
                states: vec![UiState::Enabled],
                value: None,
                available_actions: vec![],
                children: vec![
                    UiNode {
                        id: ElementId { platform_handle: "a".into(), index: 1 },
                        role: UiRole::Button,
                        name: "Visible".into(),
                        description: None,
                        bounds: None,
                        states: vec![UiState::Enabled],
                        value: None,
                        available_actions: vec!["click".into()],
                        children: vec![],
                    },
                    UiNode {
                        id: ElementId { platform_handle: "b".into(), index: 2 },
                        role: UiRole::Button,
                        name: "Hidden".into(),
                        description: None,
                        bounds: None,
                        states: vec![UiState::Invisible],
                        value: None,
                        available_actions: vec![],
                        children: vec![],
                    },
                ],
            },
        };
        let json = tree.to_llm_json(8);
        assert!(json.contains("Visible"));
        assert!(!json.contains("Hidden"));
    }

    #[test]
    fn offscreen_nodes_pruned() {
        let tree = UiTree {
            window_title: "Test".into(),
            process_name: "test".into(),
            element_count: 2,
            root: UiNode {
                id: ElementId { platform_handle: "r".into(), index: 0 },
                role: UiRole::Window,
                name: "Root".into(),
                description: None,
                bounds: None,
                states: vec![UiState::Enabled],
                value: None,
                available_actions: vec![],
                children: vec![UiNode {
                    id: ElementId { platform_handle: "off".into(), index: 1 },
                    role: UiRole::Pane,
                    name: "Offscreen".into(),
                    description: None,
                    bounds: None,
                    states: vec![UiState::Offscreen],
                    value: None,
                    available_actions: vec![],
                    children: vec![],
                }],
            },
        };
        let json = tree.to_llm_json(8);
        assert!(!json.contains("Offscreen"));
    }
}
