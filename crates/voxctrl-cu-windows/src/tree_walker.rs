//! Walk the Windows UIA tree and build `UiNode` trees.

use std::collections::HashMap;

use uiautomation::UIAutomation;
use uiautomation::UIElement;

use voxctrl_cu::tree::{ElementId, UiNode, UiRect, UiState};

use crate::roles::control_type_to_role;

/// Recursively walk a UIA element and its children, building a `UiNode` tree.
///
/// - `depth`: current depth in the tree.
/// - `max_depth`: maximum depth to recurse.
/// - `index_counter`: mutable counter for assigning sequential indices.
/// - `element_map`: stores index → UIElement for later action lookup.
pub fn walk_element(
    uia: &UIAutomation,
    element: &UIElement,
    depth: usize,
    max_depth: usize,
    index_counter: &mut usize,
    element_map: &mut HashMap<usize, UIElement>,
) -> UiNode {
    let index = *index_counter;
    *index_counter += 1;

    // Store element handle for action lookup.
    element_map.insert(index, element.clone());

    let name = element.get_name().unwrap_or_default();
    let control_type = element.get_control_type().ok().map(|ct| ct as i32).unwrap_or(0);
    let role = control_type_to_role(control_type);

    // Platform handle: use runtime ID as string.
    let runtime_id = element
        .get_runtime_id()
        .map(|ids| {
            ids.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(".")
        })
        .unwrap_or_default();

    let id = ElementId {
        platform_handle: runtime_id,
        index,
    };

    let bounds = element.get_bounding_rectangle().ok().map(|r| UiRect {
        x: r.get_left() as f64,
        y: r.get_top() as f64,
        width: r.get_width() as f64,
        height: r.get_height() as f64,
    });

    let value = get_element_value(element);
    let states = get_element_states(element);
    let available_actions = get_available_actions(element);
    let description = element.get_help_text().ok().filter(|s| !s.is_empty());

    // Recurse into children if below max depth.
    let children = if depth < max_depth {
        get_children(uia, element)
            .into_iter()
            .map(|child| {
                walk_element(uia, &child, depth + 1, max_depth, index_counter, element_map)
            })
            .collect()
    } else {
        Vec::new()
    };

    UiNode {
        id,
        role,
        name,
        description,
        bounds,
        states,
        value,
        available_actions,
        children,
    }
}

/// Get child elements using the control view walker.
fn get_children(uia: &UIAutomation, parent: &UIElement) -> Vec<UIElement> {
    let mut children = Vec::new();
    let walker = match uia.get_control_view_walker() {
        Ok(w) => w,
        Err(_) => return children,
    };

    let first = match walker.get_first_child(parent) {
        Ok(c) => c,
        Err(_) => return children,
    };

    children.push(first.clone());
    let mut current = first;

    loop {
        match walker.get_next_sibling(&current) {
            Ok(sibling) => {
                children.push(sibling.clone());
                current = sibling;
            }
            Err(_) => break,
        }
    }

    children
}

/// Extract the value from a UIA element (tries ValuePattern, then RangeValuePattern).
fn get_element_value(element: &UIElement) -> Option<String> {
    // Try ValuePattern.
    if let Ok(val) = element.get_property_value(uiautomation::types::UIProperty::ValueValue) {
        let s = val.to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    // Try RangeValue.
    if let Ok(val) =
        element.get_property_value(uiautomation::types::UIProperty::RangeValueValue)
    {
        let s = val.to_string();
        if !s.is_empty() && s != "0" {
            return Some(s);
        }
    }
    None
}

/// Determine element state flags from UIA properties.
fn get_element_states(element: &UIElement) -> Vec<UiState> {
    let mut states = Vec::new();

    if element.is_enabled().unwrap_or(true) {
        states.push(UiState::Enabled);
    } else {
        states.push(UiState::Disabled);
    }

    if element.has_keyboard_focus().unwrap_or(false) {
        states.push(UiState::Focused);
    }

    if element.is_offscreen().unwrap_or(false) {
        states.push(UiState::Offscreen);
    }

    // Check toggle state for checked.
    if let Ok(val) =
        element.get_property_value(uiautomation::types::UIProperty::ToggleToggleState)
    {
        let state = val.to_string();
        if state == "1" || state.to_lowercase() == "on" {
            states.push(UiState::Checked);
        }
    }

    // Check expand/collapse state.
    if let Ok(val) = element
        .get_property_value(uiautomation::types::UIProperty::ExpandCollapseExpandCollapseState)
    {
        let state = val.to_string();
        if state == "1" || state.to_lowercase().contains("expanded") {
            states.push(UiState::Expanded);
        } else if state == "0" || state.to_lowercase().contains("collapsed") {
            states.push(UiState::Collapsed);
        }
    }

    // Check selection state.
    if let Ok(val) =
        element.get_property_value(uiautomation::types::UIProperty::SelectionItemIsSelected)
    {
        if val.to_string() == "true" || val.to_string() == "1" {
            states.push(UiState::Selected);
        }
    }

    // Check read-only for value pattern.
    if let Ok(val) =
        element.get_property_value(uiautomation::types::UIProperty::ValueIsReadOnly)
    {
        if val.to_string() == "true" || val.to_string() == "1" {
            states.push(UiState::ReadOnly);
        }
    }

    states
}

/// Determine available actions from supported UIA patterns.
fn get_available_actions(element: &UIElement) -> Vec<String> {
    let mut actions = Vec::new();

    // Check InvokePattern → click.
    if is_pattern_available(element, uiautomation::types::UIProperty::IsInvokePatternAvailable) {
        actions.push("click".into());
    }

    // Check ValuePattern → set_value.
    if is_pattern_available(element, uiautomation::types::UIProperty::IsValuePatternAvailable) {
        actions.push("set_value".into());
    }

    // Check TogglePattern → toggle.
    if is_pattern_available(element, uiautomation::types::UIProperty::IsTogglePatternAvailable) {
        actions.push("toggle".into());
    }

    // Check ExpandCollapsePattern → expand/collapse.
    if is_pattern_available(
        element,
        uiautomation::types::UIProperty::IsExpandCollapsePatternAvailable,
    ) {
        actions.push("expand".into());
        actions.push("collapse".into());
    }

    // Check SelectionItemPattern → select.
    if is_pattern_available(
        element,
        uiautomation::types::UIProperty::IsSelectionItemPatternAvailable,
    ) {
        actions.push("select".into());
    }

    // Check ScrollPattern → scroll.
    if is_pattern_available(element, uiautomation::types::UIProperty::IsScrollPatternAvailable) {
        actions.push("scroll".into());
    }

    // Focus is always available for focusable elements.
    if element.is_keyboard_focusable().unwrap_or(false) {
        actions.push("focus".into());
    }

    actions
}

fn is_pattern_available(element: &UIElement, property: uiautomation::types::UIProperty) -> bool {
    element
        .get_property_value(property)
        .map(|v| v.to_string() == "true" || v.to_string() == "1")
        .unwrap_or(false)
}
