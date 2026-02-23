//! Execute UiAction on UIA elements.

use std::collections::HashMap;

use anyhow::{Context, Result};
use uiautomation::UIElement;
use uiautomation::patterns::{
    UIExpandCollapsePattern, UIInvokePattern, UIScrollPattern,
    UISelectionItemPattern, UITogglePattern, UIValuePattern,
};

use voxctrl_cu::actions::{ScrollDirection, UiAction, UiActionResult};

/// Execute a `UiAction` using the stored element map.
pub fn execute_action(
    action: &UiAction,
    element_map: &HashMap<usize, UIElement>,
) -> Result<UiActionResult> {
    match action {
        UiAction::Click { element_id } => {
            let elem = lookup(element_map, element_id.index)?;
            // Try InvokePattern first, fall back to click().
            if let Ok(invoke) = elem.get_pattern::<UIInvokePattern>() {
                invoke.invoke().context("InvokePattern::invoke()")?;
            } else {
                elem.click().context("UIElement::click()")?;
            }
            Ok(UiActionResult::ok(format!("Clicked element #{}", element_id.index)))
        }

        UiAction::SetValue { element_id, value } => {
            let elem = lookup(element_map, element_id.index)?;
            let pattern = elem
                .get_pattern::<UIValuePattern>()
                .context("element does not support ValuePattern")?;
            pattern
                .set_value(value)
                .context("ValuePattern::set_value()")?;
            Ok(UiActionResult::ok(format!(
                "Set value on #{} to {:?}",
                element_id.index, value
            )))
        }

        UiAction::Toggle { element_id } => {
            let elem = lookup(element_map, element_id.index)?;
            let pattern = elem
                .get_pattern::<UITogglePattern>()
                .context("element does not support TogglePattern")?;
            pattern.toggle().context("TogglePattern::toggle()")?;
            Ok(UiActionResult::ok(format!("Toggled element #{}", element_id.index)))
        }

        UiAction::Focus { element_id } => {
            let elem = lookup(element_map, element_id.index)?;
            elem.set_focus().context("UIElement::set_focus()")?;
            Ok(UiActionResult::ok(format!("Focused element #{}", element_id.index)))
        }

        UiAction::Expand { element_id } => {
            let elem = lookup(element_map, element_id.index)?;
            let pattern = elem
                .get_pattern::<UIExpandCollapsePattern>()
                .context("element does not support ExpandCollapsePattern")?;
            pattern.expand().context("ExpandCollapsePattern::expand()")?;
            Ok(UiActionResult::ok(format!("Expanded element #{}", element_id.index)))
        }

        UiAction::Collapse { element_id } => {
            let elem = lookup(element_map, element_id.index)?;
            let pattern = elem
                .get_pattern::<UIExpandCollapsePattern>()
                .context("element does not support ExpandCollapsePattern")?;
            pattern
                .collapse()
                .context("ExpandCollapsePattern::collapse()")?;
            Ok(UiActionResult::ok(format!(
                "Collapsed element #{}",
                element_id.index
            )))
        }

        UiAction::Select { element_id } => {
            let elem = lookup(element_map, element_id.index)?;
            let pattern = elem
                .get_pattern::<UISelectionItemPattern>()
                .context("element does not support SelectionItemPattern")?;
            pattern.select().context("SelectionItemPattern::select()")?;
            Ok(UiActionResult::ok(format!("Selected element #{}", element_id.index)))
        }

        UiAction::SendKeys { keys } => {
            // Focus the active element and simulate keyboard input.
            // The `uiautomation` crate doesn't directly support SendKeys;
            // we use the Windows SendInput API via uiautomation utilities.
            uiautomation::inputs::Keyboard::new().send_keys(keys).context("send_keys")?;
            Ok(UiActionResult::ok(format!("Sent keys: {:?}", keys)))
        }

        UiAction::Scroll {
            element_id,
            direction,
            amount,
        } => {
            let elem = lookup(element_map, element_id.index)?;
            let pattern = elem
                .get_pattern::<UIScrollPattern>()
                .context("element does not support ScrollPattern")?;

            for i in 0..*amount {
                match direction {
                    ScrollDirection::Up => {
                        pattern.scroll(
                            uiautomation::types::ScrollAmount::NoAmount,
                            uiautomation::types::ScrollAmount::LargeDecrement,
                        ).with_context(|| format!("scroll up (step {}/{})", i + 1, amount))?;
                    }
                    ScrollDirection::Down => {
                        pattern.scroll(
                            uiautomation::types::ScrollAmount::NoAmount,
                            uiautomation::types::ScrollAmount::LargeIncrement,
                        ).with_context(|| format!("scroll down (step {}/{})", i + 1, amount))?;
                    }
                    ScrollDirection::Left => {
                        pattern.scroll(
                            uiautomation::types::ScrollAmount::LargeDecrement,
                            uiautomation::types::ScrollAmount::NoAmount,
                        ).with_context(|| format!("scroll left (step {}/{})", i + 1, amount))?;
                    }
                    ScrollDirection::Right => {
                        pattern.scroll(
                            uiautomation::types::ScrollAmount::LargeIncrement,
                            uiautomation::types::ScrollAmount::NoAmount,
                        ).with_context(|| format!("scroll right (step {}/{})", i + 1, amount))?;
                    }
                }
            }
            Ok(UiActionResult::ok(format!(
                "Scrolled #{} {:?} x{}",
                element_id.index, direction, amount
            )))
        }

        UiAction::Wait { ms } => {
            std::thread::sleep(std::time::Duration::from_millis(*ms));
            Ok(UiActionResult::ok(format!("Waited {ms}ms")))
        }
    }
}

/// Look up a UIA element by its index in the element map.
fn lookup(map: &HashMap<usize, UIElement>, index: usize) -> Result<&UIElement> {
    map.get(&index).ok_or_else(|| {
        anyhow::anyhow!(
            "Element #{index} not found — tree may have changed. \
             Try get_focused_tree() first to refresh the element map."
        )
    })
}
