//! Windows UI Automation accessibility provider.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, Result};
use uiautomation::types::TreeScope;
use uiautomation::UIAutomation;
use uiautomation::UIElement;

use voxctrl_cu::actions::{UiAction, UiActionResult};
use voxctrl_cu::tree::{UiNode, UiTree};
use voxctrl_cu::AccessibilityProvider;

use crate::actions::execute_action;
use crate::tree_walker::walk_element;

/// Maximum tree depth to walk (prevents runaway recursion on deep UIs).
const MAX_WALK_DEPTH: usize = 25;

/// Windows UI Automation provider.
///
/// Stores a map from element index → UIA element handle so that actions
/// can look up elements by the index the LLM references.
pub struct WindowsUiaProvider {
    element_map: Mutex<HashMap<usize, UIElement>>,
    creator_thread_id: std::thread::ThreadId,
}

impl WindowsUiaProvider {
    pub fn new() -> Result<Self> {
        let _uia = UIAutomation::new().context("Failed to initialize UI Automation")?;
        Ok(Self {
            element_map: Mutex::new(HashMap::new()),
            creator_thread_id: std::thread::current().id(),
        })
    }

    /// Assert that we're on the creator thread (COM apartment-threading requirement).
    fn assert_creator_thread(&self) {
        debug_assert_eq!(
            std::thread::current().id(),
            self.creator_thread_id,
            "WindowsUiaProvider must be used from the thread that created it \
             (COM apartment-threading). Current thread: {:?}, creator: {:?}",
            std::thread::current().id(),
            self.creator_thread_id,
        );
    }

    /// Replace the stored element map with a new one.
    fn store_element_map(&self, map: HashMap<usize, UIElement>) {
        let mut guard = self.element_map.lock().unwrap();
        *guard = map;
    }
}

impl AccessibilityProvider for WindowsUiaProvider {
    fn get_focused_tree(&self) -> Result<UiTree> {
        self.assert_creator_thread();
        let uia = UIAutomation::new().context("UI Automation init")?;
        let focused = uia.get_focused_element().context("get focused element")?;

        // Walk up to the top-level window.
        let walker = uia.get_control_view_walker().context("get tree walker")?;
        let mut current = focused.clone();
        loop {
            match walker.get_parent(&current) {
                Ok(parent) => {
                    // The desktop root has no name and HWND 0 — stop there.
                    let parent_rid = parent.get_runtime_id().unwrap_or_default();
                    let root_rid = uia.get_root_element().ok()
                        .and_then(|r| r.get_runtime_id().ok())
                        .unwrap_or_default();
                    if parent_rid == root_rid {
                        break;
                    }
                    current = parent;
                }
                Err(_) => break,
            }
        }

        let window = current;
        let window_title = window.get_name().unwrap_or_default();
        let pid = window.get_process_id().unwrap_or(0);
        let process_name = process_name_from_pid(pid);

        let mut index_counter = 0usize;
        let mut element_map = HashMap::new();
        let root = walk_element(
            &uia,
            &window,
            0,
            MAX_WALK_DEPTH,
            &mut index_counter,
            &mut element_map,
        );

        let element_count = index_counter;
        self.store_element_map(element_map);

        Ok(UiTree {
            root,
            window_title,
            process_name,
            element_count,
        })
    }

    fn get_tree_for_pid(&self, pid: u32) -> Result<UiTree> {
        self.assert_creator_thread();
        let uia = UIAutomation::new().context("UI Automation init")?;
        let root = uia.get_root_element().context("get desktop root")?;

        // Find windows belonging to this PID.
        let condition = uia
            .create_property_condition(
                uiautomation::types::UIProperty::ProcessId,
                uiautomation::variants::Variant::from(pid as i32),
                None,
            )
            .context("create PID condition")?;
        let windows = root
            .find_all(TreeScope::Children, &condition)
            .context("find windows by PID")?;

        let window = windows
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No window found for PID {pid}"))?;

        let window_title = window.get_name().unwrap_or_default();
        let process_name = process_name_from_pid(pid);

        let mut index_counter = 0usize;
        let mut element_map = HashMap::new();
        let root_node = walk_element(
            &uia,
            &window,
            0,
            MAX_WALK_DEPTH,
            &mut index_counter,
            &mut element_map,
        );

        let element_count = index_counter;
        self.store_element_map(element_map);

        Ok(UiTree {
            root: root_node,
            window_title,
            process_name,
            element_count,
        })
    }

    fn find_elements(&self, query: &str) -> Result<Vec<UiNode>> {
        self.assert_creator_thread();
        let uia = UIAutomation::new().context("UI Automation init")?;
        let root = uia.get_root_element().context("get desktop root")?;

        let condition = uia
            .create_property_condition(
                uiautomation::types::UIProperty::Name,
                uiautomation::variants::Variant::from(query),
                None,
            )
            .context("create name condition")?;

        let elements = root
            .find_all(TreeScope::Descendants, &condition)
            .unwrap_or_default();

        let mut results = Vec::new();
        let mut index_counter = 0usize;
        let mut element_map = HashMap::new();

        for elem in elements.iter().take(50) {
            let node = walk_element(&uia, elem, 0, 0, &mut index_counter, &mut element_map);
            results.push(node);
        }

        self.store_element_map(element_map);
        Ok(results)
    }

    fn perform_action(&self, action: &UiAction) -> Result<UiActionResult> {
        self.assert_creator_thread();
        // Take a snapshot of the element map to release the lock before
        // potentially-blocking COM calls.
        let snapshot = {
            let guard = self.element_map.lock().unwrap();
            guard.clone()
        };
        execute_action(action, &snapshot)
    }

    fn capture_screenshot(&self) -> Result<Option<Vec<u8>>> {
        crate::screenshot::capture_screen()
    }

    fn platform_name(&self) -> &str {
        "windows-uia"
    }
}

/// Best-effort process name lookup from PID.
fn process_name_from_pid(pid: u32) -> String {
    if pid == 0 {
        return String::new();
    }
    // /proc/{pid}/comm only exists on Linux/WSL — expected to fail on native Windows.
    match std::fs::read_to_string(format!("/proc/{pid}/comm")) {
        Ok(s) => s.trim().to_string(),
        Err(_) => {
            log::debug!("Could not read /proc/{pid}/comm (expected on Windows) — using pid:{pid}");
            format!("pid:{pid}")
        }
    }
}

// SAFETY: WindowsUiaProvider wraps COM UIAutomation objects which use apartment
// threading (STA). This is safe because:
// 1. All COM calls go through methods that debug_assert we're on the creator thread
// 2. The Mutex<HashMap<usize, UIElement>> serialises access to element handles
// 3. In the current architecture, the provider is created and used on a single thread
//    within the agent loop pipeline
//
// If this provider is ever used from multiple threads, the debug_assert will catch
// the violation in dev/test builds. For production multi-thread use, consider wrapping
// in a dedicated COM STA thread with message pumping.
unsafe impl Send for WindowsUiaProvider {}
unsafe impl Sync for WindowsUiaProvider {}
