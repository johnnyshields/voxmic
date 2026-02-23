//! Global hotkey — configurable toggle with graceful fallback.

use std::sync::Arc;

use anyhow::{Context, Result};
use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

use voxctrl_core::config::{Config, HotkeyConfig};
use voxctrl_core::pipeline::SharedPipeline;
use voxctrl_core::SharedState;

/// Registered global hotkeys (HotKey is Copy; derive IDs via `.id()`).
pub struct HotkeyIds {
    pub dictation: Option<HotKey>,
    pub computer_use: Option<HotKey>,
}

impl HotkeyIds {
    pub fn none() -> Self {
        Self { dictation: None, computer_use: None }
    }
}

/// Parse a shortcut string like "Ctrl+Super+Space" into a `HotKey`.
pub fn parse_shortcut(s: &str) -> Result<HotKey> {
    let mut modifiers = Modifiers::empty();
    let mut key_code: Option<Code> = None;

    for token in s.split('+') {
        let token = token.trim();
        match token.to_lowercase().as_str() {
            "ctrl" | "control" => modifiers |= Modifiers::CONTROL,
            "alt" => modifiers |= Modifiers::ALT,
            "shift" => modifiers |= Modifiers::SHIFT,
            "super" | "win" | "meta" | "cmd" => modifiers |= Modifiers::SUPER,
            _ => {
                if key_code.is_some() {
                    anyhow::bail!("multiple key codes in shortcut: {s:?}");
                }
                key_code = Some(parse_key_code(token)?);
            }
        }
    }

    let code = key_code.context(format!("no key code found in shortcut: {s:?}"))?;
    let mods = if modifiers.is_empty() { None } else { Some(modifiers) };
    Ok(HotKey::new(mods, code))
}

/// Map a key name to a `Code` variant.
fn parse_key_code(token: &str) -> Result<Code> {
    // Single letter A-Z
    if token.len() == 1 {
        let ch = token.chars().next().unwrap();
        if ch.is_ascii_alphabetic() {
            let code = match ch.to_ascii_uppercase() {
                'A' => Code::KeyA,
                'B' => Code::KeyB,
                'C' => Code::KeyC,
                'D' => Code::KeyD,
                'E' => Code::KeyE,
                'F' => Code::KeyF,
                'G' => Code::KeyG,
                'H' => Code::KeyH,
                'I' => Code::KeyI,
                'J' => Code::KeyJ,
                'K' => Code::KeyK,
                'L' => Code::KeyL,
                'M' => Code::KeyM,
                'N' => Code::KeyN,
                'O' => Code::KeyO,
                'P' => Code::KeyP,
                'Q' => Code::KeyQ,
                'R' => Code::KeyR,
                'S' => Code::KeyS,
                'T' => Code::KeyT,
                'U' => Code::KeyU,
                'V' => Code::KeyV,
                'W' => Code::KeyW,
                'X' => Code::KeyX,
                'Y' => Code::KeyY,
                'Z' => Code::KeyZ,
                _ => unreachable!(),
            };
            return Ok(code);
        }
        // Single digit 0-9
        if ch.is_ascii_digit() {
            let code = match ch {
                '0' => Code::Digit0,
                '1' => Code::Digit1,
                '2' => Code::Digit2,
                '3' => Code::Digit3,
                '4' => Code::Digit4,
                '5' => Code::Digit5,
                '6' => Code::Digit6,
                '7' => Code::Digit7,
                '8' => Code::Digit8,
                '9' => Code::Digit9,
                _ => unreachable!(),
            };
            return Ok(code);
        }
    }

    // Named keys (case-insensitive)
    match token.to_lowercase().as_str() {
        "space" => Ok(Code::Space),
        "enter" | "return" => Ok(Code::Enter),
        "tab" => Ok(Code::Tab),
        "escape" | "esc" => Ok(Code::Escape),
        "backspace" => Ok(Code::Backspace),
        "delete" | "del" => Ok(Code::Delete),
        "insert" | "ins" => Ok(Code::Insert),
        "home" => Ok(Code::Home),
        "end" => Ok(Code::End),
        "pageup" => Ok(Code::PageUp),
        "pagedown" => Ok(Code::PageDown),
        "up" => Ok(Code::ArrowUp),
        "down" => Ok(Code::ArrowDown),
        "left" => Ok(Code::ArrowLeft),
        "right" => Ok(Code::ArrowRight),
        "f1" => Ok(Code::F1),
        "f2" => Ok(Code::F2),
        "f3" => Ok(Code::F3),
        "f4" => Ok(Code::F4),
        "f5" => Ok(Code::F5),
        "f6" => Ok(Code::F6),
        "f7" => Ok(Code::F7),
        "f8" => Ok(Code::F8),
        "f9" => Ok(Code::F9),
        "f10" => Ok(Code::F10),
        "f11" => Ok(Code::F11),
        "f12" => Ok(Code::F12),
        _ => anyhow::bail!("unknown key: {token:?}"),
    }
}

/// Register the configured hotkey, returning `None` if registration fails.
pub fn setup_hotkeys(cfg: &HotkeyConfig) -> Result<Option<(GlobalHotKeyManager, HotkeyIds)>> {
    let hotkey = match parse_shortcut(&cfg.dict_shortcut) {
        Ok(h) => h,
        Err(e) => {
            log::warn!(
                "Invalid hotkey {:?}: {e}. Continuing without hotkey — \
                 set a valid hotkey in Settings.",
                cfg.dict_shortcut
            );
            return Ok(None);
        }
    };

    let manager = GlobalHotKeyManager::new().context("create hotkey manager")?;

    match manager.register(hotkey) {
        Ok(()) => {
            log::info!("Global dictation hotkey registered: {}", cfg.dict_shortcut);
        }
        Err(e) => {
            log::warn!(
                "Failed to register hotkey {:?}: {e}. Continuing without hotkey — \
                 use the tray menu or change hotkey.dict_shortcut in config.json.",
                cfg.dict_shortcut
            );
            return Ok(None);
        }
    }

    // Register computer-use hotkey if configured
    let cu_hotkey = if let Some(ref cu_shortcut) = cfg.cu_shortcut {
        match parse_shortcut(cu_shortcut) {
            Ok(cu_hk) => match manager.register(cu_hk) {
                Ok(()) => {
                    log::info!("Global computer-use hotkey registered: {}", cu_shortcut);
                    Some(cu_hk)
                }
                Err(e) => {
                    log::warn!(
                        "Failed to register CU hotkey {:?}: {e}. CU hotkey disabled.",
                        cu_shortcut
                    );
                    None
                }
            },
            Err(e) => {
                log::warn!("Invalid CU hotkey {:?}: {e}. CU hotkey disabled.", cu_shortcut);
                None
            }
        }
    } else {
        None
    };

    Ok(Some((manager, HotkeyIds {
        dictation: Some(hotkey),
        computer_use: cu_hotkey,
    })))
}

/// Unregister all active hotkeys (e.g. before opening Settings subprocess).
pub fn unregister_hotkeys(manager: &GlobalHotKeyManager, ids: &HotkeyIds) {
    let hotkeys: Vec<HotKey> = [ids.dictation, ids.computer_use].into_iter().flatten().collect();
    if hotkeys.is_empty() { return; }
    if let Err(e) = manager.unregister_all(&hotkeys) {
        log::warn!("Failed to unregister hotkeys: {e}");
    } else {
        log::info!("Hotkeys unregistered ({} total)", hotkeys.len());
    }
}

/// Handle a hotkey event: toggle Idle → Recording → Transcribing.
pub fn handle_hotkey_event(
    event: &GlobalHotKeyEvent,
    ids: &HotkeyIds,
    state: &Arc<SharedState>,
    cfg: &Config,
    pipeline: &Arc<SharedPipeline>,
) {
    if event.state != HotKeyState::Pressed {
        return;
    }
    if ids.dictation.map(|hk| hk.id()) == Some(event.id) {
        voxctrl_core::recording::toggle_recording(state, cfg, pipeline);
    } else if ids.computer_use.map(|hk| hk.id()) == Some(event.id) {
        log::info!("Computer-use hotkey pressed");
        // TODO: Route to CU pipeline when connected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ctrl_super_space() {
        let hk = parse_shortcut("Ctrl+Super+Space").unwrap();
        let expected = HotKey::new(Some(Modifiers::CONTROL | Modifiers::SUPER), Code::Space);
        assert_eq!(hk.id(), expected.id());
    }

    #[test]
    fn parse_ctrl_alt_v() {
        let hk = parse_shortcut("Ctrl+Alt+V").unwrap();
        let expected = HotKey::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::KeyV);
        assert_eq!(hk.id(), expected.id());
    }

    #[test]
    fn parse_case_insensitive() {
        let hk = parse_shortcut("ctrl+shift+F1").unwrap();
        let expected = HotKey::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::F1);
        assert_eq!(hk.id(), expected.id());
    }

    #[test]
    fn parse_single_key() {
        let hk = parse_shortcut("F12").unwrap();
        let expected = HotKey::new(None, Code::F12);
        assert_eq!(hk.id(), expected.id());
    }

    #[test]
    fn parse_no_key_code_errors() {
        assert!(parse_shortcut("Ctrl+Alt").is_err());
    }

    #[test]
    fn parse_unknown_key_errors() {
        assert!(parse_shortcut("Ctrl+Banana").is_err());
    }

    #[test]
    fn parse_duplicate_key_errors() {
        assert!(parse_shortcut("Ctrl+A+B").is_err());
    }

    // ── handle_hotkey_event tests ────────────────────────────────────

    use voxctrl_core::{AppStatus, SharedState};

    fn make_test_ids() -> (HotkeyIds, u32) {
        let hk = HotKey::new(Some(Modifiers::CONTROL), Code::Space);
        let id = hk.id();
        let ids = HotkeyIds { dictation: Some(hk), computer_use: None };
        (ids, id)
    }

    fn make_test_pipeline() -> Arc<SharedPipeline> {
        use voxctrl_core::action::ActionExecutor;
        use voxctrl_core::pipeline::Pipeline;
        use voxctrl_core::router::{Intent, IntentRouter};
        use voxctrl_core::stt::Transcriber;

        struct Noop;
        impl Transcriber for Noop {
            fn transcribe(&self, _: &std::path::Path) -> anyhow::Result<String> { Ok(String::new()) }
            fn transcribe_pcm(&self, _: &[f32], _: u32) -> anyhow::Result<String> { Ok(String::new()) }
            fn name(&self) -> &str { "noop" }
            fn is_available(&self) -> bool { true }
        }
        impl IntentRouter for Noop {
            fn route(&self, t: &str) -> anyhow::Result<Intent> { Ok(Intent::Dictate(t.into())) }
            fn name(&self) -> &str { "noop" }
        }
        impl ActionExecutor for Noop {
            fn execute(&self, _: &Intent) -> anyhow::Result<()> { Ok(()) }
            fn name(&self) -> &str { "noop" }
        }

        Arc::new(SharedPipeline::new(Pipeline {
            stt: Box::new(Noop),
            router: Box::new(Noop),
            action: Box::new(Noop),
        }))
    }

    #[test]
    fn pressed_event_triggers_toggle() {
        let (ids, id) = make_test_ids();
        let state = Arc::new(SharedState::new());
        let cfg = voxctrl_core::config::Config::default();
        let pipeline = make_test_pipeline();

        let event = GlobalHotKeyEvent { id, state: HotKeyState::Pressed };
        handle_hotkey_event(&event, &ids, &state, &cfg, &pipeline);

        assert_eq!(*state.status.lock().unwrap(), AppStatus::Recording);
    }

    #[test]
    fn released_event_is_ignored() {
        let (ids, id) = make_test_ids();
        let state = Arc::new(SharedState::new());
        let cfg = voxctrl_core::config::Config::default();
        let pipeline = make_test_pipeline();

        let event = GlobalHotKeyEvent { id, state: HotKeyState::Released };
        handle_hotkey_event(&event, &ids, &state, &cfg, &pipeline);

        assert_eq!(*state.status.lock().unwrap(), AppStatus::Idle);
    }

    #[test]
    fn unrelated_hotkey_id_is_ignored() {
        let (ids, _) = make_test_ids();
        let state = Arc::new(SharedState::new());
        let cfg = voxctrl_core::config::Config::default();
        let pipeline = make_test_pipeline();

        let unrelated = GlobalHotKeyEvent { id: 99999, state: HotKeyState::Pressed };
        handle_hotkey_event(&unrelated, &ids, &state, &cfg, &pipeline);

        assert_eq!(*state.status.lock().unwrap(), AppStatus::Idle);
    }
}
