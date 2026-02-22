//! Global hotkey — configurable toggle with graceful fallback.

use std::sync::Arc;

use anyhow::{Context, Result};
use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager};

use crate::config::{Config, HotkeyConfig};
use crate::pipeline::Pipeline;
use crate::SharedState;

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
pub fn setup_hotkeys(cfg: &HotkeyConfig) -> Result<Option<(GlobalHotKeyManager, u32)>> {
    let hotkey = match parse_shortcut(&cfg.shortcut) {
        Ok(h) => h,
        Err(e) => {
            log::warn!(
                "Invalid hotkey {:?}: {e}. Continuing without hotkey — \
                 set a valid hotkey in Settings.",
                cfg.shortcut
            );
            return Ok(None);
        }
    };

    let manager = GlobalHotKeyManager::new().context("create hotkey manager")?;
    let id = hotkey.id();

    match manager.register(hotkey) {
        Ok(()) => {
            log::info!("Global hotkey registered: {}", cfg.shortcut);
            Ok(Some((manager, id)))
        }
        Err(e) => {
            log::warn!(
                "Failed to register hotkey {:?}: {e}. Continuing without hotkey — \
                 use the tray menu or change hotkey.shortcut in config.json.",
                cfg.shortcut
            );
            Ok(None)
        }
    }
}

/// Handle a hotkey event: toggle Idle → Recording → Transcribing.
pub fn handle_hotkey_event(
    event: &GlobalHotKeyEvent,
    hotkey_id: Option<u32>,
    state: &Arc<SharedState>,
    cfg: &Config,
    pipeline: Arc<Pipeline>,
) {
    if Some(event.id) != hotkey_id {
        return;
    }

    crate::recording::toggle_recording(state, cfg, pipeline);
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
}
