//! AT-SPI2 action execution — will use the `atspi` crate to perform UI actions.
//!
//! Maps `UiAction` variants to AT-SPI2 action interface calls:
//! - Click → DoAction("click") or "press"
//! - SetValue → Value interface / EditableText interface
//! - SendKeys → DeviceEventController / simulated key events
//! - Focus, Select, etc. → corresponding AT-SPI2 actions
