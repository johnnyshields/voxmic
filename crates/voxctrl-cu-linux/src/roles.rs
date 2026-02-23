//! AT-SPI2 role mapping — converts AT-SPI2 roles to cross-platform `UiRole`.
//!
//! AT-SPI2 defines roles in the `atspi::Role` enum. This module maps them
//! to the unified `UiRole` enum used by the rest of the system.
//! Unmapped roles become `UiRole::Custom(name)` or `UiRole::Unknown`.
