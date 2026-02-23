//! AT-SPI2 tree walker — will use the `atspi` crate to traverse the accessibility tree.
//!
//! The atspi crate provides async access to the AT-SPI2 D-Bus interface.
//! This module will walk the tree starting from a given accessible object
//! and convert it into the cross-platform `UiTree`/`UiNode` representation.
