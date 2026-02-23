//! Embedded tokio runtime for async→sync bridge.
//!
//! The `atspi` crate is fully async (requires tokio). However, the
//! `AccessibilityProvider` trait is synchronous. This module provides
//! an embedded single-threaded tokio runtime that the provider uses
//! to block on async AT-SPI2 calls.
//!
//! Pattern:
//! ```ignore
//! use tokio::runtime::Runtime;
//!
//! let rt = Runtime::new()?; // or Builder::new_current_thread()
//! let result = rt.block_on(async { atspi_call().await })?;
//! ```
//!
//! The runtime is created once in `LinuxAtspiProvider::new()` and stored
//! as a field, reused across all method calls.
