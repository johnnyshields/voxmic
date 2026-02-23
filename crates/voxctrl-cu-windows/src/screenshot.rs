//! Screen capture using Windows GDI.
//!
//! Stub implementation — returns `None` for now. A full implementation would
//! use `GetDC`/`BitBlt`/`CreateDIBSection` to capture the screen and encode
//! to PNG.

use anyhow::Result;

/// Capture a screenshot of the primary monitor.
///
/// Returns `Ok(None)` — screenshot capture is not yet implemented.
/// A future version will use the Windows GDI or DXGI APIs to capture
/// the screen and return PNG bytes.
pub fn capture_screen() -> Result<Option<Vec<u8>>> {
    log::debug!("Screenshot capture not yet implemented on Windows");
    Ok(None)
}
