//! Screenshot capture for Linux — X11/Wayland screenshot support.
//!
//! Strategies to investigate:
//! - X11: use `xcb` or `x11rb` to capture via XGetImage / SHM extension
//! - Wayland: use the `wlr-screencopy` protocol or `xdg-desktop-portal` D-Bus API
//! - Fallback: shell out to `grim` (Wayland) or `import`/`xwd` (X11)
//!
//! Returns PNG bytes for inclusion in LLM context.
