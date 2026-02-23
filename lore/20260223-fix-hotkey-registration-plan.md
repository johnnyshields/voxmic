# Fix Hotkey Registration (2026-02-23)

## Problem

The Settings Test tab fails with `"HotKey already registerd"` when the main app is running.
The `global_hotkey` crate uses Win32 `RegisterHotKey` which is system-wide — only one process
can hold a given hotkey combo. The Settings subprocess (launched via `--settings`) conflicts
with the main app's registration.

## Root Cause

- `global_hotkey` v0.6.4 Windows impl calls `RegisterHotKey(hwnd, id, mods, vk)` per hotkey
- When OS returns `ERROR_HOTKEY_ALREADY_REGISTERED`, crate maps it to `Error::AlreadyRegistered`
- The Settings subprocess creates its own `GlobalHotKeyManager` (separate HWND, separate process)
- OS rejects the duplicate registration since the main app already holds it

## Fix

Main app unregisters its global hotkeys before spawning the Settings subprocess.
It monitors the child process handle via `try_wait()` in `about_to_wait` and
re-registers hotkeys when Settings exits.

### Key changes
- `hotkey.rs`: Store `HotKey` objects in `HotkeyIds` for unregister/re-register
- `main.rs`: Add `settings_child: Option<Child>` to `App`, lifecycle management
- `model_table.rs`: Improve `AlreadyRegistered` error message as fallback

### Files
- `crates/voxctrl/src/hotkey.rs`
- `crates/voxctrl/src/main.rs`
- `crates/voxctrl/src/ui/model_table.rs`
