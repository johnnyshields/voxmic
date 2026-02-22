# Voxctrl MSI Installer Plan

**Date**: 2026-02-22
**Branch**: feat-rust-crossplatform-tray-app/supervisor

## Summary

Add a WiX-based MSI installer for voxctrl on Windows. The installer:
- Installs `voxctrl.exe` + `config.json` to `%LOCALAPPDATA%\Voxctrl\`
- Per-user install (no UAC elevation required)
- Creates a `VoxctrlDictation` scheduled task for auto-start at logon
- Removes the scheduled task on uninstall
- Supports clean version upgrades via `<MajorUpgrade>`

## Files

| File | Action | Purpose |
|------|--------|---------|
| `voxctrl/wix/main.wxs` | Created | WiX XML manifest |
| `voxctrl/Cargo.toml` | Edited | Added `[package.metadata.wix]` section |

## Constraints

- `cargo-wix` requires WiX Toolset (candle.exe, light.exe) — Windows-only
- Cannot compile MSI from WSL2; manifest + build instructions provided
- User tests actual MSI build + install on Windows

## Build Instructions

```powershell
cargo install cargo-wix
cargo wix    # → target\wix\voxctrl-0.1.0-x86_64.msi
```

## Design Decisions

- **KeyPath**: Uses HKCU registry entries as KeyPath (WiX requirement for user-profile directories)
- **Scheduled task**: Uses `schtasks.exe` custom actions with `/RL LIMITED` (no elevation)
- **Source path**: Points to `target\x86_64-pc-windows-gnu\release\voxctrl.exe` (cross-compilation output)
- **config.json**: Sourced from project root alongside Cargo.toml
