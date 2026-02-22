# NSIS Windows Installer for voxctrl

**Date**: 2026-02-22
**Branch**: feat-test-build-windows/supervisor

## Summary

Replaced the WiX MSI approach with an NSIS installer that can be cross-compiled from
Linux using `makensis`. This avoids WiX licensing concerns and runs natively on Ubuntu.

## Design Decisions

- **Per-user install** to `$LOCALAPPDATA\Voxctrl\` — no admin/UAC required
- **HKCU Run key** for auto-start instead of scheduled task (simpler, same effect)
- **No config.json bundled** — app creates defaults when no config found
- **No .ico file** — app generates tray icons programmatically; installer uses NSIS default
- **LZMA compression** for smaller installer size

## Files

- `voxctrl/installer.nsi` — new NSIS installer script
- `Dockerfile` — added `nsis` package and installer build step

## Verification

```bash
makensis voxctrl/installer.nsi    # produces voxctrl/voxctrl-0.2.0-setup.exe
file voxctrl/voxctrl-0.2.0-setup.exe  # PE32 executable
```
