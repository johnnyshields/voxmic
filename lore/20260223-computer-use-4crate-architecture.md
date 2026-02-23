# Computer-Use: 4-Crate Accessibility Architecture

**Date**: 2026-02-23
**Branch**: refactor-split-build-targets/supervisor

## Summary

Implement real desktop automation via OS accessibility APIs (not screenshots) + Claude API agent loop.
Replaces the stub `computer_use.rs` in voxctrl-core with 4 dedicated crates.

## Crate Graph

```
voxctrl (binary) → voxctrl-cu → voxctrl-core
   ├── voxctrl-cu-windows (uiautomation 0.24)
   ├── voxctrl-cu-macos   (objc2-application-services 0.3) [stub]
   └── voxctrl-cu-linux   (atspi 0.29 + tokio bridge) [stub]
```

## Integration Pattern

Mirrors the existing `voxctrl-stt` factory pattern:
- `ActionFactory` type alias in voxctrl-core (like `SttFactory`)
- `create_action(cfg, extra_factory)` gains an optional factory param
- Platform crates export provider factories, binary chains them under feature flags

## Scope

- Fully implement: voxctrl-cu (types + agent loop) + voxctrl-cu-windows (UIA provider)
- Stub only: voxctrl-cu-macos, voxctrl-cu-linux

## Implementation Order

1. voxctrl-core changes (ActionFactory, config, pipeline, delete stub)
2. voxctrl-cu (types, agent loop, executor)
3. voxctrl-cu-windows (real UIA tree walking + actions)
4. voxctrl-cu-macos (scaffold/stub)
5. voxctrl-cu-linux (scaffold/stub)
6. voxctrl binary wiring (feature flags, factory chain)
