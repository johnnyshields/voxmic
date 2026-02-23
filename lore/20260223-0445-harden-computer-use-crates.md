# Harden: Computer-Use 4-Crate Architecture

**Date**: 2026-02-23
**Branch**: refactor-split-build-targets/supervisor
**Scope**: voxctrl-cu, voxctrl-cu-windows, related core changes

## Effort/Impact Assessment Table

| # | Opportunity | Effort | Impact | Action |
|---|-------------|--------|--------|--------|
| 1 | API key masked in Debug output | Quick | High | Auto-fix |
| 2 | `/proc` path on Windows — always fails silently | Quick | Low | Auto-fix |
| 3 | LLM input bounds validation (element IDs, wait ms, scroll amount) | Easy | High | Auto-fix |
| 4 | Improve error messages from LLM tool call parsing | Easy | Medium | Auto-fix |
| 5 | Add boundary/edge-case tests for tree pruning + tool parsing | Easy | Medium | Auto-fix |
| 6 | Tool definitions ↔ UiAction enum sync test | Easy | Medium | Auto-fix |
| 7 | Scroll `amount` parameter silently ignored | Moderate | Medium | Ask first |
| 8 | Unsafe Send/Sync for WindowsUiaProvider — justify or fix | Moderate | High | Ask first |
| 9 | Mutex held across I/O in perform_action | Moderate | Medium | Ask first |

## Opportunity Details

### #1 — API key masked in Debug output
- **What**: Add `ApiKey` newtype with `Debug` impl that masks the value, preventing accidental log exposure
- **Where**: `crates/voxctrl-cu/src/agent.rs` — `AgentConfig.api_key` field
- **Why**: Security — API key could be logged via `{:?}` formatting

### #2 — `/proc` path on Windows always fails silently
- **What**: `process_name_from_pid()` reads `/proc/{pid}/comm` which doesn't exist on Windows. Always returns `"pid:N"` with no log. Replace with a Windows API call or at minimum log a warning.
- **Where**: `crates/voxctrl-cu-windows/src/provider.rs:183-191`
- **Why**: Diagnostics — misleading process_name in tree context

### #3 — LLM input bounds validation
- **What**: Add reasonable caps: element_id < 100,000, scroll amount < 100, wait ms < 60,000. Reject out-of-range values from LLM tool calls.
- **Where**: `crates/voxctrl-cu/src/agent.rs` — `parse_tool_call()` function
- **Why**: Prevents denial-of-service from malformed LLM output (e.g. `wait(u64::MAX)`)

### #4 — Improve error messages from LLM tool call parsing
- **What**: Include field name and reason in parse errors instead of generic "Invalid tool call". The LLM gets better feedback to self-correct.
- **Where**: `crates/voxctrl-cu/src/agent.rs` — `parse_tool_call()` error paths

### #5 — Add boundary/edge-case tests
- **What**: Tests for: overflow element_id, excessive wait ms, all-invisible tree, mixed visible/invisible siblings, depth=0 pruning, missing required fields in tool calls
- **Where**: `crates/voxctrl-cu/src/agent.rs` tests, `crates/voxctrl-cu/src/tree.rs` tests

### #6 — Tool definitions ↔ UiAction enum sync test
- **What**: Add a test that parses the TOOLS_JSON, extracts tool names, and asserts they match the known action types. Catches drift if someone adds a UiAction variant without updating the prompt.
- **Where**: `crates/voxctrl-cu/src/prompt.rs` — new test

### #7 — Scroll `amount` parameter silently ignored
- **What**: The `amount` field in `UiAction::Scroll` is accepted from the LLM but the Windows provider ignores it (uses `LargeIncrement`/`LargeDecrement` only). Either loop scroll N times or remove amount from the API.
- **Where**: `crates/voxctrl-cu-windows/src/actions.rs:96-138`, `crates/voxctrl-cu/src/actions.rs`, `crates/voxctrl-cu/src/prompt.rs`
- **Trade-offs**: Looping is more correct but may feel janky. Removing the param simplifies but reduces LLM control.

### #8 — Unsafe Send/Sync for WindowsUiaProvider
- **What**: The `unsafe impl Send/Sync` is hand-waved. COM objects are apartment-threaded. Options: (a) document that the provider is always used on a single thread via the pipeline (correct today), (b) add runtime thread-ID check, (c) wrap in a thread-local.
- **Where**: `crates/voxctrl-cu-windows/src/provider.rs:195-196`
- **Trade-offs**: Adding thread-ID check is safest but adds runtime overhead. Documenting is zero-cost but relies on caller discipline.

### #9 — Mutex held across I/O in perform_action
- **What**: `perform_action()` holds `element_map` lock during the entire UIA action (which may block on COM calls). This blocks `get_focused_tree()` from a different thread.
- **Where**: `crates/voxctrl-cu-windows/src/provider.rs:168-171`
- **Trade-offs**: Releasing the lock early requires cloning UIElement handles, which may have COM lifetime implications. In practice, the current pipeline is single-threaded for actions, so this may be a non-issue today.

## Execution Protocol
**DO NOT implement any changes without user approval.**
For EACH opportunity, use `AskUserQuestion`.
Options: "Implement" / "Skip (add to TODO.md)" / "Do not implement"
Ask ALL questions before beginning any implementation work.
(do NOT do alternating ask then implement, ask then implement, etc.)
Quick items may be batched into one multi-select AskUserQuestion.
After all items resolved, run the project's test suite.
