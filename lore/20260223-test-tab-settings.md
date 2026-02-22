# Test Tab in Settings Window + Mic Input Selector

**Date**: 2026-02-23

## Goal
Add a "Test" tab (between Settings and Models) to the Settings window with 5 test
sections, plus a mic input device selector in the Settings tab.

## Architecture

The Settings window runs as a **subprocess** (`--settings`) with its own eframe
event loop. It does NOT share state with the main voxctrl process. This means:
- Audio capture must be set up independently in the settings process
- Pipeline components (STT, VAD) must be instantiated locally
- No access to the main app's `SharedState` or `Pipeline`

### Key insight: Test tab needs its own audio pipeline
The test tab creates its own cpal stream, VAD instance, and STT transcriber
on demand. These are created when tests start and cleaned up when tests finish.

## Design

### Settings tab addition: Mic Input selector
- Enumerate `cpal::default_host().input_devices()` at startup
- Store as `Vec<String>` of device names
- Combo box showing all available devices, selected value = `audio.device_pattern`
- Save writes to `config.json` `audio.device_pattern`

### Test tab state (`TestState`)
```
struct TestState {
    // Mic test
    mic_active: bool,
    mic_level: f32,           // 0.0-1.0 RMS from last chunk
    mic_stream: Option<cpal::Stream>,  // kept alive while testing
    mic_level_rx: Option<Receiver<f32>>,

    // Hotkey test
    hotkey_bypass: bool,
    hotkey_detected: bool,

    // VAD test
    vad_bypass: bool,         // bypass = always-on
    vad_active: bool,         // is speech detected
    vad_instance: Option<Box<dyn VoiceDetector>>,

    // STT test
    stt_status: String,       // "Ready" / "Recording..." / "Transcribing..." / result

    // Final output
    final_status: String,

    // Shared recording buffer for tests
    test_chunks: Arc<Mutex<Vec<f32>>>,
    test_recording: bool,
}
```

### Test flow
1. **Mic Test**: Start → opens cpal stream, sends RMS levels via channel → volume bar
2. **Hotkey Test**: bypass checkbox OR "Press now..." waits for keypress detection
3. **VAD Test**: bypass checkbox (always-on) OR shows live speech/silence from VAD
4. **STT Test**: Record → stop → write WAV → run STT → show text
5. **Final Output**: Combines all: hotkey → record with VAD gating → STT → show result

### Implementation approach
Since cpal::Stream is !Send, the audio stream must live on the main thread.
Use channels to communicate audio levels and chunks between the stream callback
and the UI. The test tab will:
- Use `std::sync::mpsc` for mic level readings
- Use `Arc<Mutex<Vec<f32>>>` for recording chunks (same pattern as main app)

## Files to modify
- `src/ui/model_table.rs` — add Test tab, TestState, mic selector
- `src/audio.rs` — add `list_input_devices()` and `start_test_capture()` helpers
- `Cargo.toml` — no new deps needed (cpal, egui already available)
