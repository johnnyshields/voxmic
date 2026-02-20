# TODO

## Voice isolation (filter out system audio / YouTube)

Currently the mic captures all audio including bleed from speakers (YouTube, music, etc.).

Options to explore:
- **Noise gate on input device**: DJI Mic Mini supports hardware noise gate — check if it can be enabled via Windows audio properties or DJI app
- **Echo cancellation**: Windows audio stack has AEC (Acoustic Echo Cancellation) — can be enabled via `sounddevice` by selecting the `Communications` device variant which has AEC applied
- **Microphone-only VAD**: Compare energy on mic input vs. system loopback (Stereo Mix / WASAPI loopback capture) — only transcribe when mic energy significantly exceeds loopback energy
- **WebRTC VAD**: Replace energy-based VAD with `webrtcvad` (pip install webrtcvad) which is voice-specific and rejects non-speech sounds like music
- **Speaker diarization**: Use `pyannote.audio` to identify speaker segments and filter to a specific voice profile

## MSI installer

Package everything into a double-click `.msi` for easy distribution.

Approach:
- **WiX Toolset v4** — industry standard, free, generates MSI from XML
- Bundle contents:
  - Python embeddable runtime (no system Python required)
  - `transcribe.py`, `config.json`, `list-devices.py`, `monitor.bat`
  - Pre-downloaded faster-whisper `small` model weights
- Custom actions (PowerShell via WiX `Exec`):
  - Install `VoxtralMic` scheduled task on install
  - Remove task on uninstall
- Add to Programs and Features with uninstall support
- Optional: GUI config screen (device selection, model size) using WiX UI extension
