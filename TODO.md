# TODO

## Voice isolation (filter out system audio / YouTube)

Currently the mic captures all audio including bleed from speakers (YouTube, music, etc.).

Options to explore:
- **Noise gate on input device**: DJI Mic Mini supports hardware noise gate — check if it can be enabled via Windows audio properties or DJI app
- **Echo cancellation**: Windows audio stack has AEC (Acoustic Echo Cancellation) — can be enabled via `sounddevice` by selecting the `Communications` device variant which has AEC applied
- **Microphone-only VAD**: Compare energy on mic input vs. system loopback (Stereo Mix / WASAPI loopback capture) — only transcribe when mic energy significantly exceeds loopback energy
- **WebRTC VAD**: Replace energy-based VAD with `webrtcvad` (pip install webrtcvad) which is voice-specific and rejects non-speech sounds like music
- **Speaker diarization**: Use `pyannote.audio` to identify speaker segments and filter to a specific voice profile

## Windows installer

NSIS installer (`voxctrl/installer.nsi`) — cross-compiled from Linux via `makensis`.
Installs per-user to `%LOCALAPPDATA%\Voxctrl\`, auto-starts via HKCU Run key.
