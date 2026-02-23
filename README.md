# voxctrl

Multi-platform service for real-time mic transcription and computer use.

## What it does

- Captures audio from your DJI mic (or any input device)
- Detects speech via VAD model or algorithm (RMS, Silero)
- Transcribes each utterance locally using pluggable STT (Whisper, Voxtral)
- Generates use computer actions via an LLM
- Appends timestamped lines to a live transcription log

## Requirements

- Windows 10/11
- Python 3.9+
- Administrator PowerShell (for setup only)

## Install

```powershell
# Run as Administrator
.\setup.ps1
```

Setup will:
1. Download llama.cpp to `C:\workspace\voxtral\llama-win\`
2. Install Python packages: `sounddevice numpy scipy requests faster-whisper`
3. Deploy scripts to `C:\workspace\voxtral\`
4. Install two Windows Scheduled Tasks:
   - `VoxtralLLM` — runs at boot as SYSTEM (optional, for llama-server)
   - `VoxtralMic` — runs at logon as your user (audio capture + transcription)

## Usage

```powershell
# Find your mic device index
python C:\workspace\voxtral\list-devices.py

# Start transcription (or just log out/back in)
Start-ScheduledTask -TaskName VoxtralMic

# Watch live output
C:\workspace\voxtral\monitor.bat
```

## Config

Edit `C:\workspace\voxtral\config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `device_pattern` | `"DJI"` | Substring match against mic name |
| `silence_threshold` | `0.015` | RMS energy cutoff — raise if noisy room |
| `silence_duration` | `1.5` | Seconds of silence before flushing utterance |
| `whisper_model` | `"small"` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `whisper_language` | `null` | `null` = auto-detect, `"en"` = English only |

## Manage

```powershell
Start-ScheduledTask / Stop-ScheduledTask -TaskName VoxtralMic
Get-ScheduledTask VoxtralMic | Select TaskName, State
```

Logs: `C:\workspace\voxtral\logs\`
Output: `C:\workspace\voxtral\output\transcription.log`

