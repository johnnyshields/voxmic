# Fix STT Test Connection — Plan

**Date:** 2026-02-23
**Branch:** fix-stt-test-connection/supervisor

## Problem

The Settings window (`--settings`) runs as a separate subprocess without access to the main tray process's Pipeline. When a user wants to test STT, the `VoxtralHttpTranscriber` tries to connect to `http://127.0.0.1:5200` (an external llama-server) which may not be running, resulting in "Connection refused" errors.

## Solution: Embedded STT Proxy Server + Test Tab

### Architecture

```
Main Tray Process                        Settings Subprocess
+---------------------------+            +---------------------------+
| Pipeline (Arc<Pipeline>)  |            | Test Tab                  |
|   .stt.transcribe(wav)    |            |   Record button           |
|                           |            |   cpal audio capture      |
| SttServer (background)    |<-- TCP --> |   SttClient               |
|   127.0.0.1:5201          |            |     connect, send WAV     |
|   reads WAV, transcribes  |            |     receive text          |
+---------------------------+            +---------------------------+
```

### Wire Protocol (length-prefixed TCP, one request per connection)

- **Request:** `[4 bytes: WAV len u32 BE] [WAV bytes]`
- **Response:** `[1 byte: status 0=ok 1=error] [4 bytes: text len u32 BE] [UTF-8 text]`

### Files Changed

1. `src/config.rs` — Add `stt_server_port: u16` (default 5201) to `SttConfig`
2. `src/stt_server.rs` — NEW: TCP server using `std::net::TcpListener`, one thread per connection
3. `src/stt_client.rs` — NEW: TCP client using `std::net::TcpStream`
4. `src/main.rs` — Wire up new modules, start server before event loop
5. `src/ui/model_table.rs` — Add Test tab with Record/Stop/Transcribe buttons, audio capture via cpal, async transcription via mpsc channel

### Key Decisions

- **TCP over HTTP:** Simpler protocol, no new dependencies, ~30 LOC total for client+server
- **No new crate dependencies:** Uses only `std::net`, `std::io`, existing `cpal`, `hound`, `tempfile`
- **Per-connection threading:** Each transcription request gets its own thread (low frequency, simple)
- **Configurable port:** `stt_server_port` in `SttConfig`, default 5201
- **Future-compatible:** Protocol can be extended with command bytes for VAD-over-IPC
