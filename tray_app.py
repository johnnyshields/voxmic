#!/usr/bin/env python3
"""
Voxtral System Tray Dictation App

Hold Ctrl+Win to record, release Win to transcribe and type at cursor.
Transcription backend: faster-whisper (local) or Voxtral Mini (llama-server).

Tray icon:  green = idle,  red = recording,  yellow = transcribing
Hotkey:     Hold Ctrl+Win while speaking, release Win to transcribe and type
Log:        %LOCALAPPDATA%/Voxtral/logs/tray.log
"""

# ── Section 1: Constants, paths, config loading ────────────────────────────────

import ctypes
import ctypes.wintypes
import glob
import json
import logging
import os
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd

LOCALAPPDATA = Path(os.environ["LOCALAPPDATA"])
MODELS_DIR   = LOCALAPPDATA / "Voxtral" / "models" / "whisper"
LOGS_DIR     = LOCALAPPDATA / "Voxtral" / "logs"
CONFIG_PATH  = Path(__file__).parent / "config.json"

KNOWN_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

_DEFAULTS: dict = {
    "backend":              "whisper",
    "device_pattern":       "DJI",
    "sample_rate":          16000,
    "chunk_duration":       0.1,
    "whisper_model":        "small",
    "whisper_language":     None,
    "whisper_device":       "cpu",
    "whisper_compute_type": "int8",
}


def load_config() -> dict:
    cfg = dict(_DEFAULTS)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ── Section 2: Logging (file-only) ────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("voxtral")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOGS_DIR / "tray.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger


log = logging.getLogger("voxtral")


# ── Section 3: Backend abstraction + implementations ───────────────────────────

class TranscriptionBackend(ABC):
    @abstractmethod
    def transcribe(self, wav_path: str) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def name(self) -> str: ...


class FasterWhisperBackend(TranscriptionBackend):
    def __init__(self, cfg: dict):
        self._cfg        = dict(cfg)
        self._model      = None
        self._model_lock = threading.Lock()

    def reload(self, cfg: dict) -> None:
        """Force re-load of the model on next transcription (after config change)."""
        with self._model_lock:
            self._cfg  = dict(cfg)
            self._model = None

    def _get_model(self):
        with self._model_lock:
            if self._model is None:
                from faster_whisper import WhisperModel
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                log.info("Loading Whisper model '%s' …", self._cfg["whisper_model"])
                t0 = time.monotonic()
                self._model = WhisperModel(
                    self._cfg["whisper_model"],
                    device=self._cfg.get("whisper_device", "cpu"),
                    compute_type=self._cfg.get("whisper_compute_type", "int8"),
                    download_root=str(MODELS_DIR),
                )
                log.info("Model loaded in %.1fs", time.monotonic() - t0)
            return self._model

    def transcribe(self, wav_path: str) -> str:
        model    = self._get_model()
        language = self._cfg.get("whisper_language") or None
        segments, _ = model.transcribe(
            wav_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def is_available(self) -> bool:
        return True

    def name(self) -> str:
        return "faster-whisper"


class VoxtralBackend(TranscriptionBackend):
    BASE_URL = "http://127.0.0.1:5200"

    def transcribe(self, wav_path: str) -> str:
        import requests
        with open(wav_path, "rb") as f:
            resp = requests.post(
                f"{self.BASE_URL}/v1/audio/transcriptions",
                files={"file": ("audio.wav", f, "audio/wav")},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()

    def is_available(self) -> bool:
        try:
            import requests
            resp = requests.get(f"{self.BASE_URL}/health", timeout=1)
            return resp.status_code == 200
        except Exception:
            return False

    def name(self) -> str:
        return "Voxtral Mini"


def make_backend(cfg: dict) -> TranscriptionBackend:
    if cfg.get("backend", "whisper") == "voxtral":
        return VoxtralBackend()
    return FasterWhisperBackend(cfg)


# ── Section 4: Audio capture helpers ──────────────────────────────────────────

def find_input_device(pattern: str):
    """Return (index, name) of first input device whose name contains pattern."""
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and pattern.lower() in dev["name"].lower():
            return idx, dev["name"]
    return None, None


# ── Section 5: State machine ───────────────────────────────────────────────────

class State(Enum):
    IDLE         = auto()
    RECORDING    = auto()
    TRANSCRIBING = auto()


@dataclass
class AppState:
    state:     State                          = State.IDLE
    lock:      threading.Lock                 = field(default_factory=threading.Lock)
    chunks:    List[np.ndarray]               = field(default_factory=list)
    src_hwnd:  int                            = 0
    backend:   Optional[TranscriptionBackend] = None
    tray_icon: object                         = None   # pystray.Icon, set in main()
    cfg:       dict                           = field(default_factory=dict)


_app = AppState()


def _set_tray_icon(state: State) -> None:
    if _app.tray_icon is not None:
        try:
            _app.tray_icon.icon = _make_icon(state)
        except Exception as exc:
            log.debug("Icon update failed: %s", exc)


# ── Section 6: Action handlers ─────────────────────────────────────────────────

def start_recording() -> None:
    with _app.lock:
        if _app.state != State.IDLE:
            return
        _app.src_hwnd = ctypes.windll.user32.GetForegroundWindow()
        _app.chunks   = []
        _app.state    = State.RECORDING
    log.info("Recording started (hwnd=%d)", _app.src_hwnd)
    _set_tray_icon(State.RECORDING)


def stop_recording() -> None:
    with _app.lock:
        if _app.state != State.RECORDING:
            return
        chunks_snapshot = list(_app.chunks)
        src_hwnd        = _app.src_hwnd
        _app.chunks     = []
        _app.state      = State.TRANSCRIBING

    duration = len(chunks_snapshot) * _app.cfg.get("chunk_duration", 0.1)
    log.info("Recording stopped — %.1fs captured (%d chunks)", duration, len(chunks_snapshot))
    _set_tray_icon(State.TRANSCRIBING)

    if not chunks_snapshot:
        log.info("No audio captured, returning to idle")
        with _app.lock:
            _app.state = State.IDLE
        _set_tray_icon(State.IDLE)
        return

    threading.Thread(
        target=_transcribe_and_type,
        args=(chunks_snapshot, src_hwnd),
        daemon=True,
        name="transcription",
    ).start()


def _transcribe_and_type(chunks: List[np.ndarray], src_hwnd: int) -> None:
    sr         = int(_app.cfg.get("sample_rate", 16000))
    audio_data = np.clip(
        np.concatenate(chunks) * 32767, -32768, 32767
    ).astype(np.int16)

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wavfile.write(tmp.name, sr, audio_data)
            tmp_path = tmp.name

        t0   = time.monotonic()
        text = _app.backend.transcribe(tmp_path)
        preview = (text[:80] + "…") if len(text) > 80 else (text or "(empty)")
        log.info("Transcribed in %.1fs: %s", time.monotonic() - t0, preview)

        if text:
            _type_at_window(src_hwnd, text)
    except Exception:
        log.exception("Transcription error")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        with _app.lock:
            _app.state = State.IDLE
        _set_tray_icon(State.IDLE)


def _type_at_window(hwnd: int, text: str) -> None:
    """Re-focus the source window and type transcribed text at cursor."""
    from pynput.keyboard import Controller

    user32   = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    if hwnd:
        try:
            cur_tid = kernel32.GetCurrentThreadId()
            tgt_tid = user32.GetWindowThreadProcessId(hwnd, None)
            if tgt_tid and tgt_tid != cur_tid:
                user32.AttachThreadInput(cur_tid, tgt_tid, True)
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.05)   # brief pause for focus to settle
            if tgt_tid and tgt_tid != cur_tid:
                user32.AttachThreadInput(cur_tid, tgt_tid, False)
        except Exception as exc:
            log.debug("Window focus error: %s", exc)

    Controller().type(text)
    log.info("Typed %d chars into hwnd=%d", len(text), hwnd)


# ── Section 7a: Tray icon images ───────────────────────────────────────────────

def _make_icon(state: State):
    """Generate a coloured mic icon with Pillow — no external .ico file needed."""
    from PIL import Image, ImageDraw

    size = 64
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg = {
        State.IDLE:         "#22BB44",   # green
        State.RECORDING:    "#CC2222",   # red
        State.TRANSCRIBING: "#CC9900",   # amber
    }.get(state, "#888888")

    # Coloured circle background
    m = 2
    draw.ellipse([m, m, size - m, size - m], fill=bg)

    # White mic silhouette
    cx, cy = size // 2, size // 2
    # Capsule body
    body_r = 6
    draw.rounded_rectangle(
        [cx - body_r, cy - 18, cx + body_r, cy + 2],
        radius=body_r, fill="white",
    )
    # Stand arc (semicircle below the capsule)
    draw.arc([cx - 12, cy - 8, cx + 12, cy + 14], start=0, end=180, fill="white", width=3)
    # Vertical stand line
    draw.line([cx, cy + 14, cx, cy + 20], fill="white", width=3)
    # Horizontal base
    draw.line([cx - 7, cy + 20, cx + 7, cy + 20], fill="white", width=3)

    return img


# ── Section 7b: Model installation detection + download ───────────────────────

def _is_model_installed(model_name: str) -> bool:
    """True if faster-whisper has already cached this model in MODELS_DIR."""
    # HF hub stores models as models--{org}--{repo}/snapshots/{hash}/model.bin
    pattern = str(
        MODELS_DIR / f"models--Systran--faster-whisper-{model_name}"
                   / "snapshots" / "*" / "model.bin"
    )
    return len(glob.glob(pattern)) > 0


def _download_model_bg(model_name: str) -> None:
    """Background thread: download model, rebuild tray menu when done."""
    icon = _app.tray_icon
    log.info("Downloading whisper/%s …", model_name)
    try:
        if icon:
            icon.notify(f"Downloading whisper/{model_name} …", "Voxtral")
    except Exception:
        pass

    try:
        from faster_whisper import WhisperModel
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # Instantiating triggers HF hub download; model object is discarded after
        WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(MODELS_DIR),
        )
        log.info("whisper/%s downloaded successfully", model_name)
        try:
            if icon:
                icon.notify(f"whisper/{model_name} ready!", "Voxtral")
        except Exception:
            pass
    except Exception:
        log.exception("Model download failed: %s", model_name)
        try:
            if icon:
                icon.notify(f"Download failed: whisper/{model_name}", "Voxtral")
        except Exception:
            pass
    finally:
        # Rebuild menu so [installed] status reflects the new download
        if icon:
            try:
                icon.menu = _build_menu()
            except Exception:
                pass


# ── Section 7c: Tray menu ──────────────────────────────────────────────────────

def _build_menu():
    """Build (or rebuild) the full tray context menu from current _app state."""
    import pystray

    cfg         = _app.cfg
    cur_backend = cfg.get("backend", "whisper")
    cur_model   = cfg.get("whisper_model", "small")

    # --- callbacks ---

    def _switch_backend(b):
        def _do(icon, item):
            if b == "voxtral":
                vb = VoxtralBackend()
                if not vb.is_available():
                    log.warning("Voxtral server unreachable at %s", VoxtralBackend.BASE_URL)
                    try:
                        icon.notify(
                            "Voxtral server unreachable (127.0.0.1:5200)", "Voxtral"
                        )
                    except Exception:
                        pass
            cfg["backend"] = b
            save_config(cfg)
            _app.backend = make_backend(cfg)
            icon.menu = _build_menu()
            log.info("Backend → %s", b)
        return _do

    def _switch_model(m):
        def _do(icon, item):
            cfg["whisper_model"] = m
            save_config(cfg)
            if isinstance(_app.backend, FasterWhisperBackend):
                _app.backend.reload(cfg)
            icon.menu = _build_menu()
            log.info("Model → whisper/%s", m)
        return _do

    def _download_model(m):
        def _do(icon, item):
            threading.Thread(
                target=_download_model_bg,
                args=(m,),
                daemon=True,
                name=f"download-{m}",
            ).start()
        return _do

    def _open_log(icon, item):
        os.startfile(str(LOGS_DIR / "tray.log"))

    def _quit(icon, item):
        log.info("User quit")
        icon.stop()

    # --- backend submenu ---
    backend_menu = pystray.Menu(
        pystray.MenuItem(
            "faster-whisper",
            _switch_backend("whisper"),
            checked=lambda item, _b=cur_backend: _b == "whisper",
            radio=True,
        ),
        pystray.MenuItem(
            "Voxtral Mini",
            _switch_backend("voxtral"),
            checked=lambda item, _b=cur_backend: _b == "voxtral",
            radio=True,
        ),
    )

    # --- model submenu ---
    model_items = []
    for m in KNOWN_MODELS:
        if _is_model_installed(m):
            model_items.append(pystray.MenuItem(
                f"whisper/{m}  [installed]",
                _switch_model(m),
                checked=lambda item, _m=m, _c=cur_model: _c == _m,
                radio=True,
            ))
        else:
            model_items.append(pystray.MenuItem(
                f"Download whisper/{m}",
                _download_model(m),
            ))

    if not model_items:
        model_items = [pystray.MenuItem("(no models)", None, enabled=False)]

    return pystray.Menu(
        pystray.MenuItem("Voxtral Dictation", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Backend", backend_menu),
        pystray.MenuItem("Whisper Model", pystray.Menu(*model_items)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open Log", _open_log),
        pystray.MenuItem("Quit", _quit),
    )


# ── Section 7d: Hotkey listener (pynput) ──────────────────────────────────────

def _start_hotkey_listener() -> None:
    from pynput import keyboard

    held: set = set()

    # Safely enumerate keys that may not exist in all pynput versions
    ctrl_keys = frozenset(filter(None, [
        getattr(keyboard.Key, "ctrl",   None),
        getattr(keyboard.Key, "ctrl_l", None),
        getattr(keyboard.Key, "ctrl_r", None),
    ]))
    win_keys = frozenset(filter(None, [
        getattr(keyboard.Key, "cmd",   None),
        getattr(keyboard.Key, "cmd_l", None),
        getattr(keyboard.Key, "cmd_r", None),
    ]))

    def _on_press(key):
        held.add(key)
        # Win DOWN while Ctrl held → start recording (only if currently idle)
        if key in win_keys and (held & ctrl_keys):
            start_recording()   # no-op if not IDLE

    def _on_release(key):
        # Snapshot state before modifying `held`
        with _app.lock:
            recording_now = _app.state == State.RECORDING
        is_win = key in win_keys
        held.discard(key)
        # Win UP while recording → stop and transcribe
        if is_win and recording_now:
            stop_recording()

    listener = keyboard.Listener(
        on_press=_on_press,
        on_release=_on_release,
        suppress=False,
    )
    listener.daemon = True
    listener.start()
    log.info("Hotkey listener active — hold Ctrl+Win to dictate")


# ── Section 8: main() ──────────────────────────────────────────────────────────

def main() -> None:
    import pystray

    setup_logging()
    log.info("─── Voxtral tray app starting ───")

    cfg          = load_config()
    _app.cfg     = cfg
    _app.backend = make_backend(cfg)

    # --- audio stream (always-open to avoid per-recording init latency) ---
    sr            = int(cfg.get("sample_rate", 16000))
    chunk_samples = int(sr * cfg.get("chunk_duration", 0.1))
    pattern       = cfg.get("device_pattern", "DJI")

    device_idx, device_name = find_input_device(pattern)
    if device_idx is None:
        log.warning("No device matching '%s' found — using system default input", pattern)
    else:
        log.info("Mic: [%d] %s", device_idx, device_name)

    def _audio_cb(indata, frames, time_info, status):
        if status:
            log.debug("sd status: %s", status)
        # GIL-safe read of enum; acquire lock only for list append
        if _app.state == State.RECORDING:
            chunk = indata[:, 0].copy()
            with _app.lock:
                _app.chunks.append(chunk)

    stream = sd.InputStream(
        device=device_idx,
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
        callback=_audio_cb,
    )
    stream.start()
    log.info("Audio stream open (always-on, device=%s)", device_name or "default")

    # --- hotkeys ---
    _start_hotkey_listener()

    # --- tray icon ---
    icon = pystray.Icon(
        name="voxtral",
        icon=_make_icon(State.IDLE),
        title="Voxtral Dictation",
        menu=_build_menu(),
    )
    _app.tray_icon = icon

    log.info("Tray icon running — green=idle  red=recording  yellow=transcribing")
    try:
        icon.run()
    finally:
        stream.stop()
        stream.close()
        log.info("─── Tray app stopped ───")


if __name__ == "__main__":
    main()
