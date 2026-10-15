"""Окно-визуализатор: HTML-страница с анимированным piano-roll и звуком.

Страница самодостаточная — данные разбора и mp3 встраиваются внутрь,
поэтому файл можно открыть где угодно или переслать. Открываем её в окне
Edge/Chrome без рамок браузера (режим --app), иначе — в браузере по умолчанию.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

import librosa
import soundfile as sf

from soundfunc.chords import NO_CHORD
from soundfunc.models import Analysis
from soundfunc.palette import HEAT_STOPS, TRACK_COLORS, TRACK_LABELS

TEMPLATE = Path(__file__).with_name("viewer.html")
PLACEHOLDER = "/*__SFUNC_DATA__*/null"
TRACK_ORDER = ("vocals", "guitar", "piano", "other", "bass", "mix")


def build_viewer(result: Analysis, out_path: str | Path, *, audio_path: str | Path | None = None) -> Path:
    """Пишет HTML-визуализатор; audio_path — откуда взять звук (по умолчанию исходный файл)."""
    tracks = sorted(result.note_tracks().items(),
                    key=lambda kv: TRACK_ORDER.index(kv[0]) if kv[0] in TRACK_ORDER else 99)
    data = {
        "title": Path(result.path).name,
        "offset": result.offset,
        "duration": result.duration,
        "tempo": result.rhythm.tempo,
        "timeSignature": result.rhythm.time_signature,
        "meterConfidence": result.rhythm.meter_confidence,
        "key": result.key.label,
        "keyAlternatives": result.key.alternatives,
        "beats": result.rhythm.beats,
        "downbeats": result.rhythm.downbeats,
        "chords": [{"label": c.label, "start": c.start, "end": c.end, "root": c.root,
                    "quality": c.quality} for c in result.chords if c.label != NO_CHORD],
        "tracks": [{"name": name, "label": TRACK_LABELS.get(name, name),
                    "color": TRACK_COLORS.get(name, "#7dd3fc"),
                    "notes": [[n.pitch, n.start, n.end, n.velocity, n.confidence] for n in notes]}
                   for name, notes in tracks],
        "drums": [{"kind": h.kind, "time": h.time, "strength": h.strength} for h in result.drums],
        "drumColor": TRACK_COLORS["drums"],
        "heatStops": HEAT_STOPS,
        "audio": _audio_data_uri(audio_path or result.path, result.offset, result.duration),
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, payload)
    out = Path(out_path)
    out.write_text(html, encoding="utf-8")
    return out


def _audio_data_uri(path: str | Path, offset: float, duration: float) -> str | None:
    try:
        y, sr = librosa.load(str(path), sr=44100, mono=False, offset=offset, duration=duration)
        buf = io.BytesIO()
        sf.write(buf, y.T if y.ndim == 2 else y, sr, format="MP3")
    except Exception:  # без звука визуализатор всё равно работает — на внутренних часах
        return None
    return "data:audio/mpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def open_window(path: str | Path) -> str:
    """Открывает страницу отдельным окном. Возвращает, чем открыли."""
    uri = Path(path).resolve().as_uri()
    browser = _find_chromium()
    if browser:
        subprocess.Popen([browser, f"--app={uri}", "--window-size=1440,880"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return Path(browser).stem
    webbrowser.open(uri)
    return "браузер по умолчанию"


def _find_chromium() -> str | None:
    candidates: list[str] = []
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                     os.environ.get("LOCALAPPDATA")):
            if base:
                candidates += [rf"{base}\Microsoft\Edge\Application\msedge.exe",
                               rf"{base}\Google\Chrome\Application\chrome.exe"]
    elif sys.platform == "darwin":
        candidates += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                       "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    return next((c for c in candidates if Path(c).exists()), None)
