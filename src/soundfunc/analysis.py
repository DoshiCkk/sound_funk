"""Оркестратор: из Audio получаем полный разбор."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from soundfunc.audio import Audio, load
from soundfunc.chords import recognize_chords
from soundfunc.drums import detect_drums
from soundfunc.key import detect_key
from soundfunc.models import Analysis
from soundfunc.notes import transcribe
from soundfunc.rhythm import analyze_rhythm

CHROMA_HOP = 2048
SILENT_STEM_DB = -30.0  # дорожки тише самой громкой на 30 дБ считаем пустыми


def analyze(audio: Audio, *, with_notes: bool = False,
            stems: dict[str, Path] | None = None) -> Analysis:
    """stems — дорожки из soundfunc.stems.separate: тогда ноты ищутся по инструментам."""
    y, sr, t0 = audio.y, audio.sr, audio.offset
    y_harm = librosa.effects.harmonic(y, margin=2.0)

    chroma_mix = _chroma(y, sr)
    chroma_harm = _chroma(y_harm, sr)
    rms = librosa.feature.rms(y=y, frame_length=2 * CHROMA_HOP, hop_length=CHROMA_HOP)[0]

    rhythm = analyze_rhythm(y, sr, chroma_mix, CHROMA_HOP, t0=t0)
    key = detect_key(chroma_harm)
    chords = recognize_chords(chroma_harm, rms, sr, CHROMA_HOP, beats=rhythm.beats,
                              t0=t0, duration=audio.duration)
    result = Analysis(path=audio.path, offset=t0, duration=round(audio.duration, 3),
                      rhythm=rhythm, key=key, chords=chords)

    if stems:
        _analyze_stems(result, stems, audio)
    elif with_notes:
        result.notes = transcribe(y_harm, sr, t0=t0)
    return result


def _analyze_stems(result: Analysis, stems: dict[str, Path], audio: Audio) -> None:
    duration = audio.duration
    loaded = {name: load(p, offset=audio.offset, duration=duration, sr=audio.sr)
              for name, p in stems.items()}
    level = {name: float(np.sqrt(np.mean(a.y ** 2)) + 1e-12) for name, a in loaded.items()}
    loudest = max(level.values())

    for name, stem in loaded.items():
        if 20 * np.log10(level[name] / loudest) < SILENT_STEM_DB:
            continue
        if name == "drums":
            result.drums = detect_drums(stem.y, stem.sr, t0=audio.offset)
        else:
            harm = librosa.effects.harmonic(stem.y, margin=2.0)
            notes = transcribe(harm, stem.sr, t0=audio.offset)
            if notes:
                result.tracks[name] = notes


def _chroma(y, sr: int):
    # от C2 (65 Гц): гудящий суббас бочки/808 иначе перетягивает хрому на одну ноту
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=CHROMA_HOP,
                                      fmin=librosa.note_to_hz("C2"), n_octaves=6)
