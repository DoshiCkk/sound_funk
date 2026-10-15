"""Оркестратор: из Audio получаем полный разбор."""

from __future__ import annotations

import librosa

from soundfunc.audio import Audio
from soundfunc.chords import recognize_chords
from soundfunc.key import detect_key
from soundfunc.models import Analysis
from soundfunc.notes import transcribe
from soundfunc.rhythm import analyze_rhythm

CHROMA_HOP = 2048


def analyze(audio: Audio, *, with_notes: bool = False) -> Analysis:
    y, sr, t0 = audio.y, audio.sr, audio.offset
    y_harm = librosa.effects.harmonic(y, margin=2.0)

    chroma_mix = _chroma(y, sr)
    chroma_harm = _chroma(y_harm, sr)
    rms = librosa.feature.rms(y=y, frame_length=2 * CHROMA_HOP, hop_length=CHROMA_HOP)[0]

    rhythm = analyze_rhythm(y, sr, chroma_mix, CHROMA_HOP, t0=t0)
    key = detect_key(chroma_harm)
    chords = recognize_chords(chroma_harm, rms, sr, CHROMA_HOP, beats=rhythm.beats,
                              t0=t0, duration=audio.duration)
    notes = transcribe(y_harm, sr, t0=t0) if with_notes else []

    return Analysis(path=audio.path, offset=t0, duration=round(audio.duration, 3),
                    rhythm=rhythm, key=key, chords=chords, notes=notes)


def _chroma(y, sr: int):
    # от C2 (65 Гц): гудящий суббас бочки/808 иначе перетягивает хрому на одну ноту
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=CHROMA_HOP,
                                      fmin=librosa.note_to_hz("C2"), n_octaves=6)
