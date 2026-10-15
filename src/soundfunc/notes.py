"""Наивная полифоническая транскрипция в ноты (CQT + пики + подавление обертонов).

Это базовый вариант без нейросетей: годится для чистых записей и синтеза,
на плотном миксе даёт шум. В планах — бэкенд basic-pitch (см. ROADMAP.md).
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import median_filter

from soundfunc.models import Note

MIDI_LO, MIDI_HI = 21, 108  # A0..C8 — диапазон фортепиано
BINS_PER_SEMITONE = 3
# интервалы до 2..6-й гармоники в полутонах
HARMONIC_OFFSETS = (12, 19, 24, 28, 31)


def transcribe(y: np.ndarray, sr: int, *, t0: float = 0.0, hop: int = 512,
               rel_db: float = -30.0, abs_db: float = -55.0,
               min_dur: float = 0.08) -> list[Note]:
    n_pitches = MIDI_HI - MIDI_LO + 1
    cqt = np.abs(librosa.cqt(
        y, sr=sr, hop_length=hop,
        fmin=librosa.midi_to_hz(MIDI_LO - 1 / BINS_PER_SEMITONE),
        n_bins=n_pitches * BINS_PER_SEMITONE,
        bins_per_octave=12 * BINS_PER_SEMITONE,
    ))
    mag = cqt.reshape(n_pitches, BINS_PER_SEMITONE, -1).max(axis=1)  # (88, T)
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)

    active = (mag_db > mag_db.max(axis=0, keepdims=True) + rel_db) & (mag_db > abs_db)
    active &= _pitch_peaks(mag)
    for off in HARMONIC_OFFSETS:
        below_active = np.zeros_like(active)
        below_active[off:] = active[:-off]
        below_mag = np.zeros_like(mag)
        below_mag[off:] = mag[:-off]
        active &= ~(below_active & (mag < below_mag))
    active = median_filter(active.astype(np.uint8), size=(1, 3)).astype(bool)

    times = librosa.frames_to_time(np.arange(active.shape[1] + 1), sr=sr, hop_length=hop)
    notes = []
    for i, row in enumerate(active):
        edges = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            start, end = float(times[s]), float(times[e])
            if end - start < min_dur:
                continue
            velocity = float(np.clip((mag_db[i, s:e].max() - abs_db) / -abs_db, 0.0, 1.0))
            notes.append(Note(MIDI_LO + i, round(start + t0, 3), round(end + t0, 3),
                              round(velocity, 2)))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def _pitch_peaks(mag: np.ndarray) -> np.ndarray:
    """Локальные максимумы по оси высоты: соседние полутона одной ноты не дублируются."""
    peaks = np.ones(mag.shape, dtype=bool)
    peaks[1:] &= mag[1:] >= mag[:-1]
    peaks[:-1] &= mag[:-1] >= mag[1:]
    return peaks
