"""Тональность по алгоритму Крумхансла–Шмуклера."""

from __future__ import annotations

import numpy as np

from soundfunc.models import Key

# Krumhansl & Kessler (1982)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def detect_key(chroma: np.ndarray) -> Key:
    """chroma: (12, T) — хромаграмма гармонической части."""
    profile = chroma.sum(axis=1)
    candidates = []
    for mode, ref, names in (("major", MAJOR_PROFILE, MAJOR_NAMES),
                             ("minor", MINOR_PROFILE, MINOR_NAMES)):
        for tonic in range(12):
            r = float(np.corrcoef(profile, np.roll(ref, tonic))[0, 1])
            candidates.append((r, names[tonic], mode))
    candidates.sort(reverse=True)

    r, tonic, mode = candidates[0]
    alternatives = [Key(t, m, c).label for c, t, m in candidates[1:4]]
    return Key(tonic=tonic, mode=mode, confidence=round(r, 3), alternatives=alternatives)
