"""Синтетические сигналы с заранее известным ответом."""

from __future__ import annotations

import numpy as np
import pytest

from soundfunc.audio import Audio

SR = 22050


def tone(midi: int, dur: float, amp: float = 0.3) -> np.ndarray:
    """Нота с 6 гармониками и мягкой огибающей."""
    t = np.arange(int(dur * SR)) / SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    y = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 7) if f0 * h < SR / 2)
    env = np.minimum(1.0, t / 0.01) * np.exp(-0.8 * t)
    return amp * y * env


def chord(pitches: list[int], dur: float) -> np.ndarray:
    return sum(tone(p, dur) for p in pitches) / len(pitches)


def kick(dur: float, amp: float) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    f = 50 + 120 * np.exp(-40 * t)
    return amp * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-12 * t)


def hat(dur: float, amp: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    noise = np.diff(rng.uniform(-1, 1, len(t) + 1))  # грубый ФВЧ
    return amp * noise * np.exp(-60 * t)


def make_song(progression: list[list[int]], *, bpm: float, beats_per_bar: int,
              bars_per_chord: int = 1, repeats: int = 2, seed: int = 0) -> Audio:
    """Аккорды меняются на сильную долю, бочка на 1-й доле, хэт на остальных."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar_len = int(round(beat * beats_per_bar * SR))
    beat_len = int(round(beat * SR))
    parts = []
    for _ in range(repeats):
        for pitches in progression:
            for _ in range(bars_per_chord):
                bar = chord(pitches, bar_len / SR)[:bar_len]
                for b in range(beats_per_bar):
                    hit = kick(beat, 0.9) if b == 0 else hat(beat, 0.15, rng)
                    bar[b * beat_len:b * beat_len + len(hit)] += hit[:bar_len - b * beat_len]
                parts.append(bar)
    y = np.concatenate(parts).astype(np.float32)
    return Audio(y=y / np.abs(y).max() * 0.9, sr=SR, path="<synthetic>")


C, F, G = [48, 60, 64, 67], [53, 60, 65, 69], [55, 59, 62, 67]
AM, DM, E = [45, 60, 64, 69], [50, 62, 65, 69], [52, 59, 64, 68]


@pytest.fixture(scope="session")
def song_c_major_44() -> Audio:
    return make_song([C, F, G, C], bpm=110, beats_per_bar=4, repeats=3)


@pytest.fixture(scope="session")
def song_a_minor_34() -> Audio:
    return make_song([AM, DM, E, AM], bpm=120, beats_per_bar=3, repeats=4)
