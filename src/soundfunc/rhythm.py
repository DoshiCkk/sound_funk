"""Темп, доли и размер такта.

Размер такта определяется эвристикой: на каждой доле считаем «акцент»
(удар в низах, общий onset, смена гармонии) и смотрим, при какой группировке
долей (по 2/3/4) первая доля такта выделяется сильнее всего.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

from soundfunc.models import Rhythm

HOP = 512


def analyze_rhythm(y: np.ndarray, sr: int, chroma: np.ndarray, chroma_hop: int,
                   *, t0: float = 0.0) -> Rhythm:
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=64))
    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=HOP)
    # без сглаживания острые пики (электроника, синтез) «промахиваются» мимо целого
    # лага автокорреляции, и трекер выбирает неверный темп
    _, beat_frames = librosa.beat.beat_track(onset_envelope=gaussian_filter1d(onset_env, 1.0),
                                             sr=sr, hop_length=HOP)
    beat_frames = beat_frames[beat_frames > 0]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)

    if len(beat_times) > 2:
        # наклон регрессии точнее медианы интервалов: кадры квантованы по 23 мс
        period = float(np.polyfit(np.arange(len(beat_times)), beat_times, 1)[0])
        tempo = 60.0 / period
    else:
        tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr))[0])
        period = 60.0 / tempo

    # нижние ~8 мел-полос — бочка и бас
    low_env = librosa.onset.onset_strength(S=mel_db[:8], sr=sr, hop_length=HOP)
    beat_chroma_idx = beat_frames * HOP // chroma_hop
    meter, phase, confidence = estimate_meter(onset_env, low_env, chroma, beat_frames, beat_chroma_idx)

    # трекер обрезает первые доли — достраиваем их назад до начала фрагмента
    leading = []
    if len(beat_times):
        t = beat_times[0] - period
        while t > -0.25 * period:
            leading.insert(0, max(t, 0.0))
            t -= period
    phase = (phase + len(leading)) % meter
    beats = [round(float(t + t0), 3) for t in [*leading, *beat_times]]
    return Rhythm(
        tempo=round(tempo, 1),
        beats_per_bar=meter,
        meter_confidence=round(confidence, 2),
        beats=beats,
        downbeats=beats[phase::meter],
    )


def estimate_meter(onset_env: np.ndarray, low_env: np.ndarray, chroma: np.ndarray,
                   beat_frames: np.ndarray, beat_chroma_idx: np.ndarray) -> tuple[int, int, float]:
    """Возвращает (долей в такте, индекс первой сильной доли, уверенность 0..1)."""
    if len(beat_frames) < 12:
        return 4, 0, 0.0

    harmonic_change = _harmonic_change(chroma, beat_chroma_idx)
    # смена аккорда — самый надёжный признак сильной доли, бочка — второй;
    # общий onset (хэты, мелодия) часто акцентирует слабые доли, поэтому вес меньше
    accent = (2.0 * _zscore(harmonic_change)
              + 1.0 * _zscore(_sample_at(low_env, beat_frames))
              + 0.5 * _zscore(_sample_at(onset_env, beat_frames)))

    s2, _ = _bar_contrast(accent, 2)
    s3, p3 = _bar_contrast(accent, 3)
    s4, p4 = _bar_contrast(accent, 4)
    duple = max(s2, s4)

    if s3 > duple:
        meter, phase, win, lose = 3, p3, s3, duple
    else:
        meter, phase, win, lose = 4, p4, duple, s3
    confidence = float(np.clip((win - lose) / (abs(win) + abs(lose) + 1e-9), 0.0, 1.0))
    return meter, phase, confidence


def _bar_contrast(accent: np.ndarray, m: int) -> tuple[float, int]:
    """Насколько доли с шагом m (лучшая фаза) громче остальных."""
    best, best_phase = -np.inf, 0
    for p in range(m):
        mask = np.zeros(len(accent), dtype=bool)
        mask[p::m] = True
        c = float(accent[mask].mean() - accent[~mask].mean())
        if c > best:
            best, best_phase = c, p
    return best, best_phase


def _harmonic_change(chroma: np.ndarray, beat_idx: np.ndarray, window: int = 2) -> np.ndarray:
    """Косинусное расстояние между хромой `window` долей до и после каждой доли.

    Окно в несколько долей нужно, чтобы арпеджио и мелодия внутри доли не
    выглядели сменой гармонии — меняться должен «аккорд целиком».
    """
    idx = np.clip(beat_idx, 1, chroma.shape[1] - 1)
    segs = librosa.util.sync(chroma, idx, aggregate=np.mean)  # segs[i+1] — от доли i до i+1
    change = np.zeros(len(beat_idx))
    for i in range(min(len(beat_idx), segs.shape[1] - 1)):
        before = segs[:, max(0, i + 1 - window):i + 1].mean(axis=1)
        after = segs[:, i + 1:i + 1 + window].mean(axis=1)
        denom = np.linalg.norm(before) * np.linalg.norm(after)
        change[i] = 1.0 - before @ after / denom if denom > 1e-9 else 0.0
    return change


def _sample_at(env: np.ndarray, frames: np.ndarray, radius: int = 2) -> np.ndarray:
    return np.array([env[max(0, f - radius):f + radius + 1].max() for f in frames])


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)
