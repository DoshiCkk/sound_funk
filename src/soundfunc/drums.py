"""Удары барабанов по дорожке ударных: бочка, малый, хэты.

Наивный подход без нейросети. В каждой из трёх полос (низ, середина, верх)
считается прирост энергии (spectral flux) — он реагирует на новый удар,
а не на хвост предыдущего. Полосы нормируются отдельно, поэтому тихие хэты
не теряются на фоне громкой бочки. Удар классифицируется по тому, в каких
полосах был всплеск:
  • бочка — всплеск в низах (до 150 Гц);
  • малый — всплеск в середине, и он шумовой: у перегруженной бочки середина
    тоже вспыхивает, но гармониками — это тональный, а не шумовой спектр;
  • хэт — всплеск выше 5 кГц.
Один удар может быть сразу несколькими (бочка + хэт).
"""

from __future__ import annotations

import librosa
import numpy as np

from soundfunc.models import DrumHit

HOP = 256
BANDS = {"low": (30.0, 150.0), "mid": (300.0, 5000.0), "high": (5000.0, 16000.0)}
WINDOW_S = 0.04
PRESENT = {"low": 0.3, "mid": 0.3, "high": 0.18}  # доля самого сильного всплеска в полосе
NOISY = 0.2  # спектральная плоскостность, выше которой середина — шум, а не гармоники
LEVEL_GATE = 20.0  # дБ: удар в полосе не тише самого громкого на столько
HAT_GATE = 8.0  # дБ: хэт не тише типичного «чистого» хэта на столько
PAD_S = 0.1  # тишина в начале: иначе удар в первом кадре не с чем сравнить


def detect_drums(y: np.ndarray, sr: int, *, t0: float = 0.0) -> list[DrumHit]:
    pad = int(PAD_S * sr)
    y = np.concatenate([np.zeros(pad, dtype=y.dtype), y])
    t0 -= pad / sr
    power = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2
    if power.max() <= 0:
        return []
    db = librosa.power_to_db(power, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    rows = {name: (freqs >= lo) & (freqs < min(hi, sr / 2)) for name, (lo, hi) in BANDS.items()}

    flux = {}
    for name, r in rows.items():
        env = librosa.onset.onset_strength(S=db[r], sr=sr, hop_length=HOP) if r.any() else np.zeros(1)
        flux[name] = env / env.max() if env.max() > 0 else env
    n = min(len(f) for f in flux.values())
    combined = sum(f[:n] for f in flux.values())
    if combined.max() <= 0:
        return []
    onsets = librosa.onset.onset_detect(onset_envelope=combined, sr=sr, hop_length=HOP,
                                        backtrack=False, wait=4, delta=0.05)

    width = max(1, int(WINDOW_S * sr / HOP))
    # поток считается в дБ, и еле слышный щелчок бочки на верхах даёт большой всплеск,
    # поэтому дополнительно требуем, чтобы удар в полосе был не тише лучших на LEVEL_GATE дБ
    level = {name: np.array([10 * np.log10(power[r, f:f + width].sum() + 1e-12) for f in onsets])
             for name, r in rows.items()}
    peaks = [{name: float(env[max(0, f - 1):min(n, f + 2)].max()) for name, env in flux.items()}
             for f in onsets]
    hat_ref = _hat_reference(peaks, level["high"])
    hits: list[DrumHit] = []
    for i, (f, peak) in enumerate(zip(onsets, peaks)):
        loud = {name: level[name][i] > level[name].max() - LEVEL_GATE for name in rows}
        time = round(float(librosa.frames_to_time(f, sr=sr, hop_length=HOP)) + t0, 3)
        strength = round(float(np.clip(combined[f] / combined.max(), 0.0, 1.0)), 2)
        if peak["low"] > PRESENT["low"] and loud["low"]:
            hits.append(DrumHit("kick", time, strength))
        if peak["mid"] > PRESENT["mid"] and loud["mid"] and _flatness(power, rows["mid"], f, width) > NOISY:
            hits.append(DrumHit("snare", time, strength))
        if (peak["high"] > PRESENT["high"] and level["high"][i] >= hat_ref - HAT_GATE
                and _flatness(power, rows["high"], f, width) > NOISY):
            hits.append(DrumHit("hat", time, strength))
    return [h for h in hits if h.time >= t0 + pad / sr - 0.01]


def _hat_reference(peaks: list[dict[str, float]], high_level: np.ndarray) -> float:
    """Типичная громкость верхов у «чистых» хэтов — без бочки и малого в тот же момент.

    Щелчок бочки и артефакты разделения тоже дают немного верхов, но заметно
    тише настоящего хэта — с этой громкостью их и сравниваем.
    """
    pure = [i for i, p in enumerate(peaks)
            if p["high"] > PRESENT["high"] and p["low"] < PRESENT["low"] and p["mid"] < PRESENT["mid"]]
    return float(np.median(high_level[pure])) if pure else float(high_level.max())


def _flatness(power: np.ndarray, rows: np.ndarray, frame: int, width: int) -> float:
    """Спектральная плоскостность сразу после удара: шум → ближе к 1, гармоники → к 0."""
    spec = power[rows, frame:frame + width].mean(axis=1) + 1e-12
    return float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
