from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

DEFAULT_SR = 22050


@dataclass
class Audio:
    y: np.ndarray  # моно, float32
    sr: int
    path: str
    offset: float = 0.0  # с какой секунды файла начинается y

    @property
    def duration(self) -> float:
        return len(self.y) / self.sr


def load(path: str | Path, *, offset: float = 0.0, duration: float | None = None,
         sr: int = DEFAULT_SR) -> Audio:
    y, sr = librosa.load(str(path), sr=sr, mono=True, offset=offset, duration=duration)
    if y.size == 0:
        raise ValueError(f"Пустой аудиофрагмент: {path}")
    return Audio(y=y, sr=sr, path=str(path), offset=offset)
