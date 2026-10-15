"""Разделение трека на инструменты моделью Demucs `htdemucs_6s` (Meta).

Шесть дорожек: drums, bass, other, vocals, guitar, piano. Нужна опциональная
зависимость: `uv sync --extra stems` (или `pip install soundfunc[stems]`).
Дорожки кешируются по хешу файла в ~/.cache/soundfunc/stems, поэтому
повторный анализ того же трека не запускает нейросеть заново.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

MODEL = "htdemucs_6s"
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
CACHE_DIR = Path(os.environ.get("SOUNDFUNC_CACHE", Path.home() / ".cache" / "soundfunc")) / "stems"


class StemsUnavailable(RuntimeError):
    pass


def file_hash(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def separate(path: str | Path, *, model: str = MODEL, cache_dir: Path | None = None,
             on_status: Callable[[str], None] | None = None) -> dict[str, Path]:
    """Возвращает {имя дорожки: путь к wav}. Считает один раз, дальше берёт из кеша."""
    out_dir = (cache_dir or CACHE_DIR) / f"{file_hash(path)}-{model}"
    paths = {name: out_dir / f"{name}.wav" for name in STEMS}
    if all(p.exists() for p in paths.values()):
        return paths

    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise StemsUnavailable(
            "для разделения на инструменты нужен Demucs: uv sync --extra stems "
            "(или pip install soundfunc[stems])") from exc

    # чекпойнты Demucs — pickle с классами модели; torch>=2.6 по умолчанию грузит только веса
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if on_status:
        on_status(f"Загружаю модель {model}...")
    net = get_model(model)
    net.eval()

    wav, _ = librosa.load(str(path), sr=net.samplerate, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    mix = torch.from_numpy(np.ascontiguousarray(wav[:2], dtype=np.float32))
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if on_status:
        on_status(f"Разделяю на инструменты ({device}, первый раз может занять минуту-другую)...")
    with torch.no_grad():
        sources = apply_model(net, ((mix - mean) / std)[None], device=device,
                              split=True, overlap=0.25, progress=False)[0]
    sources = sources * std + mean

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, src in zip(net.sources, sources):
        sf.write(out_dir / f"{name}.wav", src.cpu().numpy().T, net.samplerate)
    return paths


def stem_loudness(paths: dict[str, Path]) -> dict[str, float]:
    """Громкость каждой дорожки в дБ относительно самой громкой — чтобы пропускать пустые."""
    rms = {}
    for name, p in paths.items():
        y, _ = sf.read(p, dtype="float32", always_2d=True)
        rms[name] = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    top = max(rms.values())
    return {name: round(float(20 * np.log10(v / top)), 1) for name, v in rms.items()}
