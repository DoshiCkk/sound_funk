import json
import re
import sys

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, chord, kick
from soundfunc import stems as stems_mod
from soundfunc.analysis import analyze
from soundfunc.audio import Audio
from soundfunc.models import Note
from soundfunc.notes import transcribe
from soundfunc.termroll import terminal_roll
from soundfunc.viz import PLACEHOLDER, build_viewer


def test_confidence_in_range_and_hot_for_clear_chord():
    import librosa
    y = librosa.effects.harmonic(chord([60, 64, 67], 1.5).astype(np.float32), margin=2.0)
    notes = transcribe(y, SR)
    assert all(0.0 <= n.confidence <= 1.0 for n in notes)
    main = [n for n in notes if n.pitch in (60, 64, 67) and n.end - n.start > 0.5]
    assert main and min(n.confidence for n in main) > 0.6


def test_terminal_roll_draws_half_blocks():
    notes = [Note(60, 0.0, 1.0, 0.8, 0.9), Note(64, 1.0, 2.0, 0.5, 0.2)]
    lines = terminal_roll(notes, 0.0, 2.0, width=60, downbeats=[0.0, 1.0])
    text = "\n".join(line.plain for line in lines)
    assert "▀" in text and "C4" in text and "горячо" in text
    assert terminal_roll([], 0.0, 1.0)[0].plain.strip() == "(нот не найдено)"


@pytest.fixture(scope="module")
def stem_files(tmp_path_factory):
    """Синтетические «дорожки»: бас (A2 → F2) и барабаны (бочка на каждую долю)."""
    d = tmp_path_factory.mktemp("stems")
    t = np.arange(int(4 * SR)) / SR
    bass = 0.5 * np.sin(2 * np.pi * np.where(t < 2, 110.0, 87.31) * t)
    drums = np.zeros_like(t)
    for b in range(8):
        k = kick(0.4, 0.9)
        i = int(b * 0.5 * SR)
        drums[i:i + len(k)] += k[:len(drums) - i]
    other = chord([57, 60, 64], 4.0)[:len(t)]
    silent = np.zeros_like(t)
    paths = {}
    for name, y in {"bass": bass, "drums": drums, "other": other, "vocals": silent,
                    "guitar": silent, "piano": silent}.items():
        paths[name] = d / f"{name}.wav"
        sf.write(paths[name], y.astype(np.float32), SR)
    mix = (bass + drums + other) / 3
    mix_path = d / "mix.wav"
    sf.write(mix_path, mix.astype(np.float32), SR)
    return mix_path, paths


def test_analysis_by_stems(stem_files):
    from soundfunc.audio import load
    mix_path, paths = stem_files
    result = analyze(load(mix_path), with_notes=True, stems=paths)

    assert set(result.tracks) == {"bass", "other"}  # пустые вокал/гитара/пианино пропущены
    bass_pitches = {n.pitch for n in result.tracks["bass"] if n.end - n.start > 1.0}
    assert {45, 41} <= bass_pitches  # A2 и F2
    kicks = [h for h in result.drums if h.kind == "kick"]
    assert len(kicks) >= 7


def test_viewer_embeds_data_and_audio(tmp_path, stem_files):
    from soundfunc.audio import load
    mix_path, paths = stem_files
    result = analyze(load(mix_path), with_notes=True, stems=paths)
    out = build_viewer(result, tmp_path / "v.html")

    html = out.read_text(encoding="utf-8")
    assert PLACEHOLDER not in html
    payload = re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1)
    data = json.loads(payload)
    assert {t["name"] for t in data["tracks"]} == {"bass", "other"}
    assert data["audio"].startswith("data:audio/mpeg;base64,")
    assert data["drums"] and len(data["heatStops"]) >= 2
    assert all(len(n) == 5 for t in data["tracks"] for n in t["notes"])


def test_missing_demucs_gives_friendly_error(tmp_path, monkeypatch, stem_files):
    mix_path, _ = stem_files
    monkeypatch.setitem(sys.modules, "demucs", None)  # import demucs → ImportError
    monkeypatch.setitem(sys.modules, "demucs.apply", None)
    monkeypatch.setitem(sys.modules, "demucs.pretrained", None)
    with pytest.raises(stems_mod.StemsUnavailable, match="uv sync --extra stems"):
        stems_mod.separate(mix_path, cache_dir=tmp_path)


def test_separate_uses_cache(tmp_path, stem_files):
    mix_path, paths = stem_files
    cached = tmp_path / f"{stems_mod.file_hash(mix_path)}-{stems_mod.MODEL}"
    cached.mkdir()
    for name, p in paths.items():
        (cached / f"{name}.wav").write_bytes(p.read_bytes())
    result = stems_mod.separate(mix_path, cache_dir=tmp_path)  # нейросеть не запускается
    assert set(result) == set(stems_mod.STEMS)
