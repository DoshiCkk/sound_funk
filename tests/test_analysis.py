from pathlib import Path

import numpy as np
import pytest

from conftest import SR, chord
from soundfunc.analysis import analyze
from soundfunc.audio import Audio
from soundfunc.notes import transcribe
from soundfunc.report import chords_by_bar
from soundfunc.synth import render


@pytest.fixture(scope="module")
def c_major(song_c_major_44):
    return analyze(song_c_major_44)


@pytest.fixture(scope="module")
def a_minor(song_a_minor_34):
    return analyze(song_a_minor_34)


def test_tempo(c_major, a_minor):
    assert c_major.rhythm.tempo == pytest.approx(110, rel=0.04)
    assert a_minor.rhythm.tempo == pytest.approx(120, rel=0.04)


def test_time_signature(c_major, a_minor):
    assert c_major.rhythm.time_signature == "4/4"
    assert a_minor.rhythm.time_signature == "3/4"


def test_key(c_major, a_minor):
    assert c_major.key.label == "C major"
    assert a_minor.key.label == "A minor"


def test_chords_follow_progression(c_major):
    labels = [s.label for s in c_major.chords if s.label != "N"]
    assert labels[:4] == ["C", "F", "G", "C"] or labels[:3] == ["C", "F", "G"]


def test_chords_by_bar(a_minor):
    bars = chords_by_bar(a_minor.chords, a_minor.rhythm.downbeats,
                         a_minor.offset + a_minor.duration)
    flat = [b[0] for b in bars[:8] if b]
    assert flat[:4] == ["Am", "Dm", "E", "Am"]


def test_transcribe_triad():
    y = chord([60, 64, 67], 1.5)
    notes = transcribe(librosa_harmonic(y), SR)
    pitches = {n.pitch for n in notes if n.end - n.start > 0.5}
    assert {60, 64, 67} <= pitches
    assert len(pitches) <= 5  # без россыпи обертонов


def test_offset_shifts_times(song_c_major_44):
    sr = song_c_major_44.sr
    part = Audio(song_c_major_44.y[sr * 4:sr * 12], sr, "<x>", offset=4.0)
    result = analyze(part)
    assert result.rhythm.beats[0] >= 4.0
    assert result.chords[0].start == pytest.approx(4.0, abs=0.01)


def test_formula_patch_roundtrip():
    """Синтезированное формулой арпеджио Am–C–G–F разбирается обратно."""
    patch = Path(__file__).parent.parent / "examples" / "patches" / "arp.sfx"
    synth = render(patch.read_text(encoding="utf-8"), duration=16, sr=SR)
    result = analyze(Audio(synth.y, synth.sr, "arp.sfx"))

    assert result.rhythm.tempo == pytest.approx(120, rel=0.02)
    assert result.rhythm.time_signature == "4/4"
    bars = chords_by_bar(result.chords, result.rhythm.downbeats, result.duration)
    assert bars[:4] == [["Am"], ["C"], ["G"], ["F"]]


def librosa_harmonic(y):
    import librosa
    return librosa.effects.harmonic(np.asarray(y, dtype=np.float32), margin=2.0)
