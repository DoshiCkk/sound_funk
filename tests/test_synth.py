import textwrap
from pathlib import Path

import numpy as np
import pytest

from soundfunc.synth import FormulaError, render

PATCHES = sorted((Path(__file__).parent.parent / "examples" / "patches").glob("*.sfx"))


def dominant_freq(y: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.fft.rfftfreq(len(y), 1 / sr)[spectrum.argmax()])


def test_sine_frequency():
    r = render("sin(2*pi*440*t)", duration=1.0, sr=8000)
    assert len(r.y) == 8000
    assert dominant_freq(r.y, r.sr) == pytest.approx(440, abs=2)
    assert np.abs(r.y).max() == pytest.approx(0.9, abs=1e-3)


def test_note_and_midi_agree():
    a = render('sin(phase(note("A4")))', duration=0.5, sr=8000).y
    b = render("sin(phase(midi(69)))", duration=0.5, sr=8000).y
    assert np.allclose(a, b, atol=1e-5)


def test_phase_of_constant_matches_closed_form():
    a = render("sin(phase(300))", duration=0.5, sr=8000, fade=0).y
    b = render("sin(2*pi*300*t)", duration=0.5, sr=8000, fade=0).y
    assert np.allclose(a, b, atol=1e-4)


def test_multiline_patch_with_params():
    src = textwrap.dedent("""
        # комментарий
        f = base * 2
        out = sin(2*pi*f*t)
    """)
    r = render(src, duration=1.0, sr=8000, params={"base": 220})
    assert dominant_freq(r.y, r.sr) == pytest.approx(440, abs=2)


def test_seq_steps_through_values():
    r = render("seq(0.5, 1, 2, 3)", duration=2.0, sr=100, normalize=False, fade=0)
    # 1, 2, 3, 1 — нормализация выключена, пик 3 > 1 обрезается
    assert r.y[10] == pytest.approx(1.0)
    assert r.y[160] == pytest.approx(1.0)
    assert any("обрезан" in w for w in r.warnings)


def test_nan_replaced_with_warning():
    r = render("sin(t)/t", duration=0.1, sr=1000)
    assert np.isfinite(r.y).all()
    assert r.warnings


def test_noise_is_reproducible_with_seed():
    a = render("noise()", duration=0.1, sr=8000, seed=1).y
    b = render("noise()", duration=0.1, sr=8000, seed=1).y
    assert np.array_equal(a, b)


@pytest.mark.parametrize("src", [
    "__import__('os').system('echo hi')",
    "t.__class__",
    "[x for x in range(3)]",
    "(lambda: 1)()",
    "open('x')",
    "sin(x=t)",
    "t[0]",
    "import os",
])
def test_unsafe_constructs_rejected(src):
    with pytest.raises(FormulaError):
        render(src, duration=0.1, sr=1000)


@pytest.mark.parametrize("src, match", [
    ("sin", "со скобками"),
    ("foo(t)", "неизвестная функция"),
    ("t = 1\nt", "зарезервировано"),
    ("sin(2*pi*440*t", "синтаксическая"),
    ("x = 1", "нет результата"),
])
def test_helpful_errors(src, match):
    with pytest.raises(FormulaError, match=match):
        render(src, duration=0.1, sr=1000)


def test_huge_power_does_not_hang():
    r = render("9**9**9 * 0 + sin(t)", duration=0.1, sr=1000)
    assert np.isfinite(r.y).all()


def test_param_cannot_shadow_time():
    with pytest.raises(FormulaError, match="зарезервированы"):
        render("sin(t)", duration=0.1, sr=1000, params={"t": 1.0})


@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.name)
def test_example_patches_render(patch):
    r = render(patch.read_text(encoding="utf-8"), duration=1.0, sr=22050, params={"f": 440})
    assert np.abs(r.y).max() > 0.1
    assert not r.warnings
