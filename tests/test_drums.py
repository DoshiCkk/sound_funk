import numpy as np
import pytest
from scipy import signal

from soundfunc.drums import detect_drums

SR = 22050
BEAT = 0.5


def _kick(dur=0.45):
    t = np.arange(int(dur * SR)) / SR
    f = 45 + 160 * np.exp(-35 * t)
    return np.tanh(2 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-9 * t))


def _snare(rng, dur=0.25):
    t = np.arange(int(dur * SR)) / SR
    sos = signal.butter(2, [200, 8000], btype="band", fs=SR, output="sos")
    noise = signal.sosfilt(sos, rng.uniform(-1, 1, len(t)))
    return 0.7 * (noise * 2 + 0.5 * np.sin(2 * np.pi * 190 * t)) * np.exp(-18 * t)


def _hat(rng, dur=0.08):
    t = np.arange(int(dur * SR)) / SR
    sos = signal.butter(4, 7000, btype="high", fs=SR, output="sos")
    return 0.25 * signal.sosfilt(sos, rng.uniform(-1, 1, len(t))) * np.exp(-60 * t)


def _place(y, x, at):
    i = int(at * SR)
    n = min(len(x), len(y) - i)
    y[i:i + n] += x[:n]


@pytest.fixture(scope="module")
def kit():
    """4 такта: бочка на 1 и 3, малый на 2 и 4, хэт на каждую восьмую."""
    rng = np.random.default_rng(0)
    bars = 4
    y = np.zeros(int(bars * 4 * BEAT * SR) + SR)
    expected = {"kick": [], "snare": [], "hat": []}
    for beat in range(bars * 4):
        t = beat * BEAT
        if beat % 2 == 0:
            _place(y, _kick(), t)
            expected["kick"].append(t)
        else:
            _place(y, _snare(rng), t)
            expected["snare"].append(t)
    for eighth in range(bars * 8):
        t = eighth * BEAT / 2
        _place(y, _hat(rng), t)
        expected["hat"].append(t)
    return y / np.abs(y).max(), expected


@pytest.mark.parametrize("kind", ["kick", "snare", "hat"])
def test_drum_kinds_found(kit, kind):
    y, expected = kit
    found = [h.time for h in detect_drums(y, SR) if h.kind == kind]
    hits = sum(any(abs(f - e) < 0.04 for f in found) for e in expected[kind])
    false = sum(not any(abs(f - e) < 0.04 for e in expected[kind]) for f in found)
    assert hits / len(expected[kind]) >= 0.8, f"{kind}: найдено {hits}/{len(expected[kind])}"
    assert false <= 0.2 * len(expected[kind]), f"{kind}: {false} лишних"


def test_saturated_kick_is_not_snare():
    """Гармоники перегруженной бочки в середине спектра — не малый барабан."""
    y = np.zeros(SR * 4)
    for k in range(8):
        _place(y, _kick(), k * BEAT)
    kinds = [h.kind for h in detect_drums(y, SR)]
    assert kinds.count("kick") >= 7
    assert kinds.count("snare") == 0


def test_silence_has_no_hits():
    assert detect_drums(np.zeros(SR), SR) == []
