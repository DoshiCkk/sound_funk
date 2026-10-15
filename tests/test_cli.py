import json

import soundfile as sf
from typer.testing import CliRunner

from soundfunc.cli import app

runner = CliRunner()


def test_synth_writes_wav(tmp_path):
    out = tmp_path / "beep.wav"
    res = runner.invoke(app, ["synth", "sin(2*pi*440*t) * exp(-3*t)", "-d", "0.5", "-o", str(out)])
    assert res.exit_code == 0, res.output
    y, sr = sf.read(out)
    assert sr == 44100 and len(y) == 22050


def test_synth_reports_formula_error(tmp_path):
    res = runner.invoke(app, ["synth", "foo(t)", "-o", str(tmp_path / "x.wav")])
    assert res.exit_code == 1
    assert not (tmp_path / "x.wav").exists()


def test_synth_bad_param(tmp_path):
    res = runner.invoke(app, ["synth", "sin(t)", "-p", "f440", "-o", str(tmp_path / "x.wav")])
    assert res.exit_code != 0


def test_analyze_and_roll_on_synthesized_file(tmp_path):
    wav, png, mid = tmp_path / "c.wav", tmp_path / "c.png", tmp_path / "c.mid"
    chord = 'seq(2, 1, 0) * (sin(phase(note("C4"))) + sin(phase(note("E4"))) + sin(phase(note("G4"))))'
    assert runner.invoke(app, ["synth", chord, "-d", "4", "-o", str(wav)]).exit_code == 0

    res = runner.invoke(app, ["analyze", str(wav), "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["key"]["tonic"] == "C"
    assert data["chords"][0]["label"] == "C"

    res = runner.invoke(app, ["roll", str(wav), "-o", str(png), "--midi", str(mid)])
    assert res.exit_code == 0, res.output
    assert png.stat().st_size > 0 and mid.stat().st_size > 0


def test_functions_lists_reference():
    res = runner.invoke(app, ["functions"])
    assert res.exit_code == 0
    assert "phase(f)" in res.output
