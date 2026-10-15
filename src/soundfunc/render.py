"""Piano-roll в PNG и экспорт нот в MIDI."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mido  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from soundfunc.chords import NO_CHORD  # noqa: E402
from soundfunc.models import Analysis, Note, midi_to_name  # noqa: E402

BLACK_KEYS = {1, 3, 6, 8, 10}


def render_piano_roll(result: Analysis, out_path: str | Path) -> None:
    notes, rhythm = result.notes, result.rhythm
    t_start, t_end = result.offset, result.offset + result.duration
    if notes:
        lo, hi = min(n.pitch for n in notes) - 2, max(n.pitch for n in notes) + 2
    else:
        lo, hi = 48, 84

    width = min(40.0, max(10.0, result.duration * 0.5))
    fig, (ax_ch, ax) = plt.subplots(
        2, 1, sharex=True, figsize=(width, 7),
        gridspec_kw={"height_ratios": [1, 12], "hspace": 0.05})

    for p in range(lo, hi + 1):
        if p % 12 in BLACK_KEYS:
            ax.axhspan(p - 0.5, p + 0.5, color="#eef0f3", zorder=0, lw=0)
    for b in rhythm.beats:
        ax.axvline(b, color="#d5d8dd", lw=0.5, zorder=1)
    for d in rhythm.downbeats:
        ax.axvline(d, color="#8a9099", lw=1.0, zorder=1)

    cmap = plt.get_cmap("viridis")
    rects = [Rectangle((n.start, n.pitch - 0.4), n.end - n.start, 0.8) for n in notes]
    ax.add_collection(PatchCollection(
        rects, facecolors=[cmap(0.25 + 0.7 * n.velocity) for n in notes],
        edgecolors="none", zorder=2))

    c_rows = [p for p in range(lo, hi + 1) if p % 12 == 0]
    ax.set_yticks(c_rows, [midi_to_name(p) for p in c_rows])
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_xlim(t_start, t_end)
    ax.set_xlabel("время, с")

    palette = plt.get_cmap("tab20")
    for seg in result.chords:
        if seg.label == NO_CHORD:
            continue
        color = palette((2 * seg.root + (seg.quality == "m")) % 20)
        ax_ch.axvspan(seg.start, seg.end, color=color, alpha=0.8, lw=0)
        ax_ch.text((seg.start + seg.end) / 2, 0.5, seg.label, ha="center", va="center",
                   fontsize=9, fontweight="bold", clip_on=True)
    ax_ch.set_yticks([])
    ax_ch.set_ylim(0, 1)
    ax_ch.set_title(
        f"{Path(result.path).name} — {result.key.label}, {rhythm.tempo:.0f} BPM, "
        f"{rhythm.time_signature}", loc="left")

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_waveform(y, sr: int, out_path: str | Path, *, title: str = "") -> None:
    """Осциллограмма + спектрограмма (для синтезированного звука)."""
    import librosa
    import librosa.display
    import numpy as np

    fig, (ax_w, ax_s) = plt.subplots(2, 1, sharex=True, figsize=(12, 6),
                                     gridspec_kw={"height_ratios": [1, 2]})
    librosa.display.waveshow(y, sr=sr, ax=ax_w, color="#3b6ea8")
    ax_w.set_title(title, loc="left", fontsize=10)
    ax_w.set_xlabel("")
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="log", ax=ax_s, cmap="magma")
    fig.colorbar(img, ax=ax_s, format="%+.0f dB", pad=0.01)
    ax_s.set_xlabel("время, с")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_midi(notes: list[Note], out_path: str | Path, *, tempo: float = 120.0) -> None:
    ticks_per_beat = 480
    midi_tempo = mido.bpm2tempo(tempo)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=midi_tempo, time=0))

    events = []
    for n in notes:
        events.append((n.start, 1, n.pitch, max(1, round(n.velocity * 127))))
        events.append((n.end, 0, n.pitch, 0))
    events.sort()  # при равном времени note_off (0) раньше note_on (1)

    last_tick = 0
    for t, is_on, pitch, velocity in events:
        tick = round(mido.second2tick(t, ticks_per_beat, midi_tempo))
        kind = "note_on" if is_on else "note_off"
        track.append(mido.Message(kind, note=pitch, velocity=velocity, time=tick - last_tick))
        last_tick = tick

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    mid.tracks.append(track)
    mid.save(str(out_path))
