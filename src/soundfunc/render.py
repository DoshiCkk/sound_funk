"""Piano-roll в PNG, осциллограмма/спектрограмма и экспорт нот в MIDI."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mido  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from soundfunc.chords import NO_CHORD  # noqa: E402
from soundfunc.models import Analysis, DrumHit, Note, midi_to_name  # noqa: E402
from soundfunc.palette import TRACK_COLORS, TRACK_LABELS, heat_cmap  # noqa: E402

BLACK_KEYS = {1, 3, 6, 8, 10}
BG = "#0e1015"
BG_BLACK_KEY = "#151922"
FG = "#c9d1d9"
GRID_BEAT = "#1d222b"
GRID_BAR = "#3a4150"
TRACK_ORDER = ("vocals", "guitar", "piano", "other", "bass", "mix")
DRUM_ROWS = ("kick", "snare", "hat")
DRUM_LABELS = {"kick": "бочка", "snare": "малый", "hat": "хэт"}
GM_DRUMS = {"kick": 36, "snare": 38, "hat": 42}


def render_piano_roll(result: Analysis, out_path: str | Path) -> None:
    tracks = {name: notes for name, notes in sorted(
        result.note_tracks().items(), key=lambda kv: TRACK_ORDER.index(kv[0])
        if kv[0] in TRACK_ORDER else len(TRACK_ORDER))}
    ranges = {name: _pitch_range(notes) for name, notes in tracks.items()}
    t_start, t_end = result.offset, result.offset + result.duration

    heights = [0.45] + [max(1.3, 0.1 * (hi - lo + 1)) for lo, hi in ranges.values()]
    if result.drums:
        heights.append(0.75)
    width = min(40.0, max(12.0, result.duration * 0.6))
    fig, axes = plt.subplots(len(heights), 1, sharex=True, figsize=(width, sum(heights) + 0.9),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.08})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        _style_axis(ax)

    _draw_chords(axes[0], result)
    cmap = heat_cmap()
    for ax, (name, notes) in zip(axes[1:], tracks.items()):
        lo, hi = ranges[name]
        _draw_grid(ax, result, rows=range(lo, hi + 1))
        rects = [Rectangle((n.start, n.pitch - 0.42), n.end - n.start, 0.84) for n in notes]
        ax.add_collection(PatchCollection(rects, facecolors=[cmap(n.confidence) for n in notes],
                                          edgecolors="none", zorder=3))
        c_rows = [p for p in range(lo, hi + 1) if p % 12 == 0]
        ax.set_yticks(c_rows, [midi_to_name(p) for p in c_rows])
        ax.set_ylim(lo - 0.5, hi + 0.5)
        ax.set_ylabel(TRACK_LABELS.get(name, name), color=TRACK_COLORS.get(name, FG),
                      fontsize=10, fontweight="bold")

    if result.drums:
        _draw_drums(axes[-1], result, result.drums)

    axes[-1].set_xlim(t_start, t_end)
    axes[-1].set_xlabel("время, с", color=FG)
    axes[0].set_title(
        f"{Path(result.path).name} — {result.key.label}, {result.rhythm.tempo:.0f} BPM, "
        f"{result.rhythm.time_signature}", loc="left", color=FG, fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.025, pad=0.06,
                        aspect=60, anchor=(1.0, 1.0), shrink=0.3)
    cbar.set_label("уверенность ноты: холодно → горячо", color=FG, fontsize=9)
    cbar.ax.tick_params(colors=FG, labelsize=8)
    cbar.outline.set_visible(False)

    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _pitch_range(notes: list[Note]) -> tuple[int, int]:
    if not notes:
        return 48, 72
    lo, hi = min(n.pitch for n in notes) - 2, max(n.pitch for n in notes) + 2
    pad = max(0, 8 - (hi - lo + 1))  # не меньше 8 рядов, чтобы панель читалась
    return lo - pad // 2, hi + pad - pad // 2


def _style_axis(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_grid(ax, result: Analysis, rows) -> None:
    for p in rows:
        if p % 12 in BLACK_KEYS:
            ax.axhspan(p - 0.5, p + 0.5, color=BG_BLACK_KEY, zorder=0, lw=0)
    for b in result.rhythm.beats:
        ax.axvline(b, color=GRID_BEAT, lw=0.6, zorder=1)
    for d in result.rhythm.downbeats:
        ax.axvline(d, color=GRID_BAR, lw=1.0, zorder=1)


def _draw_chords(ax, result: Analysis) -> None:
    palette = plt.get_cmap("tab20")
    for seg in result.chords:
        if seg.label == NO_CHORD:
            continue
        color = palette((2 * seg.root + (seg.quality == "m")) % 20)
        ax.axvspan(seg.start, seg.end, color=color, alpha=0.75, lw=0)
        ax.text((seg.start + seg.end) / 2, 0.5, seg.label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#0b0d12", clip_on=True)
    ax.set_yticks([])
    ax.set_ylim(0, 1)


def _draw_drums(ax, result: Analysis, hits: list[DrumHit]) -> None:
    _draw_grid(ax, result, rows=())
    color = TRACK_COLORS["drums"]
    for row, kind in enumerate(DRUM_ROWS):
        ts = [h.time for h in hits if h.kind == kind]
        alphas = [0.35 + 0.65 * h.strength for h in hits if h.kind == kind]
        for t, a in zip(ts, alphas):
            ax.add_patch(Rectangle((t, row - 0.32), 0.06, 0.64, color=color, alpha=a, lw=0, zorder=3))
    ax.set_yticks(range(len(DRUM_ROWS)), [DRUM_LABELS[k] for k in DRUM_ROWS])
    ax.set_ylim(-0.6, len(DRUM_ROWS) - 0.4)
    ax.set_ylabel(TRACK_LABELS["drums"], color=color, fontsize=10, fontweight="bold")


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


def write_midi(result: Analysis, out_path: str | Path) -> None:
    """Отдельная MIDI-дорожка на инструмент; барабаны — на 10-м канале (General MIDI)."""
    ticks_per_beat = 480
    tempo = mido.bpm2tempo(result.rhythm.tempo)
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, charset="utf-8")  # названия дорожек по-русски

    channel = 0
    for name, notes in result.note_tracks().items():
        events = []
        for n in notes:
            events.append((n.start, 1, n.pitch, max(1, round(n.velocity * 127))))
            events.append((n.end, 0, n.pitch, 0))
        mid.tracks.append(_midi_track(TRACK_LABELS.get(name, name), events, channel, tempo,
                                      ticks_per_beat))
        channel = channel + 1 if channel != 8 else 10  # канал 9 (10-й) — ударные

    if result.drums:
        events = []
        for h in result.drums:
            events.append((h.time, 1, GM_DRUMS[h.kind], max(1, round(h.strength * 127))))
            events.append((h.time + 0.05, 0, GM_DRUMS[h.kind], 0))
        mid.tracks.append(_midi_track(TRACK_LABELS["drums"], events, 9, tempo, ticks_per_beat))

    mid.tracks[0].insert(0, mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    mid.save(str(out_path))


def _midi_track(name: str, events: list, channel: int, tempo: int, ticks_per_beat: int) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    events.sort()  # при равном времени note_off (0) раньше note_on (1)
    last_tick = 0
    for t, is_on, pitch, velocity in events:
        tick = round(mido.second2tick(t, ticks_per_beat, tempo))
        kind = "note_on" if is_on else "note_off"
        track.append(mido.Message(kind, note=pitch, velocity=velocity, channel=channel,
                                  time=max(0, tick - last_tick)))
        last_tick = max(last_tick, tick)
    return track
