"""Распознавание аккордов: шаблоны трезвучий по хромаграмме + сглаживание Витерби."""

from __future__ import annotations

import librosa
import numpy as np

from soundfunc.models import PITCH_NAMES, ChordSegment

QUALITIES = {"": (0, 4, 7), "m": (0, 3, 7)}
NO_CHORD = "N"


def _templates() -> tuple[np.ndarray, list[tuple[str, int | None, str]]]:
    vectors, meta = [], []
    for quality, intervals in QUALITIES.items():
        for root in range(12):
            t = np.zeros(12)
            t[[(root + i) % 12 for i in intervals]] = 1.0
            vectors.append(t / np.linalg.norm(t))
            meta.append((PITCH_NAMES[root] + quality, root, quality))
    meta.append((NO_CHORD, None, ""))
    return np.array(vectors), meta


TEMPLATES, TEMPLATE_META = _templates()


def recognize_chords(chroma: np.ndarray, rms: np.ndarray, sr: int, hop: int, *,
                     beats: list[float] | None = None, t0: float = 0.0,
                     duration: float | None = None, p_self: float = 0.92,
                     sharpness: float = 25.0, min_dur: float = 0.4) -> list[ChordSegment]:
    n = min(chroma.shape[1], len(rms))
    chroma, rms = chroma[:, :n], rms[:n]

    norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    scores = TEMPLATES @ norm
    # «нет аккорда» выигрывает, когда хрома размазана по всем нотам или тишина
    no_chord = 0.9 * norm.sum(axis=0) / np.sqrt(12)
    silent = librosa.amplitude_to_db(rms, ref=np.max) < -40
    no_chord[silent] = 2.0
    scores = np.vstack([scores, no_chord])

    prob = np.exp(sharpness * (scores - scores.max(axis=0, keepdims=True)))
    prob /= prob.sum(axis=0, keepdims=True)
    path = librosa.sequence.viterbi(prob, librosa.sequence.transition_loop(len(TEMPLATE_META), p_self))

    times = librosa.frames_to_time(np.arange(n + 1), sr=sr, hop_length=hop)
    end_time = duration if duration is not None else float(times[-1])
    segments = _runs(path, times, end_time)
    segments = _absorb_short(segments, min_dur)
    if beats:
        segments = _snap_to_beats(segments, np.asarray(beats) - t0)
    for s in segments:
        s.start, s.end = round(s.start + t0, 3), round(s.end + t0, 3)
    return segments


def _runs(path: np.ndarray, times: np.ndarray, end_time: float) -> list[ChordSegment]:
    segments = []
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            label, root, quality = TEMPLATE_META[path[start]]
            end = end_time if i == len(path) else float(times[i])
            segments.append(ChordSegment(label, float(times[start]), end, root, quality))
            start = i
    return segments


def _absorb_short(segments: list[ChordSegment], min_dur: float) -> list[ChordSegment]:
    """Слишком короткие сегменты отдаём предыдущему соседу."""
    out: list[ChordSegment] = []
    for s in segments:
        if out and (s.end - s.start < min_dur or s.label == out[-1].label):
            out[-1].end = s.end
        else:
            out.append(s)
    if len(out) > 1 and out[0].end - out[0].start < min_dur:
        out[1].start = out[0].start
        out.pop(0)
    return out


def _snap_to_beats(segments: list[ChordSegment], beats: np.ndarray) -> list[ChordSegment]:
    if len(beats) < 2:
        return segments
    tolerance = 0.5 * float(np.median(np.diff(beats)))
    for prev, cur in zip(segments, segments[1:]):
        nearest = beats[np.argmin(np.abs(beats - cur.start))]
        if abs(nearest - cur.start) < tolerance:
            prev.end = cur.start = float(nearest)
    segments = [s for s in segments if s.end - s.start > 1e-6]
    merged: list[ChordSegment] = []
    for s in segments:
        if merged and merged[-1].label == s.label:
            merged[-1].end = s.end
        else:
            merged.append(s)
    return merged
