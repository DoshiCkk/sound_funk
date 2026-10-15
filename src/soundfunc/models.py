"""Результаты анализа. Все времена — секунды от начала файла."""

from __future__ import annotations

from dataclasses import dataclass, field

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def midi_to_name(pitch: int) -> str:
    """60 -> 'C4'."""
    return f"{PITCH_NAMES[pitch % 12]}{pitch // 12 - 1}"


@dataclass
class Note:
    pitch: int  # MIDI-номер
    start: float
    end: float
    velocity: float  # 0..1

    @property
    def name(self) -> str:
        return midi_to_name(self.pitch)


@dataclass
class ChordSegment:
    label: str  # "C", "Am", "N" (нет аккорда)
    start: float
    end: float
    root: int | None = None  # pitch class 0..11
    quality: str = ""  # "" — мажор, "m" — минор


@dataclass
class Rhythm:
    tempo: float  # BPM
    beats_per_bar: int
    meter_confidence: float  # 0..1
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)

    @property
    def time_signature(self) -> str:
        return f"{self.beats_per_bar}/4"


@dataclass
class Key:
    tonic: str
    mode: str  # "major" | "minor"
    confidence: float  # корреляция с профилем, -1..1
    alternatives: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.tonic} {'major' if self.mode == 'major' else 'minor'}"


@dataclass
class Analysis:
    path: str
    offset: float
    duration: float
    rhythm: Rhythm
    key: Key
    chords: list[ChordSegment]
    notes: list[Note] = field(default_factory=list)
