"""Человекочитаемый вывод разбора в терминал."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from soundfunc.chords import NO_CHORD
from soundfunc.models import Analysis, ChordSegment


def chords_by_bar(chords: list[ChordSegment], downbeats: list[float],
                  end: float) -> list[list[str]]:
    """Аккорды, звучащие в каждом такте (занимающие >25% такта)."""
    bounds = list(downbeats) + [end]
    bars = []
    for a, b in zip(bounds, bounds[1:]):
        labels: list[str] = []
        for seg in chords:
            overlap = min(seg.end, b) - max(seg.start, a)
            if seg.label != NO_CHORD and overlap > 0.25 * (b - a) and seg.label not in labels[-1:]:
                labels.append(seg.label)
        bars.append(labels)
    return bars


def print_report(result: Analysis, console: Console) -> None:
    r, k = result.rhythm, result.key
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("Файл", result.path)
    summary.add_row("Длительность", f"{result.duration:.1f} c")
    summary.add_row("Темп", f"{r.tempo:.1f} BPM")
    summary.add_row("Размер", f"{r.time_signature}  (уверенность {r.meter_confidence:.0%})")
    summary.add_row("Тональность", f"{k.label}  (r={k.confidence:.2f}; ещё: {', '.join(k.alternatives)})")
    console.print(summary)

    if not r.downbeats:
        return
    bars = chords_by_bar(result.chords, r.downbeats, result.offset + result.duration)
    console.print("\n[bold cyan]Аккорды по тактам[/]")
    per_line = 4
    for i in range(0, len(bars), per_line):
        cells = [" ".join(b) if b else "—" for b in bars[i:i + per_line]]
        line = "| " + " | ".join(f"{c:<8}" for c in cells) + " |"
        console.print(f"[dim]{i + 1:>4}[/] {line}", highlight=False)
