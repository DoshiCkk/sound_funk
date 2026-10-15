"""Piano-roll прямо в терминале.

Каждая строка текста — две ноты по высоте: символ «▀» красится цветом
верхней ноты, а фон — цветом нижней. Цвет ноты — тепловая шкала уверенности.
"""

from __future__ import annotations

import math

import numpy as np
from rich.style import Style
from rich.text import Text

from soundfunc.models import PITCH_NAMES, Note
from soundfunc.palette import heat_rgb

BG = (22, 24, 29)
BG_BLACK_KEY = (28, 31, 38)
BG_BAR = (52, 57, 68)
LABEL_W = 5


def terminal_roll(notes: list[Note], t_start: float, t_end: float, *, width: int = 80,
                  max_rows: int = 18, downbeats: list[float] | None = None) -> list[Text]:
    if not notes or t_end <= t_start:
        return [Text("  (нот не найдено)", style="dim")]
    cols = max(10, width - LABEL_W - 1)
    lo = min(n.pitch for n in notes)
    hi = max(n.pitch for n in notes)
    per_px = max(1, math.ceil((hi - lo + 1) / (2 * max_rows)))  # полутонов на «пиксель»
    n_px = math.ceil((hi - lo + 1) / per_px)
    n_px += n_px % 2
    dt = (t_end - t_start) / cols

    heat = np.full((n_px, cols), -1.0)
    for n in notes:
        px = (n.pitch - lo) // per_px
        c0 = int((n.start - t_start) / dt)
        c1 = max(c0 + 1, math.ceil((n.end - t_start) / dt))
        c0, c1 = max(0, c0), min(cols, c1)
        if c0 < c1:
            heat[px, c0:c1] = np.maximum(heat[px, c0:c1], n.confidence)

    bar_cols = {int((d - t_start) / dt) for d in (downbeats or []) if t_start <= d < t_end}

    def color(px: int, col: int) -> tuple[int, int, int]:
        if heat[px, col] >= 0:
            return heat_rgb(heat[px, col])
        if col in bar_cols:
            return BG_BAR
        if per_px == 1 and (lo + px) % 12 in (1, 3, 6, 8, 10):
            return BG_BLACK_KEY
        return BG

    lines = []
    for top in range(n_px - 1, 0, -2):  # сверху вниз, по два пикселя на строку
        upper, lower = top, top - 1
        label = _label(lo, per_px, lower, upper)
        line = Text(f"{label:>{LABEL_W}} ", style="dim")
        for col in range(cols):
            fg, bg = color(upper, col), color(lower, col)
            line.append("▀", Style(color=f"rgb{fg}", bgcolor=f"rgb{bg}"))
        lines.append(line)

    axis = Text(" " * (LABEL_W + 1) + f"{t_start:.1f} с", style="dim")
    tail = f"{t_end:.1f} с"
    axis.append(" " * max(1, cols - len(axis.plain) + LABEL_W + 1 - len(tail)) + tail)
    lines.append(axis)
    lines.append(heat_legend())
    return lines


def heat_legend(width: int = 20) -> Text:
    legend = Text(" " * (LABEL_W + 1) + "уверенность: холодно ", style="dim")
    for i in range(width):
        legend.append("█", Style(color=f"rgb{heat_rgb(i / (width - 1))}"))
    legend.append(" горячо", style="dim")
    return legend


def _label(lo: int, per_px: int, *pixels: int) -> str:
    for px in pixels:
        for pitch in range(lo + px * per_px, lo + (px + 1) * per_px):
            if pitch % 12 == 0:
                return f"{PITCH_NAMES[0]}{pitch // 12 - 1}"
    return ""
