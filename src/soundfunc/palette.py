"""Тепловая палитра «как в тепловизоре»: чем увереннее нота, тем она горячее.

Одни и те же цвета используют терминальный piano-roll, PNG и окно-визуализатор
(viewer.html получает их через JSON).
"""

from __future__ import annotations

# (позиция 0..1, цвет): холодный синий → фиолетовый → красный → оранжевый → жёлтый → белый
HEAT_STOPS: list[tuple[float, str]] = [
    (0.00, "#2440ff"),
    (0.25, "#8e2de2"),
    (0.50, "#ec3050"),
    (0.70, "#ff7b1c"),
    (0.85, "#ffd23f"),
    (1.00, "#fff8e1"),
]

TRACK_LABELS = {
    "mix": "Микс",
    "bass": "Бас",
    "piano": "Пианино",
    "guitar": "Гитара",
    "vocals": "Вокал",
    "other": "Другое",
    "drums": "Барабаны",
}

# цвета инструментов — для режима «по инструментам» в визуализаторе
TRACK_COLORS = {
    "mix": "#7dd3fc",
    "bass": "#f472b6",
    "piano": "#a78bfa",
    "guitar": "#34d399",
    "vocals": "#fbbf24",
    "other": "#60a5fa",
    "drums": "#f87171",
}


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def heat_rgb(x: float) -> tuple[int, int, int]:
    """Цвет для уверенности x ∈ [0, 1]."""
    x = min(1.0, max(0.0, x))
    for (p0, c0), (p1, c1) in zip(HEAT_STOPS, HEAT_STOPS[1:]):
        if x <= p1:
            k = (x - p0) / (p1 - p0)
            a, b = _hex_to_rgb(c0), _hex_to_rgb(c1)
            return tuple(round(u + (v - u) * k) for u, v in zip(a, b))  # type: ignore[return-value]
    return _hex_to_rgb(HEAT_STOPS[-1][1])


def heat_hex(x: float) -> str:
    return "#%02x%02x%02x" % heat_rgb(x)


def heat_cmap():
    """Та же палитра как matplotlib-колормап."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("heat", HEAT_STOPS)
