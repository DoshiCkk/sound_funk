"""Перегенерирует картинки и звук для README в docs/.

    uv sync --extra stems          # нужен Demucs: картинки показывают разбор по инструментам
    uv run python scripts/make_docs.py

Демо-трек синтезируется формулой из examples/patches/demo.sfx, поэтому
всё воспроизводимо без сторонних аудиофайлов. Вывод терминала — настоящие
команды CLI, сохранённые в SVG через Rich. Скриншот окна-визуализатора
снимается безголовым Edge/Chrome.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
from rich.console import Console

import soundfunc.cli as cli
import soundfunc.viz as viz

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PATCHES = ROOT / "examples" / "patches"
VIEWER_TIME = 5.1  # момент трека на скриншоте визуализатора


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    cli.err = Console(file=io.StringIO())  # спиннеры статуса в скриншот не нужны
    viz.open_window = lambda path: "msedge"  # окно на этапе сборки документации не открываем

    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        session = _recorder()
        cli.console = session
        _run(session, "sfunc synth -f examples/patches/demo.sfx -d 16 --seed 1 -o demo.wav --plot spectrogram.png",
             lambda: cli.synth(file=PATCHES / "demo.sfx", output=Path("demo.wav"), duration=16, seed=1,
                               plot=Path("spectrogram.png")))
        _run(session, "sfunc view demo.wav --stems",
             lambda: cli.view(path=Path("demo.wav"), stems=True))
        session.save_svg(str(DOCS / "terminal.svg"), title="sfunc")

        rolls = _recorder()
        cli.console = rolls
        _run(rolls, "sfunc roll demo.wav --stems -o roll.png --midi demo.mid",
             lambda: cli.roll(path=Path("demo.wav"), stems=True, output=Path("roll.png"),
                              midi=Path("demo.mid")))
        rolls.save_svg(str(DOCS / "terminal_roll.svg"), title="sfunc roll --stems")

        shutil.copy("spectrogram.png", DOCS / "spectrogram.png")
        shutil.copy("roll.png", DOCS / "roll.png")
        y, sr = sf.read("demo.wav")
        sf.write(DOCS / "demo.mp3", y, sr, format="MP3")  # чтобы послушать с GitHub
        _screenshot(Path("demo.view.html").resolve(), DOCS / "viewer.png")
        os.chdir(ROOT)

    for name in sorted(p.name for p in DOCS.iterdir()):
        print(f"docs/{name}")


def _recorder() -> Console:
    return Console(record=True, width=100, force_terminal=True, color_system="truecolor",
                   file=io.StringIO())


def _run(term: Console, cmd: str, action) -> None:
    term.print(f"[bold green]❯[/] [bold]{cmd}[/]", highlight=False)
    action()
    term.print()


def _screenshot(html: Path, out: Path) -> None:
    browser = viz._find_chromium()
    if not browser:
        print("Edge/Chrome не найден — скриншот визуализатора пропущен")
        return
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([browser, "--headless", "--disable-gpu", "--hide-scrollbars",
                        f"--user-data-dir={profile}", "--window-size=1440,880",
                        "--virtual-time-budget=3000", f"--screenshot={out}",
                        f"{html.as_uri()}#t={VIEWER_TIME}&static"],
                       check=True, capture_output=True, timeout=120)


if __name__ == "__main__":
    main()
