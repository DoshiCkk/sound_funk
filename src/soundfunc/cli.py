from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from soundfunc import __version__

# На Windows при выводе в файл/пайп stdout по умолчанию в cp1251 — `--json > x.json` должен быть UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    help="sfunc — разбор музыки (темп, размер, тональность, аккорды, piano-roll) "
         "и синтез звука из математических формул.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err = Console(stderr=True)

AudioPath = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Аудиофайл: wav, mp3, flac, ogg...")]
Start = Annotated[float, typer.Option("--start", "-s", help="Начало фрагмента, с")]
Duration = Annotated[Optional[float], typer.Option("--duration", "-d", help="Длина фрагмента, с")]


def _version(value: bool) -> None:
    if value:
        console.print(f"sfunc {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True,
                                          help="Показать версию")] = False,
) -> None:
    pass


@app.command()
def analyze(path: AudioPath, start: Start = 0.0, duration: Duration = None,
            as_json: Annotated[bool, typer.Option("--json", help="Вывод в JSON")] = False) -> None:
    """Темп, размер такта, тональность и аккорды по тактам."""
    from soundfunc.analysis import analyze as run
    from soundfunc.audio import load
    from soundfunc.report import print_report

    with err.status("Анализирую..."):
        result = run(load(path, offset=start, duration=duration))
    if as_json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print_report(result, console)


@app.command()
def roll(path: AudioPath, start: Start = 0.0, duration: Duration = None,
         output: Annotated[Optional[Path], typer.Option("--output", "-o", help="PNG (по умолчанию <файл>.roll.png)")] = None,
         midi: Annotated[Optional[Path], typer.Option("--midi", help="Сохранить ноты в MIDI")] = None) -> None:
    """Ноты в виде piano-roll (PNG) с аккордами и сеткой тактов; опционально MIDI."""
    from soundfunc.analysis import analyze as run
    from soundfunc.audio import load
    from soundfunc.render import render_piano_roll, write_midi

    with err.status("Распознаю ноты..."):
        result = run(load(path, offset=start, duration=duration), with_notes=True)
    out = output or path.with_name(path.stem + ".roll.png")
    render_piano_roll(result, out)
    console.print(f"Piano-roll: [bold]{out}[/] ({len(result.notes)} нот)")
    if midi:
        write_midi(result.notes, midi, tempo=result.rhythm.tempo)
        console.print(f"MIDI: [bold]{midi}[/]")


@app.command()
def synth(
    formula: Annotated[Optional[str], typer.Argument(help="Формула от t, напр. \"sin(2*pi*440*t)*exp(-3*t)\"")] = None,
    file: Annotated[Optional[Path], typer.Option("--file", "-f", exists=True, dir_okay=False, help="Взять формулу из файла (.sfx)")] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Куда сохранить WAV")] = Path("out.wav"),
    duration: Annotated[float, typer.Option("--duration", "-d", help="Длительность, с")] = 2.0,
    sr: Annotated[int, typer.Option("--sr", help="Частота дискретизации")] = 44100,
    param: Annotated[Optional[list[str]], typer.Option("--param", "-p", help="Параметр формулы: name=value (можно несколько)")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed", help="Seed для noise()")] = None,
    no_normalize: Annotated[bool, typer.Option("--no-normalize", help="Не нормализовать громкость")] = False,
    plot: Annotated[Optional[Path], typer.Option("--plot", help="Сохранить осциллограмму и спектрограмму в PNG")] = None,
) -> None:
    """Синтез звука из математической формулы. Список функций: sfunc functions."""
    import soundfile as sf

    from soundfunc.synth import FormulaError, render

    if (formula is None) == (file is None):
        raise typer.BadParameter("укажите формулу аргументом или файл через --file (что-то одно)")
    source = formula if formula is not None else file.read_text(encoding="utf-8")
    params = _parse_params(param or [])

    try:
        result = render(source, duration=duration, sr=sr, params=params, seed=seed,
                        normalize=not no_normalize)
    except FormulaError as exc:
        err.print(f"[red]Ошибка в формуле:[/] {exc}")
        raise typer.Exit(1)
    for w in result.warnings:
        err.print(f"[yellow]Внимание:[/] {w}")

    sf.write(str(output), result.y, result.sr, subtype="PCM_16")
    console.print(f"WAV: [bold]{output}[/] ({duration:g} c, {sr} Гц)")
    if plot:
        from soundfunc.render import render_waveform
        render_waveform(result.y, result.sr, plot, title=source.strip().splitlines()[-1][:100])
        console.print(f"График: [bold]{plot}[/]")


@app.command()
def functions() -> None:
    """Функции и переменные, доступные в формулах synth."""
    from soundfunc.synth import FUNCTIONS, VARIABLES_HELP

    table = Table(title="Функции", show_header=False, box=None, padding=(0, 2))
    for f in FUNCTIONS.values():
        table.add_row(f"[bold cyan]{f.signature}[/]", f.help)
    console.print(table)
    variables = Table(title="Переменные", show_header=False, box=None, padding=(0, 2))
    for name, text in VARIABLES_HELP.items():
        variables.add_row(f"[bold cyan]{name}[/]", text)
    console.print(variables)


def _parse_params(items: list[str]) -> dict[str, float]:
    params = {}
    for item in items:
        name, sep, value = item.partition("=")
        try:
            if not sep or not name.strip().isidentifier():
                raise ValueError
            params[name.strip()] = float(value)
        except ValueError:
            raise typer.BadParameter(f"ожидается name=число, получено {item!r}", param_hint="--param")
    return params
