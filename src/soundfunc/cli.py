from __future__ import annotations

import json
import os
import subprocess
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
    help="sfunc — разбор музыки (темп, размер, тональность, аккорды, ноты по инструментам) "
         "и синтез звука из математических формул.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err = Console(stderr=True)

STEM_NAMES = ("drums", "bass", "other", "vocals", "guitar", "piano")

AudioPath = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Аудиофайл: wav, mp3, flac, ogg...")]
Start = Annotated[float, typer.Option("--start", "-s", help="Начало фрагмента, с")]
Duration = Annotated[Optional[float], typer.Option("--duration", "-d", help="Длина фрагмента, с")]
Stem = Annotated[Optional[str], typer.Option(
    "--stem", help="Разобрать одну дорожку: bass, piano, guitar, vocals, other, drums (нужен Demucs)")]
Stems = Annotated[bool, typer.Option("--stems", help="Разделить на инструменты и показать каждый отдельно (нужен Demucs)")]


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
def analyze(path: AudioPath, start: Start = 0.0, duration: Duration = None, stem: Stem = None,
            as_json: Annotated[bool, typer.Option("--json", help="Вывод в JSON")] = False) -> None:
    """Темп, размер такта, тональность и аккорды по тактам."""
    from soundfunc.analysis import analyze as run
    from soundfunc.audio import load
    from soundfunc.report import print_report

    source = _stem_path(path, stem) if stem else path
    with err.status("Анализирую..."):
        result = run(load(source, offset=start, duration=duration))
    if as_json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print_report(result, console)


@app.command()
def roll(path: AudioPath, start: Start = 0.0, duration: Duration = None, stem: Stem = None,
         stems: Stems = False,
         output: Annotated[Optional[Path], typer.Option("--output", "-o", help="PNG (по умолчанию <файл>.roll.png)")] = None,
         midi: Annotated[Optional[Path], typer.Option("--midi", help="Сохранить ноты в MIDI")] = None,
         open_png: Annotated[bool, typer.Option("--open", help="Открыть картинку после сохранения")] = False,
         preview: Annotated[bool, typer.Option("--preview/--no-preview", help="Piano-roll прямо в терминале")] = True) -> None:
    """Ноты в виде piano-roll (PNG + превью в терминале) с аккордами и сеткой тактов; опционально MIDI."""
    from soundfunc.render import render_piano_roll, write_midi

    result = _analyze_notes(path, start, duration, stem=stem, stems=stems)
    out = output or path.with_name(path.stem + ".roll.png")
    render_piano_roll(result, out)
    if preview:
        _print_terminal_roll(result)
    total = sum(len(n) for n in result.note_tracks().values())
    console.print(f"Piano-roll: [bold]{out}[/] ({total} нот)")
    if midi:
        write_midi(result, midi)
        console.print(f"MIDI: [bold]{midi}[/]")
    if open_png:
        _open_file(out)


@app.command()
def view(path: AudioPath, start: Start = 0.0, duration: Duration = None, stems: Stems = False,
         output: Annotated[Optional[Path], typer.Option("--output", "-o", help="HTML (по умолчанию <файл>.view.html)")] = None,
         open_window: Annotated[bool, typer.Option("--open/--no-open", help="Открыть окно визуализатора")] = True) -> None:
    """Разбор в терминале + окно-визуализатор: ноты светятся по уверенности, как в тепловизоре."""
    from soundfunc.report import print_report
    from soundfunc.viz import build_viewer, open_window as open_viewer

    result = _analyze_notes(path, start, duration, stems=stems)
    print_report(result, console)
    _print_track_summary(result)
    out = output or path.with_name(path.stem + ".view.html")
    with err.status("Собираю визуализатор..."):
        build_viewer(result, out, audio_path=path)
    console.print(f"\nВизуализатор: [bold]{out}[/]")
    if open_window:
        how = open_viewer(out)
        console.print(f"[dim]Открыто в окне: {how}. Пробел — пауза, колесо — масштаб, H — цвет.[/]")


@app.command()
def stems(path: AudioPath,
          output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Папка для дорожек (по умолчанию <файл>_stems)")] = None) -> None:
    """Разделить трек на барабаны, бас, гитару, пианино, вокал и остальное (Demucs htdemucs_6s)."""
    import shutil

    from soundfunc.palette import TRACK_LABELS
    from soundfunc.stems import stem_loudness

    paths = _separate(path)
    out_dir = output or path.with_name(path.stem + "_stems")
    out_dir.mkdir(parents=True, exist_ok=True)
    loudness = stem_loudness(paths)
    table = Table(title="Дорожки", show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("инструмент")
    table.add_column("громкость", justify="right")
    table.add_column("файл")
    for name, src in paths.items():
        dst = out_dir / src.name
        shutil.copy(src, dst)
        db = loudness[name]
        style = "dim" if db < -30 else ""
        table.add_row(TRACK_LABELS.get(name, name), f"{db:+.1f} дБ", str(dst), style=style)
    console.print(table)
    console.print("[dim]Тише −30 дБ — дорожка почти пустая: такого инструмента в треке, скорее всего, нет.[/]")


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
    open_files: Annotated[bool, typer.Option("--open", help="Сразу проиграть WAV (и открыть график, если есть)")] = False,
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
    if open_files:
        if plot:
            _open_file(plot)
        _open_file(output)


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


def _analyze_notes(path: Path, start: float, duration: Optional[float], *,
                   stem: Optional[str] = None, stems: bool = False):
    from soundfunc.analysis import analyze as run
    from soundfunc.audio import load

    if stem and stems:
        raise typer.BadParameter("--stem и --stems вместе не используются")
    stem_paths = _separate(path) if stems else None
    source = _stem_path(path, stem) if stem else path
    with err.status("Распознаю ноты по инструментам..." if stems else "Распознаю ноты..."):
        return run(load(source, offset=start, duration=duration), with_notes=True, stems=stem_paths)


def _separate(path: Path) -> dict[str, Path]:
    from soundfunc.stems import StemsUnavailable, separate

    try:
        with err.status("Разделяю на инструменты...") as status:
            return separate(path, on_status=status.update)
    except StemsUnavailable as exc:
        err.print(f"[red]Нет Demucs:[/] {exc}")
        raise typer.Exit(1)


def _stem_path(path: Path, stem: str) -> Path:
    if stem not in STEM_NAMES:
        raise typer.BadParameter(f"неизвестная дорожка {stem!r}; есть: {', '.join(STEM_NAMES)}",
                                 param_hint="--stem")
    return _separate(path)[stem]


def _print_terminal_roll(result) -> None:
    from soundfunc.palette import TRACK_COLORS, TRACK_LABELS
    from soundfunc.termroll import terminal_roll

    t0, t1 = result.offset, result.offset + result.duration
    for name, notes in result.note_tracks().items():
        if len(result.note_tracks()) > 1:
            console.print(f"[bold {TRACK_COLORS.get(name, 'white')}]{TRACK_LABELS.get(name, name)}[/]")
        for line in terminal_roll(notes, t0, t1, width=min(console.width, 120),
                                  max_rows=10 if len(result.note_tracks()) > 1 else 16,
                                  downbeats=result.rhythm.downbeats):
            console.print(line, no_wrap=True, crop=True, highlight=False)


def _print_track_summary(result) -> None:
    from soundfunc.palette import TRACK_LABELS

    tracks = result.note_tracks()
    if not tracks and not result.drums:
        return
    console.print()
    for name, notes in tracks.items():
        if not notes:
            continue
        mean_conf = sum(n.confidence for n in notes) / len(notes)
        console.print(f"[bold cyan]{TRACK_LABELS.get(name, name):<12}[/] {len(notes):>4} нот, "
                      f"средняя уверенность {mean_conf:.0%}", highlight=False)
    if result.drums:
        kinds = {k: sum(1 for h in result.drums if h.kind == k) for k in ("kick", "snare", "hat")}
        console.print(f"[bold cyan]{'Барабаны':<12}[/] бочка {kinds['kick']}, малый {kinds['snare']}, "
                      f"хэт {kinds['hat']}", highlight=False)


def _open_file(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


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
