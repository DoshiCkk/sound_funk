"""Синтез звука из математических формул.

Формула — выражение от времени t (секунды). Можно писать несколько строк
с присваиваниями и комментариями; результат — последнее выражение
(или переменная out):

    f = note("A3") + 3 * sin(2*pi*5*t)     # вибрато
    sin(phase(f)) * adsr(0.01, 0.2, 0.6, 0.5)

Формулы не выполняются через eval: дерево разбирается вручную и разрешены
только числа, арифметика, сравнения, переменные и функции из FUNCTIONS.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Callable

import librosa
import numpy as np
from scipy import signal

MAX_DURATION = 300.0
MAX_SOURCE_LEN = 10_000
DEFAULT_SR = 44100


class FormulaError(ValueError):
    pass


@dataclass
class SynthResult:
    y: np.ndarray
    sr: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Ctx:
    t: np.ndarray
    sr: int
    duration: float
    rng: np.random.Generator


@dataclass
class Func:
    impl: Callable  # первым аргументом получает _Ctx
    signature: str
    help: str


def _scalar(x, name: str) -> float:
    arr = np.asarray(x)
    if arr.ndim != 0:
        raise FormulaError(f"{name}: ожидается число, а не сигнал")
    return float(arr)


def _phase(ctx: _Ctx, f) -> np.ndarray:
    f = np.broadcast_to(np.asarray(f, dtype=float), ctx.t.shape)
    return 2 * np.pi * np.concatenate([[0.0], np.cumsum(f[:-1])]) / ctx.sr


def _saw(_, x):
    return 2.0 * np.mod(np.asarray(x) / (2 * np.pi), 1.0) - 1.0


def _square(_, x, duty=0.5):
    return np.where(np.mod(np.asarray(x) / (2 * np.pi), 1.0) < duty, 1.0, -1.0)


def _tri(ctx, x):
    return 2.0 * np.abs(_saw(ctx, x)) - 1.0


def _adsr(ctx: _Ctx, a, d, s, r) -> np.ndarray:
    a, d, s, r = (_scalar(v, "adsr") for v in (a, d, s, r))
    total = ctx.duration
    a_end = min(a, total)
    d_end = min(a_end + d, total)
    r_start = max(d_end, total - r)
    return np.interp(ctx.t, [0.0, a_end, d_end, r_start, total], [0.0, 1.0, s, s, 0.0])


def _filter(ctx: _Ctx, x, fc, btype: str) -> np.ndarray:
    fc = float(np.clip(_scalar(fc, btype), 1.0, ctx.sr / 2 - 1))
    sos = signal.butter(2, fc, btype=btype, fs=ctx.sr, output="sos")
    return signal.sosfilt(sos, np.broadcast_to(np.asarray(x, dtype=float), ctx.t.shape))


def _echo(ctx: _Ctx, x, delay, feedback=0.5) -> np.ndarray:
    lag = max(1, int(_scalar(delay, "echo") * ctx.sr))
    fb = float(np.clip(_scalar(feedback, "echo"), 0.0, 0.99))
    a = np.zeros(lag + 1)
    a[0], a[lag] = 1.0, -fb
    return signal.lfilter([1.0], a, np.broadcast_to(np.asarray(x, dtype=float), ctx.t.shape))


def _seq(ctx: _Ctx, step, *values) -> np.ndarray:
    if not values:
        raise FormulaError("seq: нужны значения, напр. seq(0.25, 60, 64, 67)")
    step = _scalar(step, "seq")
    if step <= 0:
        raise FormulaError("seq: шаг должен быть > 0")
    vals = np.array([_scalar(v, "seq") for v in values])
    return vals[(ctx.t // step).astype(int) % len(vals)]


def _note(_, name) -> float:
    if not isinstance(name, str):
        raise FormulaError('note: ожидается имя ноты в кавычках, напр. note("A4")')
    try:
        return float(librosa.note_to_hz(name))
    except Exception as exc:  # librosa кидает ParameterError
        raise FormulaError(f"note: не понимаю ноту {name!r}") from exc


def _ufunc(fn) -> Callable:
    return lambda _ctx, *args: fn(*args)


FUNCTIONS: dict[str, Func] = {
    # математика
    "sin": Func(_ufunc(np.sin), "sin(x)", "синус"),
    "cos": Func(_ufunc(np.cos), "cos(x)", "косинус"),
    "tan": Func(_ufunc(np.tan), "tan(x)", "тангенс"),
    "tanh": Func(_ufunc(np.tanh), "tanh(x)", "мягкое ограничение (сатурация)"),
    "exp": Func(_ufunc(np.exp), "exp(x)", "экспонента; exp(-k*t) — затухание"),
    "log": Func(_ufunc(np.log), "log(x)", "натуральный логарифм"),
    "sqrt": Func(_ufunc(np.sqrt), "sqrt(x)", "корень"),
    "abs": Func(_ufunc(np.abs), "abs(x)", "модуль"),
    "sign": Func(_ufunc(np.sign), "sign(x)", "знак: -1, 0, 1"),
    "floor": Func(_ufunc(np.floor), "floor(x)", "округление вниз"),
    "min": Func(_ufunc(np.minimum), "min(a, b)", "поэлементный минимум"),
    "max": Func(_ufunc(np.maximum), "max(a, b)", "поэлементный максимум"),
    "clip": Func(_ufunc(np.clip), "clip(x, lo, hi)", "ограничить диапазоном"),
    "where": Func(_ufunc(np.where), "where(cond, a, b)", "a там, где cond, иначе b"),
    # осцилляторы (принимают фазу в радианах, как sin)
    "saw": Func(_saw, "saw(x)", "пила"),
    "square": Func(_square, "square(x, duty=0.5)", "меандр / импульс со скважностью duty"),
    "tri": Func(_tri, "tri(x)", "треугольник"),
    "noise": Func(lambda ctx: ctx.rng.uniform(-1.0, 1.0, ctx.t.shape), "noise()", "белый шум"),
    "phase": Func(_phase, "phase(f)", "фаза для частоты f (может меняться во времени) — для FM и глиссандо"),
    # музыка
    "note": Func(_note, 'note("A4")', "частота ноты, Гц"),
    "midi": Func(_ufunc(lambda n: 440.0 * 2.0 ** ((np.asarray(n) - 69) / 12)), "midi(n)", "частота MIDI-ноты (60 = C4)"),
    "seq": Func(_seq, "seq(step, v1, v2, ...)", "по очереди v1, v2, ... каждые step секунд (по кругу)"),
    "adsr": Func(_adsr, "adsr(a, d, s, r)", "огибающая на весь звук: атака, спад, уровень сустейна, затухание"),
    # обработка
    "lowpass": Func(lambda ctx, x, fc: _filter(ctx, x, fc, "lowpass"), "lowpass(x, fc)", "ФНЧ 12 дБ/окт"),
    "highpass": Func(lambda ctx, x, fc: _filter(ctx, x, fc, "highpass"), "highpass(x, fc)", "ФВЧ 12 дБ/окт"),
    "echo": Func(_echo, "echo(x, delay, feedback=0.5)", "эхо с обратной связью"),
}

CONSTANTS = {"pi": np.pi, "tau": 2 * np.pi, "e": np.e}
RESERVED = FUNCTIONS.keys() | CONSTANTS.keys() | {"t", "n", "sr", "T"}

VARIABLES_HELP = {
    "t": "время, с",
    "n": "номер сэмпла",
    "sr": "частота дискретизации",
    "T": "длительность звука, с",
    "pi, tau, e": "константы",
}

_BINOPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
    ast.FloorDiv: np.floor_divide, ast.Mod: np.mod, ast.Pow: np.power,
}
_UNARY = {ast.USub: np.negative, ast.UAdd: np.positive}
_COMPARE = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}


def parse(source: str) -> list[tuple[str | None, ast.expr]]:
    """Разбирает текст формулы в список (имя переменной или None, выражение)."""
    if len(source) > MAX_SOURCE_LEN:
        raise FormulaError("слишком длинная формула")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise FormulaError(f"синтаксическая ошибка в строке {exc.lineno}: {exc.msg}") from exc

    program: list[tuple[str | None, ast.expr]] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr):
            program.append((None, stmt.value))
        elif (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
              and isinstance(stmt.targets[0], ast.Name)):
            name = stmt.targets[0].id
            if name in RESERVED:
                raise FormulaError(f"строка {stmt.lineno}: имя {name!r} зарезервировано")
            program.append((name, stmt.value))
        else:
            raise FormulaError(f"строка {stmt.lineno}: допустимы только выражения и «имя = выражение»")
    if not program:
        raise FormulaError("пустая формула")
    return program


def render(source: str, *, duration: float = 2.0, sr: int = DEFAULT_SR,
           params: dict[str, float] | None = None, seed: int | None = None,
           normalize: bool = True, fade: float = 0.005) -> SynthResult:
    if not 0 < duration <= MAX_DURATION:
        raise FormulaError(f"длительность должна быть в (0, {MAX_DURATION:.0f}] с")
    program = parse(source)
    clash = sorted(RESERVED & (params or {}).keys())
    if clash:
        raise FormulaError(f"параметры не могут называться {', '.join(clash)} — имена зарезервированы")

    n = np.arange(int(round(duration * sr)))
    ctx = _Ctx(t=n / sr, sr=sr, duration=duration, rng=np.random.default_rng(seed))
    names: dict[str, object] = {**CONSTANTS, "t": ctx.t, "n": n.astype(float),
                                "sr": float(sr), "T": duration, **(params or {})}

    result = None
    with np.errstate(all="ignore"):
        for target, expr in program:
            value = _Evaluator(ctx, names).eval(expr)
            if target is None:
                result = value
            else:
                names[target] = value
    if result is None:
        if "out" not in names:
            raise FormulaError("нет результата: последней строкой должно быть выражение или out = ...")
        result = names["out"]
    if isinstance(result, str):
        raise FormulaError("результат формулы — строка, а не сигнал")

    y = np.array(np.broadcast_to(np.asarray(result, dtype=float), ctx.t.shape))
    warnings = []
    bad = ~np.isfinite(y)
    if bad.any():
        warnings.append(f"{int(bad.sum())} сэмплов NaN/inf заменены нулями (деление на 0? log(0)?)")
        y[bad] = 0.0

    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak == 0.0:
        warnings.append("получилась тишина")
    elif normalize:
        y *= 0.9 / peak
    elif peak > 1.0:
        warnings.append(f"пик {peak:.2f} > 1 — сигнал обрезан (уберите --no-normalize или уменьшите громкость)")
        y = np.clip(y, -1.0, 1.0)

    fade_len = min(int(fade * sr), len(y) // 2)
    if fade_len > 0:
        ramp = np.linspace(0.0, 1.0, fade_len)
        y[:fade_len] *= ramp
        y[-fade_len:] *= ramp[::-1]
    return SynthResult(y=y.astype(np.float32), sr=sr, warnings=warnings)


class _Evaluator:
    def __init__(self, ctx: _Ctx, names: dict[str, object]):
        self.ctx = ctx
        self.names = names

    def eval(self, node: ast.AST):
        match node:
            case ast.Constant(value=bool()):
                raise FormulaError("логические константы не поддерживаются")
            case ast.Constant(value=int() | float() as v):
                return float(v)  # float, чтобы 9**9**9 не считался длинной арифметикой
            case ast.Constant(value=str() as s):
                return s
            case ast.Name(id=name):
                if name in self.names:
                    return self.names[name]
                if name in FUNCTIONS:
                    raise FormulaError(f"{name} — функция, вызывайте её со скобками: {FUNCTIONS[name].signature}")
                raise FormulaError(f"неизвестное имя {name!r}")
            case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINOPS:
                return _BINOPS[type(op)](self._num(left), self._num(right))
            case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY:
                return _UNARY[type(op)](self._num(operand))
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                acc, prev = 1.0, self._num(left)
                for op, comp in zip(ops, comparators):
                    if type(op) not in _COMPARE:
                        raise FormulaError("недопустимое сравнение")
                    cur = self._num(comp)
                    acc = acc * np.asarray(_COMPARE[type(op)](prev, cur), dtype=float)
                    prev = cur
                return acc
            case ast.Call(func=ast.Name(id=fname), args=args, keywords=[]):
                if fname not in FUNCTIONS:
                    raise FormulaError(f"неизвестная функция {fname!r}")
                values = [self.eval(a) for a in args]
                if fname != "note" and any(isinstance(v, str) for v in values):
                    raise FormulaError(f"{fname}: строки допустимы только в note(...)")
                try:
                    return FUNCTIONS[fname].impl(self.ctx, *values)
                except TypeError as exc:
                    raise FormulaError(f"неверные аргументы: {FUNCTIONS[fname].signature}") from exc
            case ast.Call():
                raise FormulaError("вызывать можно только встроенные функции, без именованных аргументов")
        raise FormulaError(f"недопустимая конструкция: {type(node).__name__}")

    def _num(self, node: ast.AST):
        value = self.eval(node)
        if isinstance(value, str):
            raise FormulaError("строку нельзя использовать в арифметике — оберните её в note(...)")
        return value
