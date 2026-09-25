"""Color overlays: gradients and tints laid over ASCII art.

An overlay recolors every visible character by its position: a solid color
or a multi-stop gradient (horizontal, vertical, diagonal, radial), blended
with the character's existing color (from the image, a theme, or the
terminal's default for plain text)::

    ascii-magic text HELLO -s figlet --overlay sunset
    ascii-magic image cat.png --color --overlay "#00e5ff,#ff2d55" --overlay-mode multiply
    ascii-magic compose ... --overlay rainbow --overlay-direction diagonal

It works as a post-process on finished ANSI text (so every renderer gets it
for free) or directly on a cell grid (compose).
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

RGB = Tuple[int, int, int]

DIRECTIONS = ("horizontal", "vertical", "diagonal", "diagonal-up", "radial")
MODES = ("tint", "multiply", "screen", "overlay")

PALETTES = {
    "rainbow": ["#ff0000", "#ff8000", "#ffee00", "#00c040", "#0080ff", "#8000ff"],
    "sunset": ["#ff5e62", "#ff9966", "#ffd86b"],
    "ocean": ["#00c6ff", "#0072ff", "#3f2b96"],
    "fire": ["#ffe259", "#ff8c00", "#e52d27"],
    "forest": ["#a8e063", "#56ab2f", "#134e2a"],
    "neon": ["#00e5ff", "#ff00e5"],
    "matrix": ["#00ff41", "#008f11"],
    "mono": ["#ffffff", "#606060"],
}

# Plain text has no color of its own; treat the terminal default as white so
# a full-strength overlay shows its true colors.
DEFAULT_FG: RGB = (255, 255, 255)


def _parse_color(value: str) -> RGB:
    from .colorize_ascii import parse_matrix_color

    try:
        return parse_matrix_color(value.strip())
    except ValueError:
        raise ValueError(
            f"Unknown overlay color {value.strip()!r}. Use a palette ({', '.join(PALETTES)}), "
            "a theme (green, amber, cyan, crimson, violet, white), #RRGGBB, "
            "or comma-separated colors for a gradient."
        ) from None


@dataclass
class Overlay:
    stops: List[RGB]
    direction: str = "horizontal"
    mode: str = "tint"
    strength: float = 1.0

    @classmethod
    def parse(cls, spec: str, direction: str = "horizontal", mode: str = "tint",
              strength: float = 1.0) -> "Overlay":
        """`spec`: a palette name, one color (solid), or comma-separated
        colors (gradient). Colors are theme names or #RRGGBB."""
        spec = (spec or "").strip()
        if not spec:
            raise ValueError("empty overlay")
        if spec.lower() in PALETTES:
            stops = [_parse_color(c) for c in PALETTES[spec.lower()]]
        else:
            stops = [_parse_color(c) for c in spec.split(",") if c.strip()]
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown overlay direction {direction!r}; expected one of {', '.join(DIRECTIONS)}")
        if mode not in MODES:
            raise ValueError(f"unknown overlay mode {mode!r}; expected one of {', '.join(MODES)}")
        if not 0.0 <= strength <= 1.0 or math.isnan(strength):
            raise ValueError(f"overlay strength must be 0..1, got {strength}")
        return cls(stops=stops, direction=direction, mode=mode, strength=strength)

    # ---- geometry ----

    def position(self, x: int, y: int, w: int, h: int) -> float:
        """0..1 along the gradient for cell (x, y) of a w x h block."""
        fx = x / (w - 1) if w > 1 else 0.0
        fy = y / (h - 1) if h > 1 else 0.0
        if self.direction == "horizontal":
            return fx
        if self.direction == "vertical":
            return fy
        if self.direction == "diagonal":
            return (fx + fy) / 2
        if self.direction == "diagonal-up":
            return (fx + (1 - fy)) / 2
        # radial: centre -> corners, on the block's own ellipse
        dx, dy = fx - 0.5, fy - 0.5
        return min(1.0, math.hypot(dx, dy) / math.hypot(0.5, 0.5))

    def color_at(self, t: float) -> RGB:
        stops = self.stops
        if len(stops) == 1:
            return stops[0]
        t = min(1.0, max(0.0, t)) * (len(stops) - 1)
        i = min(int(t), len(stops) - 2)
        f = t - i
        a, b = stops[i], stops[i + 1]
        return tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))

    def blend(self, base: Optional[RGB], over: RGB) -> RGB:
        base = base or DEFAULT_FG
        s = self.strength
        if self.mode == "tint":
            mixed = over
        else:
            mixed = tuple(round(255 * _blend_ch(self.mode, base[k] / 255, over[k] / 255)) for k in range(3))
        return tuple(round(base[k] * (1 - s) + mixed[k] * s) for k in range(3))

    def apply_grid(self, fg: List[List[Optional[RGB]]], visible: List[List[bool]]) -> None:
        """Recolor a grid in place; only visible (inked) cells change."""
        h = len(fg)
        w = max((len(r) for r in fg), default=0)
        for y, row in enumerate(fg):
            for x in range(len(row)):
                if visible[y][x]:
                    row[x] = self.blend(row[x], self.color_at(self.position(x, y, w, h)))


def _blend_ch(mode: str, b: float, o: float) -> float:
    if mode == "multiply":
        return b * o
    if mode == "screen":
        return 1 - (1 - b) * (1 - o)
    # overlay
    return 2 * b * o if b < 0.5 else 1 - 2 * (1 - b) * (1 - o)


# =============================
# ANSI <-> cells
# =============================

_SGR = re.compile(r"\x1b\[([0-9;]*)m")
_BLANKS = frozenset(" ⠀")


@dataclass
class Cell:
    ch: str
    fg: Optional[RGB] = None
    bg: Optional[RGB] = None


def parse_ansi(text: str) -> List[List[Cell]]:
    """Split ANSI art into rows of cells with their fg/bg colors. Understands
    the escapes the renderers emit (24-bit fg/bg, resets); other SGR codes
    are ignored. Double-width characters get a continuation cell."""
    from .textwidth import CONT, char_width

    rows: List[List[Cell]] = []
    fg: Optional[RGB] = None
    bg: Optional[RGB] = None
    for line in text.split("\n"):
        row: List[Cell] = []
        pos = 0
        for m in _SGR.finditer(line + "\x1b[m"):
            for ch in line[pos:m.start()]:
                w = char_width(ch)
                if w == 0 and row:
                    row[-1].ch += ch
                    continue
                row.append(Cell(ch, fg, bg))
                if w == 2:
                    row.append(Cell(CONT, fg, bg))
            pos = m.end()
            if m.start() >= len(line):
                break
            params = [p for p in m.group(1).split(";")] if m.group(1) else ["0"]
            i = 0
            while i < len(params):
                p = params[i]
                if p in ("", "0"):
                    fg = bg = None
                elif p == "39":
                    fg = None
                elif p == "49":
                    bg = None
                elif p in ("38", "48") and i + 4 < len(params) and params[i + 1] == "2":
                    rgb = tuple(min(255, int(v or 0)) for v in params[i + 2:i + 5])
                    if p == "38":
                        fg = rgb
                    else:
                        bg = rgb
                    i += 4
                i += 1
        rows.append(row)
    while rows and not rows[-1]:
        rows.pop()
    return rows


def cells_to_ansi(rows: Sequence[Sequence[Cell]]) -> str:
    out = []
    for row in rows:
        parts: List[str] = []
        cur_fg: Optional[RGB] = None
        cur_bg: Optional[RGB] = None
        for c in row:
            if c.bg != cur_bg:
                parts.append("\x1b[49m" if c.bg is None else "\x1b[48;2;%d;%d;%dm" % c.bg)
                cur_bg = c.bg
            fg = c.fg if c.ch not in _BLANKS else cur_fg  # spaces needn't switch fg
            if fg != cur_fg:
                parts.append("\x1b[39m" if fg is None else "\x1b[38;2;%d;%d;%dm" % fg)
                cur_fg = fg
            parts.append(c.ch)
        if cur_fg is not None or cur_bg is not None:
            parts.append("\x1b[0m")
        out.append("".join(parts))
    return "\n".join(out) + "\n"


def cells_to_html_lines(rows: Sequence[Sequence[Cell]]) -> List[str]:
    lines = []
    for row in rows:
        parts: List[str] = []
        run: List[str] = []
        key: Tuple[Optional[RGB], Optional[RGB]] = (None, None)

        def flush():
            if not run:
                return
            text = html.escape("".join(run))
            fg, bg = key
            style = []
            if fg is not None:
                style.append("color:#%02x%02x%02x" % fg)
            if bg is not None:
                style.append("background-color:#%02x%02x%02x" % bg)
            parts.append(f'<span style="{";".join(style)}">{text}</span>' if style else text)

        for c in row:
            k = (c.fg if c.ch not in _BLANKS else key[0], c.bg)
            if k != key:
                flush()
                run, key = [], k
            run.append(c.ch)
        flush()
        lines.append("".join(parts))
    return lines


def apply_to_cells(ov: Overlay, rows: List[List[Cell]]) -> None:
    from .textwidth import CONT

    w = max((len(r) for r in rows), default=0)
    fg = [[c.fg for c in r] + [None] * (w - len(r)) for r in rows]
    vis = [[c.ch not in _BLANKS and c.ch != CONT for c in r] + [False] * (w - len(r)) for r in rows]
    ov.apply_grid(fg, vis)
    for r, frow in zip(rows, fg):
        for x, c in enumerate(r):
            if c.ch not in _BLANKS:
                c.fg = frow[x] if c.ch != CONT else (frow[x - 1] if x else c.fg)


def apply_to_ansi(ov: Overlay, text: str) -> str:
    rows = parse_ansi(text)
    apply_to_cells(ov, rows)
    return cells_to_ansi(rows)


def ansi_to_html(text: str, title: str = "ASCII Art", font_size_px: int = 12) -> str:
    from .colorize_ascii import wrap_html

    return wrap_html(cells_to_html_lines(parse_ansi(text)), title=title, font_size_px=font_size_px)


# =============================
# CLI
# =============================


def add_overlay_args(parser) -> None:
    g = parser.add_argument_group("color overlay")
    g.add_argument("--overlay", default=None, metavar="COLORS",
                   help="Recolor the art: a palette (" + ", ".join(PALETTES) + "), one color "
                        "(theme or #RRGGBB), or comma-separated colors for a gradient")
    g.add_argument("--overlay-direction", choices=DIRECTIONS, default="horizontal",
                   help="Gradient direction (default: horizontal)")
    g.add_argument("--overlay-mode", choices=MODES, default="tint",
                   help="How the overlay combines with existing colors: tint replaces them "
                        "(mixed by strength); multiply darkens; screen lightens; overlay adds contrast")
    g.add_argument("--overlay-strength", type=_strength, default=1.0, metavar="0..1",
                   help="How much of the overlay to apply (default: 1)")


def _strength(value: str) -> float:
    import argparse

    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}")
    if not 0.0 <= f <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return f


def finish_output(ansi_text: str, args, output_path: Optional[str], to_terminal: bool,
                  font_size_px: int = 12) -> str:
    """Apply --overlay to finished ANSI art, then convert for the destination:
    HTML for .html outputs, else ANSI at --color-depth."""
    ov = from_args(args)
    if ov is not None:
        ansi_text = apply_to_ansi(ov, ansi_text)
    if output_path and output_path.lower().endswith((".html", ".htm")):
        import os

        return ansi_to_html(ansi_text, title=os.path.basename(output_path), font_size_px=font_size_px)
    from .ansi import downsample, resolve_depth

    return downsample(ansi_text, resolve_depth(getattr(args, "color_depth", "truecolor"), to_terminal))


def from_args(args) -> Optional[Overlay]:
    """The overlay requested on the command line, or None."""
    spec = getattr(args, "overlay", None)
    if not spec:
        return None
    return Overlay.parse(spec, args.overlay_direction, args.overlay_mode, args.overlay_strength)
