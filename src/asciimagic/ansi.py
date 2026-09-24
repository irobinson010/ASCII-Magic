"""ANSI color-depth conversion.

Every renderer emits 24-bit "truecolor" escapes (``ESC[38;2;R;G;Bm``). Many
terminals -- notably over SSH, in older macOS Terminal.app, the Linux console,
and tmux/screen without truecolor enabled -- only understand the 256-color or
16-color palettes and show garbage (or wrong colors) for truecolor. Rather
than teach every renderer about palettes, ``downsample`` rewrites finished
ANSI text to the requested depth.
"""

from __future__ import annotations

import os
import re
from typing import Mapping, Optional, Tuple

DEPTHS = ("truecolor", "256", "16")
DEPTH_CHOICES = ("auto",) + DEPTHS

RGB = Tuple[int, int, int]

_TRUECOLOR_RE = re.compile(r"\x1b\[(38|48);2;(\d{1,3});(\d{1,3});(\d{1,3})m")

# xterm's 6x6x6 cube levels and the standard 16-color palette (xterm defaults).
_CUBE = (0, 95, 135, 175, 215, 255)
_BASE16 = (
    (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0),
    (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
    (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
)


def _dist2(a: RGB, b: RGB) -> int:
    # Weighted RGB distance ("redmean"-lite): green matters most to the eye.
    dr, dg, db = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return 2 * dr * dr + 4 * dg * dg + 3 * db * db


def _nearest_cube_level(v: int) -> int:
    return min(range(6), key=lambda i: abs(_CUBE[i] - v))


def rgb_to_256(rgb: RGB) -> int:
    """Nearest xterm-256 index, choosing between the color cube (16-231) and
    the grayscale ramp (232-255). The 16 system colors are skipped: terminal
    themes redefine them, so they are not reliable for exact colors."""
    r, g, b = rgb
    ri, gi, bi = _nearest_cube_level(r), _nearest_cube_level(g), _nearest_cube_level(b)
    cube = (_CUBE[ri], _CUBE[gi], _CUBE[bi])
    cube_idx = 16 + 36 * ri + 6 * gi + bi

    avg = (r + g + b) // 3
    gray_i = max(0, min(23, round((avg - 8) / 10)))
    gray = (8 + 10 * gray_i,) * 3
    gray_idx = 232 + gray_i
    return gray_idx if _dist2(rgb, gray) < _dist2(rgb, cube) else cube_idx


def rgb_to_16(rgb: RGB) -> int:
    """Nearest of the 16 standard colors, as an index 0-15."""
    return min(range(16), key=lambda i: _dist2(rgb, _BASE16[i]))


def _sgr_16(layer: str, idx: int) -> str:
    base = 30 if layer == "38" else 40
    code = base + idx if idx < 8 else base + 60 + (idx - 8)  # 90-97 / 100-107
    return f"\x1b[{code}m"


def downsample(text: str, depth: str) -> str:
    """Rewrite truecolor escapes in `text` to `depth` ("truecolor" = as is)."""
    if depth == "truecolor":
        return text
    if depth not in DEPTHS:
        raise ValueError(f"unknown color depth {depth!r}; expected one of {', '.join(DEPTH_CHOICES)}")
    cache = {}

    def repl(m: "re.Match[str]") -> str:
        key = m.group(0)
        hit = cache.get(key)
        if hit is None:
            layer = m.group(1)
            rgb = tuple(min(255, int(v)) for v in m.group(2, 3, 4))
            if depth == "256":
                hit = f"\x1b[{layer};5;{rgb_to_256(rgb)}m"
            else:
                hit = _sgr_16(layer, rgb_to_16(rgb))
            cache[key] = hit
        return hit

    return _TRUECOLOR_RE.sub(repl, text)


def detect_depth(env: Optional[Mapping[str, str]] = None) -> str:
    """Best guess at what the current terminal supports.

    ASCII_MAGIC_COLOR_DEPTH overrides. COLORTERM=truecolor/24bit is the
    standard signal; a few terminals are known truecolor without setting it.
    Over SSH, COLORTERM is usually not forwarded, so this falls back to 256
    for "*-256color" TERMs -- the safe choice for a login greeting.
    """
    env = os.environ if env is None else env
    forced = (env.get("ASCII_MAGIC_COLOR_DEPTH") or "").strip().lower()
    if forced in DEPTHS:
        return forced
    if (env.get("COLORTERM") or "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    if env.get("WT_SESSION") or (env.get("TERM_PROGRAM") or "") in (
        "iTerm.app", "WezTerm", "vscode", "ghostty",
    ):
        return "truecolor"
    term = (env.get("TERM") or "").lower()
    if "direct" in term or "truecolor" in term:
        return "truecolor"
    if "256" in term:
        return "256"
    return "16"


def resolve_depth(depth: Optional[str], to_terminal: bool) -> str:
    """'auto' detects only when writing to a terminal; files keep truecolor
    (the eventual viewer is unknown -- convert them explicitly if needed)."""
    if depth in (None, "auto"):
        return detect_depth() if to_terminal else "truecolor"
    if depth not in DEPTHS:
        raise ValueError(f"unknown color depth {depth!r}")
    return depth


def add_depth_arg(parser, default: str = "truecolor") -> None:
    parser.add_argument(
        "--color-depth", choices=DEPTH_CHOICES, default=default,
        help="ANSI color palette: truecolor (24-bit), 256, 16, or auto (detect the "
             "terminal; files stay truecolor). Use 256 or 16 for SSH sessions and "
             f"older terminals (default: {default})",
    )
