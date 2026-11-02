"""Still-image (PNG) export of finished art.

Every renderer can produce 24-bit ANSI, so the PNG path starts from that:
parse the escapes into cells (``overlay.parse_ansi``), then draw each
character into a fixed cell with ``glyphs.GlyphAtlas``, the same cells the
GIF and MP4 sinks use, braille dots included. Uncolored characters use
``fg``; colored ones keep their own color; cell backgrounds (an opaque
compose layer, a canvas background) are filled first.

    ascii-magic image cat.png --color -o cat-art.png
    ascii-magic text HELLO --overlay sunset -o hello.png --png-bg transparent
"""

from __future__ import annotations

import io
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

RGB = Tuple[int, int, int]

DEFAULT_BG: RGB = (0, 0, 0)        # the same page the .html export uses
DEFAULT_FG: RGB = (224, 224, 224)  # its default text color (#e0e0e0)
FONT_SIZES = (6, 72)
# A 4096 x 4096 canvas: a 200-column, 16 px braille picture is ~2000 x 1700.
MAX_PIXELS = 16_777_216


class TooLarge(ValueError):
    pass


def parse_color(value: Union[str, RGB, None], default: Optional[RGB]) -> Optional[RGB]:
    """A theme name, ``#RRGGBB``, an RGB tuple, or None for ``default``.
    ``"transparent"`` (or ``"none"``) returns None: no background."""
    if value is None or value == "":
        return default
    if isinstance(value, str) and value.strip().lower() in ("transparent", "none"):
        return None
    from .colorize_ascii import parse_matrix_color

    return parse_matrix_color(value)


def ansi_to_image(
    text: str,
    font_size: int = 16,
    font_path: Optional[str] = None,
    bg: Optional[RGB] = DEFAULT_BG,
    fg: RGB = DEFAULT_FG,
    padding: int = 1,
    max_pixels: int = MAX_PIXELS,
    max_cells: Optional[int] = None,
) -> Image.Image:
    """Draw ANSI (or plain) art to an image. ``bg=None`` gives a transparent
    RGBA image; otherwise RGB. ``padding`` is in character cells."""
    from .glyphs import GlyphAtlas
    from .overlay import _SGR, parse_ansi
    from .textwidth import CONT

    lo, hi = FONT_SIZES
    if not lo <= font_size <= hi:
        raise ValueError(f"font size must be between {lo} and {hi}, got {font_size}")
    if not 0 <= padding <= 8:
        raise ValueError(f"padding must be between 0 and 8 cells, got {padding}")

    text = text.rstrip("\n")
    if max_cells is not None:
        # Cheap upper bound first: parse_ansi walks every character in Python.
        visible = len(_SGR.sub("", text)) - text.count("\n")
        if visible > max_cells * 2:  # x2: wide characters take two cells
            raise TooLarge(f"too many characters to draw (limit {max_cells:,} cells)")
    rows = parse_ansi(text)
    atlas = GlyphAtlas(font_path, font_size)
    cw, chh = atlas.cell_w, atlas.cell_h
    cols = max((len(r) for r in rows), default=0)
    if max_cells is not None and cols * len(rows) > max_cells:
        raise TooLarge(f"{cols} x {len(rows)} characters is too many to draw (limit {max_cells:,})")
    W, H = (cols + 2 * padding) * cw, (len(rows) + 2 * padding) * chh
    if W * H > max_pixels:
        raise TooLarge(
            f"the image would be {W} x {H} pixels, too large (limit {max_pixels:,}); "
            f"use a smaller font size or fewer columns"
        )
    W, H = max(W, 1), max(H, 1)

    # Premultiplied color + coverage (Porter-Duff "over"), then over bg.
    color = np.zeros((H, W, 3), np.float32)
    cover = np.zeros((H, W), np.float32)
    fg_arr = np.asarray(fg, np.float32)

    for y, row in enumerate(rows):
        top = (y + padding) * chh
        for x, cell in enumerate(row):
            if cell.ch == CONT:
                continue
            left = (x + padding) * cw
            span = 2 if x + 1 < len(row) and row[x + 1].ch == CONT else 1
            sl = (slice(top, top + chh), slice(left, left + cw * span))
            if cell.bg is not None:
                color[sl] = cell.bg
                cover[sl] = 1.0
            if not cell.ch.strip():
                continue
            a = atlas.alpha(cell.ch[0], span)
            ink = fg_arr if cell.fg is None else np.asarray(cell.fg, np.float32)
            # "over": the glyph on top of whatever this cell already holds.
            color[sl] = ink * a[:, :, None] + color[sl] * (1 - a[:, :, None])
            cover[sl] = a + cover[sl] * (1 - a)

    if bg is None:
        # Un-premultiply where there's coverage; the rest stays transparent.
        safe = np.where(cover > 0, cover, 1)[:, :, None]
        rgb = np.where(cover[:, :, None] > 0, color / safe, 0)
        out = np.dstack([rgb, cover * 255]).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGBA")
    out = color + np.asarray(bg, np.float32) * (1 - cover[:, :, None])
    return Image.fromarray(out.round().clip(0, 255).astype(np.uint8), "RGB")


def ansi_to_png(text: str, **kw) -> bytes:
    buf = io.BytesIO()
    # optimize=True would cost ~10x the time for ~1% smaller files.
    ansi_to_image(text, **kw).save(buf, format="PNG")
    return buf.getvalue()


# =============================
# CLI
# =============================


def is_png_path(path: Optional[str]) -> bool:
    return bool(path) and path.lower().endswith(".png")


def add_png_args(parser) -> None:
    g = parser.add_argument_group("image output (-o FILE.png)")
    g.add_argument("--png-font-size", type=int, default=16, metavar="PX",
                   help=f"Character height in the PNG, {FONT_SIZES[0]}-{FONT_SIZES[1]} (default: 16)")
    g.add_argument("--png-bg", default="#000000", metavar="COLOR",
                   help="PNG background: #RRGGBB, a theme name, or 'transparent' (default: black)")
    g.add_argument("--png-fg", default="#e0e0e0", metavar="COLOR",
                   help="Color for uncolored characters (default: #e0e0e0)")


def write_output(path: str, art: str, args) -> None:
    """Write finished art to ``path``: a PNG for ``.png``, else UTF-8 text."""
    if not is_png_path(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(art if art.endswith("\n") else art + "\n")
        return
    try:
        bg = parse_color(getattr(args, "png_bg", None), DEFAULT_BG)
        fg = parse_color(getattr(args, "png_fg", None), DEFAULT_FG) or DEFAULT_FG
        data = ansi_to_png(art, font_size=getattr(args, "png_font_size", 16), bg=bg, fg=fg)
    except ValueError as e:
        raise SystemExit(f"error: {e}") from None
    with open(path, "wb") as f:
        f.write(data)

