#!/usr/bin/env python3
import argparse
import sys
import os
import html
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageFilter
import random

ESC = "\x1b"


# -----------------------------
# Data model / options
# -----------------------------

@dataclass
class HtmlOptions:
    font_size_px: int = 12
    line_height_px: Optional[int] = None  # None => match font-size
    fill_spaces: bool = False


@dataclass
class SizeOptions:
    # exact sizing of the ART block
    rows: Optional[int] = None
    cols: Optional[int] = None

    # max sizing (ART block height is affected by header size)
    max_rows: Optional[int] = None
    max_cols: Optional[int] = None

_MATRIX_THEMES = {
    "green": (0, 255, 0),
    "amber": (255, 176, 0),
    "cyan": (0, 229, 255),
    "crimson": (255, 45, 85),
    "violet": (186, 85, 255),
    "white": (255, 255, 255),
}


def parse_matrix_color(value) -> Tuple[int, int, int]:
    """Accepts a theme name, '#RRGGBB', or an RGB tuple."""
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(max(0, min(255, int(c))) for c in value)
    s = str(value).strip().lower()
    if s in _MATRIX_THEMES:
        return _MATRIX_THEMES[s]
    if s.startswith("#") and len(s) == 7:
        try:
            return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            pass
    raise ValueError(
        f"Unknown matrix color: {value!r}. Use one of "
        f"{', '.join(_MATRIX_THEMES)} or #RRGGBB."
    )


def tint_rgb(intensity: int, tint: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Scale a tint color by an intensity byte (0..255)."""
    return (
        intensity * tint[0] // 255,
        intensity * tint[1] // 255,
        intensity * tint[2] // 255,
    )


@dataclass
class MatrixOptions:
    enabled: bool = False
    top: bool = False          # apply to header too
    seed: Optional[int] = None
    gamma: float = 2.0
    tint: Tuple[int, int, int] = (0, 255, 0)  # see parse_matrix_color / _MATRIX_THEMES

    # green intensity ranges (0..255)
    fg_min: int = 20
    fg_max: int = 255
    bg_min: int = 0
    bg_max: int = 60

    # glyph behavior (raw string: single literal backslash in the set)
    chars: str = r"01ABCDEFGHIJKLMNOPQRSTUVWXYZ@$%&*+;:,.?/\|[]{}()<>"
    fill_spaces: bool = False  # keep background color even on spaces?

    use_mask: bool = False
    mask_boost: float = 0.30          # add to subject score on non-space pixels (0..1)
    mask_density_floor: float = 0.35  # minimum glyph probability on subject pixels
    bg_dim: float = 0.80              # multiply subject score on background pixels
    bg_density: float = 0.75          # multiply glyph probability on background pixels

@dataclass
class CaptionOptions:
    """Text rendered as ASCII and stitched above/below the art, un-colorized
    by the image (optionally tinted a uniform color)."""

    text: Optional[str] = None
    position: str = "bottom"   # "top" | "bottom"
    style: str = "block"       # block | small | shadow | box | banner | figlet
    scale: float = 0.6         # AUTO sizing: fraction of art width for rendered styles
    cols: Optional[int] = None  # EXACT width in chars (overrides scale; free transform)
    rows: Optional[int] = None  # EXACT height in rows
    gap: int = 1               # blank lines between caption and art
    color: Optional[str] = None  # theme, #RRGGBB, or "image" (sample the picture); None = default fg
    align: str = "center"      # left | center | right


@dataclass
class Options:
    out_format: Optional[str] = None  # "ansi" | "html" (None => infer from output extension)

    keep_top: int = 0
    color_top: bool = False

    rotate: int = 0  # CLI-only: clockwise rotation of the reference image

    debug: bool = False
    log_path: Optional[str] = None

    # Animation (implies matrix mode). Serialized here rather than as an
    # AnimationOptions to avoid a circular import with animate.py.
    animate: bool = False
    anim_frames: int = 60
    anim_fps: float = 12.0
    anim_tail: float = 6.0
    anim_loops: int = 3
    anim_reveal: bool = False

    size: SizeOptions = field(default_factory=SizeOptions)
    html: HtmlOptions = field(default_factory=HtmlOptions)
    matrix: MatrixOptions = field(default_factory=MatrixOptions)
    caption: CaptionOptions = field(default_factory=CaptionOptions)

# -----------------------------
# Utilities / core logic
# -----------------------------

LOG = logging.getLogger("colorize_ascii")
def setup_logging(debug: bool, log_path: str | None = None) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    LOG.setLevel(level)

    fmt = logging.Formatter("%(levelname)s: %(message)s")

    handlers: list[logging.Handler] = []

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    handlers.append(sh)

    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        handlers.append(fh)

    LOG.handlers[:] = handlers
    LOG.propagate = False       # prevent double logging via root logger

def scale_grid(lines, target_h, target_w):
    """Nearest-neighbor scale of a rectangular character grid."""
    src_h = len(lines)
    src_w = max(len(l) for l in lines) if lines else 0
    padded = [l.ljust(src_w) for l in lines]

    out = []
    for y2 in range(target_h):
        y = int(y2 * src_h / target_h)
        row = []
        for x2 in range(target_w):
            x = int(x2 * src_w / target_w)
            row.append(padded[y][x])
        out.append("".join(row))
    return out


_OUT_EXTS = (".ans", ".html", ".gif", ".frames")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="colorize-ascii",
        description="Colorize ASCII art from an image, as ANSI, HTML, or animated matrix rain.",
        epilog=(
            "If OUT is omitted: defaults to ANSI on stdout (pass --format html for HTML). "
            "--animate plays matrix rain in the terminal, or writes OUT.gif / OUT.html / OUT.frames."
        ),
    )
    ap.add_argument("image", help="Reference image (colors are sampled from it)")
    ap.add_argument("ascii", help="ASCII text file to colorize")
    ap.add_argument("out", nargs="?", default=None,
                    help="Output file: .ans, .html, .gif, .frames, or '-' (default: stdout)")

    ap.add_argument("--format", dest="out_format", choices=["ansi", "html"], default=None,
                    help="Output format (default: inferred from OUT extension)")

    g = ap.add_argument_group("sizing")
    g.add_argument("--max-rows", type=int, default=None, help="Fit art within N rows (after header)")
    g.add_argument("--max-cols", type=int, default=None, help="Fit art within N columns")
    g.add_argument("--rows", type=int, default=None, help="Exact art height")
    g.add_argument("--cols", type=int, default=None, help="Exact art width")

    g = ap.add_argument_group("header")
    g.add_argument("--keep-top", type=int, default=0, metavar="N",
                   help="Preserve top N lines uncolorized")
    g.add_argument("--color-top", action="store_true", help="Colorize the kept top lines")

    g = ap.add_argument_group("html output")
    g.add_argument("--html-font-size", type=int, default=12, metavar="PX")
    g.add_argument("--html-line-height", type=int, default=None, metavar="PX")
    g.add_argument("--html-fill-spaces", action="store_true",
                   help="Give spaces a background color in HTML output")

    g = ap.add_argument_group("matrix mode")
    g.add_argument("--matrix", action="store_true", help="Green glyphs driven by luminance/edges")
    g.add_argument("--matrix-top", action="store_true", help="Apply matrix mode to kept top lines")
    g.add_argument("--matrix-seed", type=int, default=None, metavar="N")
    g.add_argument("--matrix-gamma", type=float, default=2.0, metavar="F")
    g.add_argument("--matrix-fg-min", type=int, default=20, metavar="N")
    g.add_argument("--matrix-fg-max", type=int, default=255, metavar="N")
    g.add_argument("--matrix-bg-min", type=int, default=0, metavar="N")
    g.add_argument("--matrix-bg-max", type=int, default=60, metavar="N")
    g.add_argument("--matrix-color", default="green", metavar="COLOR",
                   help=f"Rain color: {', '.join(_MATRIX_THEMES)}, or #RRGGBB (default: green)")
    g.add_argument("--matrix-chars", default=MatrixOptions.chars, metavar="STR")
    g.add_argument("--matrix-fill-spaces", action="store_true")
    g.add_argument("--matrix-mask", action="store_true",
                   help="Bias glyph placement toward inked ASCII characters")
    g.add_argument("--matrix-mask-boost", type=float, default=0.30, metavar="F")
    g.add_argument("--matrix-mask-density-floor", type=float, default=0.35, metavar="F")
    g.add_argument("--matrix-bg-dim", type=float, default=0.80, metavar="F")
    g.add_argument("--matrix-bg-density", type=float, default=0.75, metavar="F")

    g = ap.add_argument_group("caption")
    g.add_argument("--caption", default=None, metavar="TEXT",
                   help="Render TEXT as ASCII and stitch it onto the art")
    g.add_argument("--caption-pos", choices=["top", "bottom"], default="bottom")
    g.add_argument("--caption-style", choices=["block", "small", "shadow", "box", "banner", "figlet"],
                   default="block")
    g.add_argument("--caption-scale", type=float, default=0.6, metavar="F",
                   help="Caption width as a fraction of art width (rendered styles)")
    g.add_argument("--caption-cols", type=int, default=None, metavar="N",
                   help="Exact caption width in chars (free transform; overrides --caption-scale)")
    g.add_argument("--caption-rows", type=int, default=None, metavar="N",
                   help="Exact caption height in rows")
    g.add_argument("--caption-gap", type=int, default=1, metavar="N",
                   help="Blank lines between caption and art")
    g.add_argument("--caption-color", default=None, metavar="COLOR",
                   help="Caption color: theme name, #RRGGBB, 'image' (nearby strip), or "
                        "'image-full' (whole picture stretched over the text)")
    g.add_argument("--caption-align", choices=["left", "center", "right"], default="center")

    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="Rotate the reference image clockwise (EXIF orientation "
                    "is applied automatically)")

    g = ap.add_argument_group("animation")
    g.add_argument("--animate", action="store_true",
                   help="Matrix rain animation (implies --matrix)")
    g.add_argument("--frames", type=int, default=60, metavar="N", help="Frames per loop")
    g.add_argument("--fps", type=float, default=12.0, metavar="F")
    g.add_argument("--tail", type=float, default=6.0, metavar="F", help="Drop tail fade length")
    g.add_argument("--loops", type=int, default=3, metavar="N",
                   help="Terminal playback repeats (0 = until Ctrl-C)")
    g.add_argument("--reveal", action="store_true",
                   help="Rain uncovers the colorized art, which persists beneath it")

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log", dest="log_path", default=None, metavar="FILE")
    return ap


def parse_args(argv) -> Tuple[str, str, Optional[str], Options]:
    ns = build_arg_parser().parse_args(argv[1:])

    out_path = ns.out
    if out_path is not None and out_path != "-":
        ext = os.path.splitext(out_path)[1].lower()
        if ext not in _OUT_EXTS:
            raise SystemExit(
                f"Unexpected output file: {out_path}\n"
                f"Expected one of {', '.join(_OUT_EXTS)} or '-' for stdout."
            )

    opt = Options(
        out_format=ns.out_format,
        keep_top=ns.keep_top,
        color_top=ns.color_top,
        debug=ns.debug,
        log_path=ns.log_path,
        rotate=ns.rotate,
        animate=ns.animate,
        anim_frames=ns.frames,
        anim_fps=ns.fps,
        anim_tail=ns.tail,
        anim_loops=ns.loops,
        anim_reveal=ns.reveal,
        size=SizeOptions(rows=ns.rows, cols=ns.cols, max_rows=ns.max_rows, max_cols=ns.max_cols),
        html=HtmlOptions(
            font_size_px=ns.html_font_size,
            line_height_px=ns.html_line_height,
            fill_spaces=ns.html_fill_spaces,
        ),
        caption=CaptionOptions(
            text=ns.caption,
            position=ns.caption_pos,
            style=ns.caption_style,
            scale=ns.caption_scale,
            cols=ns.caption_cols,
            rows=ns.caption_rows,
            gap=ns.caption_gap,
            color=ns.caption_color,
            align=ns.caption_align,
        ),
        matrix=MatrixOptions(
            enabled=ns.matrix,
            top=ns.matrix_top,
            seed=ns.matrix_seed,
            gamma=ns.matrix_gamma,
            tint=parse_matrix_color(ns.matrix_color),
            fg_min=ns.matrix_fg_min,
            fg_max=ns.matrix_fg_max,
            bg_min=ns.matrix_bg_min,
            bg_max=ns.matrix_bg_max,
            chars=ns.matrix_chars,
            fill_spaces=ns.matrix_fill_spaces,
            use_mask=ns.matrix_mask,
            mask_boost=ns.matrix_mask_boost,
            mask_density_floor=ns.matrix_mask_density_floor,
            bg_dim=ns.matrix_bg_dim,
            bg_density=ns.matrix_bg_density,
        ),
    )

    # Infer output format if not explicitly set
    if opt.out_format is None:
        if out_path is None:
            opt.out_format = "ansi"
        else:
            ext = os.path.splitext(out_path)[1].lower()
            opt.out_format = "html" if ext == ".html" else "ansi"

    return ns.image, ns.ascii, out_path, opt


def read_ascii_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f]


def split_header(lines: Sequence[str], keep_top: int) -> Tuple[List[str], List[str]]:
    keep_top = max(0, min(keep_top, len(lines)))
    return list(lines[:keep_top]), list(lines[keep_top:])


def compute_target_art_height(max_rows: Optional[int], header_len: int, art_len: int) -> int:
    if max_rows is None:
        return art_len
    return max(0, max_rows - header_len)


def scale_art_block(art_lines: Sequence[str], target_art_h: int, opt: SizeOptions) -> List[str]:
    """
    Scale art:
      - If rows/cols is set: exact mode (optionally deriving the other dimension to preserve aspect)
      - Else: fit mode using target_art_h and max_cols preserving aspect
    """
    if not art_lines or target_art_h <= 0:
        return []

    src_h = len(art_lines)
    src_w = max(len(ln) for ln in art_lines)
    art_rect = [ln.ljust(src_w) for ln in art_lines]

    # EXACT size mode (wins over max-* constraints)
    if opt.rows is not None or opt.cols is not None:
        target_h = opt.rows if opt.rows is not None else src_h
        target_w = opt.cols if opt.cols is not None else src_w

        # If only one provided, keep aspect by deriving the other
        if opt.rows is not None and opt.cols is None:
            target_w = max(1, round(src_w * (target_h / src_h)))
        elif opt.cols is not None and opt.rows is None:
            target_h = max(1, round(src_h * (target_w / src_w)))

        return scale_grid(art_rect, max(1, target_h), max(1, target_w))

    # FIT mode (preserve aspect)
    scale = 1.0
    scale = min(scale, target_art_h / src_h)
    if opt.max_cols is not None and opt.max_cols > 0:
        scale = min(scale, opt.max_cols / src_w)

    if scale < 1.0:
        target_h = max(1, round(src_h * scale))
        target_w = max(1, round(src_w * scale))
        return scale_grid(art_rect, target_h, target_w)

    return art_rect

def _flat_pixels(img: Image.Image) -> list:
    """Flat pixel list; Pillow before 11.3 has no get_flattened_data."""
    if hasattr(img, "get_flattened_data"):
        return list(img.get_flattened_data())
    return list(img.getdata())


def _percentile_stretch(vals, lo=0.02, hi=0.98):
    # vals: list of 0..255 ints
    if not vals:
        return 0, 255
    s = sorted(vals)
    n = len(s)
    lo_v = s[int(lo * (n - 1))]
    hi_v = s[int(hi * (n - 1))]
    if hi_v <= lo_v:
        return 0, 255
    return lo_v, hi_v

_DENSITY_RAMP = " .'`^\",:;Il!i~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

_BLOCK_MAP = {
    " ": 0.0,
    "░": 0.25,
    "▒": 0.50,
    "▓": 0.75,
    "█": 1.00,
}

def ink_strength(ch: str) -> float:
    """Estimate how 'filled' a character is: 0.0 (empty) .. 1.0 (solid)."""
    if not ch:
        return 0.0

    # Common block characters
    if ch in _BLOCK_MAP:
        return _BLOCK_MAP[ch]

    o = ord(ch)

    # Braille patterns U+2800..U+28FF (8-dot). Density = number of raised dots / 8.
    if 0x2800 <= o <= 0x28FF:
        bits = o - 0x2800
        # Python 3.8+: int.bit_count()
        return bits.bit_count() / 8.0

    # ASCII density ramp (best-effort)
    idx = _DENSITY_RAMP.find(ch)
    if idx != -1:
        return idx / (len(_DENSITY_RAMP) - 1)

    # Anything else: treat as a “medium” ink by default (tweak if you want)
    if ch.isspace():
        return 0.0
    return 0.35

def matrix_field(lines, img, m: MatrixOptions):
    """Per-cell matrix scoring shared by the static renderers and animation.

    Returns (grid, field) where grid is the space-padded character grid and
    field[y][x] = (fg_g, bg_g, p, subject): glyph green, background green,
    glyph probability, and the raw subject score (0..~1).
    """
    h = len(lines)
    w = max(len(ln) for ln in lines)
    grid = [ln.ljust(w) for ln in lines]

    # Resize once for sampling
    img = img.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")

    # Build luminance + edge maps
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)

    gpx = gray.load()
    epx = edges.load()

    # Percentile stretch luminance (2%..98%)
    lum_vals = _flat_pixels(gray)  # 0..255
    lo_v, hi_v = _percentile_stretch(lum_vals, lo=0.02, hi=0.98)
    denom = (hi_v - lo_v) if hi_v > lo_v else 1

    # Tunables (hardcoded for now)
    edge_weight = 0.35   # how much edges contribute to "subjectness"
    edge_gamma = 0.7     # emphasize edges a bit
    base_density = 0.12  # minimum glyph probability

    field = []
    for y in range(h):
        frow = []
        for x in range(w):
            lum_byte = gpx[x, y]  # 0..255
            # stretched luminance 0..1
            lum = (lum_byte - lo_v) / denom
            lum = 0.0 if lum < 0.0 else (1.0 if lum > 1.0 else lum)

            edge = epx[x, y] / 255.0
            edge = edge ** edge_gamma

            # subject score: brightness + edges
            subject = (1.0 - edge_weight) * lum + edge_weight * max(lum, edge)
            subject = subject ** m.gamma

            # Background stays driven mainly by luminance (prevents noisy backgrounds)
            bg_score = (lum ** max(0.1, (m.gamma * 0.9)))

            ink = ink_strength(grid[y][x]) if m.use_mask else 0.0

            if m.use_mask:
                subject_bg = subject * m.bg_dim
                subject_fg = min(1.0, subject + m.mask_boost)

                subject = subject_bg * (1.0 - ink) + subject_fg * ink
                # Blend toward a boosted background on inked cells; clamp so
                # bg_g never exceeds the configured bg range.
                bg_boosted = min(1.0, bg_score + 0.15)
                bg_score = min(1.0, bg_score * (1.0 - 0.25 * ink) + bg_boosted * ink)

            fg_g = int(m.fg_min + subject * (m.fg_max - m.fg_min))
            bg_g = int(m.bg_min + bg_score * (m.bg_max - m.bg_min))

            # Glyph density follows subjectness
            p = base_density + (1.0 - base_density) * subject
            if m.use_mask:
                p_bg = p * m.bg_density
                p_fg = max(p, m.mask_density_floor)

                p = p_bg * (1.0 - ink) + p_fg * ink

            frow.append((fg_g, bg_g, p, subject))
        field.append(frow)

    return grid, field


def matrix_render_cells(lines, img, m: MatrixOptions, rng):
    """Per-cell (char, fg_rgb|None) for matrix mode, sampling in the same rng
    order as matrix_lines_ansi so ANSI and bitmap sinks agree. Background
    color is omitted (bitmap sinks draw on black)."""
    grid, field = matrix_field(lines, img, m)
    rows = []
    for frow in field:
        row = []
        for (fg_g, _bg_g, p, _subject) in frow:
            ch = rng.choice(m.chars) if (rng.random() < p) else " "
            row.append((ch, tint_rgb(fg_g, m.tint) if ch != " " else None))
        rows.append(row)
    return rows


def matrix_lines_ansi(lines, img, m: MatrixOptions):
    """ANSI Matrix mode: green glyphs with subject emphasis (edges + stretched luminance)."""
    if not lines:
        return []

    grid, field = matrix_field(lines, img, m)
    rng = random.Random(m.seed)

    out_lines = []
    for y, frow in enumerate(field):
        prev_style = None  # (fg_g, bg_g) or None
        row = []

        for x, (fg_g, bg_g, p, _subject) in enumerate(frow):
            ch = rng.choice(m.chars) if (rng.random() < p) else " "

            if ch == " " and not m.fill_spaces:
                if prev_style is not None:
                    row.append(f"{ESC}[0m")
                    prev_style = None
                row.append(" ")
                continue

            style = (fg_g, bg_g)
            if style != prev_style:
                fr, fg, fb = tint_rgb(fg_g, m.tint)
                br, bg, bb = tint_rgb(bg_g, m.tint)
                row.append(f"{ESC}[0m{ESC}[38;2;{fr};{fg};{fb}m{ESC}[48;2;{br};{bg};{bb}m")
                prev_style = style

            row.append(ch)

        row.append(f"{ESC}[0m")
        out_lines.append("".join(row))

    return out_lines

def matrix_lines_html(lines, img, m: MatrixOptions, fill_spaces=False):
    """HTML Matrix mode: green glyphs with subject emphasis (edges + stretched luminance)."""
    if not lines:
        return []

    grid, field = matrix_field(lines, img, m)
    rng = random.Random(m.seed)

    out_lines = []
    for y, frow in enumerate(field):
        prev_style = None
        span_open = False
        row = []

        def close():
            nonlocal span_open
            if span_open:
                row.append("</span>")
                span_open = False

        for x, (fg_g, bg_g, p, _subject) in enumerate(frow):
            ch = rng.choice(m.chars) if (rng.random() < p) else " "

            effective_fill = fill_spaces or m.fill_spaces
            if ch == " " and not effective_fill:
                close()
                prev_style = None
                row.append(" ")
                continue

            style = (fg_g, bg_g)
            if style != prev_style:
                close()
                fr, fg, fb = tint_rgb(fg_g, m.tint)
                br, bg, bb = tint_rgb(bg_g, m.tint)
                row.append(
                    f'<span style="color: rgb({fr},{fg},{fb}); '
                    f'background-color: rgb({br},{bg},{bb})">'
                )
                span_open = True
                prev_style = style

            row.append("&nbsp;" if ch == " " else html.escape(ch))

        close()
        out_lines.append("".join(row))

    return out_lines

# -----------------------------
# Rendering (unchanged logic)
# -----------------------------

def colorize_lines_ansi(lines, img, color_spaces=False):
    """Return list of ANSI-colored lines."""
    if not lines:
        return []

    h = len(lines)
    w = max(len(ln) for ln in lines)
    grid = [ln.ljust(w) for ln in lines]

    img = img.resize((w, h), Image.Resampling.LANCZOS)
    px = img.load()

    out_lines = []
    for y in range(h):
        prev = None
        row = []
        for x, ch in enumerate(grid[y]):
            r, g, b = px[x, y]

            if ch == " " and not color_spaces:
                if prev is not None:
                    row.append(f"{ESC}[0m")
                    prev = None
                row.append(" ")
                continue

            if prev != (r, g, b):
                row.append(f"{ESC}[38;2;{r};{g};{b}m")
                prev = (r, g, b)

            row.append(ch)

        row.append(f"{ESC}[0m")
        out_lines.append("".join(row))
    return out_lines


def colorize_lines_html(lines, img, color_spaces=False, fill_spaces=False):
    """Return list of HTML lines (no surrounding <pre>)."""
    if not lines:
        return []

    h = len(lines)
    w = max(len(ln) for ln in lines)
    grid = [ln.ljust(w) for ln in lines]

    img = img.resize((w, h), Image.Resampling.LANCZOS)
    px = img.load()

    out_lines = []
    for y in range(h):
        prev = None
        span_open = False
        row = []

        for x, ch in enumerate(grid[y]):
            r, g, b = px[x, y]

            if ch == " " and not color_spaces:
                if span_open:
                    row.append("</span>")
                    span_open = False
                    prev = None
                if fill_spaces:
                    row.append(f'<span style="background-color: rgb({r},{g},{b})">&nbsp;</span>')
                else:
                    row.append(" ")
                continue

            if prev != (r, g, b):
                if span_open:
                    row.append("</span>")
                row.append(f'<span style="color: rgb({r},{g},{b})">')
                span_open = True
                prev = (r, g, b)

            row.append(html.escape(ch))

        if span_open:
            row.append("</span>")

        out_lines.append("".join(row))

    return out_lines


def wrap_html(pre_lines, title="ASCII Art", font_size_px=12, line_height_px=None):
    # Browsers can drift if line-height is not locked; keep px values.
    if line_height_px is None:
        line_height_px = font_size_px

    return (
        "<!doctype html>\n"
        "<html>\n<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{html.escape(title)}</title>\n"
        "  <style>\n"
        "    html, body { margin: 0; background: #000; }\n"
        "    .wrap { padding: 16px; }\n"
        "    pre {\n"
        "      margin: 0;\n"
        "      white-space: pre;\n"
        "      overflow: auto;\n"
        "      color: #e0e0e0;\n"  # default text must contrast the black page
        '      font-family: "Hack", "JetBrains Mono", "Cascadia Mono", "Fira Code", Consolas, monospace;\n'
        "      font-variant-ligatures: none;\n"
        f"      font-size: {font_size_px}px;\n"
        f"      line-height: {line_height_px}px;\n"
        "      letter-spacing: 0;\n"
        "    }\n"
        "  </style>\n"
        "</head>\n<body>\n"
        '  <div class="wrap">\n'
        "    <pre>\n" + "\n".join(pre_lines) + "\n    </pre>\n"
        "  </div>\n"
        "</body>\n</html>\n"
    )


def _build_caption_lines(cap: CaptionOptions, width: int) -> List[str]:
    from . import text_to_ascii as text_mod

    if not cap.text:
        return []
    return text_mod.caption_lines(
        cap.text, width, style=cap.style, scale=cap.scale, align=cap.align,
        cols=cap.cols, rows=cap.rows,
    )


def _caption_lines_ansi(lines: Sequence[str], cap: CaptionOptions) -> List[str]:
    if not cap.color:
        return list(lines)
    r, g, b = parse_matrix_color(cap.color)
    return [f"{ESC}[38;2;{r};{g};{b}m{ln}{ESC}[0m" if ln.strip() else ln for ln in lines]


def _caption_lines_html(lines: Sequence[str], cap: CaptionOptions) -> List[str]:
    escaped = [html.escape(ln) for ln in lines]
    if not cap.color:
        return escaped
    r, g, b = parse_matrix_color(cap.color)
    return [
        f'<span style="color: rgb({r},{g},{b})">{ln}</span>' if ln.strip() else ln
        for ln in escaped
    ]


def _with_caption(body: List[str], cap_lines: List[str], cap: CaptionOptions) -> List[str]:
    if not cap_lines:
        return body
    spacer = [""] * max(0, int(cap.gap))
    if cap.position == "top":
        return cap_lines + spacer + body
    return body + spacer + cap_lines


def caption_image_strip(img: Image.Image, position: str) -> Image.Image:
    """The quarter of the picture adjacent to the caption, so image-colored
    captions flow from the nearest art rows instead of the whole image."""
    w, h = img.size
    strip_h = max(1, h // 4)
    if position == "top":
        return img.crop((0, 0, w, strip_h))
    return img.crop((0, h - strip_h, w, h))


_CAPTION_IMAGE_COLORS = ("image", "image-full")


def caption_ref_image(img: Image.Image, cap: "CaptionOptions") -> Image.Image:
    """The image used to colorize an image-colored caption: the whole picture
    stretched over the text (image-full) or the strip nearest the caption."""
    if cap.color == "image-full":
        return img
    return caption_image_strip(img, cap.position)


def render_ansi(
        header: Sequence[str],
        art: Sequence[str],
        img: Image.Image,
        color_top: bool,
        m: MatrixOptions,
        cap: Optional[CaptionOptions] = None,
        cap_lines: Optional[List[str]] = None,
) -> List[str]:
    out_lines: List[str] = []
    if header:
        if m.enabled and m.top:
            out_lines.extend(matrix_lines_ansi(header, img, m))
        elif color_top:
            out_lines.extend(colorize_lines_ansi(header, img, color_spaces=False))
        else:
            out_lines.extend(header)
    if art:
        if m.enabled:
            out_lines.extend(matrix_lines_ansi(art, img, m))
        else:
            out_lines.extend(colorize_lines_ansi(art, img, color_spaces=False))

    if cap and cap_lines:
        if cap.color in _CAPTION_IMAGE_COLORS:
            rendered = colorize_lines_ansi(
                cap_lines, caption_ref_image(img, cap), color_spaces=False
            )
        else:
            rendered = _caption_lines_ansi(cap_lines, cap)
        out_lines = _with_caption(out_lines, rendered, cap)
    return out_lines

def render_html(
        header: Sequence[str],
        art: Sequence[str],
        img: Image.Image,
        color_top: bool,
        html_opt: HtmlOptions,
        m: MatrixOptions,
        cap: Optional[CaptionOptions] = None,
        cap_lines: Optional[List[str]] = None,
) -> str:
    pre_lines: List[str] = []

    if header:
        if m.enabled and m.top:
            pre_lines.extend(matrix_lines_html(header, img, m, fill_spaces=html_opt.fill_spaces))
        elif color_top:
            pre_lines.extend(colorize_lines_html(header, img, color_spaces=False, fill_spaces=html_opt.fill_spaces))
        else:
            pre_lines.extend([html.escape(ln) for ln in header])

    if art:
        if m.enabled:
            pre_lines.extend(matrix_lines_html(art, img, m, fill_spaces=html_opt.fill_spaces))
        else:
            pre_lines.extend(colorize_lines_html(art, img, color_spaces=False, fill_spaces=html_opt.fill_spaces))

    if cap and cap_lines:
        if cap.color in _CAPTION_IMAGE_COLORS:
            rendered = colorize_lines_html(
                cap_lines, caption_ref_image(img, cap),
                color_spaces=False, fill_spaces=False,
            )
        else:
            rendered = _caption_lines_html(cap_lines, cap)
        pre_lines = _with_caption(pre_lines, rendered, cap)

    return wrap_html(
        pre_lines,
        title="ASCII Art",
        font_size_px=html_opt.font_size_px,
        line_height_px=html_opt.line_height_px,
    )


def colorize_ascii_text(
    image: Image.Image,
    ascii_text: str,
    opt: Optional[Options] = None,
) -> str:
    """
    Colorize an ASCII text block using an in-memory PIL image.

    This is a programmatic API intended for orchestration layers where
    different modules share runtime context.
    """
    if opt is None:
        opt = Options()

    lines = [ln.rstrip("\n") for ln in ascii_text.splitlines()]
    if not lines:
        return ""

    header, art_lines = split_header(lines, opt.keep_top)
    target_art_h = compute_target_art_height(opt.size.max_rows, len(header), len(art_lines))
    scaled_art = scale_art_block(art_lines, target_art_h, opt.size)

    cap_lines: List[str] = []
    if opt.caption.text:
        width = max([len(ln) for ln in scaled_art] + [len(ln) for ln in header] + [1])
        cap_lines = _build_caption_lines(opt.caption, width)

    if opt.out_format == "html":
        return render_html(
            header, scaled_art, image.convert("RGB"), opt.color_top, opt.html, opt.matrix,
            cap=opt.caption, cap_lines=cap_lines,
        )

    out_lines = render_ansi(
        header, scaled_art, image.convert("RGB"), opt.color_top, opt.matrix,
        cap=opt.caption, cap_lines=cap_lines,
    )
    return "\n".join(out_lines) + "\n"


# -----------------------------
# main
# -----------------------------

def main():
    img_path, ascii_path, out_path, opt = parse_args(sys.argv)

    t0 = time.perf_counter()
    setup_logging(opt.debug, opt.log_path)

    LOG.debug("Starting")
    LOG.debug("Args: out_format=%s keep_top=%d color_top=%s", opt.out_format, opt.keep_top, opt.color_top)
    LOG.debug("Size: max_rows=%s max_cols=%s rows=%s cols=%s",
              opt.size.max_rows, opt.size.max_cols, opt.size.rows, opt.size.cols)
    LOG.debug("HTML: font=%spx line_height=%s fill_spaces=%s",
              opt.html.font_size_px, opt.html.line_height_px, opt.html.fill_spaces)


    lines = read_ascii_file(ascii_path)
    LOG.debug("Loaded ASCII: %d lines", len(lines))
    if not lines:
        if out_path:
            open(out_path, "w", encoding="utf-8").close()
        return

    header, art_lines = split_header(lines, opt.keep_top)
    LOG.debug("Header lines: %d | Art lines: %d", len(header), len(art_lines))
    from .image_to_ascii import open_oriented, rotate_cw

    base_img = rotate_cw(open_oriented(img_path, "RGB"), opt.rotate)

    target_art_h = compute_target_art_height(opt.size.max_rows, len(header), len(art_lines))
    LOG.debug("Target art height (after header): %d", target_art_h)
    
    LOG.debug("Scaling art...")
    scaled_art = scale_art_block(art_lines, target_art_h, opt.size)
    LOG.debug("Scaled art: %d lines", len(scaled_art))

    ext = os.path.splitext(out_path)[1].lower() if out_path else None
    if ext in (".gif", ".frames") and not opt.animate:
        raise SystemExit(f"{ext} output requires --animate")

    if opt.animate:
        from .animate import AnimationOptions, generate

        opt.matrix.enabled = True
        anim_opt = AnimationOptions(
            frames=opt.anim_frames,
            fps=opt.anim_fps,
            tail=opt.anim_tail,
            loops=opt.anim_loops,
            reveal=opt.anim_reveal,
        )
        animation = generate(
            "\n".join(header + scaled_art), base_img, m=opt.matrix, a=anim_opt,
            caption=opt.caption,
        )
        if out_path is None:
            animation.play(loops=opt.anim_loops)
        elif ext == ".gif":
            with open(out_path, "wb") as out:
                out.write(animation.to_gif_bytes())
        elif ext == ".html":
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(animation.to_html(title=os.path.basename(out_path)))
        elif ext == ".frames":
            from .greet import write_frames_file
            from pathlib import Path

            write_frames_file(
                Path(out_path), animation.frames_ansi(), fps=opt.anim_fps, loops=opt.anim_loops
            )
        else:
            raise SystemExit("--animate output must be .gif, .html, .frames, or omitted for terminal playback")
        LOG.debug("Done in %.3fs", time.perf_counter() - t0)
        return

    LOG.debug("Writing %s output to %s", opt.out_format, out_path)
    if opt.out_format == "ansi":
        cap_lines = []
        if opt.caption.text:
            width = max([len(ln) for ln in scaled_art] + [len(ln) for ln in header] + [1])
            cap_lines = _build_caption_lines(opt.caption, width)
        out_lines = render_ansi(
            header, scaled_art, base_img, opt.color_top, opt.matrix,
            cap=opt.caption, cap_lines=cap_lines,
        )
        text = "\n".join(out_lines) + "\n"
        if out_path:
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(text)
        else:
            sys.stdout.write(text)
    else:
        cap_lines = []
        if opt.caption.text:
            width = max([len(ln) for ln in scaled_art] + [len(ln) for ln in header] + [1])
            cap_lines = _build_caption_lines(opt.caption, width)
        doc = render_html(
            header, scaled_art, base_img, opt.color_top, opt.html, opt.matrix,
            cap=opt.caption, cap_lines=cap_lines,
        )
        title = html.escape(os.path.basename(out_path)) if out_path else "ASCII Art"
        doc = doc.replace("<title>ASCII Art</title>", f"<title>{title}</title>", 1)

        if out_path:
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(doc)
        else:
            sys.stdout.write(doc)

    LOG.debug("Done in %.3fs", time.perf_counter() - t0)

if __name__ == "__main__":
    main()
