#!/usr/bin/env python3
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

@dataclass
class MatrixOptions:
    enabled: bool = False
    top: bool = False          # apply to header too
    seed: Optional[int] = None
    gamma: float = 2.0

    # green intensity ranges (0..255)
    fg_min: int = 20
    fg_max: int = 255
    bg_min: int = 0
    bg_max: int = 60

    # glyph behavior
    chars: str = r"01ABCDEFGHIJKLMNOPQRSTUVWXYZ@$%&*+;:,.?/\\|[]{}()<>"
    fill_spaces: bool = False  # keep background color even on spaces?

    use_mask: bool = False
    mask_boost: float = 0.30          # add to subject score on non-space pixels (0..1)
    mask_density_floor: float = 0.35  # minimum glyph probability on subject pixels
    bg_dim: float = 0.80              # multiply subject score on background pixels
    bg_density: float = 0.75          # multiply glyph probability on background pixels

@dataclass
class Options:
    out_format: Optional[str] = None  # "ansi" | "html" (None => infer from output extension)

    keep_top: int = 0
    color_top: bool = False

    debug: bool = False
    log_path: Optional[str] = None

    size: SizeOptions = field(default_factory=SizeOptions)
    html: HtmlOptions = field(default_factory=HtmlOptions)
    matrix: MatrixOptions = field(default_factory=MatrixOptions)

# -----------------------------
# Utilities / core logic
# -----------------------------

def print_usage(file=sys.stderr):
    print(
        "usage: colorize_ascii.py <image> <ascii.txt> <out.ans|out.html> "
        "[--format ansi|html] "
        "[--max-rows N] [--max-cols N] "
        "[--rows N] [--cols N] "
        "[--keep-top N] [--color-top] "
        "[--debug] [--log FILE] "
        "[--html-font-size PX] [--html-line-height PX] [--html-fill-spaces] "
        "[--matrix] [--matrix-top] [--matrix-seed N] [--matrix-gamma F] "
        "[--matrix-fg-min N] [--matrix-fg-max N] [--matrix-bg-min N] [--matrix-bg-max N] "
        "[--matrix-chars STR] [--matrix-fill-spaces] "
        "[--matrix-mask] [--matrix-mask-boost F] [--matrix-mask-density-floor F] "
        "[--matrix-bg-dim F] [--matrix-bg-density F] "
        "\n"
        "If out.* is omitted: defaults to ANSI and prints to stdout.\n"
        "To print HTML to stdout, pass --format html.\n",
        file=file,
    )


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

def require_value(argv, i, flag):
    if i + 1 >= len(argv):
        raise SystemExit(f"{flag} requires a value")
    v = argv[i + 1]
    if v.startswith("--"):
        raise SystemExit(f"{flag} requires a value")
    return v


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


def parse_args(argv) -> Tuple[str, str, Optional[str], Options]:
    if "-h" in argv or "--help" in argv:
        print_usage(file=sys.stdout)
        sys.exit(0)

    if len(argv) < 3:
        print_usage(file=sys.stderr)
        sys.exit(2)

    img_path = argv[1]
    ascii_path = argv[2]
    out_path: Optional[str] = None
    i = 3
    if i < len(argv) and not argv[i].startswith("-"):
        cand = argv[i]
        ext = os.path.splitext(cand)[1].lower()
        if cand == "-" or ext in (".ans", ".html"):
            out_path = argv[i]
            i += 1

    opt = Options()

    while i < len(argv):
        a = argv[i]
        if a == "--max-rows":
            opt.size.max_rows = int(require_value(argv, i, a)); i += 2
        elif a == "--max-cols":
            opt.size.max_cols = int(require_value(argv, i, a)); i += 2
        elif a == "--rows":
            opt.size.rows = int(require_value(argv, i, a)); i += 2
        elif a == "--cols":
            opt.size.cols = int(require_value(argv, i, a)); i += 2
        elif a == "--keep-top":
            opt.keep_top = int(require_value(argv, i, a)); i += 2
        elif a == "--color-top":
            opt.color_top = True; i += 1
        elif a == "--format":
            opt.out_format = require_value(argv, i, a).lower(); i += 2
        elif a == "--html-font-size":
            opt.html.font_size_px = int(require_value(argv, i, a)); i += 2
        elif a == "--html-line-height":
            opt.html.line_height_px = int(require_value(argv, i, a)); i += 2
        elif a == "--html-fill-spaces":
            opt.html.fill_spaces = True; i += 1
        elif a == "--debug":
            opt.debug = True
            i += 1
        elif a == "--log":
            opt.log_path = require_value(argv, i, a)
            i += 2
        elif a == "--matrix":
            opt.matrix.enabled = True; i += 1
        elif a == "--matrix-top":
            opt.matrix.top = True; i += 1
        elif a == "--matrix-seed":
            opt.matrix.seed = int(require_value(argv, i, a)); i += 2
        elif a == "--matrix-gamma":
            opt.matrix.gamma = float(require_value(argv, i, a)); i += 2
        elif a == "--matrix-fg-min":
            opt.matrix.fg_min = int(require_value(argv, i, a)); i += 2
        elif a == "--matrix-fg-max":
            opt.matrix.fg_max = int(require_value(argv, i, a)); i += 2
        elif a == "--matrix-bg-min":
            opt.matrix.bg_min = int(require_value(argv, i, a)); i += 2
        elif a == "--matrix-bg-max":
            opt.matrix.bg_max = int(require_value(argv, i, a)); i += 2
        elif a == "--matrix-chars":
            opt.matrix.chars = require_value(argv, i, a); i += 2
        elif a == "--matrix-fill-spaces":
            opt.matrix.fill_spaces = True; i += 1
        elif a == "--matrix-mask":
            opt.matrix.use_mask = True; i += 1
        elif a == "--matrix-mask-boost":
            opt.matrix.mask_boost = float(require_value(argv, i, a)); i += 2
        elif a == "--matrix-mask-density-floor":
            opt.matrix.mask_density_floor = float(require_value(argv, i, a)); i += 2
        elif a == "--matrix-bg-dim":
            opt.matrix.bg_dim = float(require_value(argv, i, a)); i += 2
        elif a == "--matrix-bg-density":
            opt.matrix.bg_density = float(require_value(argv, i, a)); i += 2
        else:
            if not a.startswith("-"):
                raise SystemExit(
                        f"Unexpected bare value: {a}\n"
                        f"Did you forget a flag (e.g. --keep-top {a}) or mean output file (e.g out.ans)?"
                    )
            raise SystemExit(f"Unknown arg: {a}")

    # Infer output format if not explicitly set
    if opt.out_format is None:
        if out_path is None:
            opt.out_format = "ansi"
        else:
            ext = os.path.splitext(out_path)[1].lower()
            opt.out_format = "html" if ext == ".html" else "ansi"

    if opt.out_format not in ("ansi", "html"):
        raise SystemExit("--format must be 'ansi' or 'html'")

    return img_path, ascii_path, out_path, opt


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
def matrix_lines_ansi(lines, img, m: MatrixOptions):
    """ANSI Matrix mode: green glyphs with subject emphasis (edges + stretched luminance)."""
    if not lines:
        return []

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
    lum_vals = list(gray.get_flattened_data())  # 0..255
    lo_v, hi_v = _percentile_stretch(lum_vals, lo=0.02, hi=0.98)
    denom = (hi_v - lo_v) if hi_v > lo_v else 1

    # Tunables (hardcoded for now)
    edge_weight = 0.35   # how much edges contribute to "subjectness"
    edge_gamma = 0.7     # emphasize edges a bit
    base_density = 0.12  # minimum glyph probability

    rng = random.Random(m.seed)

    out_lines = []
    for y in range(h):
        prev_style = None  # (fg_g, bg_g) or None
        row = []

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

            is_subject = (grid[y][x] != " ") if m.use_mask else False

            if m.use_mask:
                subject = min(1.0, subject + ink * m.mask_boost)
                subject *= (m.bg_dim + (1.0 - m.bg_dim) * ink)

            fg_g = int(m.fg_min + subject * (m.fg_max - m.fg_min))
            bg_g = int(m.bg_min + bg_score * (m.bg_max - m.bg_min))

            # Glyph density follows subjectness
            p = base_density + (1.0 - base_density) * subject
            if m.use_mask:
                p = max(p, ink * m.mask_density_floor)
                p *= (m.bg_density + (1.0 - m.bg_density) * ink)

            ch = rng.choice(m.chars) if (rng.random() < p) else " "

            if ch == " " and not m.fill_spaces:
                if prev_style is not None:
                    row.append(f"{ESC}[0m")
                    prev_style = None
                row.append(" ")
                continue

            style = (fg_g, bg_g)
            if style != prev_style:
                row.append(f"{ESC}[0m{ESC}[38;2;0;{fg_g};0m{ESC}[48;2;0;{bg_g};0m")
                prev_style = style

            row.append(ch)

        row.append(f"{ESC}[0m")
        out_lines.append("".join(row))

    return out_lines

def matrix_lines_html(lines, img, m: MatrixOptions, fill_spaces=False):
    """HTML Matrix mode: green glyphs with subject emphasis (edges + stretched luminance)."""
    if not lines:
        return []

    h = len(lines)
    w = max(len(ln) for ln in lines)
    grid = [ln.ljust(w) for ln in lines]  # shape only

    img = img.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")

    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)

    gpx = gray.load()
    epx = edges.load()

    lum_vals = list(gray.get_flattened_data())
    lo_v, hi_v = _percentile_stretch(lum_vals, lo=0.02, hi=0.98)
    denom = (hi_v - lo_v) if hi_v > lo_v else 1

    edge_weight = 0.35
    edge_gamma = 0.7
    base_density = 0.12

    rng = random.Random(m.seed)

    out_lines = []
    for y in range(h):
        prev_style = None
        span_open = False
        row = []

        def close():
            nonlocal span_open
            if span_open:
                row.append("</span>")
                span_open = False

        for x in range(w):
            lum_byte = gpx[x, y]
            lum = (lum_byte - lo_v) / denom
            lum = 0.0 if lum < 0.0 else (1.0 if lum > 1.0 else lum)

            edge = epx[x, y] / 255.0
            edge = edge ** edge_gamma

            subject = (1.0 - edge_weight) * lum + edge_weight * max(lum, edge)
            subject = subject ** m.gamma

            bg_score = (lum ** max(0.1, (m.gamma * 0.9)))

            ink = ink_strength(grid[y][x]) if m.use_mask else 0.0

            is_subject = (grid[y][x] != " ") if m.use_mask else False

            if m.use_mask:
                subject = min(1.0, subject + ink * m.mask_boost)
                subject *= (m.bg_dim + (1.0 - m.bg_dim) * ink)
                    
            fg_g = int(m.fg_min + subject * (m.fg_max - m.fg_min))
            bg_g = int(m.bg_min + bg_score * (m.bg_max - m.bg_min))

            p = base_density + (1.0 - base_density) * subject
            if m.use_mask:
                p = max(p, ink * m.mask_density_floor)
                p *= (m.bg_density + (1.0 - m.bg_density) * ink)

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
                row.append(f'<span style="color: rgb(0,{fg_g},0); background-color: rgb(0,{bg_g},0)">')
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


def render_ansi(
        header: Sequence[str],
        art: Sequence[str],
        img: Image.Image,
        color_top: bool,
        m: MatrixOptions
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
    
    return out_lines

def render_html(
        header: Sequence[str],
        art: Sequence[str],
        img: Image.Image,
        color_top: bool,
        html_opt: HtmlOptions,
        m: MatrixOptions
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

    return wrap_html(
        pre_lines,
        title="ASCII Art",
        font_size_px=html_opt.font_size_px,
        line_height_px=html_opt.line_height_px,
    )


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
    base_img = Image.open(img_path).convert("RGB")

    target_art_h = compute_target_art_height(opt.size.max_rows, len(header), len(art_lines))
    LOG.debug("Target art height (after header): %d", target_art_h)
    
    LOG.debug("Scaling art...")
    scaled_art = scale_art_block(art_lines, target_art_h, opt.size)
    LOG.debug("Scaled art: %d lines", len(scaled_art))
    
    LOG.debug("Writing %s output to %s", opt.out_format, out_path)
    if opt.out_format == "ansi":
        out_lines = render_ansi(header, scaled_art, base_img, opt.color_top, opt.matrix)
        text = "\n".join(out_lines) + "\n"
        if out_path:
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(text)
        else:
            sys.stdout.write(text)
    else:
        doc = render_html(header, scaled_art, base_img, opt.color_top, opt.html, opt.matrix)
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
