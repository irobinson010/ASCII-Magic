#!/usr/bin/env python3
import sys
import os
import html
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
from PIL import Image

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
class Options:
    out_format: Optional[str] = None  # "ansi" | "html" (None => infer from output extension)

    keep_top: int = 0
    color_top: bool = False

    size: SizeOptions = field(default_factory=SizeOptions)
    html: HtmlOptions = field(default_factory=HtmlOptions)


# -----------------------------
# Utilities / core logic
# -----------------------------

def require_value(argv, i, flag):
    if i + 1 >= len(argv):
        raise SystemExit(f"{flag} requires a value")
    return argv[i + 1]


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


def parse_args(argv) -> Tuple[str, str, str, Options]:
    # usage:
    # colorize_ascii.py image.png ascii.txt out.ans|out.html [options]
    if len(argv) < 4:
        print(
            "usage: colorize_ascii.py <image> <ascii.txt> <out.ans|out.html> "
            "[--format ansi|html] "
            "[--max-rows N] [--max-cols N] "
            "[--rows N] [--cols N] "
            "[--keep-top N] [--color-top] "
            "[--html-font-size PX] [--html-line-height PX] [--html-fill-spaces]",
            file=sys.stderr,
        )
        sys.exit(2)

    img_path, ascii_path, out_path = argv[1], argv[2], argv[3]
    opt = Options()

    i = 4
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
        else:
            raise SystemExit(f"Unknown arg: {a}")

    # Infer output format if not explicitly set
    if opt.out_format is None:
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


def render_ansi(header: Sequence[str], art: Sequence[str], img: Image.Image, color_top: bool) -> List[str]:
    out_lines: List[str] = []
    if header:
        if color_top:
            out_lines.extend(colorize_lines_ansi(header, img, color_spaces=False))
        else:
            out_lines.extend(header)
    if art:
        out_lines.extend(colorize_lines_ansi(art, img, color_spaces=False))
    return out_lines


def render_html(header: Sequence[str], art: Sequence[str], img: Image.Image, color_top: bool, html_opt: HtmlOptions) -> str:
    pre_lines: List[str] = []

    if header:
        if color_top:
            pre_lines.extend(colorize_lines_html(header, img, color_spaces=False, fill_spaces=html_opt.fill_spaces))
        else:
            pre_lines.extend([html.escape(ln) for ln in header])

    if art:
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

    lines = read_ascii_file(ascii_path)
    if not lines:
        open(out_path, "w", encoding="utf-8").close()
        return

    header, art_lines = split_header(lines, opt.keep_top)
    base_img = Image.open(img_path).convert("RGB")

    target_art_h = compute_target_art_height(opt.size.max_rows, len(header), len(art_lines))
    scaled_art = scale_art_block(art_lines, target_art_h, opt.size)

    if opt.out_format == "ansi":
        out_lines = render_ansi(header, scaled_art, base_img, opt.color_top)
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("\n".join(out_lines) + "\n")
    else:
        doc = render_html(header, scaled_art, base_img, opt.color_top, opt.html)

        # keep same title behavior as before (basename)
        doc = doc.replace("<title>ASCII Art</title>", f"<title>{html.escape(os.path.basename(out_path))}</title>", 1)

        with open(out_path, "w", encoding="utf-8") as out:
            out.write(doc)


if __name__ == "__main__":
    main()
