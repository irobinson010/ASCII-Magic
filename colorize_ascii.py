#!/usr/bin/env python3
import sys
import os
import html
from PIL import Image

ESC = "\x1b"


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


def require_value(argv, i, flag):
    if i + 1 >= len(argv):
        raise SystemExit(f"{flag} requires a value")
    return argv[i + 1]


def parse_args(argv):
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

    # sizing
    rows = None
    cols = None
    max_rows = None
    max_cols = None

    # header behavior
    keep_top = 0
    color_top = False

    # output format
    out_format = None  # "ansi" | "html"

    # html tuning
    html_font_size_px = 12
    html_line_height_px = None  # None => match font-size
    html_fill_spaces = False

    i = 4
    while i < len(argv):
        a = argv[i]
        if a == "--max-rows":
            max_rows = int(require_value(argv, i, a)); i += 2
        elif a == "--max-cols":
            max_cols = int(require_value(argv, i, a)); i += 2
        elif a == "--rows":
            rows = int(require_value(argv, i, a)); i += 2
        elif a == "--cols":
            cols = int(require_value(argv, i, a)); i += 2
        elif a == "--keep-top":
            keep_top = int(require_value(argv, i, a)); i += 2
        elif a == "--color-top":
            color_top = True; i += 1
        elif a == "--format":
            out_format = require_value(argv, i, a).lower(); i += 2
        elif a == "--html-font-size":
            html_font_size_px = int(require_value(argv, i, a)); i += 2
        elif a == "--html-line-height":
            html_line_height_px = int(require_value(argv, i, a)); i += 2
        elif a == "--html-fill-spaces":
            html_fill_spaces = True; i += 1
        else:
            raise SystemExit(f"Unknown arg: {a}")

    if out_format is None:
        ext = os.path.splitext(out_path)[1].lower()
        out_format = "html" if ext == ".html" else "ansi"

    if out_format not in ("ansi", "html"):
        raise SystemExit("--format must be 'ansi' or 'html'")

    return (
        img_path,
        ascii_path,
        out_path,
        max_rows,
        max_cols,
        rows,
        cols,
        keep_top,
        color_top,
        out_format,
        html_font_size_px,
        html_line_height_px,
        html_fill_spaces,
    )


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
                    # Use &nbsp; so background is visibly applied
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
        "  <meta charset=\"utf-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"  <title>{html.escape(title)}</title>\n"
        "  <style>\n"
        "    html, body { margin: 0; background: #000; }\n"
        "    .wrap { padding: 16px; }\n"
        "    pre {\n"
        "      margin: 0;\n"
        "      white-space: pre;\n"
        "      overflow: auto;\n"
        "      font-family: \"Hack\", \"JetBrains Mono\", \"Cascadia Mono\", \"Fira Code\", Consolas, monospace;\n"
        "      font-variant-ligatures: none;\n"
        f"      font-size: {font_size_px}px;\n"
        f"      line-height: {line_height_px}px;\n"
        "      letter-spacing: 0;\n"
        "    }\n"
        "  </style>\n"
        "</head>\n<body>\n"
        "  <div class=\"wrap\">\n"
        "    <pre>\n"
        + "\n".join(pre_lines) +
        "\n    </pre>\n"
        "  </div>\n"
        "</body>\n</html>\n"
    )


def main():
    (
        img_path,
        ascii_path,
        out_path,
        max_rows,
        max_cols,
        rows,
        cols,
        keep_top,
        color_top,
        out_format,
        html_font_size_px,
        html_line_height_px,
        html_fill_spaces,
    ) = parse_args(sys.argv)

    with open(ascii_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if not lines:
        open(out_path, "w", encoding="utf-8").close()
        return

    keep_top = max(0, min(keep_top, len(lines)))
    header = lines[:keep_top]
    art_lines = lines[keep_top:]

    base_img = Image.open(img_path).convert("RGB")

    # Art height allowed when using max-rows (header counts toward total)
    if max_rows is None:
        target_art_h = len(art_lines)
    else:
        target_art_h = max(0, max_rows - len(header))

    # Scale art (fit or force exact dimensions)
    scaled_art = []
    if art_lines and target_art_h > 0:
        src_h = len(art_lines)
        src_w = max(len(ln) for ln in art_lines)
        art_rect = [ln.ljust(src_w) for ln in art_lines]

        # EXACT size mode (wins over max-rows/max-cols)
        if rows is not None or cols is not None:
            target_h = rows if rows is not None else src_h
            target_w = cols if cols is not None else src_w

            # If only one provided, keep aspect by deriving the other
            if rows is not None and cols is None:
                target_w = max(1, round(src_w * (target_h / src_h)))
            elif cols is not None and rows is None:
                target_h = max(1, round(src_h * (target_w / src_w)))

            scaled_art = scale_grid(art_rect, target_h, target_w)

        else:
            # FIT mode (preserve aspect) using max constraints
            scale = 1.0
            if target_art_h is not None and target_art_h > 0:
                scale = min(scale, target_art_h / src_h)
            if max_cols is not None and max_cols > 0:
                scale = min(scale, max_cols / src_w)

            if scale < 1.0:
                target_h = max(1, round(src_h * scale))
                target_w = max(1, round(src_w * scale))
                scaled_art = scale_grid(art_rect, target_h, target_w)
            else:
                scaled_art = art_rect

    if out_format == "ansi":
        out_lines = []
        if header:
            if color_top:
                out_lines.extend(colorize_lines_ansi(header, base_img, color_spaces=False))
            else:
                out_lines.extend(header)
        if scaled_art:
            out_lines.extend(colorize_lines_ansi(scaled_art, base_img, color_spaces=False))

        with open(out_path, "w", encoding="utf-8") as out:
            out.write("\n".join(out_lines) + "\n")

    else:
        pre_lines = []
        if header:
            if color_top:
                pre_lines.extend(colorize_lines_html(header, base_img, color_spaces=False, fill_spaces=html_fill_spaces))
            else:
                pre_lines.extend([html.escape(ln) for ln in header])

        if scaled_art:
            pre_lines.extend(colorize_lines_html(scaled_art, base_img, color_spaces=False, fill_spaces=html_fill_spaces))

        doc = wrap_html(
            pre_lines,
            title=os.path.basename(out_path),
            font_size_px=html_font_size_px,
            line_height_px=html_line_height_px,
        )
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(doc)


if __name__ == "__main__":
    main()

