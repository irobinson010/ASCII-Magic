#!/usr/bin/env python3
"""Convert text to ASCII art using various fonts and styles."""

import argparse
import os, sys
import logging
from PIL import Image, ImageDraw, ImageFont, ImageOps


# =============================
# Font Management
# =============================


def find_default_font(font_name: str = "DejaVuSansMono"):
    """Find a suitable monospace font on the system."""
    candidates = {
        "DejaVuSansMono": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Monaco.ttf",
            "C:\\Windows\\Fonts\\consola.ttf",
        ],
        "LiberationMono": [
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ],
    }

    for path in candidates.get(font_name, []):
        if os.path.exists(path):
            return path
    return None


def load_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont:
    """Load a font, falling back to default if not found."""
    logger = logging.getLogger(__name__)
    if font_path and os.path.exists(font_path):
        logger.debug("Loading font from %s (size=%d)", font_path, font_size)
        return ImageFont.truetype(font_path, font_size)

    default = find_default_font()
    if default:
        logger.debug("Falling back to default font %s (size=%d)", default, font_size)
        return ImageFont.truetype(default, font_size)

    logger.debug("Using PIL default bitmap font (size=%d)", font_size)
    return ImageFont.load_default()


# =============================
# Text Rendering
# =============================


def measure_text(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Measure text dimensions."""
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def render_text_to_image(
    text: str,
    font_size: int = 24,
    font_path: str | None = None,
    bg_color: str = "white",
    text_color: str = "black",
    padding: int = 2,
) -> Image.Image:
    """Render text to an image."""
    logger = logging.getLogger(__name__)
    logger.debug("Rendering text to image; font_size=%d padding=%d", font_size, padding)
    font = load_font(font_path, font_size)

    # Rough text measurement to size temporary canvas
    rough_w, rough_h = measure_text(text, font)
    # Create a temporary image to compute the ink bbox accurately
    tmp_w = max(rough_w + 64, 256)
    tmp_h = max(rough_h + 64, 256)
    tmp = Image.new("L", (tmp_w, tmp_h), color=0)
    tmp_draw = ImageDraw.Draw(tmp)

    try:
        bbox = tmp_draw.textbbox((0, 0), text, font=font)
    except Exception:
        # Fallback: use font.getbbox
        bbox = font.getbbox(text)

    # bbox: (left, top, right, bottom)
    left, top, right, bottom = bbox
    text_w = right - left
    text_h = bottom - top

    # Small safety pad to account for antialiasing/rounding
    safety = max(1, int(round(font_size * 0.05)))
    pad = max(1, padding)

    img_width = text_w + (pad * 2) + (safety * 2)
    img_height = text_h + (pad * 2) + (safety * 2)

    logger.debug(
        "Measured text bbox=%s size=(%d,%d) final_size=(%d,%d) pad=%d safety=%d",
        bbox,
        text_w,
        text_h,
        img_width,
        img_height,
        pad,
        safety,
    )

    img = Image.new("RGB", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text offset so actual ink aligns within the image with padding
    x = pad + safety - left
    y = pad + safety - top
    logger.debug("Drawing text at offset=(%d,%d)", x, y)
    draw.text((x, y), text, font=font, fill=text_color)

    return img


# =============================
# ASCII Art Generation
# =============================


def text_to_ascii_art(
    text: str,
    style: str = "block",
    width: int = 80,
    font_size: int = 12,
    font_path: str | None = None,
) -> str:
    """
    Convert text to ASCII art.

    Args:
        text: Input text to convert
        style: ASCII art style ('block', 'small', 'shadow')
        width: Output width in characters
        font_size: Font size for rendering

    Returns:
        ASCII art string
    """
    # Basic density-based ASCII renderer.
    # Validate style
    styles = ("block", "small", "shadow")
    if style not in styles:
        raise ValueError(f"unknown style: {style}")

    logger = logging.getLogger(__name__)
    # Render text to an image
    logger.info(
        "Generating ASCII art (style=%s width=%s font_size=%s)", style, width, font_size
    )
    img = render_text_to_image(text, font_size=font_size, font_path=font_path)
    img = img.convert("L")

    # Target character width (number of characters per line)
    target_w = max(1, int(width))

    # Map image pixels -> characters. Characters are taller than wide,
    # so reduce the height when computing rows to keep aspect ratio.
    scale = target_w / max(1, img.width)
    target_h = max(1, int(img.height * scale * 0.5))

    img_small = img.resize((target_w, target_h))
    logger.debug("Resized image to %dx%d for ascii mapping", target_w, target_h)

    # Choose character ramps per style
    if style == "block":
        ramp = "@%#*+=-:. "
    elif style == "small":
        ramp = "@#*.- "
    else:  # shadow
        ramp = " .:-=+*#%@"

    # Use the new flattened data API when available (Pillow deprecation),
    # otherwise fall back to getdata for compatibility.
    if hasattr(img_small, "get_flattened_data"):
        pixels = list(img_small.get_flattened_data())
    else:
        pixels = list(img_small.getdata())
    lines = []
    for row in range(target_h):
        line_chars = []
        for col in range(target_w):
            val = pixels[row * target_w + col]
            idx = int((val / 255) * (len(ramp) - 1))
            ch = ramp[idx]
            line_chars.append(ch)
        lines.append("".join(line_chars))

    return "\n".join(lines)


def text_to_box(text: str, width: int = 80) -> str:
    """Draw text in a simple box."""
    lines = text.split("\n")
    max_len = max(len(line) for line in lines) if lines else 0

    # content width is space for text inside the box
    content_width = max(1, min(max_len, max(1, width - 4)))
    box_width = content_width + 4

    result = []
    result.append("┌" + "─" * (box_width - 2) + "┐")

    for line in lines:
        # truncate long lines to fit the requested width
        if len(line) > content_width:
            logging.getLogger(__name__).debug(
                "Truncating line from %d to %d characters", len(line), content_width
            )
            content = line[:content_width]
        else:
            content = line.ljust(content_width)
        result.append("│ " + content + " │")

    result.append("└" + "─" * (box_width - 2) + "┘")

    return "\n".join(result)


def text_to_banner(text: str, char: str = "#") -> str:
    """Create a simple text banner."""
    border = char * (len(text) + 4)
    return f"{border}\n{char} {text} {char}\n{border}"


def text_to_figlet(text: str, width: int = 80, font: str = "standard") -> str:
    """Classic figlet outline lettering (the traditional terminal-banner look)."""
    import pyfiglet

    return pyfiglet.figlet_format(text, font=font, width=max(20, int(width)))


# Real figlet fonts in ascending size — scaling picks a font instead of
# stretching character cells, so letterforms stay clean at every size.
# Ascending size. Dense on purpose: the sizer picks the fitting font closest
# to the drag/slider target, so gaps in the ladder feel like a dead knob.
_FIGLET_SIZES = (
    "mini", "small", "standard", "big", "chunky", "banner3",
    "epic", "larry3d", "roman", "univers", "colossal", "doh",
)


def _figlet_sized(text: str, width: int, scale: float) -> str:
    """Figlet text sized toward scale x width using the font ladder.

    Candidates render UNWRAPPED for measurement: pyfiglet's width-wrapping
    smushes the wrapped blocks of large fonts into each other (letters merge
    and truncate), so a wrapped render must never be chosen as "fitting".
    """
    import pyfiglet

    target = max(1, int(width * min(1.0, max(0.05, float(scale)))))
    rendered = []
    for font in _FIGLET_SIZES:
        try:
            block = pyfiglet.figlet_format(text, font=font, width=100_000)
        except Exception:
            continue
        w = max((len(ln.rstrip()) for ln in block.splitlines()), default=0)
        if w:
            rendered.append((w, block))
    if not rendered:
        return text
    fitting = [r for r in rendered if r[0] <= width]
    if fitting:
        return min(fitting, key=lambda r: abs(r[0] - target))[1]
    # Nothing fits on one line even in the smallest font: let the small font
    # wrap into stacked blocks (it wraps cleanly, unlike the giants). A single
    # overlong word still falls through to caption_lines' grid-shrink.
    try:
        return pyfiglet.figlet_format(text, font=_FIGLET_SIZES[0], width=max(20, int(width)))
    except Exception:
        return text


def caption_lines(
    text: str,
    width: int,
    style: str = "block",
    scale: float = 0.6,
    align: str = "center",
    font_path: str | None = None,
    cols: int | None = None,
    rows: int | None = None,
) -> list[str]:
    """Render text as ASCII caption lines padded to exactly `width` columns.

    Auto sizing: rendered styles occupy `scale` x width columns; box/banner
    use their natural size. Exact sizing: `cols`/`rows` free-transform the
    block to precise dimensions (GIMP-style), deriving a missing dimension
    from the block's aspect. Meant for stitching above/below art.
    """
    width = max(1, int(width))
    cols = min(width, max(2, int(cols))) if cols else None
    rows = max(1, int(rows)) if rows else None

    # With an exact width requested, aim the generators at it directly so the
    # grid transform starts from the closest natural rendering.
    eff_scale = (cols / width) if cols else scale

    if style == "box":
        block = text_to_box(text, width=cols or width)
    elif style == "banner":
        block = text_to_banner(text)
    elif style == "figlet":
        block = _figlet_sized(text, width, eff_scale)
    else:
        eff = min(1.0, max(0.05, float(eff_scale)))
        target = max(1, int(width * eff))
        block = text_to_ascii_art(text, style=style, width=target, font_path=font_path)

    lines = [ln.rstrip() for ln in block.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if lines and (cols or rows):
        # Exact free transform to cols x rows (missing dim keeps aspect).
        from .colorize_ascii import scale_grid

        nat_w = max(len(ln) for ln in lines)
        nat_h = len(lines)
        tc = cols or max(2, min(width, round(nat_w * (rows / nat_h))))
        tr = rows or max(1, round(nat_h * (tc / nat_w)))
        if (tc, tr) != (nat_w, nat_h):
            lines = [ln.rstrip() for ln in scale_grid([ln.ljust(nat_w) for ln in lines], tr, tc)]
    elif style == "figlet" and lines:
        # Emergency shrink only: even the smallest ladder font can overflow
        # very narrow art. Sizing up is handled by the font ladder itself.
        nat_w = max(len(ln) for ln in lines)
        if nat_w > width:
            from .colorize_ascii import scale_grid

            factor = width / nat_w
            new_h = max(1, round(len(lines) * factor))
            lines = [ln.rstrip() for ln in scale_grid([ln.ljust(nat_w) for ln in lines], new_h, width)]

    # Align the block as a UNIT: rows in multi-row letterforms have different
    # ink widths, so per-line centering shears the letters apart.
    block_w = max((len(ln) for ln in lines), default=0)
    if align == "right":
        pad_left = max(0, width - block_w)
    elif align == "center":
        pad_left = max(0, (width - block_w) // 2)
    else:
        pad_left = 0
    out = []
    for ln in lines:
        ln = (" " * pad_left + ln)[:width]
        out.append(ln.ljust(width))
    return out


def compose_caption(
    art: str,
    text: str,
    position: str = "bottom",
    style: str = "block",
    scale: float = 0.6,
    gap: int = 1,
    align: str = "center",
    font_path: str | None = None,
) -> str:
    """Stitch a rendered text caption above or below a block of ASCII art."""
    art_lines = art.splitlines()
    width = max((len(ln) for ln in art_lines), default=1)
    cap = caption_lines(text, width, style=style, scale=scale, align=align, font_path=font_path)
    spacer = [""] * max(0, int(gap))
    if position == "top":
        combined = cap + spacer + art_lines
    else:
        combined = art_lines + spacer + cap
    return "\n".join(combined)


# =============================
# CLI
# =============================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="text-to-ascii", description="Convert text to ASCII art")

    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to convert",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read text from standard input (supports piped input)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "-s",
        "--style",
        choices=["block", "small", "shadow", "box", "banner", "figlet"],
        default="block",
        help="ASCII art style",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=80,
        help="Output width in characters",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=24,
        help="Font size for rendering (used with block/small/shadow styles)",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Path to .ttf font file",
    )
    parser.add_argument(
        "-c",
        "--char",
        default="#",
        help="Character to use for banner style",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: WARNING)",
    )

    return parser


def main():
    """Main entry point for text-to-ASCII CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Configure logging early so other functions can emit messages
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.WARNING
    logging.basicConfig(
        stream=sys.stdout, level=numeric_level, format="%(levelname)s: %(message)s"
    )

    # If no args, show help and exit 0
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Get input text ! --stdin flag overrides CLI text
    if args.stdin:
        text = sys.stdin.read().strip()
    elif args.text:
        text = args.text
    else:
        # No input text provided; display error, show help, and exit 1
        print("\033[31mError: [1] no text provided\n\033[0m", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Generate ASCII art based on style
    if args.style == "box":
        output = text_to_box(text, width=args.width)
    elif args.style == "banner":
        output = text_to_banner(text, char=args.char)
    elif args.style == "figlet":
        output = text_to_figlet(text, width=args.width)
    else:
        # Generate ascii-art styles using the renderer
        output = text_to_ascii_art(
            text,
            style=args.style,
            width=args.width,
            font_size=args.font_size,
            font_path=args.font,
        )

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
