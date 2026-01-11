#!/usr/bin/env python3
"""Convert text to ASCII art using various fonts and styles."""

import argparse
import os, sys
import numpy as np
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
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, font_size)

    default = find_default_font()
    if default:
        return ImageFont.truetype(default, font_size)

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
    padding: int = 20,
) -> Image.Image:
    """Render text to an image."""
    font = load_font(font_path, font_size)
    text_width, text_height = measure_text(text, font)

    # Add padding
    img_width = text_width + (padding * 2)
    img_height = text_height + (padding * 2)

    # Create image
    img = Image.new("RGB", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text
    draw.text((padding, padding), text, font=font, fill=text_color)

    return img


# =============================
# ASCII Art Generation
# =============================


def text_to_ascii_art(
    text: str,
    style: str = "block",
    width: int = 80,
    font_size: int = 12,
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
    # TODO: Implement ASCII art generation
    # 1. Render text to image
    # 2. Scale to desired width
    # 3. Convert to ASCII using glyph matching or density-based approach
    pass


def text_to_box(text: str, width: int = 80) -> str:
    """Draw text in a simple box."""
    lines = text.split("\n")
    max_len = max(len(line) for line in lines) if lines else 0
    box_width = min(max_len + 4, width)

    result = []
    result.append("┌" + "─" * (box_width - 2) + "┐")

    for line in lines:
        padded = line.ljust(box_width - 4)
        result.append("│ " + padded + " │")

    result.append("└" + "─" * (box_width - 2) + "┘")

    return "\n".join(result)


def text_to_banner(text: str, char: str = "#") -> str:
    """Create a simple text banner."""
    border = char * (len(text) + 4)
    return f"{border}\n{char} {text} {char}\n{border}"


# =============================
# CLI
# =============================


def main():
    """Main entry point for text-to-ASCII CLI."""
    parser = argparse.ArgumentParser(description="Convert text to ASCII art")

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
        choices=["block", "small", "shadow", "box", "banner"],
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

    args = parser.parse_args()

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
    else:
        # TODO: Implement other styles
        output = text_to_banner(text, char=args.char)

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
