# ASCII Magic

Convert images into colorized ASCII art. This project provides two main tools for creating high-quality ASCII art from images.

## Features

- **Image to ASCII conversion** - Convert PNG images directly to ASCII art using high-quality glyph matching
- **ASCII colorization** - Add colors to existing ASCII text by mapping image pixels to characters
- **Flexible output formats** - Generate ANSI, HTML, or braille output
- **Multiple quality modes** - Choose between fast, balanced, or best-quality character matching

## Installation

```bash
pip install -e .
```

## Usage

### Image to ASCII (`image_to_ascii.py`)

Convert an image directly to ASCII art:

```bash
python -m ascii_magic.image_to_ascii input.png -o output.txt
```

**Options:**
- `-c, --cols` - Output width in characters (default: 120)
- `--mode` - Conversion mode: `glyph` (default) or `braille`
- `--quality` - Matching quality: `fast`, `balanced` (default), or `best`
- `--cell-w, --cell-h` - Cell dimensions for glyph rendering
- `--ascii` - Charset preset: `dense` (default) or `printable`
- `--unicode` - Unicode support: `off`, `blocks`, or `big`
- `--font` - Path to custom .ttf font
- `--autocontrast` - Auto-adjust image contrast
- `--gamma` - Apply gamma correction
- `--invert` - Invert image colors
- `--threshold` - Braille mode dot threshold (0-1)

### ASCII Colorization (`colorize_ascii.py`)

Colorize existing ASCII text by sampling colors from the source image. Supports ANSI (terminal) and HTML output, plus an optional “Matrix” green-code mode.

Basic usage (write to a file):

```bash
python -m ascii_magic.colorize_ascii image.png ascii.txt output.ans
If you omit the output file, it defaults to ANSI and prints to stdout (great for piping):

bash
Copy code
python -m ascii_magic.colorize_ascii image.png ascii.txt
To print HTML to stdout:

bash
Copy code
python -m ascii_magic.colorize_ascii image.png ascii.txt --format html > out.html
Output formats: .ans (ANSI), .html (HTML)

Common options
--format ansi|html — Force output format (otherwise inferred from output extension; if no output file, defaults to ansi)

--max-rows N, --max-cols N — Fit output within max dimensions (preserves aspect)

--rows N, --cols N — Exact output dimensions (derives the other dimension if only one is provided)

--keep-top N — Preserve the top N lines as a “header” (excluded from scaling)

--color-top — Apply colorization to the kept header lines too

--debug — Enable verbose debug logging (to stderr)

--log FILE — Write debug logs to a file

HTML tuning
--html-font-size PX — HTML font size (default: 12)

--html-line-height PX — HTML line height (default: same as font size)

--html-fill-spaces — In HTML output, render spaces with background colors (uses &nbsp; spans)

Matrix mode (green “code rain”)
Matrix mode ignores the original ASCII glyphs and renders green glyphs driven by the image (brightness/edges), with adjustable density and colors.

--matrix — Enable Matrix mode

--matrix-top — Apply Matrix mode to the header too

--matrix-seed N — RNG seed (same seed ⇒ same glyph pattern; omit for a different pattern each run)

--matrix-gamma F — Brightness shaping (lower = brighter mids; higher = darker mids)

--matrix-fg-min N, --matrix-fg-max N — Foreground green range (0..255)

--matrix-bg-min N, --matrix-bg-max N — Background green range (0..255)

--matrix-chars STR — Character set to sample from

--matrix-fill-spaces — Keep background color even where the chosen glyph is a space

Matrix subject mask (uses your ASCII art as a subject hint)
If your ASCII text already captures the subject well (non-space characters), you can bias Matrix rendering so the subject “pops” more than the background.

--matrix-mask — Enable mask-based subject emphasis

--matrix-mask-boost F — Boost subject intensity on inked pixels (default: 0.30)

--matrix-mask-density-floor F — Minimum glyph probability on subject pixels (default: 0.35)

--matrix-bg-dim F — Dim subject score on background pixels (default: 0.80)

--matrix-bg-density F — Reduce glyph density on background pixels (default: 0.75)

Example (Matrix + mask + sizing + header):

bash
Copy code
python -m ascii_magic.colorize_ascii image.png ascii.txt out.ans \
  --rows 30 --cols 90 \
  --keep-top 6 --color-top \
  --matrix --matrix-mask \
  --matrix-seed 1 --matrix-gamma 1.2 \
  --matrix-bg-max 90 --matrix-fill-spaces
## Tips

- Cartoon or anime-styled images work best for ASCII conversion
- For colorization, use `--keep-top N` to preserve text headers
- Use `--mode braille` for high-detail output with Unicode braille characters
- HTML output is great for web display and preservation

## License

Apache License 2.0
