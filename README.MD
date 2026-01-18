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

Convert an image directly to ASCII art (preferred: installed console script):

```bash
image-to-ascii input.png -o output.txt
# or, during development:
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

Add colors to existing ASCII art (preferred: installed console script):

```bash
colorize-ascii image.png ascii.txt output.ans
# or, during development:
python -m ascii_magic.colorize_ascii image.png ascii.txt output.ans
```

If you omit the output file, the tool defaults to ANSI and prints to stdout (convenient for piping). To print HTML to stdout, pass `--format html` and redirect, for example:

```bash
python -m ascii_magic.colorize_ascii image.png ascii.txt
python -m ascii_magic.colorize_ascii image.png ascii.txt --format html > out.html
```

**Output formats:** `.ans` (ANSI), `.html` (HTML)

**Options:**
- `--format` - Output format: `ansi` or `html`
- `--max-rows, --max-cols` - Maximum output dimensions
- `--rows, --cols` - Exact output dimensions
- `--keep-top N` - Preserve top N lines without colorization
- `--color-top` - Apply colors to kept header lines
- `--html-font-size PX` - HTML font size (default: 12)
- `--html-line-height PX` - HTML line height
- `--html-fill-spaces` - Colorize spaces in HTML output
- `--log FILE` — Write debug logs to a file

### Matrix mode

A special colorization mode that produces the classic "Matrix" effect: green glyphs drawn with emphasis on subject areas (edges and stretched luminance). Enable with `--matrix` and tune behavior with the flags below.

**Options:**
- `--matrix` : Enable Matrix mode (green glyphs driven by image luminance/edges)
- `--matrix-top` : Apply Matrix mode to the header/header-lines preserved by `--keep-top`
- `--matrix-seed N` : Deterministic RNG seed (integer). Use to reproduce the same random glyph placement.
- `--matrix-gamma F` : Gamma applied to the subject score (default: 2.0). Higher values emphasize bright/edge regions.
- `--matrix-fg-min N`, `--matrix-fg-max N` : Foreground (glyph) green intensity range (0..255). Defaults: `20..255`.
- `--matrix-bg-min N`, `--matrix-bg-max N` : Background green intensity range (0..255). Defaults: `0..60`.
- `--matrix-chars STR` : String of glyphs to choose from (default includes digits, letters, and symbols).
- `--matrix-fill-spaces` : Keep background color on space characters (fills spaces with background color instead of leaving them blank).
- `--matrix-mask` : Use the input ASCII glyphs as a mask to bias glyph placement toward inked characters.
- `--matrix-mask-boost F` : When `--matrix-mask` is enabled, how much to boost subject score on inked pixels (0..1, default ~0.30).
- `--matrix-mask-density-floor F` : Minimum glyph probability on masked/subject pixels (0..1, default ~0.35).
- `--matrix-bg-dim F` : Multiply subject score on background pixels (0..1, default ~0.80).
- `--matrix-bg-density F` : Multiply glyph probability on background pixels (0..1, default ~0.75).

Example (ANSI Matrix output):

```bash
colorize-ascii photo.png art.txt output.ans --matrix --matrix-seed 42 --matrix-gamma 2.2 --matrix-fill-spaces
```

Example (HTML Matrix output with larger font):

```bash
colorize-ascii photo.png art.txt output.html --format html --html-font-size 16 --matrix --matrix-mask --matrix-mask-boost 0.4
```

Matrix subject mask (uses your ASCII art as a subject hint)

If your ASCII text already captures the subject well (non-space characters), enabling `--matrix-mask` will bias Matrix glyph placement so the subject "pops" more than the background. Use `--matrix-mask-boost` and `--matrix-mask-density-floor` to tune how strongly inked characters influence glyph density and intensity.

## Example Workflow

1. Convert image to ASCII (preferred: console script):
   ```bash
   image-to-ascii photo.png -o art.txt
   # or during development:
   python -m ascii_magic.image_to_ascii photo.png -o art.txt
   ```

2. Colorize the output (preferred: console script):
   ```bash
   colorize-ascii photo.png art.txt output.ans --max-rows 30
   # or during development:
   python -m ascii_magic.colorize_ascii photo.png art.txt output.ans --max-rows 30
   ```

3. View in terminal or browser

## Tips

- Cartoon or anime-styled images work best for ASCII conversion
- For colorization, use `--keep-top N` to preserve text headers
- Use `--mode braille` for high-detail output with Unicode braille characters
- HTML output is great for web display and preservation

## License

Apache License 2.0

## Text to ASCII (`text_to_ascii.py`)

Render plain text into an image and convert it to ASCII art. This module provides a small CLI and programmatic API for producing several simple ASCII styles.

Usage (console script):

```bash
text-to-ascii "Hello, world" -s block -w 80 --font-size 24
```

Or via the module:

```bash
python -m ascii_magic.text_to_ascii "Hello, world" --style block --width 80 --font-size 24
```

Options:
- `--stdin` : Read text from standard input (useful for piping)
- `-o, --output` : Write ASCII output to a file (default: stdout)
- `-s, --style` : Style to use (`block`, `small`, `shadow`, `box`, `banner`) — `block` is the default
- `-w, --width` : Output width in characters (default: 80)
- `--font-size` : Font size used when rendering text (default: 24)
- `--font` : Path to a .ttf font file to use for rendering
- `-c, --char` : Character to use for the `banner` style
- `--log-level` : CLI logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)

Notes about rendering behavior:
- The text renderer now measures the ink bounding box (not just a rough font box) and draws text with a small safety margin. That reduces the need for large, conservative padding and avoids clipped glyphs (accents, descenders, negative bearings).
- Debug logging is available (use `--log-level DEBUG`) and will emit measurements and drawing offsets when rendering text.

## Console script entry points

This project installs three console scripts (see `pyproject.toml`):

- `image-to-ascii` -> `ascii_magic.image_to_ascii:main`
- `colorize-ascii` -> `ascii_magic.colorize_ascii:main`
- `text-to-ascii` -> `ascii_magic.text_to_ascii:main`

Prefer calling the installed console scripts (above) after `pip install -e .` or using `pipx` for a global install. You can also invoke modules directly with `python -m ascii_magic.<module>` when developing.
