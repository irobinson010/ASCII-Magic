# ASCII Magic

Convert images into colorized ASCII art. This project provides two main tools for creating high-quality ASCII art from images.

## Features

- **Image to ASCII conversion** - Convert PNG images directly to ASCII art using high-quality glyph matching
- **ASCII colorization** - Add colors to existing ASCII text by mapping image pixels to characters
- **Flexible output formats** - Generate ANSI, HTML, or braille output
- **Multiple quality modes** - Choose between fast, balanced, or best-quality character matching

## Installation

Requires Python 3.10+.

```bash
pip install -e .          # CLI tools only
pip install -e ".[web]"   # CLI tools + web GUI
```

## Quick Start (Unified CLI)

The `ascii-magic` command is the recommended entry point. It provides all tools under one command:

```bash
# Convert image to ASCII art
ascii-magic image input.png -o output.txt

# Render text as ASCII art
ascii-magic text "Hello, world" -s block -w 80

# Colorize ASCII art with an image
ascii-magic colorize image.png ascii.txt output.ans

# Install art as a login greeting / start the web GUI
ascii-magic greet install output.ans
ascii-magic web

# Show version
ascii-magic --version
```

The individual commands (`image-to-ascii`, `colorize-ascii`, `text-to-ascii`) continue to work as before.

## Development setup

The repo pins its Python version in `.python-version` and locks exact dependency versions in `uv.lock`, so a fresh clone builds the same environment on any OS. With [uv](https://docs.astral.sh/uv/) installed (a single binary — it downloads the right Python by itself):

```bash
uv sync --extra web    # creates .venv/ with pinned Python, deps, and test tools
uv run pytest          # run the test suite
uv run ascii-magic-web # start the web GUI
```

Without uv, stock tooling works too (pip 25.1+ for `--group`):

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[web]" --group dev
.venv/bin/pytest
```

Always create the virtualenv as `.venv/` inside the repo — never at the repo root — so editors and tools auto-detect it and git ignores it.

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
- `--dither` - Braille mode: Floyd-Steinberg dithering (keeps smooth gradients a hard threshold flattens)
- `--color` - Convert and colorize in one step (ANSI, or HTML when the output ends in `.html`) — no separate `colorize` invocation needed
- `--caption TEXT` - Render TEXT as ASCII and stitch it onto the art (see Captions below)

### Captions

Attach text above or below the art — rendered as ASCII letters sized to the art, or as a simple box/banner. Works on both the `image` and `colorize` commands, and composes with `--color` (the caption keeps its own color instead of being smeared by the image):

```bash
ascii-magic image cat.png --color --caption "Whiskers 2010-2024" -o cat.ans
ascii-magic image cat.png --caption "Whiskers" --caption-style box --caption-pos top
```

**Options (prefix `--caption-`):**
- `pos` : `bottom` (default) or `top`
- `style` : `block` (default), `small`, `shadow`, `box`, `banner`
- `scale F` : caption width as a fraction of the art width for rendered styles (default 0.6) — the sizing knob
- `gap N` : blank lines between caption and art (default 1)
- `color COLOR` : caption color — theme name, `#RRGGBB`, or `image` to colorize the caption from the picture itself (it samples the strip of the image nearest the caption); default: terminal foreground
- `align` : `center` (default), `left`, `right`

The web GUI has a matching **Caption** panel (text, position, style, size slider, alignment, color).

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
- `--matrix-color COLOR` : Rain color — `green` (classic, default), `amber`, `cyan`, `crimson`, `violet`, `white`, or any `#RRGGBB`.
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

### Animated matrix mode

Add `--animate` for looping matrix rain driven by the image's subject — bright heads, fading tails, glyphs that flicker, and columns timed so the loop is seamless:

```bash
colorize-ascii photo.png art.txt rain.gif --animate --matrix-seed 42   # animated GIF
colorize-ascii photo.png art.txt rain.html --animate                   # self-playing web page
colorize-ascii photo.png art.txt --animate --loops 3                   # play in the terminal
```

**Options:**
- `--frames N` : Frames per loop (default: 60)
- `--fps F` : Playback speed (default: 12)
- `--tail F` : Fade length in rows behind each drop head (default: 6)
- `--loops N` : Terminal playback repeats; `0` plays until Ctrl-C (default: 3)
- `--reveal` : The rain uncovers the colorized image, which persists beneath it — the loop ends (and holds) on the fully revealed picture

All `--matrix-*` knobs (seed, gamma, intensity ranges, mask biasing, `--matrix-color` themes) apply to the rain too, and `--caption*` works in every animation sink — the caption stays static (matching the rain tint by default, or any `--caption-color` including `image`) while the rain falls. The web GUI exposes the same controls under *Matrix mode → Animate*, previews the animation live, and adds a `.gif` download. Programmatic use: `ascii_magic.pipeline.animate(ctx, ...)` returns an object with `frames_ansi()`, `play()`, `to_gif_bytes()`, and `to_html()`.

### Text to ASCII (`text_to_ascii.py`)

Render plain text into an image and convert it to ASCII art. This module provides a small CLI and programmatic API for producing several simple ASCII styles.

Usage (console script):

```bash
text-to-ascii "Hello, world" -s block -w 80 --font-size 24
```

Or via the unified CLI:

```bash
ascii-magic text "Hello, world" --style block --width 80 --font-size 24
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

## In-memory pipeline (no intermediate files)

You can run image -> ASCII -> colorized output entirely in memory:

```python
from PIL import Image
from ascii_magic.pipeline import AsciiPipelineContext, image_to_ascii, colorize
from ascii_magic.colorize_ascii import Options

img = Image.open("photo.png").convert("RGB")
ctx = AsciiPipelineContext(source_image=img)

ascii_text = image_to_ascii(ctx, mode="braille", cols=120)
ansi_output = colorize(ctx, opt=Options(out_format="ansi"))
html_output = colorize(ctx, opt=Options(out_format="html"))
```

There are also lower-level in-memory helpers if you do not want the context object:
- `ascii_magic.image_to_ascii.image_to_text_glyph_from_image(...)`
- `ascii_magic.image_to_ascii.image_to_braille_from_image(...)`
- `ascii_magic.colorize_ascii.colorize_ascii_text(...)`

## Login greeting (`ascii-magic-greet`)

Install any output as a shell login greeting — the art (or matrix rain) appears whenever you open a terminal or SSH in:

```bash
# Static greeting
colorize-ascii cat.png art.txt cat.ans
ascii-magic-greet install cat.ans

# Animated greeting (matrix rain, plays once per login)
colorize-ascii cat.png art.txt cat.frames --animate --loops 1
ascii-magic-greet install cat.frames
```

The install copies the file to `~/.config/ascii-magic/` and appends a marker-guarded block to your `~/.bashrc` (or `~/.zshrc`) that only fires in interactive terminals. Reinstalling replaces the previous greeting; `ascii-magic-greet remove` cleans everything up.

**Subcommands:** `install FILE`, `remove`, `show`/`preview`, `status`, `play FILE [--fps F] [--loops N]`. Use `--rc PATH` to target a different rc file, and set `ASCII_MAGIC_NO_GREETING=1` to suppress the greeting temporarily.

## Web GUI

A browser interface over the same pipeline: upload an image (or type text), turn every knob the CLIs expose, preview live, and download `.ans`, `.html`, or `.txt` results. Nothing is stored server-side.

```bash
pip install -e ".[web]"
ascii-magic-web            # http://127.0.0.1:8000
```

The **Profile** switch in the header applies sensible defaults per target: *Terminal* (SSH greetings, `.ans` output) or *Web page* (larger canvas, filled spaces, `.html` output).

## Docker

One image covers both the web GUI and one-shot CLI runs (it bundles DejaVu Sans Mono, so glyph matching works out of the box):

```bash
docker build -t ascii-magic .

# Web GUI (default command) — open http://localhost:8000
docker run --rm -p 8000:8000 ascii-magic

# One-shot CLI: mount your working directory at /data and pass any subcommand
docker run --rm -v "$(pwd):/data" ascii-magic image photo.png -o art.txt
docker run --rm -v "$(pwd):/data" ascii-magic colorize photo.png art.txt output.ans
docker run --rm -v "$(pwd):/data" ascii-magic text "Hello" -s block
```

## Tips

- Cartoon or anime-styled images work best for ASCII conversion
- For colorization, use `--keep-top N` to preserve text headers
- Use `--mode braille` for high-detail output with Unicode braille characters; add `--dither` for photos and smooth gradients
- Glyph matching is fully vectorized — `--quality best` is fast enough to use freely
- HTML output is great for web display and preservation

## License

Apache License 2.0

## Console script entry points

The unified `ascii-magic` command (recommended) wraps everything as subcommands:

```bash
ascii-magic image photo.png --mode braille --dither -o art.txt
ascii-magic colorize photo.png art.txt out.gif --animate
ascii-magic greet install out.frames
ascii-magic web
```

The standalone scripts remain and behave identically:

- `ascii-magic` -> `ascii_magic.unified_cli:main` (subcommands: image, text, colorize, greet, web)
- `image-to-ascii` -> `ascii_magic.image_to_ascii:main`
- `colorize-ascii` -> `ascii_magic.colorize_ascii:main`
- `text-to-ascii` -> `ascii_magic.text_to_ascii:main`
- `ascii-magic-greet` -> `ascii_magic.greet:main`
- `ascii-magic-web` -> `ascii_magic.webapp:main` (requires the `[web]` extra)

Prefer calling `ascii-magic` or the individual console scripts after `pip install -e .`. You can also invoke modules directly with `python -m ascii_magic.<module>` when developing.
