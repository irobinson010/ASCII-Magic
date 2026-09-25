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
pipx install ascii-magic-tools           # from PyPI (CLI tools)
pip install "ascii-magic-tools[web]"     # + web GUI
pip install "ascii-magic-tools[video]"   # + video support

# from a checkout:
pip install -e ".[web,video]"
```

The import package is `asciimagic` (the PyPI project `ascii-magic` is an unrelated package that owns the `ascii_magic` import name).

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

## Shell tab completion

Generate a static completion script for bash or zsh — subcommands, flags, and enum values all complete, with filename fallback:

```bash
eval "$(ascii-magic completion bash)"     # in ~/.bashrc
eval "$(ascii-magic completion zsh)"      # in ~/.zshrc
# or install persistently:
ascii-magic completion bash > ~/.local/share/bash-completion/completions/ascii-magic
```

The standalone commands (`image-to-ascii`, `colorize-ascii`, `text-to-ascii`, `ascii-magic-greet`, `ascii-magic-video`, `ascii-magic-web`) are covered by the same script.

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

### Releasing

Bump `version` in `pyproject.toml` and `__version__` in `src/asciimagic/__init__.py`, merge, then tag: `git tag v0.3.0 && git push origin v0.3.0`. The publish workflow runs the tests, builds sdist+wheel, and uploads via PyPI trusted publishing (configure once on PyPI: project `ascii-magic-tools` → publisher = this repo, workflow `publish.yml`, environment `pypi`).

## Usage

### Image to ASCII (`image_to_ascii.py`)

Convert an image directly to ASCII art (preferred: installed console script):

```bash
image-to-ascii input.png -o output.txt
# or, during development:
python -m asciimagic.image_to_ascii input.png -o output.txt
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
- `--rotate` - Rotate clockwise (90/180/270) before conversion; EXIF orientation from phone photos is applied automatically
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
- `style` : `block` (default), `small`, `shadow`, `box`, `banner`, `figlet` (classic terminal-banner lettering)
- `scale F` : caption width as a fraction of the art width for rendered styles (default 0.6) — the sizing knob
- `gap N` : blank lines between caption and art (default 1)
- `color COLOR` : caption color — theme name, `#RRGGBB`, `image` (colors from the strip of the picture nearest the caption), or `image-full` (the whole picture stretched over the text); default: terminal foreground
- `align` : `center` (default), `left`, `right`

The web GUI has a matching **Caption** panel (text, position, style, size slider, alignment, color).

### ASCII Colorization (`colorize_ascii.py`)

Add colors to existing ASCII art (preferred: installed console script):

```bash
colorize-ascii image.png ascii.txt output.ans
# or, during development:
python -m asciimagic.colorize_ascii image.png ascii.txt output.ans
```

If you omit the output file, the tool defaults to ANSI and prints to stdout (convenient for piping). To print HTML to stdout, pass `--format html` and redirect, for example:

```bash
python -m asciimagic.colorize_ascii image.png ascii.txt
python -m asciimagic.colorize_ascii image.png ascii.txt --format html > out.html
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

All `--matrix-*` knobs (seed, gamma, intensity ranges, mask biasing, `--matrix-color` themes) apply to the rain too. Drop positions loop seamlessly; glyph identities re-roll at the wrap point, which reads as the same flicker the rain has everywhere else. `--caption*` works in every animation sink — the caption stays static (matching the rain tint by default, or any `--caption-color` including `image`) while the rain falls. The web GUI exposes the same controls under *Matrix mode → Animate*, previews the animation live, and adds a `.gif` download. Programmatic use: `asciimagic.pipeline.animate(ctx, ...)` returns an object with `frames_ansi()`, `play()`, `to_gif_bytes()`, and `to_html()`.

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
- `-s, --style` : Style to use (`block`, `small`, `shadow`, `box`, `banner`, `figlet`) — `block` is the default
- `-w, --width` : Output width in characters (default: 80)
- `--font-size` : Font size used when rendering text (default: 24)
- `--font` : Path to a .ttf font file to use for rendering
- `-c, --char` : Character to use for the `banner` style
- `--log-level` : CLI logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)

### Japanese and other non-Latin text

Text in any script works in `text`, captions, and compose text layers:

```bash
ascii-magic text "こんにちは" -s block -w 100   # big ASCII-art letters
ascii-magic text "日本語" -s box                 # boxes and banners size to terminal columns
```

- **Rendered styles** (`block`, `small`, `shadow`) draw the text with a font and convert it. If your font lacks the characters, a Japanese/Chinese/Korean font installed on your system is used automatically (Noto Sans CJK, IPA Gothic, Hiragino, Yu Gothic, Meiryo, MS Gothic, ...). If none is installed you get a warning. Install one (e.g. `sudo apt install fonts-noto-cjk`), or point `ASCII_MAGIC_FALLBACK_FONT` or `--font` at a font file.
- **`box` and `banner`** measure width in terminal columns, where Japanese characters take two, so borders line up.
- **`figlet`** fonts only have Latin letters. `--style figlet` explains that instead of printing nothing, and figlet captions and compose layers fall back to `block`.

Translation (e.g. English to Japanese) is planned: see issue #38.

## Example Workflow

1. Convert image to ASCII (preferred: console script):
   ```bash
   image-to-ascii photo.png -o art.txt
   # or during development:
   python -m asciimagic.image_to_ascii photo.png -o art.txt
   ```

2. Colorize the output (preferred: console script):
   ```bash
   colorize-ascii photo.png art.txt output.ans --max-rows 30
   # or during development:
   python -m asciimagic.colorize_ascii photo.png art.txt output.ans --max-rows 30
   ```

3. View in terminal or browser

## In-memory pipeline (no intermediate files)

You can run image -> ASCII -> colorized output entirely in memory:

```python
from PIL import Image
from asciimagic.pipeline import AsciiPipelineContext, image_to_ascii, colorize
from asciimagic.colorize_ascii import Options

img = Image.open("photo.png").convert("RGB")
ctx = AsciiPipelineContext(source_image=img)

ascii_text = image_to_ascii(ctx, mode="braille", cols=120)
ansi_output = colorize(ctx, opt=Options(out_format="ansi"))
html_output = colorize(ctx, opt=Options(out_format="html"))
```

There are also lower-level in-memory helpers if you do not want the context object:
- `asciimagic.image_to_ascii.image_to_text_glyph_from_image(...)`
- `asciimagic.image_to_ascii.image_to_braille_from_image(...)`
- `asciimagic.colorize_ascii.colorize_ascii_text(...)`

## Getting the best look: tune, presets, and your terminal

Which settings look best depends on the picture and on your terminal. Instead of guessing flags, let `tune` show you:

```bash
ascii-magic tune photo.png                    # ~12 numbered variants, each labeled with its exact flags
ascii-magic tune photo.png --color -o sheet.html   # the same, side by side in a browser
ascii-magic tune photo.png --pick 4 --save-preset my-photos
ascii-magic image other.png --preset my-photos     # the same look on any picture
```

**Presets** save a look once and reuse it exactly. Add `--save-preset NAME` to any `image`, `colorize`, `video`, or `text` run to save every non-default flag (never the input or output file). `--preset NAME` brings them back, and flags you pass alongside still win (`--preset my-photos -c 60`). Built-ins:

| Preset (image) | For |
|---|---|
| `photo` | Photos on a light background: braille, dithered, contrast-stretched |
| `photo-dark-terminal` | Photos shown in a dark terminal: bright areas become ink |
| `line-art` | Cartoons, logos, line art: glyph matching at best quality |
| `ssh-safe` | Plain printable ASCII and 256 colors: displays correctly on any server (also for `colorize` and `video`) |

`ascii-magic presets` lists them; `presets show image NAME` and `presets delete image NAME` do what they say. Saved presets live in `~/.config/ascii-magic/presets.json`.

**Colors over SSH and older terminals.** Output uses 24-bit color by default. Terminals that can't show it (common over SSH, the Linux console, older macOS Terminal, tmux without truecolor) show wrong colors or garbage. `--color-depth 256` or `--color-depth 16` converts the output; `--color-depth auto` detects the terminal (files stay 24-bit). The login greeting always uses `auto`, so a greeting installed on a server adapts to whoever logs in. `ASCII_MAGIC_COLOR_DEPTH=256` forces a depth everywhere.

**Squashed or stretched output?** Terminal fonts differ in how tall a character is compared to its width. Run `ascii-magic tune --aspect-chart`, see which box looks square, and pass that number as `--cell-aspect` (or save it in a preset).

## Color overlays (gradients and tints)

Lay a gradient or a solid color over any art: plain text, image conversions, colorized art, or a whole composition.

```bash
ascii-magic text HELLO -s figlet --overlay sunset                  # gradient text
ascii-magic text HELLO -s block --overlay "#00e5ff,#ff2d55" --overlay-direction diagonal
ascii-magic image cat.png --color --overlay ocean --overlay-mode multiply
ascii-magic compose --image cat.png --text HI --style figlet --at top --layer-overlay fire
```

| Option | Values |
|---|---|
| `--overlay` | A palette (`rainbow`, `sunset`, `ocean`, `fire`, `forest`, `neon`, `matrix`, `mono`), one color (theme or `#RRGGBB`), or comma-separated colors for a gradient |
| `--overlay-direction` | `horizontal` (default), `vertical`, `diagonal`, `diagonal-up`, `radial` |
| `--overlay-mode` | `tint` (default: replaces colors), `multiply` (darkens: keeps the image's shading), `screen` (lightens), `overlay` (adds contrast) |
| `--overlay-strength` | `0`–`1`: how much of the overlay to mix in (default 1) |

Works on `text`, `image`, `colorize` (static output), and `compose`. In `compose`, `--overlay` covers the whole canvas and `--layer-overlay` (plus `--layer-overlay-direction/-mode/-strength`) just one layer. The canvas overlay applies last, so at full strength it replaces layer colors; lower its strength to let them show through. `.html` outputs keep the colors, and `--color-depth` still applies. The web GUI has a **Color overlay** section, and overlay fields on the Compose canvas and on each layer.

## Compose images and text (`ascii-magic compose`)

Stack any number of images and text blocks on one canvas and place each one exactly. Start a layer with `--image PATH` or `--text TEXT`; the options after it apply to that layer. Later layers paint over earlier ones.

```bash
# A photo with a figlet title across the top and a boxed label bottom-right
ascii-magic compose --canvas 100x40 --background "#000000" \
  --image cat.png --cols 90 --color image \
  --text "HELLO" --style figlet --at top --dy 1 --color amber \
  --text "est. 2026" --style box --at bottom-right --dx -2 --dy -1 --color cyan \
  -o card.ans

# Save the setup, then re-render it exactly (or edit the JSON by hand)
ascii-magic compose ... --save-scene card.json
ascii-magic compose card.json -o card.html
```

| Layer option | What it does |
|---|---|
| `--at ANCHOR` | `top-left`, `top`, `top-right`, `left`, `center` (default), `right`, `bottom-left`, `bottom`, `bottom-right` |
| `--dx N` / `--dy N` | Nudge after anchoring (negative = left/up) |
| `--x N` / `--y N` | Exact column/row of the layer's top-left corner (overrides `--at`) |
| `--cols N` / `--rows N` | Exact size in characters; set one and the other follows the aspect ratio. Text without either scales to `--scale` (default 0.5) of the canvas width |
| `--color C` | `image` (an image samples its own picture; text samples the picture beneath it), a theme (`green`, `amber`, `cyan`, `crimson`, `violet`, `white`), or `#RRGGBB` |
| `--outline N` | Clear N cells around the layer's ink so it reads over busy art (default 1 for text, 0 for images) |
| `--opaque` | Blank characters cover what's beneath (a solid box) |
| image: `--mode --dither --invert --autocontrast --gamma --threshold --quality --rotate` | Same meaning as `ascii-magic image` |
| text: `--style --align --font` | Same styles as `ascii-magic text` |

Canvas: `--canvas COLSxROWS` (default: just big enough for every layer) and `--background COLOR`. Output is ANSI to stdout, or `.txt` / `.ans` / `.html` by `-o` extension.

In the web GUI, the **Compose** tab does the same with the mouse: add image and text layers, drag a layer's box in the preview to move it, drag its corner to resize (hold **Shift** to keep the aspect ratio), and use the arrow keys to nudge the selected layer one character (**Shift** for five). **Download scene .json** saves a scene file the CLI can render; put the images next to it.

## Video to ASCII (`ascii-magic video`)

Convert a video into a colorized ASCII animation — every format ffmpeg reads (mp4, webm, mov, mkv, avi, gif, ...). Requires the `[video]` extra:

```bash
pip install -e ".[video]"        # or: uv sync --extra video

ascii-magic video clip.mp4 out.gif -c 100 --fps 10   # animated GIF of ASCII frames
ascii-magic video clip.mp4 out.mp4                   # real video WITH the original audio
ascii-magic video clip.mp4 out.frames                # playable via greet/play
ascii-magic video clip.mp4 --loops 0                 # play live in the terminal
```

Frames are sampled at the target `--fps`, converted through the braille pipeline (dithered by default; `--no-dither` to disable), and colorized from each frame. `--max-frames` caps long clips (default 300). A `.frames` output can be installed as an animated login greeting with `ascii-magic greet install`.

`.mp4` output re-encodes the ASCII frames as h264 and muxes the source's audio track back in (`--no-audio` to skip; silent sources fall back gracefully). `--caption*` stamps text onto every frame — `--caption-color image` samples the first frame.

Per-frame rendering is configurable: `--mode glyph` uses character matching instead of braille, and `--matrix` (with `--matrix-color`, `--matrix-seed`, `--matrix-gamma`, `--matrix-mask`) renders each frame as matrix glyphs driven by that frame — the subject moves through the rain. Seeds advance per frame, so seeded runs are reproducible.

### Webcam (live ASCII mirror)

Point the video command at a camera device and it streams to your terminal live until Ctrl-C — mirrored like a real mirror (`--no-mirror` to disable), with all the same mode/matrix/caption knobs:

```bash
ascii-magic video "<video0>"                          # live mirror
ascii-magic video "<video0>" --matrix --matrix-color green   # you, in the Matrix
ascii-magic video "<video0>" selfie.gif --seconds 5   # record 5s to an ASCII GIF
```

Camera recordings are silent (`.mp4` output works but has no audio path).

The web GUI has a matching **Video** tab: pick a file, set width/fps/frame cap and frame mode, press Render (video renders take a few seconds — no live auto-render), preview the animated result, and download `.gif` or `.frames`. The Matrix panel applies to video too.

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

**Resize like GIMP**: the rendered art gets drag handles — pull the right edge for width, the bottom edge to stretch/squish the height, or the corner for a free resize (hold **Shift** to keep aspect). A live `cols × rows` readout follows the drag; double-click the ring to reset the height to auto. Works on images, text, and video (video applies on the next Render).

The **Profile** switch in the header applies sensible defaults per target: *Terminal* (SSH greetings, `.ans` output) or *Web page* (larger canvas, filled spaces, `.html` output).

### Server limits

The server is safe to expose on a LAN or behind a reverse proxy: request bodies over ~102 MB get a 413 before anything touches disk, cross-site browser POSTs get a 403, and each render is checked against a work budget up front (a 400 names the knob to lower). The defaults keep one request to roughly 10 seconds of CPU. Tune them with environment variables:

| Variable | Default | Limits |
|---|---|---|
| `ASCII_MAGIC_MAX_CONCURRENT` | `min(4, cores)` | Renders running at once; extra requests wait up to 15 s, then get a 503 |
| `ASCII_MAGIC_MAX_CELLS` | `250000` | Output characters (cols × rows) |
| `ASCII_MAGIC_MAX_GLYPH_PIXELS` | `16000000` | Glyph mode: characters × cell width × cell height |
| `ASCII_MAGIC_MAX_ANIM_CELL_FRAMES` | `1000000` | Animations and video: characters × frames |
| `ASCII_MAGIC_MAX_DECODED_VIDEO_FRAMES` | `5000` | Video frames decoded per upload (sampling skips frames but still decodes them) |
| `ASCII_MAGIC_ALLOWED_ORIGINS` | *(none)* | Extra allowed `Origin`s, comma-separated, for a proxy that rewrites `Host` |

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

- `ascii-magic` -> `asciimagic.unified_cli:main` (subcommands: image, text, colorize, greet, web)
- `image-to-ascii` -> `asciimagic.image_to_ascii:main`
- `colorize-ascii` -> `asciimagic.colorize_ascii:main`
- `text-to-ascii` -> `asciimagic.text_to_ascii:main`
- `ascii-magic-greet` -> `asciimagic.greet:main`
- `ascii-magic-web` -> `asciimagic.webapp:main` (requires the `[web]` extra)

Prefer calling `ascii-magic` or the individual console scripts after `pip install -e .`. You can also invoke modules directly with `python -m asciimagic.<module>` when developing.
