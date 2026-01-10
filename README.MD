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

Add colors to existing ASCII art:

```bash
python -m ascii_magic.colorize_ascii image.png ascii.txt output.ans
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

## Example Workflow

1. Convert image to ASCII:
   ```bash
   python -m ascii_magic.image_to_ascii photo.png -o art.txt
   ```

2. Colorize the output:
   ```bash
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
