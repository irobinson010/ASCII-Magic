"""Compose images and text into one piece of ASCII art.

A scene is a canvas plus an ordered stack of layers. Each layer is an image
or a block of text, converted to characters at its own size, then painted
onto the canvas at a position: an anchor (``top``, ``center``,
``bottom-right``, ...) nudged by ``dx``/``dy``, or an exact ``x``/``y``.
Later layers paint over earlier ones; blank characters are see-through
unless the layer is ``opaque``.

Color is per layer:
  - ``None``: the terminal's default foreground (plain text)
  - ``"image"``: an image layer samples its own picture; a text layer samples
    the image colors *underneath* it (text "cut from" the photo)
  - a theme name (green, amber, ...) or ``#RRGGBB``: a solid tint

Scenes load from and save to JSON, so a good setup can be reused exactly::

    {"canvas": {"cols": 100, "rows": 40, "background": "#000000"},
     "layers": [
       {"type": "image", "src": "photo.png", "cols": 80, "at": "center", "color": "image"},
       {"type": "text", "text": "HELLO", "style": "figlet", "at": "top", "dy": 1,
        "color": "#ffcc00"}]}
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import re
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .textwidth import CONT, char_width, ljust as ljust_w, str_width, to_cells

RGB = Tuple[int, int, int]
Cell = Tuple[str, Optional[RGB]]

ANCHORS = (
    "top-left", "top", "top-right",
    "left", "center", "right",
    "bottom-left", "bottom", "bottom-right",
)
TEXT_STYLES = ("block", "small", "shadow", "box", "banner", "figlet")
IMAGE_MODES = ("braille", "glyph")
QUALITIES = ("fast", "balanced", "best")
ALIGNS = ("left", "center", "right")
FORMATS = ("text", "ansi", "html")

# Characters that count as empty space when layers are stacked.
_BLANKS = frozenset(" ⠀")


@dataclass
class Layer:
    type: str = "text"                 # "image" | "text"
    name: Optional[str] = None

    # placement
    at: str = "center"                 # anchor on the canvas (ignored when x/y set)
    x: Optional[int] = None            # exact column of the layer's left edge
    y: Optional[int] = None            # exact row of the layer's top edge
    dx: int = 0                        # nudge after anchoring (+ right)
    dy: int = 0                        # nudge after anchoring (+ down)

    # size (characters); unset dimensions follow the source's aspect
    cols: Optional[int] = None
    rows: Optional[int] = None
    scale: float = 0.5                 # text only: fraction of the canvas width when cols/rows unset

    color: Optional[str] = None        # None | "image" | theme | "#RRGGBB"
    opaque: bool = False               # blanks overwrite what is beneath
    # Knock out N cells around the layer's ink so it reads over busy art.
    # None = auto: 1 for text, 0 for images.
    outline: Optional[int] = None

    # Color overlay across this layer (see asciimagic.overlay)
    overlay: Optional[str] = None
    overlay_direction: str = "horizontal"
    overlay_mode: str = "tint"
    overlay_strength: float = 1.0

    # image layers
    src: Optional[str] = None
    mode: str = "braille"
    quality: str = "balanced"
    dither: bool = False
    invert: bool = False
    autocontrast: bool = False
    gamma: float = 1.0
    threshold: float = 0.5
    rotate: int = 0

    # text layers
    text: Optional[str] = None
    translate: Optional[str] = None    # translate the text from English to this language first
    style: str = "block"
    align: str = "left"
    font: Optional[str] = None

    def validate(self) -> None:
        if self.type not in ("image", "text"):
            raise ValueError(f"layer type must be 'image' or 'text', got {self.type!r}")
        if self.at not in ANCHORS:
            raise ValueError(f"unknown anchor {self.at!r}; expected one of {', '.join(ANCHORS)}")
        for dim in ("cols", "rows"):
            v = getattr(self, dim)
            if v is not None and v < 1:
                raise ValueError(f"{dim} must be >= 1, got {v}")
        self.overlay_obj()  # raises ValueError for a bad spec
        if self.outline is not None and not (0 <= self.outline <= 10):
            raise ValueError(f"outline must be 0..10, got {self.outline}")
        if not (0 < self.scale <= 1):
            raise ValueError(f"scale must be in (0, 1], got {self.scale}")
        if self.color not in (None, "image"):
            from .colorize_ascii import parse_matrix_color

            parse_matrix_color(self.color)  # raises ValueError with the allowed names
        if self.type == "image":
            if self.mode not in IMAGE_MODES:
                raise ValueError(f"unknown image mode {self.mode!r}")
            if self.quality not in QUALITIES:
                raise ValueError(f"unknown quality {self.quality!r}")
        else:
            if not self.text:
                raise ValueError("text layer needs non-empty text")
            if self.style not in TEXT_STYLES:
                raise ValueError(f"unknown text style {self.style!r}; expected one of {', '.join(TEXT_STYLES)}")
            if self.align not in ALIGNS:
                raise ValueError(f"unknown align {self.align!r}")
            if self.translate is not None and not re.fullmatch(r"[a-z]{2,3}", self.translate):
                raise ValueError(f"translate must be a language code like 'ja', got {self.translate!r}")

    def overlay_obj(self):
        if not self.overlay:
            return None
        from .overlay import Overlay

        return Overlay.parse(self.overlay, self.overlay_direction, self.overlay_mode, self.overlay_strength)

    @property
    def effective_outline(self) -> int:
        if self.outline is not None:
            return self.outline
        return 1 if self.type == "text" else 0

    def label(self, index: int) -> str:
        if self.name:
            return self.name
        if self.type == "text":
            return f"text {index + 1}: {self.text[:20]!r}"
        return f"image {index + 1}: {os.path.basename(self.src or '?')}"


@dataclass
class Canvas:
    cols: Optional[int] = None         # None: fit the layers
    rows: Optional[int] = None
    background: Optional[str] = None   # theme or #RRGGBB; ANSI/HTML only
    # Color overlay across the whole canvas, applied after every layer
    overlay: Optional[str] = None
    overlay_direction: str = "horizontal"
    overlay_mode: str = "tint"
    overlay_strength: float = 1.0

    def overlay_obj(self):
        if not self.overlay:
            return None
        from .overlay import Overlay

        return Overlay.parse(self.overlay, self.overlay_direction, self.overlay_mode, self.overlay_strength)


@dataclass
class Scene:
    canvas: Canvas = field(default_factory=Canvas)
    layers: List[Layer] = field(default_factory=list)

    # ---- (de)serialization ----

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[str] = None) -> "Scene":
        if not isinstance(data, dict):
            raise ValueError("scene must be a JSON object")
        canvas = _dataclass_from(Canvas, data.get("canvas") or {}, "canvas")
        raw_layers = data.get("layers") or []
        if not isinstance(raw_layers, list):
            raise ValueError("'layers' must be a list")
        layers = []
        for i, raw in enumerate(raw_layers):
            layer = _dataclass_from(Layer, raw, f"layers[{i}]")
            if layer.type == "image" and layer.src and base_dir and not os.path.isabs(layer.src):
                layer.src = os.path.join(base_dir, layer.src)
            layers.append(layer)
        return cls(canvas=canvas, layers=layers)

    def to_dict(self) -> Dict[str, Any]:
        """Only non-default fields, so saved scenes stay readable."""
        return {
            "canvas": _non_defaults(self.canvas, Canvas()),
            "layers": [
                {"type": layer.type, **_non_defaults(layer, Layer(type=layer.type))}
                for layer in self.layers
            ],
        }

    @classmethod
    def load(cls, path: str) -> "Scene":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)))

    def save(self, path: str) -> None:
        """Image paths are written relative to the scene file, so a folder
        holding both can be moved or shared."""
        data = self.to_dict()
        base = os.path.dirname(os.path.abspath(path))
        for raw in data["layers"]:
            if raw.get("src"):
                try:
                    rel = os.path.relpath(os.path.abspath(raw["src"]), base)
                    # Forward slashes load on every OS (Windows accepts them),
                    # so a scene made on Windows still renders on Linux.
                    raw["src"] = rel.replace(os.sep, "/")
                except ValueError:  # different drive on Windows: keep absolute
                    raw["src"] = os.path.abspath(raw["src"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")


def _dataclass_from(cls, raw: Any, where: str):
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object")
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(names))
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {', '.join(unknown)}")
    return cls(**raw)


def _non_defaults(obj, default) -> Dict[str, Any]:
    return {
        f.name: getattr(obj, f.name)
        for f in dataclasses.fields(obj)
        if f.name != "type" and getattr(obj, f.name) != getattr(default, f.name)
    }


# =============================
# Layer rendering
# =============================


@dataclass
class Block:
    """A rendered layer: characters, per-cell colors, and (for images) the
    picture's color at every cell, so text above can sample it."""

    lines: List[str]
    colors: List[List[Optional[RGB]]]
    under: Optional[List[List[RGB]]] = None

    @property
    def width(self) -> int:
        """In terminal columns (a CJK character is two)."""
        return max((str_width(ln) for ln in self.lines), default=0)

    @property
    def cells(self) -> List[List[str]]:
        """One entry per column; a wide character is followed by CONT."""
        w = self.width
        rows = [to_cells(ln) for ln in self.lines]
        return [r + [" "] * (w - len(r)) for r in rows]

    @property
    def height(self) -> int:
        return len(self.lines)


def _cell_colors(img: Image.Image, w: int, h: int) -> List[List[RGB]]:
    small = img.convert("RGB").resize((max(1, w), max(1, h)), Image.Resampling.LANCZOS)
    px = small.load()
    return [[px[x, y] for x in range(w)] for y in range(h)]


def _solid(color: Optional[str]) -> Optional[RGB]:
    if color in (None, "image"):
        return None
    from .colorize_ascii import parse_matrix_color

    return parse_matrix_color(color)


def render_image_layer(layer: Layer, img: Optional[Image.Image] = None) -> Block:
    """Convert an image layer. `img` overrides `layer.src` (web uploads)."""
    from .image_to_ascii import (
        image_to_braille_from_image,
        image_to_text_glyph_from_image,
        make_charset,
        open_oriented,
        rotate_cw,
    )

    if img is None:
        if not layer.src:
            raise ValueError("image layer needs 'src'")
        img = open_oriented(layer.src, "RGB")
    img = rotate_cw(img.convert("RGB"), layer.rotate)

    cw, ch = (8, 16) if layer.mode == "glyph" else (2, 4)
    w_img, h_img = img.size
    cols, rows = layer.cols, layer.rows
    if cols is None and rows is None:
        cols = 80
    if cols is None:
        # Derive width from the requested height using the converters' formula.
        cols = max(1, round(rows * w_img * ch / (h_img * cw)))
    if rows is not None:
        # Exact height: pre-stretch onto the converter's cell grid so its
        # natural-rows formula lands exactly on `rows`.
        img = img.resize((cols * cw, rows * ch), Image.Resampling.LANCZOS)

    if layer.mode == "glyph":
        art = image_to_text_glyph_from_image(
            img=img, cols=cols, cell_w=cw, cell_h=ch,
            charset=make_charset(unicode_mode="off", ascii_preset="dense"),
            quality=layer.quality, font_path=None, font_size=None,
            autocontrast=layer.autocontrast, gamma=layer.gamma, invert=layer.invert, topk=24,
        )
    else:
        art = image_to_braille_from_image(
            img, cols=cols, autocontrast=layer.autocontrast, gamma=layer.gamma,
            invert=layer.invert, threshold=layer.threshold, dither=layer.dither,
        )

    lines = art.splitlines()
    w = max((len(ln) for ln in lines), default=0)
    lines = [ln.ljust(w) for ln in lines]
    under = _cell_colors(img, w, len(lines))
    solid = _solid(layer.color)
    if layer.color == "image":
        colors = [list(row) for row in under]
    else:
        colors = [[solid] * w for _ in lines]
    return Block(lines=lines, colors=colors, under=under)


_NATURAL_WIDTH = 4096


def render_text_layer(layer: Layer, ref_width: int) -> Block:
    """Render a text layer to a tight block (no surrounding padding)."""
    from .text_to_ascii import caption_lines

    if layer.cols:
        width = layer.cols
    elif layer.style in ("box", "banner") and not layer.rows:
        # Natural-size styles: render uncropped; the canvas clips instead.
        width = _NATURAL_WIDTH
    else:
        # Rendered styles size to `scale` x the reference width (or to `rows`).
        width = max(2, ref_width)
    lines = caption_lines(
        layer.text, width, style=layer.style, scale=layer.scale, align="left",
        font_path=layer.font, cols=layer.cols, rows=layer.rows,
    )
    lines = _trim_block(lines)
    if layer.align != "left" and lines:
        # Align lines within the block itself (multi-line text).
        w = max(str_width(ln) for ln in lines)
        out = []
        for ln in lines:
            s = ln.rstrip()
            pad = w - str_width(s)
            if layer.align == "center":
                s = " " * (pad // 2) + s
            else:
                s = " " * pad + s
            out.append(s)
        lines = out
    w = max((str_width(ln) for ln in lines), default=0)
    lines = [ljust_w(ln, w) for ln in lines]
    solid = _solid(layer.color)
    return Block(lines=lines, colors=[[solid] * w for _ in lines])


def _trim_block(lines: Sequence[str]) -> List[str]:
    """Drop blank rows at the edges and the common left margin."""
    lines = [ln.rstrip() for ln in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    cut = min(indents, default=0)
    return [ln[cut:] for ln in lines]


# =============================
# Composition
# =============================


@dataclass
class Placed:
    index: int
    label: str
    x: int
    y: int
    w: int
    h: int


@dataclass
class Composition:
    cells: List[List[Cell]]
    background: Optional[RGB]
    placed: List[Placed]

    @property
    def cols(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    @property
    def rows(self) -> int:
        return len(self.cells)

    def to_text(self) -> str:
        return "\n".join("".join(ch for ch, _ in row).rstrip() for row in self.cells)

    def to_ansi(self) -> str:
        bg = self.background
        bg_code = f"\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m" if bg else ""
        out = []
        for row in self.cells:
            parts = [bg_code]
            prev: Optional[RGB] = None  # the row starts in the default color
            for ch, fg in row:
                color = fg if ch not in _BLANKS else None
                if color != prev:
                    if color is None:
                        parts.append("\x1b[39m")
                    else:
                        parts.append(f"\x1b[38;2;{color[0]};{color[1]};{color[2]}m")
                    prev = color
                parts.append(ch)
            parts.append("\x1b[0m")
            out.append("".join(parts))
        return "\n".join(out) + "\n"

    def html_lines(self) -> List[str]:
        lines = []
        for row in self.cells:
            parts = []
            run_color: Any = "unset"
            run: List[str] = []

            def flush():
                if not run:
                    return
                text = html.escape("".join(run))
                if run_color in (None, "unset"):
                    parts.append(text)
                else:
                    r, g, b = run_color
                    parts.append(f'<span style="color:#{r:02x}{g:02x}{b:02x}">{text}</span>')

            for ch, fg in row:
                color = fg if ch not in _BLANKS else None
                if color != run_color:
                    flush()
                    run, run_color = [], color
                run.append(ch)
            flush()
            lines.append("".join(parts))
        return lines

    def to_html(self, title: str = "ASCII Art", font_size_px: int = 12) -> str:
        from .colorize_ascii import wrap_html

        doc = wrap_html(self.html_lines(), title=title, font_size_px=font_size_px)
        if self.background:
            r, g, b = self.background
            doc = doc.replace(
                "html, body { margin: 0; background: #000; }",
                f"html, body {{ margin: 0; background: #{r:02x}{g:02x}{b:02x}; }}",
                1,
            )
        return doc

    def render(self, fmt: str, title: str = "ASCII Art") -> str:
        if fmt == "text":
            return self.to_text() + "\n"
        if fmt == "ansi":
            return self.to_ansi()
        if fmt == "html":
            return self.to_html(title=title)
        raise ValueError(f"unknown format {fmt!r}")


def _anchor_origin(anchor: str, canvas_w: int, canvas_h: int, w: int, h: int) -> Tuple[int, int]:
    vert = "top" if anchor.startswith("top") else "bottom" if anchor.startswith("bottom") else "middle"
    horiz = "left" if anchor.endswith("left") else "right" if anchor.endswith("right") else "middle"
    x = {"left": 0, "middle": (canvas_w - w) // 2, "right": canvas_w - w}[horiz]
    y = {"top": 0, "middle": (canvas_h - h) // 2, "bottom": canvas_h - h}[vert]
    return x, y


def compose(
    scene: Scene,
    images: Optional[Dict[int, Image.Image]] = None,
    max_cells: Optional[int] = None,
) -> Composition:
    """Render a scene. `images` maps layer index -> an already-open image
    (used instead of the layer's `src`, e.g. for web uploads). `max_cells`
    caps the canvas and every layer (cols x rows) before any rendering."""
    images = images or {}
    for layer in scene.layers:
        layer.validate()
    canvas_overlay = scene.canvas.overlay_obj()
    if max_cells is not None:
        _check_cells(scene.canvas.cols, scene.canvas.rows, max_cells, "canvas")
        for layer in scene.layers:
            _check_cells(layer.cols, layer.rows, max_cells, layer.type + " layer")

    # Images first: their size is the natural reference for text scaling.
    blocks: List[Optional[Block]] = [None] * len(scene.layers)
    for i, layer in enumerate(scene.layers):
        if layer.type == "image":
            blocks[i] = render_image_layer(layer, images.get(i))
            if max_cells is not None:
                _check_cells(blocks[i].width, blocks[i].height, max_cells, "image layer")
    image_widths = [b.width for b in blocks if b is not None]
    ref_width = scene.canvas.cols or (max(image_widths) if image_widths else 80)
    for i, layer in enumerate(scene.layers):
        if layer.type == "text":
            if layer.translate:
                from .translate import translate as _translate

                layer = dataclasses.replace(layer, text=_translate(layer.text, layer.translate))
            blocks[i] = render_text_layer(layer, ref_width)

    # Canvas: explicit, or large enough for every layer.
    canvas_w = scene.canvas.cols
    canvas_h = scene.canvas.rows
    if canvas_w is None or canvas_h is None:
        need_w = max((b.width for b in blocks), default=1)
        need_h = max((b.height for b in blocks), default=1)
        for layer, b in zip(scene.layers, blocks):
            if layer.x is not None:
                need_w = max(need_w, layer.x + layer.dx + b.width)
            if layer.y is not None:
                need_h = max(need_h, layer.y + layer.dy + b.height)
        canvas_w = canvas_w or max(1, need_w)
        canvas_h = canvas_h or max(1, need_h)
    if max_cells is not None:
        _check_cells(canvas_w, canvas_h, max_cells, "canvas")

    cells: List[List[Cell]] = [[(" ", None)] * canvas_w for _ in range(canvas_h)]
    under: List[List[Optional[RGB]]] = [[None] * canvas_w for _ in range(canvas_h)]
    placed: List[Placed] = []

    for i, (layer, b) in enumerate(zip(scene.layers, blocks)):
        ax, ay = _anchor_origin(layer.at, canvas_w, canvas_h, b.width, b.height)
        x0 = (layer.x if layer.x is not None else ax) + layer.dx
        y0 = (layer.y if layer.y is not None else ay) + layer.dy
        placed.append(Placed(i, layer.label(i), x0, y0, b.width, b.height))
        sample_under = layer.type == "text" and layer.color == "image"
        layer_overlay = layer.overlay_obj()
        margin = layer.effective_outline
        if margin and not layer.opaque:
            for cy, cx in _knockout_cells(b, x0, y0, margin, canvas_w, canvas_h):
                cells[cy][cx] = (" ", None)
        for by, line in enumerate(b.cells):
            cy = y0 + by
            if not 0 <= cy < canvas_h:
                continue
            for bx, ch in enumerate(line):
                cx = x0 + bx
                if not 0 <= cx < canvas_w:
                    continue
                if b.under is not None:
                    under[cy][cx] = b.under[by][bx]
                if ch in _BLANKS and not layer.opaque:
                    continue
                color = under[cy][cx] if sample_under else b.colors[by][bx]
                if layer_overlay is not None and ch not in _BLANKS:
                    t = layer_overlay.position(bx, by, b.width, b.height)
                    color = layer_overlay.blend(color, layer_overlay.color_at(t))
                cells[cy][cx] = (ch, color)

    for row in cells:
        _repair_wide(row)
    if canvas_overlay is not None:
        fg = [[c for _, c in row] for row in cells]
        vis = [[ch not in _BLANKS and ch != CONT for ch, _ in row] for row in cells]
        canvas_overlay.apply_grid(fg, vis)
        for y, row in enumerate(cells):
            for x, (ch, _) in enumerate(row):
                if vis[y][x]:
                    row[x] = (ch, fg[y][x])
                elif ch == CONT and x:
                    row[x] = (ch, row[x - 1][1])
    return Composition(cells=cells, background=_solid(scene.canvas.background), placed=placed)


def _knockout_cells(b: Block, x0: int, y0: int, margin: int, canvas_w: int, canvas_h: int):
    """Canvas cells within `margin` (Chebyshev distance) of the block's ink."""
    out = set()
    for by, line in enumerate(b.cells):
        for bx, ch in enumerate(line):
            if ch in _BLANKS:
                continue
            for cy in range(y0 + by - margin, y0 + by + margin + 1):
                if not 0 <= cy < canvas_h:
                    continue
                for cx in range(x0 + bx - margin, x0 + bx + margin + 1):
                    if 0 <= cx < canvas_w:
                        out.add((cy, cx))
    return out


def _repair_wide(row: List[Cell]) -> None:
    """A later layer (or the canvas edge) can cut a double-width character in
    half. Keep every row exactly canvas-wide: an orphaned half becomes a
    space."""
    n = len(row)
    for x, (ch, color) in enumerate(row):
        if ch == CONT:
            lead = row[x - 1][0] if x else ""
            if not lead or char_width(lead[0]) != 2:
                row[x] = (" ", None)
        elif ch and char_width(ch[0]) == 2 and (x + 1 >= n or row[x + 1][0] != CONT):
            row[x] = (" ", None)


def _check_cells(cols: Optional[int], rows: Optional[int], limit: int, what: str) -> None:
    if cols and rows and cols * rows > limit:
        raise ValueError(f"{what} is {cols}x{rows} = {cols * rows:,} characters (limit {limit:,})")


# =============================
# CLI
# =============================

_LAYER_START = {"--image": "image", "--text": "text"}


class _LayerStart(argparse.Action):
    def __call__(self, parser, ns, value, option_string=None):
        kind = _LAYER_START[option_string]
        layer = Layer(type=kind, **({"src": value} if kind == "image" else {"text": value}))
        ns.layers = (ns.layers or []) + [layer]


class _LayerOpt(argparse.Action):
    """Applies to the most recent --image/--text."""

    def __call__(self, parser, ns, value, option_string=None):
        if not ns.layers:
            parser.error(f"{option_string} must follow an --image or --text")
        setattr(ns.layers[-1], self.dest, value if self.const is None else self.const)


def _canvas_size(value: str) -> Tuple[int, int]:
    try:
        c, r = value.lower().split("x")
        c, r = int(c), int(r)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected COLSxROWS, e.g. 100x40, got {value!r}")
    if c < 1 or r < 1:
        raise argparse.ArgumentTypeError("canvas size must be positive")
    return c, r


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ascii-magic compose",
        description=(
            "Compose images and text into one piece of ASCII art. Start a layer with "
            "--image PATH or --text TEXT; the layer options after it apply to that layer. "
            "Later layers paint over earlier ones."
        ),
        epilog=(
            "example: ascii-magic compose --image cat.png --cols 80 --color image "
            "--text 'HELLO' --style figlet --at top --dy 1 --color amber -o card.ans"
        ),
    )
    ap.add_argument("scene", nargs="?", default=None,
                    help="Scene JSON file to render (layers given on the command line are added on top)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output file: .txt, .ans, or .html (default: ANSI to stdout)")
    ap.add_argument("--format", choices=FORMATS, default=None,
                    help="Output format (default: from the output extension, else ansi)")
    ap.add_argument("--canvas", type=_canvas_size, default=None, metavar="COLSxROWS",
                    help="Canvas size in characters (default: fit the layers)")
    ap.add_argument("--background", default=None, metavar="COLOR",
                    help="Canvas background: theme name or #RRGGBB (ANSI/HTML)")
    from .ansi import add_depth_arg

    add_depth_arg(ap)
    from .overlay import add_overlay_args

    add_overlay_args(ap)  # canvas-wide overlay
    ap.add_argument("--save-scene", default=None, metavar="FILE",
                    help="Write the scene (file + command-line layers) as JSON to reuse later")

    g = ap.add_argument_group("layers")
    g.add_argument("--image", action=_LayerStart, dest="layers", metavar="PATH",
                   help="Start an image layer")
    g.add_argument("--text", action=_LayerStart, dest="layers", metavar="TEXT",
                   help="Start a text layer")

    o = ap.add_argument_group("layer options (apply to the preceding --image/--text)")
    o.add_argument("--name", action=_LayerOpt, help="Layer name")
    o.add_argument("--at", action=_LayerOpt, choices=ANCHORS, help="Anchor on the canvas (default: center)")
    o.add_argument("--x", action=_LayerOpt, type=int, help="Exact left column (overrides --at)")
    o.add_argument("--y", action=_LayerOpt, type=int, help="Exact top row (overrides --at)")
    o.add_argument("--dx", action=_LayerOpt, type=int, help="Nudge right (negative: left)")
    o.add_argument("--dy", action=_LayerOpt, type=int, help="Nudge down (negative: up)")
    o.add_argument("--cols", action=_LayerOpt, type=int, help="Layer width in characters")
    o.add_argument("--rows", action=_LayerOpt, type=int, help="Layer height in characters")
    o.add_argument("--scale", action=_LayerOpt, type=float,
                   help="Text: width as a fraction of the canvas when --cols/--rows unset (default 0.5)")
    o.add_argument("--color", action=_LayerOpt,
                   help="'image' (sample the picture), a theme (green, amber, cyan, crimson, violet, white), or #RRGGBB")
    o.add_argument("--opaque", action=_LayerOpt, nargs=0, const=True,
                   help="Blank characters cover what is beneath (a solid box)")
    o.add_argument("--layer-overlay", action=_LayerOpt, default=argparse.SUPPRESS, dest="overlay", metavar="COLORS",
                   help="Overlay on this layer only: palette, color, or comma-separated gradient")
    o.add_argument("--layer-overlay-direction", action=_LayerOpt, default=argparse.SUPPRESS, dest="overlay_direction",
                   choices=("horizontal", "vertical", "diagonal", "diagonal-up", "radial"))
    o.add_argument("--layer-overlay-mode", action=_LayerOpt, default=argparse.SUPPRESS, dest="overlay_mode",
                   choices=("tint", "multiply", "screen", "overlay"))
    o.add_argument("--layer-overlay-strength", action=_LayerOpt, default=argparse.SUPPRESS, dest="overlay_strength", type=float)
    o.add_argument("--outline", action=_LayerOpt, type=int,
                   help="Clear N cells around the layer's ink so it reads over busy art "
                        "(default: 1 for text, 0 for images)")
    o.add_argument("--mode", action=_LayerOpt, choices=IMAGE_MODES, help="Image: braille (default) or glyph")
    o.add_argument("--quality", action=_LayerOpt, choices=QUALITIES, help="Image, glyph mode: match quality")
    o.add_argument("--dither", action=_LayerOpt, nargs=0, const=True, help="Image, braille: dither")
    o.add_argument("--invert", action=_LayerOpt, nargs=0, const=True, help="Image: invert brightness")
    o.add_argument("--autocontrast", action=_LayerOpt, nargs=0, const=True, help="Image: stretch contrast")
    o.add_argument("--gamma", action=_LayerOpt, type=float, help="Image: gamma")
    o.add_argument("--threshold", action=_LayerOpt, type=float, help="Image, braille: ink threshold 0..1")
    o.add_argument("--rotate", action=_LayerOpt, type=int, choices=[0, 90, 180, 270], help="Image: rotate clockwise")
    o.add_argument("--style", action=_LayerOpt, choices=TEXT_STYLES, help="Text style (default: block)")
    o.add_argument("--align", action=_LayerOpt, choices=ALIGNS, help="Text: align lines within the block")
    o.add_argument("--font", action=_LayerOpt, help="Text: .ttf font for block/small/shadow styles")
    o.add_argument("--translate", action=_LayerOpt, default=argparse.SUPPRESS, metavar="LANG",
                   help="Text: translate from English to LANG before rendering (needs the model installed)")
    ap.set_defaults(layers=None)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    from .console import utf8_stdout

    utf8_stdout()
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    try:
        scene = Scene.load(args.scene) if args.scene else Scene()
    except (OSError, ValueError, TypeError) as e:
        ap.error(f"could not load scene {args.scene}: {e}")
    scene.layers.extend(args.layers or [])
    if args.canvas:
        scene.canvas.cols, scene.canvas.rows = args.canvas
    if args.background:
        scene.canvas.background = args.background
    if args.overlay:
        scene.canvas.overlay = args.overlay
        scene.canvas.overlay_direction = args.overlay_direction
        scene.canvas.overlay_mode = args.overlay_mode
        scene.canvas.overlay_strength = args.overlay_strength
    if not scene.layers:
        ap.error("nothing to compose: give a scene file or at least one --image/--text")

    if args.save_scene:
        scene.save(args.save_scene)

    try:
        comp = compose(scene)
    except (ValueError, OSError, RuntimeError) as e:  # RuntimeError: TranslationError
        print(f"ascii-magic compose: error: {e}", file=sys.stderr)
        return 2

    fmt = args.format
    if fmt is None:
        ext = os.path.splitext(args.output or "")[1].lower()
        fmt = {".txt": "text", ".html": "html", ".htm": "html"}.get(ext, "ansi")
    out = comp.render(fmt, title=os.path.basename(args.output) if args.output else "ASCII Art")
    if fmt == "ansi":
        from .ansi import downsample, resolve_depth

        out = downsample(out, resolve_depth(args.color_depth, to_terminal=not args.output))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
