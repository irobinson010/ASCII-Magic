"""Animated Matrix rain over the subject field of an image + ASCII pair.

The rain engine reuses ``colorize_ascii.matrix_field`` for per-cell subject
scoring, then simulates falling drops per column: a bright head, a fading
tail, glyphs that mutate every few frames. Columns loop exactly at the
requested frame count, so every sink produces a seamless loop.

Sinks:
- ``frames_ansi()``   -> list of ANSI frames (truecolor)
- ``play()``          -> terminal playback with cursor control
- ``to_gif_bytes()``  -> animated GIF
- ``to_html()``       -> self-contained HTML player
"""

from __future__ import annotations

import html as html_mod
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .colorize_ascii import ESC, MatrixOptions, matrix_field
from .image_to_ascii import find_default_mono_font

# Rain look tunables
_BRIGHT_FLOOR = 0.15     # background cells still get this share of rain brightness
_HEAD_MIN = 0.85         # heads are at least this bright
_VIS_BASE = 0.45         # base share of glyph probability applied to rain visibility
_MIN_INTENSITY = 0.02    # below this the cell is blank


@dataclass
class AnimationOptions:
    frames: int = 60
    fps: float = 12.0
    tail: float = 6.0        # fade length in rows behind each head
    loops: int = 3           # terminal playback repeats (0 = until Ctrl-C)
    font_size: int = 14      # GIF sink glyph size
    mutate_every: int = 4    # frames between glyph re-rolls


class MatrixAnimation:
    """Generated frames plus the sinks that serialize them."""

    def __init__(
        self,
        frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        chars: str,
        fps: float,
    ):
        self.frames = frames  # per frame: (glyph idx int16, green uint8, head bool)
        self.chars = chars
        self.fps = fps

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.frames[0][0].shape
        return w, h

    # ---- ANSI ----

    def frames_ansi(self) -> List[str]:
        out = []
        for idx, green, head in self.frames:
            h, w = idx.shape
            lines = []
            for y in range(h):
                prev = None
                row = []
                for x in range(w):
                    i = idx[y, x]
                    if i < 0:
                        if prev is not None:
                            row.append(f"{ESC}[0m")
                            prev = None
                        row.append(" ")
                        continue
                    g = int(green[y, x])
                    wb = int(g * 0.8) if head[y, x] else 0  # whiten heads
                    style = (wb, g)
                    if style != prev:
                        row.append(f"{ESC}[38;2;{wb};{g};{wb}m")
                        prev = style
                    row.append(self.chars[i])
                if prev is not None:
                    row.append(f"{ESC}[0m")
                lines.append("".join(row))
            out.append("\n".join(lines))
        return out

    def play(self, loops: Optional[int] = None, out=None) -> None:
        """Terminal playback. loops=0 plays until Ctrl-C."""
        out = out or sys.stdout
        frames = self.frames_ansi()
        delay = 1.0 / max(self.fps, 0.1)
        loops = 3 if loops is None else loops
        try:
            out.write(f"{ESC}[2J{ESC}[?25l")
            n = 0
            while loops == 0 or n < loops:
                for f in frames:
                    out.write(f"{ESC}[H" + f)
                    out.flush()
                    time.sleep(delay)
                n += 1
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        finally:
            try:
                out.write(f"{ESC}[0m{ESC}[?25h\n")
                out.flush()
            except BrokenPipeError:
                # Stop the interpreter-shutdown flush from complaining too.
                if out is sys.stdout:
                    os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())

    # ---- GIF ----

    def to_gif_bytes(self, font_path: Optional[str] = None, font_size: int = 14) -> bytes:
        font_path = font_path or find_default_mono_font()
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
            ascent, descent = font.getmetrics()
            cell_h = ascent + descent
            cell_w = max(1, round(font.getlength("M")))
        else:
            font = ImageFont.load_default()
            cell_w, cell_h = 7, 13

        cache: dict[int, np.ndarray] = {}

        def glyph_alpha(i: int) -> np.ndarray:
            a = cache.get(i)
            if a is None:
                img = Image.new("L", (cell_w, cell_h), 0)
                ImageDraw.Draw(img).text((0, 0), self.chars[i], fill=255, font=font)
                a = np.asarray(img, dtype=np.float32) / 255.0
                cache[i] = a
            return a

        images = []
        for idx, green, head in self.frames:
            h, w = idx.shape
            canvas = np.zeros((h * cell_h, w * cell_w, 3), dtype=np.uint8)
            ys, xs = np.nonzero(idx >= 0)
            for y, x in zip(ys, xs):
                a = glyph_alpha(int(idx[y, x]))
                g = int(green[y, x])
                wb = int(g * 0.8) if head[y, x] else 0
                block = canvas[y * cell_h:(y + 1) * cell_h, x * cell_w:(x + 1) * cell_w]
                block[:, :, 0] = (a * wb).astype(np.uint8)
                block[:, :, 1] = (a * g).astype(np.uint8)
                block[:, :, 2] = (a * wb).astype(np.uint8)
            images.append(Image.fromarray(canvas))

        buf = io.BytesIO()
        images[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=max(20, round(1000 / self.fps)),
            loop=0,
        )
        return buf.getvalue()

    # ---- HTML player ----

    def to_html(self, title: str = "Matrix", font_size_px: int = 12) -> str:
        # Quantize green to 16 levels so span runs stay long and the file small.
        frame_strings = []
        for idx, green, head in self.frames:
            h, w = idx.shape
            lines = []
            for y in range(h):
                prev = None
                parts = []
                run: list[str] = []

                def flush():
                    nonlocal run
                    if run:
                        if prev is None:
                            parts.append("".join(run))
                        else:
                            parts.append(f'<span class="{prev}">' + "".join(run) + "</span>")
                        run = []

                for x in range(w):
                    i = idx[y, x]
                    if i < 0:
                        cls = None
                        ch = " "
                    else:
                        cls = "h" if head[y, x] else f"c{green[y, x] >> 4}"
                        ch = html_mod.escape(self.chars[i])
                    if cls != prev:
                        flush()
                        prev = cls
                    run.append(ch)
                flush()
                lines.append("".join(parts))
            frame_strings.append("\n".join(lines))

        css_levels = "\n".join(
            f"    .c{i} {{ color: rgb(0,{min(255, i * 17)},0); }}" for i in range(16)
        )
        return (
            "<!doctype html>\n<html>\n<head>\n"
            '  <meta charset="utf-8">\n'
            f"  <title>{html_mod.escape(title)}</title>\n"
            "  <style>\n"
            "    html, body { margin: 0; background: #000; }\n"
            "    pre {\n"
            "      margin: 16px;\n      white-space: pre;\n      overflow: auto;\n"
            '      font-family: "Hack", "JetBrains Mono", "Cascadia Mono", Consolas, monospace;\n'
            f"      font-size: {font_size_px}px;\n      line-height: {font_size_px}px;\n"
            "    }\n"
            f"{css_levels}\n"
            "    .h { color: #d8ffd8; }\n"
            "  </style>\n</head>\n<body>\n"
            '  <pre id="m"></pre>\n'
            "  <script>\n"
            f"    const FRAMES = {json.dumps(frame_strings)};\n"
            f"    const FPS = {self.fps};\n"
            '    const pre = document.getElementById("m");\n'
            "    let i = 0;\n"
            "    pre.innerHTML = FRAMES[0];\n"
            "    setInterval(() => { i = (i + 1) % FRAMES.length; pre.innerHTML = FRAMES[i]; },\n"
            "                Math.max(20, Math.round(1000 / FPS)));\n"
            "  </script>\n</body>\n</html>\n"
        )


def generate(
    ascii_text: str,
    image: Image.Image,
    m: Optional[MatrixOptions] = None,
    a: Optional[AnimationOptions] = None,
) -> MatrixAnimation:
    """Simulate matrix rain over the subject field. Deterministic for a given seed."""
    m = m or MatrixOptions(enabled=True)
    a = a or AnimationOptions()

    lines = [ln for ln in ascii_text.splitlines()]
    if not lines:
        raise ValueError("Empty ASCII text; nothing to animate.")
    if a.frames < 1:
        raise ValueError("frames must be >= 1")

    grid, field = matrix_field(lines, image, m)
    H = len(grid)
    W = len(grid[0])

    subject = np.array([[c[3] for c in row] for row in field], dtype=np.float32)
    prob = np.array([[c[2] for c in row] for row in field], dtype=np.float32)

    rng = random.Random(m.seed)
    nprng = np.random.Generator(np.random.PCG64(rng.getrandbits(64)))

    tail = max(float(a.tail), 1e-6)

    # Per-column drop parameters. speed = passes * cycle / frames makes every
    # column return to its start position at frame N -> seamless loop.
    gaps = nprng.uniform(0.3 * H + 2, 1.2 * H + 4, size=W).astype(np.float32)
    cycle = H + tail + gaps
    passes = nprng.integers(1, 3, size=W).astype(np.float32)  # 1 or 2
    speed = passes * cycle / a.frames
    h0 = nprng.uniform(0, cycle, size=W).astype(np.float32)

    # Glyph identity per cell mutates every `mutate_every` frames.
    n_slots = max(1, -(-a.frames // max(1, a.mutate_every)))
    idx_table = nprng.integers(0, len(m.chars), size=(n_slots, H, W), dtype=np.int16)
    vis_table = nprng.random(size=(n_slots, H, W), dtype=np.float32)

    bright = _BRIGHT_FLOOR + (1.0 - _BRIGHT_FLOOR) * np.clip(subject, 0.0, 1.0)
    vis_p = np.clip(_VIS_BASE + (1.0 - _VIS_BASE) * prob, 0.0, 1.0)
    rows = np.arange(H, dtype=np.float32)[:, None]

    frames = []
    for t in range(a.frames):
        heads = (h0 + speed * t) % cycle          # (W,)
        d = heads[None, :] - rows                 # (H, W) distance behind head
        intensity = np.where((d >= 0) & (d <= tail), 1.0 - d / tail, 0.0).astype(np.float32)
        head_mask = (d >= 0) & (d < 1.0)

        level = intensity * bright
        level[head_mask] = np.maximum(level[head_mask], _HEAD_MIN)

        slot = min(t // max(1, a.mutate_every), n_slots - 1)
        visible = (intensity > _MIN_INTENSITY) & (vis_table[slot] < vis_p)
        visible |= head_mask

        green = (m.fg_min + level * (m.fg_max - m.fg_min)).astype(np.uint8)
        idx = np.where(visible, idx_table[slot], np.int16(-1))
        frames.append((idx, np.where(visible, green, 0).astype(np.uint8), head_mask & visible))

    return MatrixAnimation(frames, m.chars, a.fps)
