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

from .colorize_ascii import (
    _CAPTION_IMAGE_COLORS,
    CaptionOptions,
    ESC,
    MatrixOptions,
    caption_ref_image,
    matrix_field,
    parse_matrix_color,
    tint_rgb,
)
from .image_to_ascii import find_default_mono_font

# Rain look tunables
_BRIGHT_FLOOR = 0.15     # background cells still get this share of rain brightness
_HEAD_MIN = 0.85         # heads are at least this bright
_VIS_BASE = 0.45         # base share of glyph probability applied to rain visibility
_MIN_INTENSITY = 0.02    # below this the cell is blank


@dataclass
class CaptionRender:
    """A caption resolved to concrete lines and colors, static across frames."""

    lines: List[str]
    position: str                                   # "top" | "bottom"
    gap: int
    uniform: Optional[Tuple[int, int, int]] = None  # single color...
    colors: Optional[list] = None                   # ...or per-char [y][x] RGB


def caption_rows_ansi(cap: "CaptionRender") -> List[str]:
    """Render a resolved caption as ANSI rows (shared by animation and video)."""
    rows = []
    for y, line in enumerate(cap.lines):
        if cap.colors is not None:
            prev = None
            row = []
            for x, ch in enumerate(line):
                if ch == " ":
                    if prev is not None:
                        row.append(f"{ESC}[0m")
                        prev = None
                    row.append(" ")
                    continue
                c = tuple(cap.colors[y][x])
                if c != prev:
                    row.append(f"{ESC}[38;2;{c[0]};{c[1]};{c[2]}m")
                    prev = c
                row.append(ch)
            if prev is not None:
                row.append(f"{ESC}[0m")
            rows.append("".join(row))
        else:
            r, g, b = cap.uniform
            rows.append(
                f"{ESC}[38;2;{r};{g};{b}m{line}{ESC}[0m" if line.strip() else line
            )
    return rows


def with_caption_rows(body: List[str], cap: "CaptionRender", cap_rows: List[str]) -> List[str]:
    spacer = [""] * cap.gap
    if cap.position == "top":
        return cap_rows + spacer + body
    return body + spacer + cap_rows


def caption_strip_array(
    cap: "CaptionRender", font, cell_w: int, cell_h: int, width_px: int
) -> np.ndarray:
    """Caption drawn as a bitmap strip for stacking onto frame canvases."""
    gap_px = cap.gap * cell_h
    strip = Image.new("RGB", (width_px, len(cap.lines) * cell_h + gap_px), (0, 0, 0))
    draw = ImageDraw.Draw(strip)
    y_off = gap_px if cap.position == "bottom" else 0
    for y, line in enumerate(cap.lines):
        for x, ch in enumerate(line):
            if ch == " ":
                continue
            color = tuple(cap.colors[y][x]) if cap.colors else cap.uniform
            draw.text((x * cell_w, y_off + y * cell_h), ch, fill=color, font=font)
    return np.asarray(strip, dtype=np.uint8)


@dataclass
class AnimationOptions:
    frames: int = 60
    fps: float = 12.0
    tail: float = 6.0        # fade length in rows behind each head
    loops: int = 3           # terminal playback repeats (0 = until Ctrl-C)
    font_size: int = 14      # GIF sink glyph size
    mutate_every: int = 4    # frames between glyph re-rolls
    reveal: bool = False     # rain uncovers the colorized art, which persists
    reveal_fade: int = 6     # frames for a revealed cell to reach full brightness


class MatrixAnimation:
    """Generated frames plus the sinks that serialize them."""

    def __init__(
        self,
        frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        chars: str,
        fps: float,
        tint: Tuple[int, int, int] = (0, 255, 0),
        caption: Optional[CaptionRender] = None,
        reveal_alpha: Optional[List[np.ndarray]] = None,  # per-frame (H,W) uint8
        art_chars: Optional[List[str]] = None,            # padded source art grid
        art_rgb: Optional[np.ndarray] = None,             # (H,W,3) uint8 image colors
    ):
        self.frames = frames  # per frame: (glyph idx int16, intensity uint8, head bool)
        self.chars = chars
        self.fps = fps
        self.tint = tint
        self.caption = caption
        self.reveal_alpha = reveal_alpha
        self.art_chars = art_chars
        self.art_rgb = art_rgb

    def _cell_rgb(self, intensity: int, is_head: bool) -> Tuple[int, int, int]:
        base = tint_rgb(intensity, self.tint)
        if not is_head:
            return base
        # Heads glow toward white, scaled by their intensity.
        return tuple(c + (intensity - c) * 4 // 5 for c in base)

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.frames[0][0].shape
        return w, h

    # ---- caption helpers ----

    def _caption_rows_ansi(self) -> List[str]:
        return caption_rows_ansi(self.caption)

    def _with_caption_rows(self, body: List[str], cap_rows: List[str]) -> List[str]:
        return with_caption_rows(body, self.caption, cap_rows)

    # ---- ANSI ----

    def _revealed_cell(self, t_alpha: np.ndarray, y: int, x: int):
        """(char, rgb) for a revealed art cell, or None if still dark."""
        av = int(t_alpha[y, x])
        if av <= 8:
            return None
        ch = self.art_chars[y][x]
        if ch == " ":
            return None
        ar, ag, ab = (int(v) * av // 255 for v in self.art_rgb[y, x])
        return ch, (ar, ag, ab)

    def frames_ansi(self) -> List[str]:
        cap_rows = self._caption_rows_ansi() if self.caption else None
        out = []
        for t, (idx, green, head) in enumerate(self.frames):
            alpha = self.reveal_alpha[t] if self.reveal_alpha is not None else None
            h, w = idx.shape
            lines = []
            for y in range(h):
                prev = None
                row = []
                for x in range(w):
                    i = idx[y, x]
                    if i < 0:
                        cell = self._revealed_cell(alpha, y, x) if alpha is not None else None
                        if cell is None:
                            if prev is not None:
                                row.append(f"{ESC}[0m")
                                prev = None
                            row.append(" ")
                            continue
                        ch, rgb = cell
                    else:
                        ch = self.chars[i]
                        rgb = self._cell_rgb(int(green[y, x]), bool(head[y, x]))
                    if rgb != prev:
                        row.append(f"{ESC}[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m")
                        prev = rgb
                    row.append(ch)
                if prev is not None:
                    row.append(f"{ESC}[0m")
                lines.append("".join(row))
            if cap_rows is not None:
                lines = self._with_caption_rows(lines, cap_rows)
            # Trailing reset so color state never leaks past a frame into
            # the terminal or downstream .frames consumers.
            out.append("\n".join(lines) + f"{ESC}[0m")
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
                    fd = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(fd, sys.stdout.fileno())
                    os.close(fd)

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

        cache: dict[str, np.ndarray] = {}

        def glyph_alpha(ch: str) -> np.ndarray:
            a = cache.get(ch)
            if a is None:
                img = Image.new("L", (cell_w, cell_h), 0)
                ImageDraw.Draw(img).text((0, 0), ch, fill=255, font=font)
                a = np.asarray(img, dtype=np.float32) / 255.0
                cache[ch] = a
            return a

        cap_strip: Optional[np.ndarray] = None
        if self.caption:
            cap_strip = caption_strip_array(
                self.caption, font, cell_w, cell_h, self.size[0] * cell_w
            )

        images = []
        for t, (idx, green, head) in enumerate(self.frames):
            h, w = idx.shape
            canvas = np.zeros((h * cell_h, w * cell_w, 3), dtype=np.uint8)

            if self.reveal_alpha is not None:
                alpha = self.reveal_alpha[t]
                ys, xs = np.nonzero((alpha > 8) & (idx < 0))
                for y, x in zip(ys, xs):
                    cell = self._revealed_cell(alpha, y, x)
                    if cell is None:
                        continue
                    ch, (r, g, b) = cell
                    a = glyph_alpha(ch)
                    block = canvas[y * cell_h:(y + 1) * cell_h, x * cell_w:(x + 1) * cell_w]
                    block[:, :, 0] = (a * r).astype(np.uint8)
                    block[:, :, 1] = (a * g).astype(np.uint8)
                    block[:, :, 2] = (a * b).astype(np.uint8)

            ys, xs = np.nonzero(idx >= 0)
            for y, x in zip(ys, xs):
                a = glyph_alpha(self.chars[int(idx[y, x])])
                r, g, b = self._cell_rgb(int(green[y, x]), bool(head[y, x]))
                block = canvas[y * cell_h:(y + 1) * cell_h, x * cell_w:(x + 1) * cell_w]
                block[:, :, 0] = (a * r).astype(np.uint8)
                block[:, :, 1] = (a * g).astype(np.uint8)
                block[:, :, 2] = (a * b).astype(np.uint8)
            if cap_strip is not None:
                if self.caption.position == "top":
                    canvas = np.vstack([cap_strip, canvas])
                else:
                    canvas = np.vstack([canvas, cap_strip])
            images.append(Image.fromarray(canvas))

        dur = max(20, round(1000 / self.fps))
        durations: object = dur
        if self.reveal_alpha is not None and len(images) > 1:
            # Hold the fully-revealed final frame before the loop restarts.
            durations = [dur] * (len(images) - 1) + [dur * 6]

        buf = io.BytesIO()
        images[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0,
        )
        return buf.getvalue()

    # ---- HTML player ----

    def _caption_html_block(self) -> str:
        cap = self.caption
        parts = []
        for y, line in enumerate(cap.lines):
            if cap.colors is not None:
                prev = None
                segs: List[str] = []
                run: List[str] = []

                def flush():
                    nonlocal run
                    if not run:
                        return
                    text = "".join(run)
                    if prev is None:
                        segs.append(text)
                    else:
                        r, g, b = prev
                        segs.append(f'<span style="color: rgb({r},{g},{b})">{text}</span>')
                    run = []

                for x, ch in enumerate(line):
                    key = None if ch == " " else tuple(cap.colors[y][x])
                    if key != prev:
                        flush()
                        prev = key
                    run.append(html_mod.escape(ch))
                flush()
                parts.append("".join(segs))
            else:
                r, g, b = cap.uniform
                esc = html_mod.escape(line)
                parts.append(
                    f'<span style="color: rgb({r},{g},{b})">{esc}</span>' if line.strip() else esc
                )
        gap = "\n" * cap.gap
        body = "\n".join(parts)
        return gap + body if cap.position == "bottom" else body + gap

    def to_html(self, title: str = "Matrix", font_size_px: int = 12) -> str:
        # Quantize colors so span runs stay long and the file small: rain green
        # becomes 16 class levels; revealed art colors round to 32-step inline styles.
        frame_strings = []
        for t, (idx, green, head) in enumerate(self.frames):
            alpha = self.reveal_alpha[t] if self.reveal_alpha is not None else None
            h, w = idx.shape
            lines = []
            for y in range(h):
                prev = None
                parts = []
                run: list[str] = []

                def flush():
                    nonlocal run
                    if run:
                        text = "".join(run)
                        if prev is None:
                            parts.append(text)
                        elif isinstance(prev, tuple):
                            r_, g_, b_ = prev
                            parts.append(
                                f'<span style="color: rgb({r_},{g_},{b_})">{text}</span>'
                            )
                        else:
                            parts.append(f'<span class="{prev}">{text}</span>')
                        run = []

                for x in range(w):
                    i = idx[y, x]
                    if i < 0:
                        cell = self._revealed_cell(alpha, y, x) if alpha is not None else None
                        if cell is None:
                            key = None
                            ch = " "
                        else:
                            ach, (r_, g_, b_) = cell
                            key = ((r_ >> 5) << 5, (g_ >> 5) << 5, (b_ >> 5) << 5)
                            ch = html_mod.escape(ach)
                    else:
                        key = f"h{green[y, x] >> 4}" if head[y, x] else f"c{green[y, x] >> 4}"
                        ch = html_mod.escape(self.chars[i])
                    if key != prev:
                        flush()
                        prev = key
                    run.append(ch)
                flush()
                lines.append("".join(parts))
            frame_strings.append("\n".join(lines))

        css_levels = "\n".join(
            "    .c{i} {{ color: rgb({r},{g},{b}); }}".format(
                i=i, r=r, g=g, b=b
            )
            for i in range(16)
            for (r, g, b) in [tint_rgb(min(255, i * 17), self.tint)]
        )
        # Head classes brightness-track the drop like the ANSI/GIF sinks
        # (_cell_rgb): base tint scaled by intensity, blended 80% toward it.
        def _head_rgb(level: int):
            g = min(255, level * 17)
            base = tint_rgb(g, self.tint)
            return tuple(c + (g - c) * 4 // 5 for c in base)

        css_heads = "\n".join(
            "    .h{i} {{ color: rgb({r},{g},{b}); }}".format(i=i, r=r, g=g, b=b)
            for i in range(16)
            for (r, g, b) in [_head_rgb(i)]
        )

        cap_top = cap_bottom = ""
        if self.caption:
            block = f'  <pre id="cap">{self._caption_html_block()}</pre>\n'
            if self.caption.position == "top":
                cap_top = block
            else:
                cap_bottom = block

        return (
            "<!doctype html>\n<html>\n<head>\n"
            '  <meta charset="utf-8">\n'
            f"  <title>{html_mod.escape(title)}</title>\n"
            "  <style>\n"
            "    html { margin: 0; }\n"
            "    body { margin: 0; padding: 16px; background: #000; }\n"
            "    pre {\n"
            "      margin: 0;\n      white-space: pre;\n      overflow: auto;\n"
            "      color: #e0e0e0;\n"  # default text must contrast the black page
            '      font-family: "Hack", "JetBrains Mono", "Cascadia Mono", Consolas, monospace;\n'
            f"      font-size: {font_size_px}px;\n      line-height: {font_size_px}px;\n"
            "    }\n"
            f"{css_levels}\n"
            f"{css_heads}\n"
            "  </style>\n</head>\n<body>\n"
            f"{cap_top}"
            '  <pre id="m"></pre>\n'
            f"{cap_bottom}"
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


def _resolve_caption(
    caption: Optional[CaptionOptions],
    image: Image.Image,
    width: int,
    tint: Tuple[int, int, int],
) -> Optional[CaptionRender]:
    if not caption or not caption.text:
        return None
    from .text_to_ascii import caption_lines

    lines = caption_lines(
        caption.text, width, style=caption.style, scale=caption.scale, align=caption.align,
        cols=caption.cols, rows=caption.rows,
    )
    if not lines:
        return None
    colors = None
    uniform: Optional[Tuple[int, int, int]] = None
    if caption.color in _CAPTION_IMAGE_COLORS:
        strip = caption_ref_image(image, caption)
        strip = strip.resize((width, len(lines))).convert("RGB")
        spx = strip.load()
        colors = [[spx[x, y] for x in range(width)] for y in range(len(lines))]
    elif caption.color:
        uniform = parse_matrix_color(caption.color)
    else:
        uniform = tint  # match the rain by default
    return CaptionRender(
        lines=lines,
        position=caption.position,
        gap=max(0, int(caption.gap)),
        uniform=uniform,
        colors=colors,
    )


def generate(
    ascii_text: str,
    image: Image.Image,
    m: Optional[MatrixOptions] = None,
    a: Optional[AnimationOptions] = None,
    caption: Optional[CaptionOptions] = None,
) -> MatrixAnimation:
    """Simulate matrix rain over the subject field.

    Deterministic for a given ``m.seed``; ``seed=None`` (the default) draws
    fresh entropy on every call and is intentionally non-reproducible.
    """
    m = m or MatrixOptions(enabled=True)
    a = a or AnimationOptions()

    lines = [ln for ln in ascii_text.splitlines()]
    if not lines or max(len(ln) for ln in lines) == 0:
        raise ValueError("Empty ASCII text; nothing to animate.")
    if a.frames < 1:
        raise ValueError("frames must be >= 1")
    if not m.chars:
        raise ValueError("matrix chars must not be empty")

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

    reveal_frames: Optional[List[np.ndarray]] = None
    art_rgb: Optional[np.ndarray] = None
    if a.reveal:
        art_rgb = np.asarray(
            image.resize((W, H), Image.Resampling.LANCZOS).convert("RGB"), dtype=np.uint8
        )
        reveal_frames = []
        revealed = np.zeros((H, W), dtype=bool)
        alpha_acc = np.zeros((H, W), dtype=np.float32)

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

        if a.reveal:
            revealed |= intensity > _MIN_INTENSITY
            alpha_acc = np.minimum(
                1.0, alpha_acc + revealed.astype(np.float32) / max(1, a.reveal_fade)
            )
            reveal_frames.append((alpha_acc * 255).astype(np.uint8))

    cap_render = _resolve_caption(caption, image, W, m.tint)
    return MatrixAnimation(
        frames, m.chars, a.fps, tint=m.tint, caption=cap_render,
        reveal_alpha=reveal_frames, art_chars=grid if a.reveal else None, art_rgb=art_rgb,
    )
