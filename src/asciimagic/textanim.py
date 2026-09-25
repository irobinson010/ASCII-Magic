"""Animated text: wave like a flag, spin, flip, stretch, bounce, and more.

The text is drawn once as a high-resolution mask. Every frame, each effect
maps output pixels back to mask pixels (an inverse transform, so nothing
tears or leaves holes), and the result is averaged down to character cells
through the same density ramp as ``ascii-magic text``. Motion is periodic in
the frame count, so every output loops seamlessly::

    ascii-magic text "HELLO" --animate wave             # plays in the terminal
    ascii-magic text "HELLO" --animate spin -o spin.gif
    ascii-magic text "HELLO" --animate flip-up,rainbow -o flip.html
    ascii-magic text "Hi" --animate bounce -o hi.frames  # for ascii-magic-greet

Effects can be chained with commas; they apply in order.
"""

from __future__ import annotations

import colorsys
import html as html_mod
import io
import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL_W, CELL_H = 8, 16   # mask pixels per character cell (terminal cells are ~1:2)
ESC = "\x1b"

RAMPS = {
    "block": " .:-=+*#%@",
    "small": " .-*#@",
    "solid": " ░▒▓█",
}
STYLES = tuple(RAMPS)


@dataclass
class Ctx:
    """What an effect knows about the text and the current frame."""

    hw: float                 # half the text width (mask px)
    hh: float                 # half the text height
    canvas_w: float
    amount: float             # strength multiplier (1 = default)
    frame: int
    frames: int
    edges: Sequence[float]    # x of each character boundary, centered (typewriter)
    mirror_back: bool = False


Coords = Tuple[np.ndarray, np.ndarray, "np.ndarray | float"]
EffectFn = Callable[[float, np.ndarray, np.ndarray, Ctx], Coords]


@dataclass
class Effect:
    fn: EffectFn
    about: str
    fill: float = 0.9         # share of the canvas width, for effects that wrap around it
    wraps: bool = False
    random: bool = False      # differs frame to frame, so sizing checks every frame


def _ease(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3 - 2 * s)


def _rotate(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


# ---- geometry effects: (t in [0,1), output X/Y) -> (source X/Y, gain) ----

def _wave(t, X, Y, c):
    # A flag on a pole at the left edge: ripples travel away from the pole
    # and grow with distance from it; folds catch light and shade.
    lam = 2 * c.hw / 1.6
    phase = 2 * math.pi * (X / lam - t)
    d = np.clip((X + c.hw) / (2 * c.hw), 0, 1)
    amp = 0.42 * c.hh * c.amount * (0.25 + 0.75 * d)
    return X, Y - amp * np.sin(phase), 0.78 + 0.22 * np.cos(phase)


def _ripple(t, X, Y, c):
    lam = 2.2 * c.hh
    return X - 0.35 * c.hh * c.amount * np.sin(2 * math.pi * (Y / lam - t)), Y, 1.0


def _twist(t, X, Y, c):
    # Each row turns about the vertical axis a little behind the one above: a ribbon.
    theta = 2 * math.pi * t + 0.9 * c.amount * Y / c.hh
    cos = np.cos(theta)
    ok = np.abs(cos) > 0.03
    sx = np.where(ok, X / np.where(ok, np.abs(cos), 1), np.nan)
    if c.mirror_back:
        sx = np.where(cos < 0, -sx, sx)
    return sx, Y, 0.45 + 0.55 * np.abs(cos)


def _card(X, Y, theta, axis, c, radius):
    """Inverse of a card turned by `theta` about an in-plane axis at angle
    `axis` (0 = vertical), seen in perspective."""
    x, y = _rotate(X, Y, -axis)
    D = 4.0 * radius
    s, co = math.sin(theta), math.cos(theta)
    den = D * co - x * s
    ok = np.abs(den) > 1e-6
    u = np.where(ok, x * D / np.where(ok, den, 1), np.nan)
    depth = D + u * s
    v = y * depth / D
    u = np.where(depth > 0.05 * D, u, np.nan)
    if co < 0 and not c.mirror_back:
        u = -u  # the back reads the right way round, like a two-sided sign
    sx, sy = _rotate(u, v, axis)
    return sx, sy, 0.5 + 0.5 * abs(co)


def _flip(sign: float, axis: float):
    def fn(t, X, Y, c):
        turns = 2 if c.mirror_back else 1
        theta = sign * math.pi * turns * _ease((t - 0.35) / 0.65)
        radius = abs(math.cos(axis)) * c.hw + abs(math.sin(axis)) * c.hh
        return _card(X, Y, theta, axis, c, radius)
    return fn


def _spin(t, X, Y, c):
    return _card(X, Y, 2 * math.pi * t, 0.0, c, c.hw)


def _rotate_fx(t, X, Y, c):
    sx, sy = _rotate(X, Y, -2 * math.pi * t)
    return sx, sy, 1.0


def _swing(t, X, Y, c):
    a = 0.3 * c.amount * math.sin(2 * math.pi * t)
    py = -c.hh * 1.6  # hanging from a point above the text
    sx, sy = _rotate(X, Y - py, -a)
    return sx, sy + py, 1.0


def _stretch(t, X, Y, c):
    s = 1 + 0.45 * c.amount * math.sin(2 * math.pi * t)
    return X / max(0.05, s), Y, 1.0


def _stretch_v(t, X, Y, c):
    s = 1 + 0.6 * c.amount * math.sin(2 * math.pi * t)
    return X, Y / max(0.05, s), 1.0


def _squash(t, X, Y, c):
    s = math.exp(0.3 * c.amount * math.sin(2 * math.pi * t))  # area-preserving, never inverts
    return X * s, Y / s, 1.0


def _zoom(t, X, Y, c):
    s = 0.12 + 0.88 * (0.5 - 0.5 * math.cos(2 * math.pi * t))
    return X / s, Y / s, 1.0


def _pulse(t, X, Y, c):
    # Heartbeat: two quick beats, then rest.
    beat = max(math.exp(-((t - 0.1) / 0.05) ** 2), 0.7 * math.exp(-((t - 0.3) / 0.05) ** 2))
    s = 1 + 0.18 * c.amount * beat
    return X / s, Y / s, 1.0


def _bounce(t, X, Y, c):
    height = 1.6 * c.hh * c.amount
    lift = height * (1 - (2 * t - 1) ** 2)            # parabola, lands at t = 0/1
    contact = max(0.0, 1 - min(t, 1 - t) / 0.08)       # squash while touching down
    sy = 1 - 0.18 * contact
    sx = 1 / sy
    base = c.hh
    return X / sx, (Y - base + lift) / sy + base, 1.0


def _shake(t, X, Y, c):
    rng = np.random.default_rng(1234 + c.frame)
    dx, dy = rng.uniform(-1, 1, 2) * c.amount * np.array([0.05 * c.hw, 0.15 * c.hh])
    a = rng.uniform(-1, 1) * 0.04 * c.amount
    sx, sy = _rotate(X - dx, Y - dy, a)
    return sx, sy, 1.0


def _glitch(t, X, Y, c):
    rng = np.random.default_rng(99 + c.frame)
    if rng.random() > 0.45:
        return X, Y, 1.0
    shift = np.zeros_like(Y)
    for _ in range(rng.integers(1, 4)):
        y0 = rng.uniform(-c.hh, c.hh)
        h = rng.uniform(0.1, 0.5) * c.hh
        band = (Y >= y0) & (Y < y0 + h)
        shift = np.where(band, rng.uniform(-0.25, 0.25) * c.hw * c.amount, shift)
    return X - shift, Y, (0.6 + 0.4 * rng.random())


def _scroll(t, X, Y, c):
    period = c.canvas_w
    sx = np.mod(X + c.canvas_w / 2 + t * period + c.hw, period) - c.hw
    return sx, Y, 1.0


def _typewriter(t, X, Y, c):
    n = len(c.edges) - 1
    shown = min(n, int(min(1.0, t / 0.7) * (n + 1)))
    edge = c.edges[shown]
    return X, Y, np.where(X < edge, 1.0, 0.0)


def _fade(t, X, Y, c):
    return X, Y, 0.5 - 0.5 * math.cos(2 * math.pi * t)


def _still(t, X, Y, c):
    return X, Y, 1.0


EFFECTS: Dict[str, Effect] = {
    "wave": Effect(_wave, "wave like a flag on a pole"),
    "ripple": Effect(_ripple, "horizontal ripples running down the text"),
    "twist": Effect(_twist, "twist like a ribbon"),
    "spin": Effect(_spin, "spin around the vertical axis, in 3D"),
    "rotate": Effect(_rotate_fx, "turn like a wheel"),
    "swing": Effect(_swing, "swing like a hanging sign"),
    "flip-left": Effect(_flip(-1, 0.0), "flip over toward the left"),
    "flip-right": Effect(_flip(1, 0.0), "flip over toward the right"),
    "flip-up": Effect(_flip(-1, math.pi / 2), "flip over upward"),
    "flip-down": Effect(_flip(1, math.pi / 2), "flip over downward"),
    "flip-diagonal": Effect(_flip(1, math.pi / 4), "flip over a diagonal"),
    "flip-antidiagonal": Effect(_flip(1, -math.pi / 4), "flip over the other diagonal"),
    "stretch": Effect(_stretch, "stretch and shrink sideways"),
    "stretch-v": Effect(_stretch_v, "stretch and shrink vertically"),
    "squash": Effect(_squash, "squash and stretch like jelly"),
    "zoom": Effect(_zoom, "grow from nothing and shrink back"),
    "pulse": Effect(_pulse, "beat like a heart"),
    "bounce": Effect(_bounce, "bounce like a ball"),
    "shake": Effect(_shake, "shake", random=True),
    "glitch": Effect(_glitch, "digital glitch", random=True),
    "scroll": Effect(_scroll, "scroll across like a marquee", wraps=True),
    "typewriter": Effect(_typewriter, "type itself out"),
    "fade": Effect(_fade, "fade in and out"),
    "rainbow": Effect(_still, "colors cycling through the text (combine with any motion)"),
}


def parse_effects(spec: str) -> List[str]:
    names = [s.strip().lower() for s in (spec or "").split(",") if s.strip()]
    if not names:
        raise ValueError("no animation effect given")
    bad = [n for n in names if n not in EFFECTS]
    if bad:
        raise ValueError(f"unknown animation {bad[0]!r}; choose from {', '.join(EFFECTS)}")
    return names


# ---- rendering ----


@dataclass
class TextAnimOptions:
    effects: Sequence[str] = ("wave",)
    cols: int = 60
    frames: int = 36
    fps: float = 15.0
    style: str = "block"
    amount: float = 1.0
    color: object = None               # None, "rainbow", an overlay spec (palette/theme/#hex/gradient), or an Overlay
    mirror_back: bool = False
    font_path: Optional[str] = None
    max_pixels: Optional[int] = None   # cap on mask pixels x frames (the web server's budget)


class TooLarge(ValueError):
    """The animation would exceed TextAnimOptions.max_pixels."""


def _text_mask(text: str, width_px: int, font_path: Optional[str]) -> Tuple[np.ndarray, List[float]]:
    """White-on-black mask exactly `width_px` wide, plus the x of each
    character boundary across its widest line (for typewriter)."""
    from .text_to_ascii import load_font_for, render_text_to_image

    probe = 96
    img = render_text_to_image(text, font_size=probe, font_path=font_path,
                               bg_color="black", text_color="white").convert("L")
    # Never tiny: small sizes are hinted into different shapes; downscaling is exact.
    size = max(48, min(600, round(probe * width_px / max(1, img.width))))
    img = render_text_to_image(text, font_size=size, font_path=font_path,
                               bg_color="black", text_color="white").convert("L")
    # Tight to the ink: the renderer's fixed padding would otherwise be a
    # different share of the mask at different sizes, skewing measurements.
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    h = max(1, round(img.height * width_px / img.width))
    mask = np.asarray(img.resize((width_px, h), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0

    font = load_font_for(text, font_path, size)
    longest = max(text.split("\n"), key=lambda s: font.getlength(s))
    total = font.getlength(longest) or 1.0
    edges = [font.getlength(longest[:i]) / total for i in range(len(longest) + 1)]
    edges = [(e - 0.5) * width_px for e in edges]
    edges[-1] = width_px  # last step shows everything, whatever the rounding
    return mask, edges


def _to_indices(cov: np.ndarray, n: int) -> np.ndarray:
    # Same contrast S-curve as the static renderer: no halo of faint characters.
    v = np.clip((cov - 0.18) / 0.64, 0, 1)
    return np.rint(v * n).astype(np.int16)


class TextAnimation:
    """Frames of animated text plus the sinks that write them."""

    def __init__(self, text: str, opt: TextAnimOptions):
        if not text.strip():
            raise ValueError("no text to animate")
        names = [n for n in opt.effects]
        for n in names:
            if n not in EFFECTS:
                raise ValueError(f"unknown animation {n!r}")
        if opt.style not in RAMPS:
            raise ValueError(f"unknown style {opt.style!r} for animation; choose from {', '.join(STYLES)}")
        self.opt = opt
        self.fps = opt.fps
        self.text = text
        self.ramp = RAMPS[opt.style]
        if opt.color is None and "rainbow" in names:
            opt.color = "rainbow"
        self._overlay = None
        if opt.color is not None and not isinstance(opt.color, str):
            self._overlay = opt.color  # an overlay.Overlay
        elif opt.color and opt.color != "rainbow":
            from .overlay import Overlay

            self._overlay = Overlay.parse(opt.color)
        self.lines, self.colors = self._render([EFFECTS[n] for n in names])

    def _frame_px(self, mask, edges, W, H, effects, f, t=None):
        """Mask pixels seen through the effects at frame f (time t, default
        f / frames), on a W x H canvas."""
        o = self.opt
        th, tw = mask.shape
        hh, hw = th / 2, tw / 2
        xs = np.arange(W, dtype=np.float32) - W / 2 + 0.5
        ys = np.arange(H, dtype=np.float32) - H / 2 + 0.5
        X0, Y0 = np.meshgrid(xs, ys)
        ctx = Ctx(hw=hw, hh=hh, canvas_w=W, amount=o.amount, frame=f, frames=o.frames,
                  edges=edges, mirror_back=o.mirror_back)
        X, Y, gain = X0, Y0, 1.0
        for e in effects:
            X, Y, g = e.fn(f / o.frames if t is None else t, X, Y, ctx)
            gain = gain * g
        # NaN marks points an effect maps nowhere (e.g. behind a turning card).
        X = np.nan_to_num(np.broadcast_to(X, X0.shape), nan=-1e7, posinf=-1e7, neginf=-1e7)
        Y = np.nan_to_num(np.broadcast_to(Y, Y0.shape), nan=-1e7, posinf=-1e7, neginf=-1e7)
        ix = np.floor(np.clip(X + hw, -1, tw)).astype(np.int64)
        iy = np.floor(np.clip(Y + hh, -1, th)).astype(np.int64)
        ok = (ix >= 0) & (ix < tw) & (iy >= 0) & (iy < th)
        return np.where(ok, mask[np.clip(iy, 0, th - 1), np.clip(ix, 0, tw - 1)], 0.0) * gain

    def _extent(self, effects) -> Tuple[float, float, float]:
        """How far the motion reaches, as multiples of the text's half width
        and half height, measured on a small, roomy test render; plus the
        text's height/width."""
        o = self.opt
        mask, edges = _text_mask(self.text, 64, o.font_path)
        if mask.shape[0] > 64:  # tall text (a narrow word, several lines): keep the probe small
            mask, edges = _text_mask(self.text, max(12, 64 * 64 // mask.shape[0]), o.font_path)
        th, tw = mask.shape
        W = int(max(tw, th) * 6)
        H = int(max(tw * 2.5, th * (4 + 2 * o.amount)))
        # Deterministic effects are sampled at fixed times, so the sizing (and
        # the picture at a given phase) doesn't depend on the frame count;
        # random ones (shake, glitch) differ per frame, so every frame is checked.
        if any(e.random for e in effects):
            samples = [(f, None) for f in range(o.frames)]
        else:
            samples = [(0, k / 64) for k in range(64)]
        reach_x = reach_y = 0.0
        for f, t in samples:
            ys, xs = np.nonzero(self._frame_px(mask, edges, W, H, effects, f, t) > 0.2)
            if xs.size:
                reach_x = max(reach_x, np.abs(xs - W / 2 + 0.5).max() + 1)
                reach_y = max(reach_y, np.abs(ys - H / 2 + 0.5).max() + 1)
        return max(reach_x / (tw / 2), 1.0), max(reach_y / (th / 2), 1.0), th / tw

    def _render(self, effects: List[Effect]):
        o = self.opt
        W = o.cols * CELL_W
        reach_x, reach_y, aspect = self._extent(effects)
        # Size the text so its motion stays on the canvas. Wrapping effects
        # (scroll) always span the canvas, so they use their own fill.
        wraps = [e.fill for e in effects if e.wraps]
        fill = min(wraps) if wraps else min(0.94, 0.94 / reach_x)
        # ...and so the animation is at most as many rows as columns (tall
        # text such as a lone "I" bouncing would otherwise be enormous).
        fill = min(fill, 2 * 0.9 / (aspect * reach_y * 1.1))
        mask, edges = _text_mask(self.text, max(8, int(W * fill)), o.font_path)
        th, tw = mask.shape
        # Room for the motion plus a margin (cropped afterwards to the rows used).
        H = (int(th * reach_y * 1.1 + 2 * CELL_H) // CELL_H + 1) * CELL_H
        if o.max_pixels is not None and W * H * o.frames > o.max_pixels:
            raise TooLarge(
                f"animation too large ({W * H * o.frames / 1e6:.0f}M pixel-frames, limit "
                f"{o.max_pixels / 1e6:.0f}M): lower the width or the number of frames"
            )
        n = len(self.ramp) - 1
        idx_frames = []
        for f in range(o.frames):
            px = self._frame_px(mask, edges, W, H, effects, f)
            cov = px.reshape(H // CELL_H, CELL_H, o.cols, CELL_W).mean(axis=(1, 3))
            idx_frames.append(_to_indices(cov, n))
        stack = np.stack(idx_frames)                       # (F, rows, cols)
        # Ink on the canvas's first/last row means the motion ran off it.
        self.clipped = bool(stack[:, 0].any() or stack[:, -1].any())
        used = np.nonzero(stack.max(axis=(0, 2)) > 0)[0]
        if used.size:
            stack = stack[:, used[0]:used[-1] + 1]
        else:
            stack = stack[:, :1]
        ramp = np.array(list(self.ramp))
        lines = [["".join(r) for r in ramp[fr]] for fr in stack]
        colors = self._colorize(stack) if o.color else None
        return lines, colors

    def _colorize(self, stack: np.ndarray) -> List[np.ndarray]:
        F, R, C = stack.shape
        out = []
        if self._overlay is not None:
            ov = self._overlay
            grid = np.array([[ov.color_at(ov.position(x, y, C, R)) for x in range(C)] for y in range(R)],
                            dtype=np.uint8)
            return [grid] * F
        for f in range(F):
            hue = (np.arange(C)[None, :] / max(1, C) * 0.8 + np.arange(R)[:, None] * 0.02 - f / F) % 1.0
            rgb = np.array([colorsys.hsv_to_rgb(h, 0.85, 1.0) for h in hue.ravel()]).reshape(R, C, 3)
            out.append((rgb * 255).astype(np.uint8))
        return out

    @property
    def size(self) -> Tuple[int, int]:
        return self.opt.cols, len(self.lines[0])

    # ---- sinks ----

    def frames_text(self) -> List[str]:
        return ["\n".join(fr) for fr in self.lines]

    def frames_ansi(self) -> List[str]:
        if self.colors is None:
            return self.frames_text()
        out = []
        for fr, col in zip(self.lines, self.colors):
            rows = []
            for y, line in enumerate(fr):
                parts, prev = [], None
                for x, ch in enumerate(line):
                    if ch != " ":
                        rgb = tuple(int(v) for v in col[y, x])
                        if rgb != prev:
                            parts.append(f"{ESC}[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m")
                            prev = rgb
                    parts.append(ch)
                rows.append("".join(parts))
            out.append("\n".join(rows) + f"{ESC}[0m")
        return out

    def play(self, loops: int = 3, out=None, color_depth: str = "truecolor") -> None:
        from .ansi import downsample
        from .greet import play_frames

        play_frames([downsample(f, color_depth) for f in self.frames_ansi()], self.fps, loops, out=out)

    def write_frames(self, path, loops: int = 0) -> None:
        from pathlib import Path

        from .greet import write_frames_file

        write_frames_file(Path(path), self.frames_ansi(), self.fps, loops)

    def to_gif_bytes(self, font_size: int = 14) -> bytes:
        from .image_to_ascii import find_default_mono_font

        path = find_default_mono_font()
        if path:
            font = ImageFont.truetype(path, font_size)
            ascent, descent = font.getmetrics()
            cw, ch = max(1, round(font.getlength("M"))), ascent + descent
        else:
            font, cw, ch = ImageFont.load_default(), 7, 13
        glyphs: Dict[str, np.ndarray] = {}

        def glyph(c: str) -> np.ndarray:
            if c not in glyphs:
                im = Image.new("L", (cw, ch), 0)
                ImageDraw.Draw(im).text((0, 0), c, fill=255, font=font)
                glyphs[c] = np.asarray(im, dtype=np.float32)[:, :, None] / 255.0
            return glyphs[c]

        cols, rows = self.size
        default = np.array([224, 224, 224], dtype=np.float32)
        images = []
        for f, fr in enumerate(self.lines):
            canvas = np.zeros((rows * ch, cols * cw, 3), dtype=np.float32)
            for y, line in enumerate(fr):
                for x, c in enumerate(line):
                    if c == " ":
                        continue
                    rgb = self.colors[f][y, x].astype(np.float32) if self.colors is not None else default
                    canvas[y * ch:(y + 1) * ch, x * cw:(x + 1) * cw] = glyph(c) * rgb
            images.append(Image.fromarray(canvas.astype(np.uint8)))
        buf = io.BytesIO()
        images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                       duration=max(20, round(1000 / self.fps)), loop=0)
        return buf.getvalue()

    def _html_frames(self) -> List[str]:
        out = []
        for f, fr in enumerate(self.lines):
            rows = []
            for y, line in enumerate(fr):
                if self.colors is None:
                    rows.append(html_mod.escape(line))
                    continue
                parts, run, key = [], [], None

                def flush():
                    if run:
                        txt = html_mod.escape("".join(run))
                        parts.append(txt if key is None else f'<span style="color:#{key}">{txt}</span>')

                for x, c in enumerate(line):
                    # Quantize so neighbouring cells share spans and the file stays small.
                    k = None if c == " " else "%02x%02x%02x" % tuple((int(v) >> 5) << 5 for v in self.colors[f][y, x])
                    if k != key and c != " ":
                        flush()
                        run, key = [], k
                    run.append(c)
                flush()
                rows.append("".join(parts))
            out.append("\n".join(rows))
        return out

    def to_html(self, title: str = "ASCII Magic", font_size_px: int = 12) -> str:
        return (
            "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
            f"<title>{html_mod.escape(title)}</title>\n<style>\n"
            "  body { margin: 0; padding: 16px; background: #000; }\n"
            "  pre { margin: 0; white-space: pre; color: #e0e0e0;\n"
            '    font-family: "Hack", "JetBrains Mono", "Cascadia Mono", "DejaVu Sans Mono", Consolas, monospace;\n'
            f"    font-size: {font_size_px}px; line-height: {font_size_px}px; }}\n"
            "</style>\n</head>\n<body>\n<pre id=\"m\"></pre>\n<script>\n"
            f"const FRAMES = {json.dumps(self._html_frames())};\n"
            f"const FPS = {self.fps};\n"
            "const pre = document.getElementById(\"m\");\nlet i = 0;\npre.innerHTML = FRAMES[0];\n"
            "setInterval(() => { i = (i + 1) % FRAMES.length; pre.innerHTML = FRAMES[i]; },\n"
            "            Math.max(20, Math.round(1000 / FPS)));\n"
            "</script>\n</body>\n</html>\n"
        )


def effects_help() -> str:
    return "\n".join(f"  {name:<18} {e.about}" for name, e in EFFECTS.items())


def add_animate_args(parser) -> None:
    g = parser.add_argument_group("animation (--animate)")
    g.add_argument("--animate", default=None, metavar="EFFECT[,EFFECT]",
                   help="Animate the text: " + ", ".join(EFFECTS) + ". Combine with commas "
                        "(e.g. wave,rainbow). Plays in the terminal, or -o .gif/.html/.frames")
    g.add_argument("--frames", type=int, default=36, help="Frames per loop (default: 36)")
    g.add_argument("--fps", type=float, default=15.0, help="Frames per second (default: 15)")
    g.add_argument("--loops", type=int, default=3, help="Terminal playback loops, 0 = until Ctrl-C (default: 3)")
    g.add_argument("--amount", type=float, default=1.0, help="Effect strength, 0.1-3 (default: 1)")
    g.add_argument("--mirror-back", action="store_true",
                   help="Spins and flips show the back of the text mirrored, like a real card")


def run_cli(text: str, args) -> int:
    """`ascii-magic text ... --animate EFFECT`."""
    import sys

    try:
        effects = parse_effects(args.animate)
    except ValueError as e:
        raise SystemExit(f"text-to-ascii: {e}")
    style = args.style if args.style in RAMPS else None
    if style is None:
        raise SystemExit(f"text-to-ascii: --animate works with styles {', '.join(STYLES)}, not {args.style!r}")
    if not 2 <= args.frames <= 600:
        raise SystemExit("text-to-ascii: --frames must be 2-600")
    if not 0.5 <= args.fps <= 60:
        raise SystemExit("text-to-ascii: --fps must be 0.5-60")
    if not 0.1 <= args.amount <= 3:
        raise SystemExit("text-to-ascii: --amount must be 0.1-3")
    if not 4 <= args.width <= 400:
        raise SystemExit("text-to-ascii: --width must be 4-400 for animations")
    color = args.overlay
    try:
        anim = TextAnimation(text, TextAnimOptions(
            effects=effects, cols=args.width, frames=args.frames, fps=args.fps, style=style,
            amount=args.amount, color=color, mirror_back=args.mirror_back, font_path=args.font,
        ))
    except ValueError as e:
        raise SystemExit(f"text-to-ascii: {e}")

    out = args.output
    if not out:
        from .ansi import resolve_depth

        anim.play(loops=args.loops, color_depth=resolve_depth(args.color_depth, to_terminal=True))
        return 0
    low = out.lower()
    if low.endswith(".gif"):
        with open(out, "wb") as f:
            f.write(anim.to_gif_bytes())
    elif low.endswith((".html", ".htm")):
        with open(out, "w", encoding="utf-8") as f:
            f.write(anim.to_html(title=text[:60]))
    elif low.endswith(".frames"):
        anim.write_frames(out, loops=args.loops)
        if args.loops == 0:
            print("Note: --loops 0 plays until Ctrl-C; as a login greeting that holds up the prompt.",
                  file=sys.stderr)
    else:
        raise SystemExit("text-to-ascii: animations write .gif, .html, or .frames (or omit -o to play)")
    cols, rows = anim.size
    print(f"Wrote {out}: {len(anim.lines)} frames, {cols}x{rows}", file=sys.stderr)
    return 0
