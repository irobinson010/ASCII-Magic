"""Character cells for image output (GIF frames, MP4 video).

GIF and MP4 sinks draw each character into a fixed-size cell. The default
monospace fonts don't cover everything the converters emit: DejaVu Sans
Mono (the usual Linux pick) has no Braille Patterns, so every braille-mode
GIF came out as a grid of "missing glyph" boxes. Braille is therefore drawn
directly from the dot pattern (exact, with no font needed), and any other
character the font lacks is drawn from a fallback font that has it.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BRAILLE_BASE = 0x2800
# (column, row) of the dot for each bit of a braille code point (ISO 11548-1).
BRAILLE_DOTS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3))
_SUPERSAMPLE = 4


def is_braille(ch: str) -> bool:
    return len(ch) == 1 and BRAILLE_BASE <= ord(ch) <= BRAILLE_BASE + 0xFF


def braille_alpha(ch: str, cell_w: int, cell_h: int) -> np.ndarray:
    """Coverage (0..1, shape cell_h x cell_w) of a braille character's dots,
    laid out 2 across by 4 down like a terminal draws them."""
    bits = ord(ch) - BRAILLE_BASE
    s = _SUPERSAMPLE
    img = Image.new("L", (cell_w * s, cell_h * s), 0)
    draw = ImageDraw.Draw(img)
    pitch_x, pitch_y = cell_w * s / 2, cell_h * s / 4
    r = max(1.0, 0.34 * min(pitch_x, pitch_y))
    for bit, (col, row) in enumerate(BRAILLE_DOTS):
        if bits >> bit & 1:
            cx, cy = (col + 0.5) * pitch_x, (row + 0.5) * pitch_y
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    img = img.resize((cell_w, cell_h), Image.Resampling.BOX)
    return np.asarray(img, dtype=np.float32) / 255.0


class GlyphAtlas:
    """A monospace font sized into cells, with per-character alpha masks."""

    def __init__(self, font_path: Optional[str] = None, font_size: int = 14):
        from .image_to_ascii import find_default_mono_font

        path = font_path or find_default_mono_font()
        if path:
            self.font = ImageFont.truetype(path, font_size)
            ascent, descent = self.font.getmetrics()
            self.cell_w = max(1, round(self.font.getlength("M")))
            self.cell_h = ascent + descent
        else:
            self.font = ImageFont.load_default()
            self.cell_w, self.cell_h = 7, 13
        self._cache: Dict[str, np.ndarray] = {}

    def alpha(self, ch: str) -> np.ndarray:
        """Coverage of `ch` in one cell, float32 0..1, shape (cell_h, cell_w)."""
        a = self._cache.get(ch)
        if a is None:
            a = self._render(ch)
            self._cache[ch] = a
        return a

    def _render(self, ch: str) -> np.ndarray:
        if is_braille(ch):
            return braille_alpha(ch, self.cell_w, self.cell_h)
        img = Image.new("L", (self.cell_w, self.cell_h), 0)
        ImageDraw.Draw(img).text((0, 0), ch, fill=255, font=self._font_for(ch))
        return np.asarray(img, dtype=np.float32) / 255.0

    def _font_for(self, ch: str):
        if ord(ch[0]) <= 126 or not isinstance(self.font, ImageFont.FreeTypeFont):
            return self.font
        from .text_to_ascii import find_fallback_font, missing_glyphs

        if not missing_glyphs(self.font, ch):
            return self.font
        return find_fallback_font(ch, self.font.size) or self.font
