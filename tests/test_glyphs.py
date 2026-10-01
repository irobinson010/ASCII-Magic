"""Glyph cells for GIF/MP4 output (issue #48): braille must not render as
"missing glyph" boxes when the monospace font lacks the Braille block."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from asciimagic.glyphs import BRAILLE_BASE, GlyphAtlas, braille_alpha, is_braille


def test_is_braille():
    assert is_braille("⠀") and is_braille("⣿")
    assert not is_braille("A") and not is_braille("⤀")


def test_blank_and_full_braille():
    assert braille_alpha("⠀", 8, 16).sum() == 0
    full = braille_alpha("⣿", 8, 16)
    one = braille_alpha("⠁", 8, 16)
    assert full.shape == (16, 8)
    assert full.sum() == pytest.approx(8 * one.sum(), rel=0.05)  # eight equal dots


@pytest.mark.parametrize("bit,quadrant", [
    (0, ("left", 0)), (1, ("left", 1)), (2, ("left", 2)), (6, ("left", 3)),
    (3, ("right", 0)), (4, ("right", 1)), (5, ("right", 2)), (7, ("right", 3)),
])
def test_dot_positions(bit, quadrant):
    a = braille_alpha(chr(BRAILLE_BASE + (1 << bit)), 8, 16)
    ys, xs = np.nonzero(a > 0.3)
    side, row = quadrant
    assert (xs.max() < 4) if side == "left" else (xs.min() >= 4)
    assert row * 4 <= ys.min() and ys.max() < (row + 1) * 4


def _notdef(atlas):
    img = Image.new("L", (atlas.cell_w, atlas.cell_h), 0)
    ImageDraw.Draw(img).text((0, 0), "\U0010fffd", fill=255, font=atlas.font)
    return np.asarray(img, dtype=np.float32) / 255.0


def test_atlas_draws_braille_dots_not_boxes():
    atlas = GlyphAtlas(None, 14)
    for ch in ("⠁", "⣿", "⡇"):
        a = atlas.alpha(ch)
        assert a.shape == (atlas.cell_h, atlas.cell_w)
        assert a.sum() > 0 and not np.array_equal(a, _notdef(atlas))


def test_atlas_ascii_uses_font_and_caches():
    atlas = GlyphAtlas(None, 14)
    a = atlas.alpha("A")
    assert a.sum() > 0 and atlas.alpha("A") is a


def test_atlas_falls_back_for_missing_glyphs(monkeypatch):
    import asciimagic.text_to_ascii as t2a

    atlas = GlyphAtlas(None, 14)
    sentinel = object()
    monkeypatch.setattr(t2a, "missing_glyphs", lambda font, text: text)
    monkeypatch.setattr(t2a, "find_fallback_font", lambda text, size: sentinel)
    assert atlas._font_for("☃") is sentinel
    assert atlas._font_for("A") is atlas.font  # ASCII never needs a fallback
    monkeypatch.setattr(t2a, "find_fallback_font", lambda text, size: None)
    assert atlas._font_for("☃") is atlas.font  # nothing better: keep the font


def test_braille_video_gif_has_dots(tmp_path):
    from asciimagic.video import AsciiVideo

    lines = ["⣿⠀⣿", "⠁⠂⠄"]
    img = Image.new("RGB", (6, 4), (255, 255, 255))
    v = AsciiVideo([(lines, img)], fps=5)
    arr = v._frame_arrays()[0]
    atlas = GlyphAtlas(None, 14)
    cell = arr[: atlas.cell_h, : atlas.cell_w].mean(axis=2) / 255.0
    np.testing.assert_allclose(cell, atlas.alpha("⣿"), atol=0.02)
