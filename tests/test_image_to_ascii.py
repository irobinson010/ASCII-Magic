import numpy as np
from PIL import Image, ImageDraw

from ascii_magic.image_to_ascii import (
    _floyd_steinberg_dots,
    image_to_braille_from_image,
    image_to_text_glyph_from_image,
    make_charset,
    pick_char_balanced,
    pick_char_best,
    pick_char_fast,
    render_glyphs,
)


def _photo():
    img = Image.new("L", (120, 80), 30)
    d = ImageDraw.Draw(img)
    d.ellipse([20, 10, 100, 70], fill=200)
    d.rectangle([50, 30, 70, 50], fill=90)
    return img


def _gradient():
    img = Image.new("L", (128, 64))
    px = img.load()
    for y in range(64):
        for x in range(128):
            px[x, y] = x * 2
    return img


CHARSET = make_charset(unicode_mode="off", ascii_preset="dense")


def _glyph(img, quality, cols=24):
    return image_to_text_glyph_from_image(
        img=img, cols=cols, cell_w=8, cell_h=16, charset=CHARSET,
        quality=quality, font_path=None, font_size=None,
        autocontrast=False, gamma=1.0, invert=False, topk=24,
    )


def test_glyph_qualities_render():
    for q in ("fast", "balanced", "best"):
        art = _glyph(_photo(), q)
        lines = art.split("\n")
        assert all(len(ln) == 24 for ln in lines)
        assert len(set(art)) > 3  # more than a couple of distinct glyphs


def test_vectorized_matches_per_cell_reference():
    """The batched matcher must agree with the original per-cell helpers."""
    img = _photo()
    cols, cell_w, cell_h, topk = 12, 8, 16, 24

    art_fast = _glyph(img, "fast", cols=cols)
    art_balanced = _glyph(img, "balanced", cols=cols)
    art_best = _glyph(img, "best", cols=cols)

    # Reference: replicate preprocessing, then match cell-by-cell.
    from ascii_magic.image_to_ascii import preprocess_image, sobel_gradients

    g = preprocess_image(img.convert("L"), autocontrast=False, gamma=1.0, invert=False)
    W, H = g.size
    rows = max(1, int((H / W) * cols * (cell_w / cell_h)))
    g = g.resize((cols * cell_w, rows * cell_h), resample=Image.Resampling.BILINEAR)
    ink = 1.0 - np.asarray(g, dtype=np.float32) / 255.0
    mag, ang = sobel_gradients(ink)
    glyph_imgs, glyph_feats_n, chars, (mu, sd) = render_glyphs(CHARSET, cell_w, cell_h, None, None)

    ref = {"fast": [], "balanced": [], "best": []}
    for r in range(rows):
        for c in range(cols):
            sl = np.s_[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            ci, cm, ca = ink[sl], mag[sl], ang[sl]
            feat = np.array(
                [ci.mean(), cm.mean(), (np.cos(ca) * cm).mean(), (np.sin(ca) * cm).mean()],
                dtype=np.float32,
            )
            fn = (feat - mu) / sd
            vec = ci.reshape(-1).astype(np.float32)
            ref["fast"].append(chars[pick_char_fast(fn, glyph_feats_n)])
            ref["balanced"].append(chars[pick_char_balanced(fn, vec, glyph_feats_n, glyph_imgs, topk)])
            ref["best"].append(chars[pick_char_best(vec, glyph_imgs)])

    for q, art in (("fast", art_fast), ("balanced", art_balanced), ("best", art_best)):
        got = art.replace("\n", "")
        assert got == "".join(ref[q]), f"quality={q} diverged from reference"


def test_glyph_cache_reused():
    a = render_glyphs(CHARSET, 8, 16, None, None)
    b = render_glyphs(CHARSET, 8, 16, None, None)
    assert a is b


def test_degenerate_cols_raise_value_error():
    import pytest

    img = _photo()
    with pytest.raises(ValueError, match="cols"):
        image_to_braille_from_image(img, cols=0, autocontrast=False, gamma=1.0,
                                    invert=False, threshold=0.5)
    with pytest.raises(ValueError, match="cols"):
        _glyph(img, "fast", cols=0)


def test_braille_dither_differs_and_preserves_gradient():
    img = _gradient()
    hard = image_to_braille_from_image(
        img, cols=32, autocontrast=False, gamma=1.0, invert=False, threshold=0.5
    )
    dithered = image_to_braille_from_image(
        img, cols=32, autocontrast=False, gamma=1.0, invert=False, threshold=0.5, dither=True
    )
    assert hard != dithered

    def dots_in_right_quarter(art):
        total = 0
        for line in art.split("\n"):
            for ch in line[-(len(line) // 4):]:
                total += (ord(ch) - 0x2800).bit_count()
        return total

    # Bright side (low ink): hard threshold drops all dots, dithering keeps some.
    assert dots_in_right_quarter(hard) == 0
    assert dots_in_right_quarter(dithered) > 0


def test_floyd_steinberg_density_tracks_input():
    ink = np.full((40, 40), 0.3, dtype=np.float32)
    dots = _floyd_steinberg_dots(ink, threshold=0.5)
    density = dots.mean()
    assert 0.2 < density < 0.4  # ~30% of dots set for 0.3 ink
