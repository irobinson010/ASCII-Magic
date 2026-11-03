"""Line art: outlines instead of shading (issue #67)."""

import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

from asciimagic import lineart
from asciimagic.image_to_ascii import image_to_braille_from_image, image_to_text_glyph_from_image, make_charset
from asciimagic.lineart import line_art

BLANK = "⠀"  # braille with no dots


def square(size=(200, 200), box=(50, 50, 150, 150)):
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).rectangle(box, fill="black")
    return img


def dots(art):
    """Per character: does it have ink?"""
    return np.array([[c not in (" ", BLANK) for c in ln] for ln in art.splitlines()])


def ink(art):
    """Total braille dots lit (a border cell with 2 dots isn't a solid block)."""
    return sum(bin(ord(c) - 0x2800).count("1") for c in art if 0x2800 <= ord(c) <= 0x28FF)


def test_outline_not_fill():
    a = np.asarray(line_art(square(), out_width=100))
    assert a.min() < 64                 # the square's edges are drawn dark
    assert a[100, 100] == 255           # its inside is left blank...
    assert a[5, 5] == 255               # ...and so is the background


def test_braille_square_is_hollow():
    shade = dots(image_to_braille_from_image(square(), 40, False, 1.0, False, 0.5))
    lines_art = image_to_braille_from_image(square(), 40, False, 1.0, False, 0.5, line_art=0.5)
    lines = dots(lines_art)
    assert ink(lines_art) < ink(image_to_braille_from_image(square(), 40, False, 1.0, False, 0.5)) / 3
    mid = lines.shape[0] // 2, lines.shape[1] // 2
    assert shade[mid] and not lines[mid]           # shading fills it; line art doesn't
    assert lines.any(axis=1)[mid[0]]               # but the row through it crosses the border



def test_glyph_mode_too():
    cs = make_charset(unicode_mode="off", ascii_preset="dense")
    args = dict(cols=40, cell_w=8, cell_h=16, charset=cs, quality="fast", font_path=None, font_size=None,
                autocontrast=False, gamma=1.0, invert=False, topk=8)
    shade = image_to_text_glyph_from_image(square(), **args)
    lines = image_to_text_glyph_from_image(square(), line_art=0.5, **args)
    assert lines != shade
    mid = lines.splitlines()[len(lines.splitlines()) // 2]
    assert mid[len(mid) // 2] == " "  # hollow middle


def test_detail_controls_how_much_texture_is_kept():
    # A strong outline (the square) plus mottled texture in one corner. Only
    # the corner is measured: lower detail also blurs more, which thickens
    # the main outline, so a whole-image count mostly measures line width.
    img = square()
    rng = np.random.default_rng(0)
    a = np.asarray(img).copy()
    # Full-contrast speckle: edge strength is relative to the picture's
    # strongest edges (here a perfect black-on-white square), so faint
    # texture next to it is light grey at any detail, below the ink threshold.
    a[:40, :40] = rng.integers(0, 256, size=(40, 40, 1))
    img = Image.fromarray(a)
    texture = {d: int((np.asarray(line_art(img, 100, d))[:20, :20] < 128).sum()) for d in (0.0, 1.0)}
    assert texture[1.0] > 2 * max(texture[0.0], 1)
    outline = np.asarray(line_art(img, 100, 0.0))
    assert (outline[20:80, 20:80] < 128).any()  # the square's outline survives at any detail


def test_flat_picture_has_no_lines():
    a = np.asarray(line_art(Image.new("RGB", (64, 64), (90, 120, 30)), 32))
    assert (a == 255).all()


def test_size_and_limits(monkeypatch):
    assert line_art(square((1000, 500)), out_width=100).size == (200, 100)  # 2x the output
    assert line_art(square((60, 60)), out_width=100).size == (60, 60)       # never upscaled
    monkeypatch.setattr(lineart, "MAX_WORK_PIXELS", 10_000)
    w, h = line_art(square((400, 4000)), out_width=400).size
    assert w * h <= 10_000 and abs(h / w - 10) < 0.5                        # capped, aspect kept
    with pytest.raises(ValueError, match="between 0 and 1"):
        line_art(square(), 10, 1.5)


def test_dither_is_skipped_for_line_art():
    a = image_to_braille_from_image(square(), 40, False, 1.0, False, 0.5, dither=True, line_art=0.5)
    b = image_to_braille_from_image(square(), 40, False, 1.0, False, 0.5, dither=False, line_art=0.5)
    assert a == b


# ---- CLI / compose / video / web ----

def test_image_cli(monkeypatch, tmp_path, capsys):
    from asciimagic.image_to_ascii import main

    src = tmp_path / "sq.png"
    square().save(src)
    monkeypatch.setattr(sys, "argv", ["image-to-ascii", str(src), "--mode", "braille", "-c", "40", "--line-art"])
    main()
    art = capsys.readouterr().out
    d = dots(art.rstrip("\n"))
    assert d.any() and not d[d.shape[0] // 2, d.shape[1] // 2]
    monkeypatch.setattr(sys, "argv", ["image-to-ascii", str(src), "--line-art", "--line-detail", "2"])
    with pytest.raises(SystemExit):
        main()


def test_compose_layer(tmp_path):
    from asciimagic.compose import Canvas, Layer, Scene, compose

    def render(**kw):
        layer = Layer(type="image", cols=40, **kw)
        return compose(Scene(canvas=Canvas(), layers=[layer]), images={0: square()}).to_text()

    assert render(line_art=True) != render()
    with pytest.raises(ValueError, match="line_detail"):
        Layer(type="image", src="x.png", line_art=True, line_detail=3).validate()


def test_video_frames(tmp_path):
    from asciimagic import video as video_mod

    path = tmp_path / "sq.gif"
    frames = [square(box=(40 + i * 5, 50, 140 + i * 5, 150)) for i in range(4)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    shade = video_mod.video_to_ascii(str(path), cols=40, max_frames=4)
    lines = video_mod.video_to_ascii(str(path), cols=40, max_frames=4, line_art=0.5)
    first_shade, first_lines = (ink("\n".join(v.frames[0][0])) for v in (shade, lines))
    assert 0 < first_lines < first_shade / 3


def test_web_render_option():
    import io

    from fastapi.testclient import TestClient

    from asciimagic.webapp import app

    buf = io.BytesIO()
    square().save(buf, "PNG")
    client = TestClient(app)

    def render(**opts):
        import json

        r = client.post("/api/render", files={"image": ("sq.png", buf.getvalue(), "image/png")},
                        data={"options": json.dumps({"mode": "braille", "cols": 40, **opts})})
        assert r.status_code == 200, r.text
        return r.json()["ascii"]

    shade, lines = render(), render(line_art=True, line_detail=0.5)
    assert 0 < ink(lines) < ink(shade) / 3
    assert render(line_art=False, line_detail=0.9) == shade  # detail alone changes nothing
