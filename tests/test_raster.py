"""PNG export of finished art (issue #63)."""

import io
import sys

import numpy as np
import pytest
from PIL import Image

from asciimagic.glyphs import GlyphAtlas
from asciimagic.raster import DEFAULT_BG, DEFAULT_FG, TooLarge, ansi_to_image, ansi_to_png, parse_color

RED = "\x1b[38;2;255;0;0m"
RESET = "\x1b[0m"


def cells(img, atlas, x, y, span=1):
    """Pixels of character cell (x, y), allowing for the 1-cell padding."""
    a = np.asarray(img)
    cw, ch = atlas.cell_w, atlas.cell_h
    return a[(y + 1) * ch:(y + 2) * ch, (x + 1) * cw:(x + 1 + span) * cw]


@pytest.fixture(scope="module")
def atlas():
    return GlyphAtlas(None, 16)


def test_size_is_grid_plus_padding(atlas):
    img = ansi_to_image("ab\nabcd", font_size=16)
    assert img.size == ((4 + 2) * atlas.cell_w, (2 + 2) * atlas.cell_h) and img.mode == "RGB"


def test_plain_text_uses_fg_on_bg(atlas):
    img = ansi_to_image("#", fg=(10, 200, 30), bg=(1, 2, 3))
    cell = cells(img, atlas, 0, 0).reshape(-1, 3)
    assert tuple(cell[0]) == (1, 2, 3)               # untouched corner: background
    brightest = cell[cell.sum(1).argmax()]
    assert tuple(brightest) == (10, 200, 30)         # full-coverage ink: exactly fg


def test_ansi_colors_and_reset(atlas):
    img = ansi_to_image(f"{RED}#{RESET}#")
    red, plain = cells(img, atlas, 0, 0).reshape(-1, 3), cells(img, atlas, 1, 0).reshape(-1, 3)
    assert tuple(red[red.sum(1).argmax()]) == (255, 0, 0)
    assert tuple(plain[plain.sum(1).argmax()]) == DEFAULT_FG


def test_cell_background_is_filled(atlas):
    img = ansi_to_image("\x1b[48;2;0;0;255m \x1b[0m ")
    assert (cells(img, atlas, 0, 0) == (0, 0, 255)).all()
    assert (cells(img, atlas, 1, 0) == DEFAULT_BG).all()


def test_braille_is_drawn_not_a_missing_glyph_box(atlas):
    full, one = ansi_to_image("⣿"), ansi_to_image("⠁")
    ink = lambda im: (cells(im, atlas, 0, 0) > 0).any(2).sum()  # noqa: E731
    assert ink(full) > 4 * ink(one) > 0


def test_wide_characters_span_two_cells(atlas):
    img = ansi_to_image("水x")
    assert img.width == (3 + 2) * atlas.cell_w      # 水 takes two columns
    right_half = cells(img, atlas, 1, 0)
    assert right_half.any()                          # drawn across, not squashed into one cell


def test_transparent_background():
    img = ansi_to_image(f"{RED}#{RESET}", bg=None)
    a = np.asarray(img)
    assert img.mode == "RGBA" and a[0, 0, 3] == 0
    solid = a[a[:, :, 3] == 255]
    assert len(solid) and (solid[:, :3] == (255, 0, 0)).all()  # no dark fringe from a black bg


def test_limits():
    with pytest.raises(ValueError, match="font size"):
        ansi_to_image("x", font_size=200)
    with pytest.raises(TooLarge, match="pixels"):
        ansi_to_image("x" * 500, font_size=72, max_pixels=1_000_000)
    with pytest.raises(TooLarge, match="characters"):
        ansi_to_image("x" * 100, max_cells=50)
    with pytest.raises(TooLarge, match="too many characters"):  # rejected before parsing
        ansi_to_image("x" * 2_000_000, max_cells=1000)


def test_empty_art_is_still_an_image():
    assert Image.open(io.BytesIO(ansi_to_png(""))).size[0] > 0


@pytest.mark.parametrize("value,want", [
    (None, DEFAULT_BG), ("", DEFAULT_BG), ("transparent", None), ("NONE", None),
    ("#102030", (16, 32, 48)), ((1, 2, 3), (1, 2, 3)),
])
def test_parse_color(value, want):
    assert parse_color(value, DEFAULT_BG) == want


def test_parse_color_theme_name():
    assert len(parse_color("amber", DEFAULT_BG)) == 3


def test_parse_color_rejects_junk():
    with pytest.raises(ValueError):
        parse_color("bogus", DEFAULT_BG)


# ---- CLI ----

def _png(path):
    img = Image.open(path)
    img.load()
    return img


def test_text_cli_png(monkeypatch, tmp_path):
    from asciimagic.text_to_ascii import main

    out = tmp_path / "hi.png"
    monkeypatch.setattr(sys, "argv", ["text-to-ascii", "Hi\nthere", "-s", "block", "-w", "30",
                                      "--overlay", "sunset", "-o", str(out), "--png-bg", "transparent"])
    main()
    img = _png(out)
    assert img.format == "PNG" and img.mode == "RGBA"
    colors = {tuple(p[:3]) for p in np.asarray(img).reshape(-1, 4) if p[3] == 255}
    assert len(colors) > 3  # the overlay gradient made it into the image


def test_text_cli_png_errors(monkeypatch, tmp_path):
    from asciimagic.text_to_ascii import main

    monkeypatch.setattr(sys, "argv", ["text-to-ascii", "Hi", "-o", str(tmp_path / "a.png"), "--png-font-size", "2"])
    with pytest.raises(SystemExit, match="font size"):
        main()


def test_image_cli_png(monkeypatch, tmp_path):
    from asciimagic.image_to_ascii import main

    src = tmp_path / "p.png"
    a = np.zeros((60, 80, 3), np.uint8)
    a[:, :40] = (250, 30, 30)
    a[:, 40:] = (30, 30, 250)
    Image.fromarray(a).save(src)
    out = tmp_path / "art.png"
    monkeypatch.setattr(sys, "argv", ["image-to-ascii", str(src), "-c", "20", "--mode", "braille",
                                      "--color", "-o", str(out)])
    main()
    px = np.asarray(_png(out).convert("RGB")).reshape(-1, 3)
    lit = px[px.sum(1) > 150]
    assert (lit[:, 0] > lit[:, 2] + 100).any() and (lit[:, 2] > lit[:, 0] + 100).any()  # red and blue halves


def test_compose_cli_png(tmp_path):
    from asciimagic.compose import main

    out = tmp_path / "c.png"
    assert main(["--text", "HI", "--style", "block", "--color", "amber", "-o", str(out)]) == 0
    assert _png(out).format == "PNG"
    with pytest.raises(SystemExit):
        main(["--text", "HI", "--format", "text", "-o", str(tmp_path / "d.png")])


def test_text_outputs_unchanged(monkeypatch, tmp_path):
    from asciimagic.text_to_ascii import main

    out = tmp_path / "hi.txt"
    monkeypatch.setattr(sys, "argv", ["text-to-ascii", "Hi", "-s", "plain", "-o", str(out)])
    main()
    assert out.read_text(encoding="utf-8") == "Hi\n"


# ---- web ----

def test_web_export_png():
    from fastapi.testclient import TestClient

    from asciimagic.webapp import app

    client = TestClient(app)
    r = client.post("/api/export/png", data={"ansi": f"{RED}HI{RESET}", "bg": "transparent"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).mode == "RGBA"
    assert client.post("/api/export/png", data={"ansi": "x", "bg": "bogus"}).status_code == 400
    assert client.post("/api/export/png", data={"ansi": "x", "font_size": "500"}).status_code == 400
    big = "\n".join(["x" * 1000] * 300)  # 300k cells, over the default budget
    assert client.post("/api/export/png", data={"ansi": big}).status_code == 413
