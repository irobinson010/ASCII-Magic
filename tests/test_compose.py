import json

import pytest
from PIL import Image, ImageDraw

from asciimagic.compose import Canvas, Layer, Scene, compose, main as compose_main
from asciimagic.unified_cli import main as cli_main


@pytest.fixture
def photo(tmp_path):
    """Dark disc on white, red top half: ink in the middle, known colors."""
    img = Image.new("RGB", (160, 120), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 159, 59], fill=(220, 30, 30))
    d.ellipse([40, 20, 120, 100], fill=(10, 10, 10))
    p = tmp_path / "photo.png"
    img.save(p)
    return p


def _text(comp):
    return comp.to_text().splitlines()


def test_single_text_layer_fits_canvas_to_block():
    comp = compose(Scene(layers=[Layer(type="text", text="Hi", style="box")]))
    assert _text(comp) == ["┌────┐", "│ Hi │", "└────┘"]


@pytest.mark.parametrize("anchor,expect_xy", [
    ("top-left", (0, 0)), ("top", (7, 0)), ("top-right", (14, 0)),
    ("left", (0, 3)), ("center", (7, 3)), ("right", (14, 3)),
    ("bottom-left", (0, 7)), ("bottom", (7, 7)), ("bottom-right", (14, 7)),
])
def test_anchors_place_block_on_canvas(anchor, expect_xy):
    scene = Scene(canvas=Canvas(cols=20, rows=10),
                  layers=[Layer(type="text", text="Hi", style="box", at=anchor)])
    comp = compose(scene)
    p = comp.placed[0]
    assert (p.x, p.y) == expect_xy and (p.w, p.h) == (6, 3)
    assert _text(comp)[expect_xy[1] + 1][expect_xy[0]:expect_xy[0] + 6] == "│ Hi │"


def test_exact_xy_and_nudge():
    scene = Scene(canvas=Canvas(cols=20, rows=6),
                  layers=[Layer(type="text", text="Hi", style="box", x=3, y=1, dx=2, dy=1)])
    p = compose(scene).placed[0]
    assert (p.x, p.y) == (5, 2)


def test_layers_clip_at_canvas_edges():
    scene = Scene(canvas=Canvas(cols=4, rows=2),
                  layers=[Layer(type="text", text="Hi", style="box", x=-2, y=-1)])
    assert _text(compose(scene)) == ["Hi │", "───┘"]


def test_auto_canvas_grows_for_absolute_positions():
    scene = Scene(layers=[
        Layer(type="text", text="A", style="box"),
        Layer(type="text", text="B", style="box", x=10, y=4),
    ])
    comp = compose(scene)
    assert (comp.cols, comp.rows) == (15, 7)


def test_image_layer_exact_size(photo):
    comp = compose(Scene(layers=[Layer(type="image", src=str(photo), cols=40, rows=12)]))
    assert (comp.cols, comp.rows) == (40, 12)


@pytest.mark.parametrize("mode", ["braille", "glyph"])
def test_image_layer_rows_only_keeps_aspect(photo, mode):
    comp = compose(Scene(layers=[Layer(type="image", src=str(photo), rows=10, mode=mode)]))
    assert comp.rows == 10
    # 160x120 photo; braille cells are 2x4 px, glyph 8x16: cols = rows*w*ch/(h*cw)
    assert comp.cols == round(10 * 160 * (4 if mode == "braille" else 16) / (120 * (2 if mode == "braille" else 8)))


def test_blanks_are_see_through_and_opaque_covers(photo):
    base = Layer(type="image", src=str(photo), cols=30, rows=10, invert=True)
    see_through = compose(Scene(layers=[base, Layer(type="text", text=" ", style="box", at="center", outline=0)]))
    opaque = compose(Scene(layers=[base, Layer(type="text", text="   ", style="box", at="center", opaque=True)]))
    p = opaque.placed[1]
    inside = _text(opaque)[p.y + 1][p.x + 1:p.x + p.w - 1]
    assert inside.strip() == ""
    assert _text(see_through) != _text(opaque)


def test_outline_knocks_out_art_around_text(photo):
    img = Layer(type="image", src=str(photo), cols=40, rows=12, threshold=0.0)  # every cell is ink
    comp = compose(Scene(layers=[img, Layer(type="text", text="X", style="box", at="center")]))
    p = comp.placed[1]
    rows = _text(comp)
    # one-cell ring around the box is cleared (default outline=1 for text)
    assert rows[p.y - 1][p.x - 1:p.x + p.w + 1].strip() == ""
    assert "⣿" in rows[0]  # art far from the text is untouched
    no_outline = compose(Scene(layers=[img, Layer(type="text", text="X", style="box", at="center", outline=0)]))
    assert _text(no_outline)[p.y - 1][p.x - 1:p.x + p.w + 1].strip() != ""


def test_image_color_samples_picture(photo):
    comp = compose(Scene(layers=[Layer(type="image", src=str(photo), cols=40, rows=12, color="image", threshold=0.0)]))
    top_left = comp.cells[0][0][1]
    assert top_left[0] > 150 and top_left[1] < 90  # red band


def test_text_color_image_samples_underlying_picture(photo):
    scene = Scene(layers=[
        Layer(type="image", src=str(photo), cols=40, rows=12, threshold=0.0),
        Layer(type="text", text="RED", style="box", at="top", color="image", outline=0),
    ])
    comp = compose(scene)
    p = comp.placed[1]
    ch, rgb = comp.cells[p.y + 1][p.x + 2]
    assert ch == "R" and rgb[0] > 150 and rgb[1] < 90


def test_solid_color_and_background_in_ansi_and_html():
    scene = Scene(canvas=Canvas(background="#102030"),
                  layers=[Layer(type="text", text="Hi", style="box", color="amber")])
    comp = compose(scene)
    ansi = comp.to_ansi()
    assert "\x1b[38;2;255;176;0m" in ansi and "\x1b[48;2;16;32;48m" in ansi
    doc = comp.to_html()
    assert "color:#ffb000" in doc and "background: #102030" in doc


def test_plain_output_has_no_escape_codes():
    comp = compose(Scene(layers=[Layer(type="text", text="Hi", style="box")]))
    assert "\x1b[3" not in comp.to_ansi()


def test_html_escapes_text():
    comp = compose(Scene(layers=[Layer(type="text", text="<b>&", style="box", color="white")]))
    assert "&lt;b&gt;&amp;" in comp.to_html()


@pytest.mark.parametrize("bad", [
    {"type": "video"}, {"type": "text", "text": "x", "at": "middle"},
    {"type": "text", "text": ""}, {"type": "text", "text": "x", "style": "comic"},
    {"type": "text", "text": "x", "color": "puce"}, {"type": "text", "text": "x", "cols": 0},
    {"type": "text", "text": "x", "scale": 2},
])
def test_invalid_layers_rejected(bad):
    with pytest.raises(ValueError):
        compose(Scene.from_dict({"layers": [bad]}))


def test_unknown_scene_fields_rejected():
    with pytest.raises(ValueError, match="unknown field"):
        Scene.from_dict({"layers": [{"type": "text", "text": "x", "colour": "red"}]})


def test_max_cells_budget():
    with pytest.raises(ValueError, match="limit"):
        compose(Scene(canvas=Canvas(cols=1000, rows=1000),
                      layers=[Layer(type="text", text="x")]), max_cells=10_000)


def test_scene_round_trip_with_relative_paths(photo, tmp_path):
    scene = Scene(canvas=Canvas(cols=50, rows=16, background="#000000"), layers=[
        Layer(type="image", src=str(photo), cols=40, color="image"),
        Layer(type="text", text="Hi", style="figlet", at="bottom", dy=-1, color="cyan"),
    ])
    out = tmp_path / "scenes" / "card.json"
    out.parent.mkdir()
    scene.save(str(out))
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["layers"][0]["src"] == "../photo.png"
    assert "mode" not in raw["layers"][0]  # defaults are omitted
    loaded = Scene.load(str(out))
    assert compose(loaded).to_ansi() == compose(scene).to_ansi()


# ---- CLI ----

def test_cli_layers_and_options(photo, tmp_path, capsys):
    out = tmp_path / "card.txt"
    rc = cli_main(["compose", "--canvas", "40x14",
                   "--image", str(photo), "--cols", "30", "--at", "left",
                   "--text", "Hi", "--style", "box", "--at", "bottom-right",
                   "-o", str(out)])
    assert rc == 0
    rows = out.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 14
    assert rows[-2].rstrip().endswith("│ Hi │")


def test_cli_option_before_layer_is_an_error(capsys):
    with pytest.raises(SystemExit):
        compose_main(["--at", "top", "--text", "Hi"])
    assert "must follow an --image or --text" in capsys.readouterr().err


def test_cli_save_then_render_scene(photo, tmp_path, capsys):
    scene_path = tmp_path / "s.json"
    assert compose_main(["--image", str(photo), "--cols", "20", "--text", "Yo", "--style", "box",
                         "--color", "violet", "--save-scene", str(scene_path)]) == 0
    first = capsys.readouterr().out
    assert compose_main([str(scene_path)]) == 0
    assert capsys.readouterr().out == first


def test_cli_html_output_by_extension(tmp_path):
    out = tmp_path / "x.html"
    assert compose_main(["--text", "Hi", "--style", "box", "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8").lower().startswith("<!doctype html>")


def test_cli_reports_bad_color(capsys):
    assert compose_main(["--text", "Hi", "--color", "puce"]) == 2
    assert "puce" in capsys.readouterr().err


def test_saved_scene_paths_use_forward_slashes_on_windows(photo, tmp_path, monkeypatch):
    import os

    import asciimagic.compose as compose_mod

    # Simulate Windows: relpath yields backslashes and os.sep is "\\".
    monkeypatch.setattr(compose_mod.os.path, "relpath", lambda p, b: "..\\imgs\\photo.png")
    monkeypatch.setattr(compose_mod.os, "sep", "\\")
    out = tmp_path / "card.json"
    Scene(layers=[Layer(type="image", src=str(photo))]).save(str(out))
    monkeypatch.undo()
    assert json.loads(out.read_text(encoding="utf-8"))["layers"][0]["src"] == "../imgs/photo.png"
    assert os.sep  # restored
