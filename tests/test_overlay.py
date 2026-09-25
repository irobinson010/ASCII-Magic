import json
import math
import re
import sys

import pytest
from PIL import Image

from asciimagic.overlay import (
    PALETTES,
    Overlay,
    ansi_to_html,
    apply_to_ansi,
    cells_to_ansi,
    parse_ansi,
)

FG = re.compile(r"\x1b\[38;2;(\d+);(\d+);(\d+)m")


def _strip(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _colors(ansi):
    return [tuple(map(int, m)) for m in FG.findall(ansi)]


# ---- model ----

def test_parse_palette_single_and_gradient():
    assert len(Overlay.parse("rainbow").stops) == len(PALETTES["rainbow"])
    assert Overlay.parse("#ff0000").stops == [(255, 0, 0)]
    assert Overlay.parse("amber, #0000ff").stops == [(255, 176, 0), (0, 0, 255)]


@pytest.mark.parametrize("spec,kw", [
    ("puce", {}), ("", {}), ("#ff0000", {"direction": "sideways"}),
    ("#ff0000", {"mode": "burn"}), ("#ff0000", {"strength": 1.5}), ("#ff0000", {"strength": math.nan}),
])
def test_parse_rejects_bad_input(spec, kw):
    with pytest.raises(ValueError):
        Overlay.parse(spec, **kw)


def test_unknown_color_message_lists_palettes():
    with pytest.raises(ValueError, match="palette"):
        Overlay.parse("puce")


@pytest.mark.parametrize("direction,pos", [
    ("horizontal", [(0, 0, 0.0), (9, 0, 1.0), (9, 4, 1.0)]),
    ("vertical", [(0, 0, 0.0), (0, 4, 1.0), (9, 0, 0.0)]),
    ("diagonal", [(0, 0, 0.0), (9, 4, 1.0), (9, 0, 0.5)]),
    ("diagonal-up", [(0, 4, 0.0), (9, 0, 1.0)]),
    ("radial", [(0, 0, 1.0), (9, 4, 1.0)]),
])
def test_positions(direction, pos):
    ov = Overlay.parse("#000000,#ffffff", direction=direction)
    for x, y, t in pos:
        assert ov.position(x, y, 10, 5) == pytest.approx(t)


def test_radial_center_is_start():
    ov = Overlay.parse("#000000,#ffffff", direction="radial")
    assert ov.position(5, 5, 11, 11) == pytest.approx(0.0)


def test_multistop_interpolation():
    ov = Overlay.parse("#000000,#ff0000,#ffffff")
    assert ov.color_at(0) == (0, 0, 0)
    assert ov.color_at(0.5) == (255, 0, 0)
    assert ov.color_at(0.25) == (128, 0, 0)
    assert ov.color_at(1) == (255, 255, 255)


@pytest.mark.parametrize("mode,base,over,expect", [
    ("tint", (10, 20, 30), (200, 100, 0), (200, 100, 0)),
    ("multiply", (128, 255, 0), (255, 128, 255), (128, 128, 0)),
    ("screen", (0, 128, 255), (0, 128, 0), (0, 192, 255)),
    ("overlay", (64, 192, 128), (128, 128, 128), (64, 192, 128)),
])
def test_blend_modes(mode, base, over, expect):
    got = Overlay.parse("#000000", mode=mode).blend(base, over)
    assert all(abs(a - b) <= 1 for a, b in zip(got, expect))


def test_strength_mixes_with_base_and_plain_text_counts_as_white():
    ov = Overlay.parse("#000000", strength=0.25)
    assert ov.blend((200, 200, 200), (0, 0, 0)) == (150, 150, 150)
    assert Overlay.parse("#ff0000", mode="multiply").blend(None, (255, 0, 0)) == (255, 0, 0)


# ---- ANSI round trip ----

def test_parse_ansi_tracks_fg_bg_and_resets():
    rows = parse_ansi("\x1b[38;2;1;2;3mAB\x1b[0m C\n\x1b[48;2;4;5;6m \x1b[0m\n")
    assert [(c.ch, c.fg, c.bg) for c in rows[0]] == [
        ("A", (1, 2, 3), None), ("B", (1, 2, 3), None), (" ", None, None), ("C", None, None)]
    assert (rows[1][0].ch, rows[1][0].bg) == (" ", (4, 5, 6))


def test_roundtrip_keeps_visible_text_and_colors():
    src = "\x1b[38;2;9;9;9mhello\x1b[0m world\n"
    again = cells_to_ansi(parse_ansi(src))
    assert _strip(again) == _strip(src)
    assert _colors(again)[0] == (9, 9, 9)


def test_overlay_colors_every_visible_char_and_skips_spaces():
    out = apply_to_ansi(Overlay.parse("#000000,#ffffff"), "ab  cd\n")
    cols = _colors(out)
    assert cols[0] == (0, 0, 0) and cols[-1] == (255, 255, 255)
    assert re.sub(r"\x1b\[[0-9;]*m", "", out) == "ab  cd\n"


def test_overlay_keeps_wide_chars_intact():
    out = apply_to_ansi(Overlay.parse("sunset"), "日本語\n")
    assert re.sub(r"\x1b\[[0-9;]*m", "", out) == "日本語\n"
    assert len(_colors(out)) == 3


def test_ansi_to_html_escapes_and_colors():
    doc = ansi_to_html(apply_to_ansi(Overlay.parse("#ff0000"), "<b>&\n"))
    assert "&lt;b&gt;&amp;" in doc and "color:#ff0000" in doc


# ---- CLI ----

@pytest.fixture
def png(tmp_path):
    p = tmp_path / "t.png"
    Image.new("RGB", (40, 30), (30, 30, 30)).save(p)
    return p


def test_text_command_gradient(monkeypatch, capsys):
    from asciimagic.text_to_ascii import main

    monkeypatch.setattr(sys, "argv", ["text-to-ascii", "HI", "-s", "box", "--overlay", "#000000,#ffffff"])
    main()
    cols = _colors(capsys.readouterr().out)
    assert cols[0] == (0, 0, 0) and cols[-1] == (255, 255, 255)


def test_text_command_overlay_html_and_depth(tmp_path, monkeypatch, capsys):
    from asciimagic.text_to_ascii import main

    out = tmp_path / "t.html"
    monkeypatch.setattr(sys, "argv", ["text-to-ascii", "HI", "-s", "box", "--overlay", "fire", "-o", str(out)])
    main()
    assert "color:#" in out.read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["text-to-ascii", "HI", "-s", "box", "--overlay", "fire", "--color-depth", "256"])
    main()
    text = capsys.readouterr().out
    assert "\x1b[38;5;" in text and "\x1b[38;2;" not in text


def test_image_command_overlay_with_and_without_color(png, monkeypatch, capsys):
    from asciimagic.image_to_ascii import main

    for extra in ([], ["--color"]):
        monkeypatch.setattr(sys, "argv", ["image-to-ascii", str(png), "-c", "10", "--mode", "braille",
                                          "--overlay", "#00ff00", *extra])
        main()
        assert (0, 255, 0) in _colors(capsys.readouterr().out)


def test_image_command_overlay_html(png, tmp_path, monkeypatch):
    from asciimagic.image_to_ascii import main

    out = tmp_path / "i.html"
    monkeypatch.setattr(sys, "argv", ["image-to-ascii", str(png), "-c", "10", "--mode", "braille",
                                      "--color", "--overlay", "#00ff00", "-o", str(out)])
    main()
    assert "color:#00ff00" in out.read_text(encoding="utf-8")


def test_colorize_command_overlay(png, tmp_path, monkeypatch, capsys):
    from asciimagic import colorize_ascii

    art = tmp_path / "a.txt"
    art.write_text("####\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["colorize-ascii", str(png), str(art), "-", "--overlay", "#000000,#ffffff"])
    colorize_ascii.main()
    cols = _colors(capsys.readouterr().out)
    assert cols[0] == (0, 0, 0) and cols[-1] == (255, 255, 255)


def test_colorize_bad_overlay_is_a_clear_error(png, tmp_path, monkeypatch):
    from asciimagic import colorize_ascii

    art = tmp_path / "a.txt"
    art.write_text("#\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["colorize-ascii", str(png), str(art), "-", "--overlay", "puce"])
    with pytest.raises(SystemExit, match="Unknown overlay color"):
        colorize_ascii.main()


# ---- compose ----

def test_compose_layer_overlay_only_touches_that_layer():
    from asciimagic.compose import Layer, Scene, compose

    scene = Scene(layers=[
        Layer(type="text", text="A", style="box", color="white", at="top-left"),
        Layer(type="text", text="B", style="box", x=6, y=0, overlay="#ff0000"),
    ])
    comp = compose(scene)
    colors = {ch: c for row in comp.cells for ch, c in row if ch in "AB"}
    assert colors["A"] == (255, 255, 255) and colors["B"] == (255, 0, 0)


def test_compose_canvas_overlay_and_scene_round_trip(tmp_path):
    from asciimagic.compose import Canvas, Layer, Scene, compose

    scene = Scene(canvas=Canvas(overlay="#000000,#ffffff"),
                  layers=[Layer(type="text", text="ABCDEF", style="box")])
    cells = compose(scene).cells
    visible = [c for row in cells for ch, c in row if ch.strip()]
    assert visible[0] == (0, 0, 0)
    path = tmp_path / "s.json"
    scene.save(str(path))
    assert json.loads(path.read_text())["canvas"]["overlay"] == "#000000,#ffffff"
    assert compose(Scene.load(str(path))).to_ansi() == compose(scene).to_ansi()


def test_compose_bad_layer_overlay_rejected():
    from asciimagic.compose import Scene, compose

    with pytest.raises(ValueError, match="overlay"):
        compose(Scene.from_dict({"layers": [{"type": "text", "text": "x", "overlay": "puce"}]}))


def test_compose_cli_flags(capsys):
    from asciimagic.compose import main

    assert main(["--text", "HI", "--style", "box", "--layer-overlay", "#ff0000",
                 "--overlay", "#0000ff", "--overlay-strength", "0.5"]) == 0
    assert (128, 0, 128) in _colors(capsys.readouterr().out)


# ---- web ----

def test_web_render_overlay_and_errors():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from asciimagic.webapp import app

    c = TestClient(app)
    r = c.post("/api/render", data={"options": json.dumps(
        {"source": "text", "text": "HI", "text_style": "box", "overlay": "sunset"})})
    body = r.json()
    assert r.status_code == 200 and "\x1b[38;2;" in body["ansi"] and "color:#" in body["html"]
    assert body["warning"] is None
    r = c.post("/api/render", data={"options": json.dumps(
        {"source": "text", "text": "HI", "text_style": "box", "overlay": "puce"})})
    assert r.status_code == 400 and "overlay" in r.json()["detail"]
    r = c.post("/api/render", data={"options": json.dumps(
        {"source": "text", "text": "HI", "text_style": "box", "overlay": "sunset", "overlay_mode": "burn"})})
    assert r.status_code == 400
