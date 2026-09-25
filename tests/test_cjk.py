"""Japanese (and other double-width) text: widths, fonts, figlet, compose."""

import pytest

from asciimagic.compose import Canvas, Layer, Scene, compose
from asciimagic.text_to_ascii import (
    caption_lines,
    find_fallback_font,
    load_font,
    missing_glyphs,
    text_to_ascii_art,
    text_to_banner,
    text_to_box,
    text_to_figlet,
)
from asciimagic.textwidth import CONT, fit, from_cells, scale_lines, str_width, to_cells, truncate


# ---- display width ----

@pytest.mark.parametrize("s,w", [
    ("abc", 3), ("日本語", 6), ("ｱｲｳ", 3), ("é", 1), ("é", 1), ("a​b", 2),
    ("🙂", 2), ("┌─┐", 3), ("", 0),
])
def test_str_width(s, w):
    assert str_width(s) == w


def test_cells_round_trip_and_repair():
    assert to_cells("日a") == ["日", CONT, "a"]
    assert from_cells(to_cells("日本a")) == "日本a"
    assert from_cells(["日", "a"]) == " a"          # lead lost its second half
    assert from_cells([CONT, "a"]) == " a"          # second half lost its lead


def test_truncate_and_fit_never_split_wide_chars():
    assert truncate("日本語", 5) == "日本"
    assert fit("日本語", 5) == "日本 "
    assert str_width(fit("日本語", 5)) == 5


def test_scale_lines_keeps_column_counts():
    out = scale_lines(["日本語", "abcdef"], 2, 9)
    assert [str_width(ln) for ln in out] == [9, 9]


# ---- box / banner / figlet ----

def test_box_fits_japanese():
    lines = text_to_box("日本語").splitlines()
    assert lines == ["┌────────┐", "│ 日本語 │", "└────────┘"]
    assert len({str_width(ln) for ln in lines}) == 1


def test_box_truncates_by_columns():
    lines = text_to_box("日本語テキスト", width=10).splitlines()
    assert len({str_width(ln) for ln in lines}) == 1
    assert str_width(lines[0]) == 10


def test_banner_fits_japanese():
    lines = text_to_banner("日本 ok").splitlines()
    assert len({str_width(ln) for ln in lines}) == 1


def test_figlet_refuses_what_it_cannot_draw():
    with pytest.raises(ValueError, match="block"):
        text_to_figlet("日本")
    assert text_to_figlet("Hi").strip()


def test_figlet_caption_falls_back_to_block():
    lines = caption_lines("日本", 40, style="figlet")
    assert any(ln.strip() for ln in lines)  # used to be empty
    assert all(str_width(ln) == 40 for ln in lines)


def test_box_caption_is_padded_by_columns():
    lines = caption_lines("日本語", 30, style="box", align="center")
    assert lines and all(str_width(ln) == 30 for ln in lines)


# ---- fonts ----

def test_default_font_reports_missing_japanese():
    font = load_font(None, 24)
    assert missing_glyphs(font, "Hello") == ""
    assert "日" in missing_glyphs(font, "日本")


def test_fallback_font_draws_japanese():
    font = find_fallback_font("日本語", 24)
    if font is None:
        pytest.skip("no Japanese-capable font installed")
    assert missing_glyphs(font, "日本語") == ""
    # Rendered through the fallback, distinct characters give distinct art
    # (with the missing-glyph box they were all the same rectangle).
    assert text_to_ascii_art("日", width=24) != text_to_ascii_art("本", width=24)


def test_fallback_font_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ASCII_MAGIC_FALLBACK_FONT", str(tmp_path / "nope.ttf"))
    find_fallback_font("日", 20)  # a missing override path is skipped, not an error


# ---- compose ----

def test_compose_places_wide_text_by_columns():
    comp = compose(Scene(layers=[Layer(type="text", text="日本語", style="box")]))
    assert comp.placed[0].w == 10
    assert comp.to_text().splitlines()[1] == "│ 日本語 │"


def test_compose_overlap_repairs_cut_wide_chars():
    scene = Scene(canvas=Canvas(cols=30, rows=5), layers=[
        Layer(type="text", text="日本語テキスト", style="box", at="center"),
        Layer(type="text", text="ABC", style="box", x=12, y=1, outline=0),
    ])
    rows = compose(scene).to_text().splitlines()
    widths = {str_width(ln.ljust(1)) for ln in rows if ln.strip()}
    assert len(widths) == 1  # every visible row still lines up


def test_compose_wide_char_at_right_edge_becomes_space():
    # "│ ab日 │" shifted so 日 starts in the canvas's last column: no room
    # for its second half.
    comp = compose(Scene(canvas=Canvas(cols=5, rows=1),
                         layers=[Layer(type="text", text="ab日", style="box", x=0, y=-1)]))
    row = comp.cells[0]
    assert [ch for ch, _ in row] == ["│", " ", "a", "b", " "]


def test_web_render_japanese_box():
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    import json

    from fastapi.testclient import TestClient

    from asciimagic.webapp import app

    r = TestClient(app).post("/api/render", data={"options": json.dumps(
        {"source": "text", "text": "日本語", "text_style": "box"})})
    assert r.status_code == 200
    assert "│ 日本語 │" in r.json()["ascii"]
