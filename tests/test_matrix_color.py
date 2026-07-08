import pytest
from PIL import Image

from asciimagic.animate import AnimationOptions, generate
from asciimagic.colorize_ascii import (
    MatrixOptions,
    Options,
    colorize_ascii_text,
    parse_matrix_color,
    tint_rgb,
)

ASCII = "\n".join("#" * 20 for _ in range(8))


def _img():
    return Image.new("RGB", (40, 16), (250, 250, 250))


def test_parse_theme_names():
    assert parse_matrix_color("green") == (0, 255, 0)
    assert parse_matrix_color("AMBER") == (255, 176, 0)
    assert parse_matrix_color("white") == (255, 255, 255)


def test_parse_hex_and_tuple():
    assert parse_matrix_color("#ff8800") == (255, 136, 0)
    assert parse_matrix_color((10, 20, 30)) == (10, 20, 30)
    assert parse_matrix_color([300, -5, 128]) == (255, 0, 128)


def test_parse_invalid_raises():
    for bad in ("chartreuse", "#12345", "#zzzzzz", 42):
        with pytest.raises(ValueError):
            parse_matrix_color(bad)


def test_tint_rgb_scales():
    assert tint_rgb(255, (0, 255, 0)) == (0, 255, 0)
    assert tint_rgb(128, (255, 176, 0)) == (128, 88, 0)
    assert tint_rgb(0, (255, 255, 255)) == (0, 0, 0)


def _static_ansi(tint):
    opt = Options(out_format="ansi")
    opt.matrix = MatrixOptions(enabled=True, seed=1, tint=tint)
    return colorize_ascii_text(_img(), ASCII, opt=opt)


def test_static_default_green_uses_green_channel_only():
    out = _static_ansi((0, 255, 0))
    assert "\x1b[38;2;0;" in out
    # every fg code is (0, X, 0)
    for chunk in out.split("\x1b[38;2;")[1:]:
        r, g, b = chunk.split("m")[0].split(";")[:3]
        assert r == "0" and b == "0"


def test_static_amber_tint_has_red_channel():
    out = _static_ansi(parse_matrix_color("amber"))
    reds = [int(c.split(";")[0]) for c in out.split("\x1b[38;2;")[1:]]
    assert any(r > 0 for r in reds)
    # blue channel stays 0 for amber
    blues = [int(c.split("m")[0].split(";")[2]) for c in out.split("\x1b[38;2;")[1:]]
    assert all(b == 0 for b in blues)


def test_same_seed_same_output_across_tints_structure():
    """Tint changes colors only — glyph placement is seed-driven."""
    import re

    strip = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)
    a = _static_ansi((0, 255, 0))
    b = _static_ansi(parse_matrix_color("cyan"))
    assert strip(a) == strip(b)
    assert a != b


def test_animation_respects_tint():
    m = MatrixOptions(enabled=True, seed=5, tint=parse_matrix_color("violet"))
    anim = generate(ASCII, _img(), m=m, a=AnimationOptions(frames=3))
    ansi = "".join(anim.frames_ansi())
    reds = [int(c.split(";")[0]) for c in ansi.split("\x1b[38;2;")[1:]]
    assert any(r > 0 for r in reds)

    html = anim.to_html()
    assert "rgb(186" in html or "rgb(185" in html  # violet-tinted css classes

    gif = anim.to_gif_bytes()
    assert gif[:4] == b"GIF8"


def test_animation_default_tint_is_green():
    m = MatrixOptions(enabled=True, seed=5)
    anim = generate(ASCII, _img(), m=m, a=AnimationOptions(frames=2))
    html = anim.to_html()
    # Head classes brightness-track the drop; full-intensity green head is
    # (0,255,0) blended 80% toward its own intensity -> (204,255,204).
    assert ".h15 { color: rgb(204,255,204); }" in html
    assert ".h0 { color: rgb(0,0,0); }" in html
