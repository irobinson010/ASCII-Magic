import re

from PIL import Image

from ascii_magic.colorize_ascii import CaptionOptions, Options, colorize_ascii_text
from ascii_magic.text_to_ascii import caption_lines, compose_caption

ART = "\n".join("#" * 40 for _ in range(10))
STRIP = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)


def _img():
    return Image.new("RGB", (40, 20), (180, 40, 40))


def test_caption_lines_padded_to_width():
    lines = caption_lines("Hi", width=40, style="box")
    assert lines
    assert all(len(ln) == 40 for ln in lines)
    assert "Hi" in "\n".join(lines)


def test_caption_lines_alignment():
    left = caption_lines("Hi", width=40, style="box", align="left")
    right = caption_lines("Hi", width=40, style="box", align="right")
    assert left[0].startswith("┌")
    assert right[0].endswith("┐")
    assert left != right


def test_caption_lines_rendered_style_scales():
    narrow = caption_lines("A", width=60, style="block", scale=0.3)
    wide = caption_lines("A", width=60, style="block", scale=0.9)
    ink = lambda lines: max(len(ln.rstrip()) - (len(ln) - len(ln.lstrip())) for ln in lines)
    assert all(len(ln) == 60 for ln in narrow + wide)
    assert ink(wide) > ink(narrow)


def test_compose_caption_bottom_and_top():
    bottom = compose_caption(ART, "Cat", position="bottom", style="box", gap=2)
    lines = bottom.splitlines()
    assert lines[0] == "#" * 40
    assert lines[10] == "" and lines[11] == ""
    assert "Cat" in "\n".join(lines[12:])

    top = compose_caption(ART, "Cat", position="top", style="box", gap=1)
    tlines = top.splitlines()
    assert "Cat" in "\n".join(tlines[:4])
    assert tlines[-1] == "#" * 40


def test_colorized_ansi_caption_is_not_image_colored():
    opt = Options(out_format="ansi")
    opt.caption = CaptionOptions(text="Cat", style="box")
    out = colorize_ascii_text(_img(), ART, opt=opt)
    plain = STRIP(out)
    assert "Cat" in plain
    # caption block (after the art) carries no color escapes by default
    caption_part = out.split("\n")[-3:]
    assert all("\x1b[38;2;" not in ln for ln in caption_part)
    # while the art itself is colorized
    assert "\x1b[38;2;" in out


def test_colorized_ansi_caption_with_color():
    opt = Options(out_format="ansi")
    opt.caption = CaptionOptions(text="Cat", style="box", color="amber")
    out = colorize_ascii_text(_img(), ART, opt=opt)
    assert "\x1b[38;2;255;176;0m" in out


def test_colorized_html_caption_escaped_and_positioned():
    opt = Options(out_format="html")
    opt.caption = CaptionOptions(text="R&D <cat>", style="box", position="top")
    out = colorize_ascii_text(_img(), ART, opt=opt)
    assert "R&amp;D &lt;cat&gt;" in out


def test_caption_width_matches_scaled_art():
    opt = Options(out_format="ansi")
    opt.size.cols = 24  # scale art narrower
    opt.caption = CaptionOptions(text="Hi", style="box")
    out = STRIP(colorize_ascii_text(_img(), ART, opt=opt))
    caption_rows = [ln for ln in out.splitlines() if "┌" in ln or "└" in ln or "│" in ln]
    assert caption_rows
    assert all(len(ln) == 24 for ln in caption_rows)
