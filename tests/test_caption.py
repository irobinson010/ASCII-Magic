import re

from PIL import Image

from asciimagic.colorize_ascii import CaptionOptions, Options, colorize_ascii_text
from asciimagic.text_to_ascii import caption_lines, compose_caption

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


def test_caption_color_image_samples_adjacent_strip():
    img = Image.new("RGB", (40, 40))
    px = img.load()
    for y in range(40):
        for x in range(40):
            px[x, y] = (255, 0, 0) if y < 20 else (0, 0, 255)  # red top, blue bottom

    opt = Options(out_format="ansi")
    opt.caption = CaptionOptions(text="Cat", style="box", color="image", position="bottom")
    out = colorize_ascii_text(img, ART, opt=opt)
    cap_part = "\n".join(out.splitlines()[-3:])
    codes = re.findall(r"\x1b\[38;2;(\d+);(\d+);(\d+)m", cap_part)
    assert codes
    assert all(int(b) > int(r) for r, _g, b in codes)  # bottom caption = blue strip

    opt.caption.position = "top"
    out2 = colorize_ascii_text(img, ART, opt=opt)
    cap_part2 = "\n".join(out2.splitlines()[:3])
    codes2 = re.findall(r"\x1b\[38;2;(\d+);(\d+);(\d+)m", cap_part2)
    assert codes2
    assert all(int(r) > int(b) for r, _g, b in codes2)  # top caption = red strip


def test_caption_figlet_style():
    lines = caption_lines("Hi", width=80, style="figlet")
    assert len(lines) >= 4  # multi-row letterforms
    assert all(len(ln) == 80 for ln in lines)
    joined = "\n".join(lines)
    # figlet outline glyphs, not the density-ramp blobs of the block style
    assert any(ch in joined for ch in "|_:")
    assert "@" not in joined


def _ink_width(lines):
    return max(len(ln.rstrip()) - (len(ln) - len(ln.lstrip())) for ln in lines if ln.strip())


def test_caption_figlet_scale_picks_bigger_fonts():
    small = caption_lines("Hi", width=200, style="figlet", scale=0.05)
    full = caption_lines("Hi", width=200, style="figlet", scale=1.0)
    assert _ink_width(full) > _ink_width(small) * 2
    assert len(full) > len(small)  # taller font, not stretched cells


def test_caption_figlet_full_scale_uses_largest_fitting_font():
    lines = caption_lines("Don't Hurry Be Happy", width=110, style="figlet", scale=1.0)
    assert 90 <= _ink_width(lines) <= 110  # the standard font, ~105 wide
    assert 5 <= len(lines) <= 8


def test_caption_figlet_downscales_to_fit():
    lines = caption_lines("Don't Hurry Be Happy", width=60, style="figlet", scale=0.6)
    assert all(len(ln) == 60 for ln in lines)
    assert _ink_width(lines) <= 60


def test_wrap_html_default_text_contrasts_background():
    from asciimagic.colorize_ascii import wrap_html

    doc = wrap_html(["hello"])
    assert "background: #000" in doc
    assert "color: #e0e0e0" in doc  # uncolored captions must be visible on black


def test_caption_color_image_full_spans_whole_picture():
    img = Image.new("RGB", (40, 40))
    px = img.load()
    for y in range(40):
        for x in range(40):
            px[x, y] = (255, 0, 0) if y < 20 else (0, 0, 255)  # red top, blue bottom

    opt = Options(out_format="ansi")
    opt.caption = CaptionOptions(text="Hello", style="figlet", color="image-full", position="top")
    out = colorize_ascii_text(img, ART, opt=opt)
    cap_part = out.split("#" * 5)[0]  # everything before the art
    codes = [tuple(map(int, c.split("m")[0].split(";")[:3]))
             for c in re.findall(r"\x1b\[38;2;([0-9;]+)m", cap_part)]
    assert any(r > b for r, _g, b in codes)  # top rows sample the red half
    assert any(b > r for r, _g, b in codes)  # lower rows sample the blue half


def test_caption_color_image_html():
    opt = Options(out_format="html")
    opt.caption = CaptionOptions(text="Cat", style="box", color="image")
    out = colorize_ascii_text(_img(), ART, opt=opt)
    assert "Cat" in out


def test_caption_width_matches_scaled_art():
    opt = Options(out_format="ansi")
    opt.size.cols = 24  # scale art narrower
    opt.caption = CaptionOptions(text="Hi", style="box")
    out = STRIP(colorize_ascii_text(_img(), ART, opt=opt))
    caption_rows = [ln for ln in out.splitlines() if "┌" in ln or "└" in ln or "│" in ln]
    assert caption_rows
    assert all(len(ln) == 24 for ln in caption_rows)


def test_figlet_never_chooses_a_wrapped_render():
    """Regression: at scale 1.0 the ladder used to pick a giant font whose
    pyfiglet-wrapped output smushed letters into each other ('I see you'
    lost its I and grew fragments of other letters)."""
    lines = caption_lines("I see you", width=110, style="figlet", scale=1.0)
    ink = max(len(ln.strip()) for ln in lines if ln.strip())
    assert ink <= 110
    # one unwrapped block, not stacked wrapped blocks (doh-wrapped was 46 rows)
    assert len(lines) <= 16


def test_figlet_narrow_width_wraps_cleanly_with_small_font():
    lines = caption_lines("I see you", width=24, style="figlet", scale=0.6)
    assert all(len(ln) == 24 for ln in lines)
    ink = max(len(ln.strip()) for ln in lines if ln.strip())
    assert ink <= 24
