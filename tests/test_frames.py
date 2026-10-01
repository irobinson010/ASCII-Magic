"""Frames around any text style, so boxes and banners scale."""

import sys

import pytest

from asciimagic.text_to_ascii import (
    FRAMES, frame_art, render_text, text_to_banner, text_to_box, trim_art,
)
from asciimagic.textwidth import str_width


def widths(art):
    return {str_width(ln) for ln in art.split("\n")}


@pytest.mark.parametrize("frame", [f for f in FRAMES if f != "banner"])
def test_frame_shapes(frame):
    tl, h, tr, v, bl, br = FRAMES[frame]
    out = frame_art("Hi", frame).split("\n")
    assert out == [tl + h * 4 + tr, f"{v} Hi {v}", bl + h * 4 + br]


def test_banner_frame_uses_char():
    assert frame_art("Hi", "banner", char="*") == "******\n* Hi *\n******"


def test_multiline_art_gets_vertical_padding_and_even_rows():
    out = frame_art("ab\nabcd", "box")
    rows = out.split("\n")
    assert len(rows) == 6 and rows[1] == "│      │"  # blank row above
    assert len(widths(out)) == 1


def test_cjk_rows_line_up():
    out = frame_art("水と火\nabc", "double")
    assert len(widths(out)) == 1  # double-width characters counted as two columns


def test_trim_art_hugs_the_ink():
    assert trim_art("\n\n   ab  \n    c\n\n") == ["ab", " c"]


def test_pad_zero_and_unknown_frame():
    assert frame_art("x", "box", pad=0) == "┌─┐\n│x│\n└─┘"
    with pytest.raises(ValueError):
        frame_art("x", "zigzag")


@pytest.mark.parametrize("style,text,grow", [
    ("block", "Hi", 1.5), ("solid", "Hi", 1.5),
    ("figlet", "Hello", 1.3),  # figlet steps through real fonts, so it grows in steps
])
def test_framed_letters_scale_with_width(style, text, grow):
    small = render_text(text, style=style, width=40, frame="box")
    big = render_text(text, style=style, width=100, frame="box")
    assert max(widths(small)) <= 40 and max(widths(big)) <= 100
    assert max(widths(big)) > max(widths(small)) * grow  # letters grew, not just the frame
    assert len(widths(big)) == 1 and big.startswith("┌") and big.endswith("┘")


def test_framed_japanese_scales():
    art = render_text("水", style="solid", width=40, frame="banner", char="#")
    assert art.startswith("#") and len(widths(art)) == 1 and len(art.split("\n")) > 5


def test_plain_text_frame_and_truncation():
    assert render_text("Hello there", style="plain", frame="heavy") == "┏━━━━━━━━━━━━━┓\n┃ Hello there ┃\n┗━━━━━━━━━━━━━┛"
    long = render_text("x" * 50, style="plain", width=20, frame="box")
    assert max(widths(long)) <= 20


def test_legacy_box_and_banner_unchanged():
    assert render_text("Hi there", style="box", width=80) == text_to_box("Hi there", width=80)
    assert render_text("Hi", style="banner", char="*") == text_to_banner("Hi", char="*")


# ---- CLI ----

def run(monkeypatch, capsys, *argv):
    from asciimagic.text_to_ascii import main

    monkeypatch.setattr(sys, "argv", ["text-to-ascii", *argv])
    main()
    return capsys.readouterr().out.rstrip("\n")


def test_cli_frame(monkeypatch, capsys):
    out = run(monkeypatch, capsys, "Hi", "-s", "block", "--frame", "rounded", "-w", "40")
    assert out.startswith("╭") and out.split("\n")[-1].startswith("╰")


def test_cli_plain_and_banner_char(monkeypatch, capsys):
    assert run(monkeypatch, capsys, "Hi", "-s", "plain", "--frame", "banner", "-c", "=") == "======\n= Hi =\n======"


def test_cli_bad_frame_pad(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, capsys, "Hi", "--frame", "box", "--frame-pad", "99")
