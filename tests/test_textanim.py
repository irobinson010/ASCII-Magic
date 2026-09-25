"""Animated text effects (issue #41)."""

import io
import re
import sys

import numpy as np
import pytest
from PIL import Image

from asciimagic import textanim as ta
from asciimagic.textanim import EFFECTS, TextAnimation, TextAnimOptions, parse_effects


def anim(text="HELLO", effects=("wave",), **kw):
    kw.setdefault("cols", 40)
    kw.setdefault("frames", 12)
    return TextAnimation(text, TextAnimOptions(effects=list(effects), **kw))


def grid(frame_lines):
    return np.array([[c != " " for c in ln] for ln in frame_lines])


@pytest.mark.parametrize("name", list(EFFECTS))
def test_every_effect_renders_on_canvas(name):
    a = anim(effects=[name])
    cols, rows = a.size
    assert cols == 40 and rows >= 1
    assert len(a.lines) == 12
    assert all(len(ln) == 40 for fr in a.lines for ln in fr)
    assert all(len(fr) == rows for fr in a.lines)
    assert any(grid(fr).any() for fr in a.lines)  # something is drawn
    assert not a.clipped  # the motion stays on the canvas vertically
    if name != "scroll":  # scroll wraps around by design
        assert not any(ln[0] != " " or ln[-1] != " " for fr in a.lines for ln in fr)


@pytest.mark.parametrize("text,name,amount", [
    ("I", "rotate", 1), ("Hi", "bounce", 3), ("Two\nlines", "swing", 3),
    ("HELLO WORLD", "flip-diagonal", 1), ("Hi", "squash", 3), ("HELLO", "stretch", 3),
])
def test_hard_cases_stay_on_canvas(text, name, amount):
    a = anim(text, [name], amount=amount, frames=16)
    assert not a.clipped
    assert not any(ln[0] != " " or ln[-1] != " " for fr in a.lines for ln in fr)
    cols, rows = a.size
    assert rows <= cols  # tall text is scaled down rather than made enormous


def test_loops_seamlessly():
    # Frame t=1 would equal frame 0: the motion is periodic in the frame count.
    a = anim(effects=["spin"], frames=8)
    b = anim(effects=["spin"], frames=16)
    assert a.lines[0] == b.lines[0]
    assert a.lines[4] == b.lines[8]  # same phase, same picture


def test_spin_goes_edge_on_and_back():
    a = anim(effects=["spin"], frames=8)
    ink = [grid(fr).sum() for fr in a.lines]
    assert ink[2] < ink[0] * 0.2  # 90 degrees: seen edge-on
    assert ink[4] > ink[0] * 0.5  # 180 degrees: the back, readable


def test_mirror_back_mirrors():
    two_sided = anim(effects=["spin"], frames=8).lines[4]
    mirrored = anim(effects=["spin"], frames=8, mirror_back=True).lines[4]
    assert two_sided != mirrored


@pytest.mark.parametrize("name,near", [("flip-left", "right"), ("flip-right", "left")])
def test_flip_direction(name, near):
    a = anim("HHHHH", [name], cols=80, frames=40)
    g = grid(a.lines[20])  # early in the turn
    cols = np.nonzero(g.any(0))[0]
    quarter = max(1, len(cols) // 4)

    def span(xs):
        rows = np.nonzero(g[:, xs].any(1))[0]
        return rows[-1] - rows[0] + 1

    left, right = span(cols[:quarter]), span(cols[-quarter:])
    # The edge swinging toward the viewer looks taller.
    assert (right > left) if near == "right" else (left > right)


@pytest.mark.parametrize("name,near", [("flip-up", "bottom"), ("flip-down", "top")])
def test_flip_vertical_direction(name, near):
    a = anim("HHHHH", [name], cols=80, frames=40)
    g = grid(a.lines[20])
    rows = np.nonzero(g.any(1))[0]
    quarter = max(1, len(rows) // 4)

    def span(ys):
        cols = np.nonzero(g[ys].any(0))[0]
        return cols[-1] - cols[0] + 1

    top, bottom = span(rows[:quarter]), span(rows[-quarter:])
    assert (bottom > top) if near == "bottom" else (top > bottom)


def test_typewriter_reveals_left_to_right():
    a = anim(effects=["typewriter"], frames=20)
    ink = [grid(fr).sum() for fr in a.lines]
    assert ink[0] == 0
    assert ink[5] < ink[10] <= ink[19]
    assert ink[14] == ink[19]  # holds once fully typed


def test_chained_effects_and_parse():
    assert parse_effects(" Wave , spin ") == ["wave", "spin"]
    with pytest.raises(ValueError, match="unknown animation"):
        parse_effects("wave,moonwalk")
    with pytest.raises(ValueError):
        parse_effects("")
    a = anim(effects=["wave", "spin"])
    assert len(a.lines) == 12


def test_styles_and_bad_style():
    for style, chars in (("solid", "░▒▓█"), ("small", ".-*#@")):
        a = anim(style=style)
        used = set("".join("".join(fr) for fr in a.lines)) - {" "}
        assert used <= set(chars)
    with pytest.raises(ValueError, match="style"):
        anim(style="box")


def test_rainbow_and_overlay_colors():
    a = anim(effects=["wave", "rainbow"])
    assert a.colors is not None
    assert not np.array_equal(a.colors[0], a.colors[3])  # colors cycle
    ansi = a.frames_ansi()[0]
    assert "\x1b[38;2;" in ansi and ansi.endswith("\x1b[0m")
    b = anim(color="sunset")
    assert np.array_equal(b.colors[0], b.colors[5])  # a static gradient
    assert anim().colors is None and "\x1b[" not in anim().frames_ansi()[0]


def test_pixel_budget():
    with pytest.raises(ta.TooLarge, match="too large"):
        anim(cols=200, frames=60, max_pixels=1_000_000)


def test_japanese_text_animates():
    a = anim("こんにちは", ["bounce"])
    assert any(grid(fr).any() for fr in a.lines)


def test_gif_html_frames(tmp_path):
    a = anim(effects=["zoom", "rainbow"])
    gif = Image.open(io.BytesIO(a.to_gif_bytes()))
    assert gif.format == "GIF" and gif.n_frames >= 2
    html = a.to_html(title="<x>")
    assert "&lt;x&gt;" in html and "const FRAMES" in html and "<x>" not in html
    from asciimagic.greet import read_frames_file

    a.write_frames(tmp_path / "a.frames", loops=2)
    frames, fps, loops = read_frames_file(tmp_path / "a.frames")
    assert len(frames) == 12 and fps == 15 and loops == 2


def test_play_writes_frames():
    out = io.StringIO()
    anim(frames=3).play(loops=1, out=out)
    assert out.getvalue().count("\x1b[H") == 3


# ---- CLI ----

def run_text_cli(monkeypatch, *argv):
    from asciimagic.text_to_ascii import main

    monkeypatch.setattr(sys, "argv", ["text-to-ascii", *argv])
    with pytest.raises(SystemExit) as e:
        main()
    return e.value.code


@pytest.mark.parametrize("ext", ["gif", "html", "frames"])
def test_cli_writes_outputs(monkeypatch, tmp_path, capsys, ext):
    out = tmp_path / f"a.{ext}"
    code = run_text_cli(monkeypatch, "HI", "--animate", "spin,rainbow", "-w", "30", "--frames", "6", "-o", str(out))
    assert code == 0 and out.stat().st_size > 0
    assert re.search(r"6 frames, 30x\d+", capsys.readouterr().err)


def test_cli_errors(monkeypatch, tmp_path, capsys):
    assert "unknown animation" in str(run_text_cli(monkeypatch, "HI", "--animate", "nope"))
    assert "styles" in str(run_text_cli(monkeypatch, "HI", "--animate", "wave", "-s", "box"))
    assert ".gif" in str(run_text_cli(monkeypatch, "HI", "--animate", "wave", "-o", str(tmp_path / "a.png")))
    assert "--frames" in str(run_text_cli(monkeypatch, "HI", "--animate", "wave", "--frames", "1"))


def test_static_solid_style():
    from asciimagic.text_to_ascii import text_to_ascii_art

    art = text_to_ascii_art("HI", style="solid", width=30)
    assert set(art) - set("\n") <= set("█▓▒░ ") and "█" in art
