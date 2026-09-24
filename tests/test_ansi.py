import re
import sys

import pytest
from PIL import Image

from asciimagic.ansi import detect_depth, downsample, resolve_depth, rgb_to_16, rgb_to_256

TRUE = re.compile(r"\x1b\[(?:38|48);2;")


@pytest.mark.parametrize("rgb,idx", [
    ((0, 0, 0), 16), ((255, 255, 255), 231), ((255, 0, 0), 196), ((0, 255, 0), 46),
    ((0, 0, 255), 21), ((128, 128, 128), 244), ((95, 135, 175), 67),
])
def test_rgb_to_256(rgb, idx):
    assert rgb_to_256(rgb) == idx


@pytest.mark.parametrize("rgb,idx", [
    ((0, 0, 0), 0), ((255, 0, 0), 9), ((200, 0, 0), 1), ((0, 255, 0), 10),
    ((255, 255, 255), 15), ((120, 120, 120), 8), ((230, 230, 230), 7),
])
def test_rgb_to_16(rgb, idx):
    assert rgb_to_16(rgb) == idx


def test_downsample_rewrites_fg_and_bg_and_keeps_text():
    s = "\x1b[38;2;255;0;0mA\x1b[48;2;0;0;255mB\x1b[0m\x1b[39mC"
    assert downsample(s, "256") == "\x1b[38;5;196mA\x1b[48;5;21mB\x1b[0m\x1b[39mC"
    assert downsample(s, "16") == "\x1b[91mA\x1b[44mB\x1b[0m\x1b[39mC"
    assert downsample(s, "truecolor") == s


def test_downsample_rejects_unknown_depth():
    with pytest.raises(ValueError):
        downsample("x", "8")


@pytest.mark.parametrize("env,expected", [
    ({"COLORTERM": "truecolor", "TERM": "xterm-256color"}, "truecolor"),
    ({"COLORTERM": "24bit"}, "truecolor"),
    ({"WT_SESSION": "abc", "TERM": "xterm"}, "truecolor"),
    ({"TERM_PROGRAM": "iTerm.app"}, "truecolor"),
    ({"TERM": "xterm-direct"}, "truecolor"),
    ({"TERM": "xterm-256color"}, "256"),     # typical SSH session: COLORTERM not forwarded
    ({"TERM": "screen-256color"}, "256"),
    ({"TERM": "linux"}, "16"),
    ({}, "16"),
    ({"ASCII_MAGIC_COLOR_DEPTH": "256", "COLORTERM": "truecolor"}, "256"),
    ({"ASCII_MAGIC_COLOR_DEPTH": "bogus", "TERM": "xterm-256color"}, "256"),
])
def test_detect_depth(env, expected):
    assert detect_depth(env) == expected


def test_resolve_depth_auto_keeps_files_truecolor(monkeypatch):
    monkeypatch.setenv("ASCII_MAGIC_COLOR_DEPTH", "16")
    assert resolve_depth("auto", to_terminal=True) == "16"
    assert resolve_depth("auto", to_terminal=False) == "truecolor"
    assert resolve_depth("256", to_terminal=False) == "256"


# ---- CLI wiring ----

@pytest.fixture
def png(tmp_path):
    p = tmp_path / "t.png"
    Image.new("RGB", (32, 24), (200, 60, 30)).save(p)
    return p


def _no_truecolor(text):
    assert "\x1b[" in text and not TRUE.search(text)


def test_image_color_depth(png, tmp_path):
    from asciimagic.unified_cli import main

    out = tmp_path / "a.ans"
    assert main(["image", str(png), "--mode", "braille", "--threshold", "0.1", "-c", "10",
                 "--color", "--color-depth", "256", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    _no_truecolor(text)
    assert "\x1b[38;5;" in text


def test_colorize_color_depth(png, tmp_path, monkeypatch, capsys):
    from asciimagic import colorize_ascii

    art = tmp_path / "art.txt"
    art.write_text("##\n##\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["colorize-ascii", str(png), str(art), "-", "--color-depth", "16"])
    colorize_ascii.main()
    _no_truecolor(capsys.readouterr().out)


def test_colorize_animation_frames_depth(png, tmp_path, monkeypatch):
    from asciimagic import colorize_ascii
    from asciimagic.greet import read_frames_file

    art = tmp_path / "art.txt"
    art.write_text("####\n####\n", encoding="utf-8")
    out = tmp_path / "a.frames"
    monkeypatch.setattr(sys, "argv", ["colorize-ascii", str(png), str(art), str(out), "--animate",
                                      "--frames", "2", "--color-depth", "256"])
    colorize_ascii.main()
    frames, _, _ = read_frames_file(out)
    for f in frames:
        assert not TRUE.search(f)


def test_compose_color_depth(capsys):
    from asciimagic.compose import main

    assert main(["--text", "Hi", "--style", "box", "--color", "amber", "--color-depth", "256"]) == 0
    out = capsys.readouterr().out
    _no_truecolor(out)
    assert "\x1b[38;5;214m" in out


def test_greet_show_auto_detects_terminal(tmp_path, monkeypatch, capsys):
    from asciimagic import greet

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(greet, "installed_greeting", lambda: art)
    art = tmp_path / "g.ans"
    art.write_text("\x1b[38;2;255;0;0mHELLO\x1b[0m\n", encoding="utf-8")

    for k in ("COLORTERM", "WT_SESSION", "TERM_PROGRAM", "ASCII_MAGIC_COLOR_DEPTH"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")  # an SSH login
    assert greet.main(["show"]) == 0
    assert "\x1b[38;5;196mHELLO" in capsys.readouterr().out

    monkeypatch.setenv("COLORTERM", "truecolor")
    assert greet.main(["show"]) == 0
    assert "\x1b[38;2;255;0;0mHELLO" in capsys.readouterr().out

    assert greet.main(["show", "--color-depth", "16"]) == 0
    assert "\x1b[91mHELLO" in capsys.readouterr().out
