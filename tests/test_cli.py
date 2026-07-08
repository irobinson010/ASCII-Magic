"""End-to-end unified CLI dispatch plus colorize argparse coverage.

Mock-based unit tests for unified_cli live in test_unified_cli.py; these
exercise the real modules through the dispatcher.
"""

import pytest
from PIL import Image

from asciimagic.unified_cli import COMMANDS, main as cli_main
from asciimagic.colorize_ascii import parse_args


def test_new_subcommands_registered():
    for cmd in ("greet", "web", "image", "text", "colorize"):
        assert cmd in COMMANDS


def test_dispatch_text_banner(capsys):
    assert cli_main(["text", "Hi", "-s", "banner", "-c", "*"]) == 0
    assert "* Hi *" in capsys.readouterr().out


def test_dispatch_greet_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert cli_main(["greet", "status"]) == 0
    assert "Greeting file" in capsys.readouterr().out


def test_unknown_command_exits_2(capsys):
    assert cli_main(["frobnicate"]) == 2
    assert "Unknown command" in capsys.readouterr().err


def _png(tmp_path, color=(200, 60, 30)):
    p = tmp_path / "t.png"
    Image.new("RGB", (24, 24), color).save(p)
    return p


def test_image_color_one_step_ansi(tmp_path):
    out = tmp_path / "art.ans"
    rc = cli_main(
        ["image", str(_png(tmp_path)), "--mode", "braille", "--threshold", "0.1",
         "-c", "10", "--color", "-o", str(out)]
    )
    assert rc == 0
    content = out.read_text(encoding="utf-8")
    assert "\x1b[38;2;" in content


def test_image_color_one_step_html(tmp_path):
    out = tmp_path / "art.html"
    rc = cli_main(
        ["image", str(_png(tmp_path)), "--mode", "braille", "--threshold", "0.1",
         "-c", "10", "--color", "-o", str(out)]
    )
    assert rc == 0
    assert "<!doctype html>" in out.read_text(encoding="utf-8").lower()


def test_image_rotate_flag(tmp_path):
    p = tmp_path / "wide.png"
    Image.new("RGB", (60, 20), (10, 200, 40)).save(p)
    out = tmp_path / "art.txt"

    cli_main(["image", str(p), "--mode", "braille", "-c", "10", "-o", str(out)])
    plain = out.read_text(encoding="utf-8").splitlines()
    assert max(len(ln) for ln in plain) > len(plain)  # landscape

    cli_main(["image", str(p), "--mode", "braille", "-c", "10", "--rotate", "90", "-o", str(out)])
    rotated = out.read_text(encoding="utf-8").splitlines()
    assert len(rotated) > max(len(ln) for ln in rotated)  # portrait


def test_image_without_color_stays_plain(tmp_path):
    out = tmp_path / "art.txt"
    rc = cli_main(
        ["image", str(_png(tmp_path)), "--mode", "braille", "-c", "10", "-o", str(out)]
    )
    assert rc == 0
    assert "\x1b[" not in out.read_text(encoding="utf-8")


# ---- colorize argparse ----

def _parse(*args):
    return parse_args(["colorize-ascii", *args])


def test_parse_minimal():
    img, ascii_path, out, opt = _parse("img.png", "art.txt")
    assert (img, ascii_path, out) == ("img.png", "art.txt", None)
    assert opt.out_format == "ansi"
    assert not opt.matrix.enabled


def test_parse_out_extension_infers_html():
    _, _, out, opt = _parse("img.png", "art.txt", "page.html")
    assert out == "page.html"
    assert opt.out_format == "html"


def test_parse_explicit_format_wins():
    _, _, _, opt = _parse("img.png", "art.txt", "--format", "html")
    assert opt.out_format == "html"


def test_parse_rejects_unknown_out_extension():
    with pytest.raises(SystemExit):
        _parse("img.png", "art.txt", "out.txt")


def test_parse_matrix_and_size_flags():
    _, _, _, opt = _parse(
        "img.png", "art.txt", "out.ans",
        "--matrix", "--matrix-seed", "42", "--matrix-gamma", "1.5",
        "--matrix-fg-min", "10", "--matrix-chars", "AB",
        "--matrix-mask", "--matrix-mask-boost", "0.5",
        "--max-rows", "30", "--cols", "80", "--keep-top", "2", "--color-top",
        "--html-font-size", "16", "--html-fill-spaces",
    )
    m = opt.matrix
    assert m.enabled and m.seed == 42 and m.gamma == 1.5
    assert m.fg_min == 10 and m.chars == "AB"
    assert m.use_mask and m.mask_boost == 0.5
    assert opt.size.max_rows == 30 and opt.size.cols == 80
    assert opt.keep_top == 2 and opt.color_top
    assert opt.html.font_size_px == 16 and opt.html.fill_spaces


def test_parse_animation_flags():
    _, _, out, opt = _parse(
        "img.png", "art.txt", "rain.gif",
        "--animate", "--frames", "24", "--fps", "10", "--tail", "4", "--loops", "0",
    )
    assert out == "rain.gif"
    assert opt.animate
    assert opt.anim_frames == 24
    assert opt.anim_fps == 10.0
    assert opt.anim_tail == 4.0
    assert opt.anim_loops == 0


def test_parse_missing_positionals_exits():
    with pytest.raises(SystemExit):
        _parse("img.png")


def test_parse_stdout_dash():
    _, _, out, opt = _parse("img.png", "art.txt", "-")
    assert out == "-"
    assert opt.out_format == "ansi"
