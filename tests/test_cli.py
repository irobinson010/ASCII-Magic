import pytest

from ascii_magic import cli
from ascii_magic.colorize_ascii import parse_args


def test_no_args_prints_command_list(capsys):
    assert cli.main(["ascii-magic"]) == 0
    out = capsys.readouterr().out
    for cmd in ("convert", "text", "colorize", "greet", "web"):
        assert cmd in out


def test_unknown_command_exits_2(capsys):
    assert cli.main(["ascii-magic", "frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_version(capsys):
    from ascii_magic import __version__

    assert cli.main(["ascii-magic", "--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_dispatch_text_banner(capsys):
    assert cli.main(["ascii-magic", "text", "Hi", "-s", "banner", "-c", "*"]) == 0
    out = capsys.readouterr().out
    assert "* Hi *" in out


def test_dispatch_greet_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert cli.main(["ascii-magic", "greet", "status"]) == 0
    assert "Greeting file" in capsys.readouterr().out


# ---- colorize argparse migration ----

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
