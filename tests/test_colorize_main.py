"""End-to-end runs of colorize-ascii's main() (argparse is covered in test_cli.py)."""

import sys

import pytest
from PIL import Image

from asciimagic import colorize_ascii
from asciimagic.colorize_ascii import MatrixOptions, Options, SizeOptions, colorize_ascii_text


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["colorize-ascii", *map(str, args)])
    colorize_ascii.main()


@pytest.fixture
def inputs(tmp_path):
    img = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (200, 60, 30)).save(img)
    art = tmp_path / "art.txt"
    art.write_text("ab\ncd\n", encoding="utf-8")
    return tmp_path, img, art


def test_dash_output_writes_stdout(inputs, monkeypatch, capsys):
    tmp_path, img, art = inputs
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, img, art, "-")
    assert "\x1b[38;2;" in capsys.readouterr().out
    assert not (tmp_path / "-").exists()


def test_dash_output_html_to_stdout(inputs, monkeypatch, capsys):
    tmp_path, img, art = inputs
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, img, art, "-", "--format", "html")
    assert "<pre" in capsys.readouterr().out
    assert not (tmp_path / "-").exists()


def test_file_output(inputs, monkeypatch):
    tmp_path, img, art = inputs
    out = tmp_path / "out.ans"
    _run(monkeypatch, img, art, out)
    assert "\x1b[38;2;" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "flags",
    [[], ["--max-cols", "10"], ["--rows", "3"], ["--cols", "5"],
     ["--format", "html"], ["--matrix"], ["--matrix", "--format", "html"]],
)
def test_blank_lines_only_art_does_not_crash(inputs, monkeypatch, capsys, flags):
    tmp_path, img, art = inputs
    art.write_text("\n\n", encoding="utf-8")
    _run(monkeypatch, img, art, *flags)
    capsys.readouterr()


@pytest.mark.parametrize(
    "opt",
    [Options(), Options(size=SizeOptions(max_cols=10)), Options(size=SizeOptions(rows=3)),
     Options(out_format="html"), Options(matrix=MatrixOptions(enabled=True))],
)
def test_colorize_ascii_text_blank_lines(opt):
    colorize_ascii_text(Image.new("RGB", (8, 8)), "\n\n\n", opt=opt)
