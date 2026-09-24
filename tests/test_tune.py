import json
import shlex
import sys

import pytest
from PIL import Image, ImageDraw

from asciimagic import tune
from asciimagic.image_to_ascii import apply_cell_aspect


@pytest.fixture(autouse=True)
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return tmp_path / "cfg" / "ascii-magic"


@pytest.fixture
def photo(tmp_path):
    img = Image.new("RGB", (120, 90), "white")
    d = ImageDraw.Draw(img)
    for x in range(120):
        d.line([(x, 0), (x, 20)], fill=(x * 2,) * 3)
    d.ellipse([20, 30, 70, 80], fill=(20, 20, 20))
    d.rectangle([80, 30, 110, 80], outline=(0, 0, 0), width=3)
    p = tmp_path / "p.png"
    img.save(p)
    return p


def _image_cli(monkeypatch, capsys, path, flags):
    from asciimagic.image_to_ascii import main

    monkeypatch.setattr(sys, "argv", ["image-to-ascii", str(path), *shlex.split(flags)])
    main()
    return capsys.readouterr().out


@pytest.mark.parametrize("n", range(1, len(tune.VARIANTS) + 1))
def test_each_variant_reproduces_with_its_printed_flags(n, photo, monkeypatch, capsys):
    assert tune.main([str(photo), "-c", "24", "--only", str(n)]) == 0
    sheet = capsys.readouterr().out
    label_line, flags_line, *art = sheet.strip("\n").splitlines()
    assert label_line.startswith(f"[{n}] ")
    assert _image_cli(monkeypatch, capsys, photo, flags_line.strip()).rstrip("\n") == "\n".join(art)


def test_pick_and_save_preset_then_reuse(photo, monkeypatch, capsys, config_home):
    assert tune.main([str(photo), "-c", "20", "--pick", "4", "--save-preset", "dark"]) == 0
    out = capsys.readouterr()
    assert "--mode braille --invert --dither -c 20" in out.out
    saved = json.loads((config_home / "presets.json").read_text())["image"]["dark"]
    assert saved == {"mode": "braille", "invert": True, "dither": True, "cols": 20,
                     "_about": "tune pick 4: braille, inverted + dithered"}
    via_preset = _image_cli(monkeypatch, capsys, photo, "--preset dark")
    via_flags = _image_cli(monkeypatch, capsys, photo, "--mode braille --invert --dither -c 20")
    assert via_preset == via_flags


def test_html_sheet_has_every_variant(photo, tmp_path, capsys):
    out = tmp_path / "s.html"
    assert tune.main([str(photo), "-c", "20", "--color", "-o", str(out)]) == 0
    doc = out.read_text(encoding="utf-8")
    for n in range(1, len(tune.VARIANTS) + 1):
        assert f"<b>[{n}]</b>" in doc
    assert "color:" in doc or "color: rgb" in doc  # colorized spans


def test_color_depth_applies_to_terminal_sheet(photo, capsys):
    assert tune.main([str(photo), "-c", "16", "--only", "1", "--color", "--color-depth", "256"]) == 0
    out = capsys.readouterr().out
    assert "\x1b[38;5;" in out and "\x1b[38;2;" not in out


def test_aspect_chart_boxes_are_square_for_their_aspect(capsys):
    assert tune.main(["--aspect-chart"]) == 0
    lines = capsys.readouterr().out.splitlines()
    top = next(ln for ln in lines if ln.startswith("┌"))
    widths = [len(seg) for seg in top.split("  ")]
    # box width w for aspect a is rows/a: 8 rows -> 20, 18, 16, 15, 13
    assert widths == [round(8 / a) for a in tune.ASPECTS]


@pytest.mark.parametrize("argv,msg", [
    (["--pick", "99", "x.png"], "no variant 99"),
    (["x.png", "--save-preset", "a"], "needs --pick"),
    ([], "image is required"),
    (["x.png", "-c", "2"], "--cols"),
])
def test_usage_errors(argv, msg, capsys):
    with pytest.raises(SystemExit):
        tune.main(argv)
    assert msg in capsys.readouterr().err


def test_only_rejects_unknown_numbers(photo):
    with pytest.raises(SystemExit, match="no variant 50"):
        tune.main([str(photo), "--only", "1,50"])


# ---- --cell-aspect ----

def test_cell_aspect_default_is_identity():
    img = Image.new("RGB", (100, 80))
    assert apply_cell_aspect(img, 0.5) is img


@pytest.mark.parametrize("aspect,expect_h", [(0.4, 64), (0.6, 96)])
def test_cell_aspect_scales_height(aspect, expect_h):
    assert apply_cell_aspect(Image.new("RGB", (100, 80)), aspect).size == (100, expect_h)


def test_image_cell_aspect_changes_row_count(photo, monkeypatch, capsys):
    base = _image_cli(monkeypatch, capsys, photo, "--mode braille -c 40").splitlines()
    tall = _image_cli(monkeypatch, capsys, photo, "--mode braille -c 40 --cell-aspect 0.6").splitlines()
    assert len(tall) > len(base)


def test_image_cell_aspect_out_of_range(photo, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _image_cli(monkeypatch, capsys, photo, "--cell-aspect 3")
    assert "between 0.2 and 1.2" in capsys.readouterr().err
