import json
import sys

import pytest
from PIL import Image

from asciimagic import presets
from asciimagic.image_to_ascii import main as image_main


@pytest.fixture(autouse=True)
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return tmp_path / "cfg" / "ascii-magic"


@pytest.fixture
def png(tmp_path):
    img = Image.new("RGB", (64, 48), "white")
    for x in range(20, 44):
        for y in range(10, 38):
            img.putpixel((x, y), (20, 20, 20))
    p = tmp_path / "t.png"
    img.save(p)
    return p


def _image(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["image-to-ascii", *map(str, args)])
    image_main()
    return capsys.readouterr()


def test_save_then_reuse_reproduces_output(png, monkeypatch, capsys, config_home):
    first = _image(monkeypatch, capsys, png, "-c", "24", "--mode", "braille", "--invert",
                   "--dither", "--save-preset", "mine")
    assert "Saved image preset 'mine'" in first.err
    saved = json.loads((config_home / "presets.json").read_text())["image"]["mine"]
    assert saved == {"cols": 24, "mode": "braille", "invert": True, "dither": True}
    again = _image(monkeypatch, capsys, png, "--preset", "mine")
    assert again.out == first.out


def test_flags_beat_the_preset(png, monkeypatch, capsys):
    _image(monkeypatch, capsys, png, "-c", "24", "--mode", "braille", "--save-preset", "wide")
    narrow = _image(monkeypatch, capsys, png, "--preset", "wide", "-c", "10")
    assert max(len(ln) for ln in narrow.out.splitlines()) == 10


def test_inputs_outputs_and_caption_text_are_not_saved(png, tmp_path, monkeypatch, capsys, config_home):
    _image(monkeypatch, capsys, png, "-o", tmp_path / "o.txt", "--caption", "Hi",
           "--caption-style", "box", "--save-preset", "cap")
    saved = json.loads((config_home / "presets.json").read_text())["image"]["cap"]
    assert saved == {"caption_style": "box"}


def test_builtin_preset_applies(png, monkeypatch, capsys):
    out = _image(monkeypatch, capsys, png, "-c", "20", "--preset", "ssh-safe").out
    assert all(ord(ch) < 128 for ch in out)  # printable ASCII only


def test_saved_preset_shadows_builtin(monkeypatch):
    presets.save("image", "photo", {"cols": 33})
    assert presets.get("image", "photo") == {"cols": 33}
    assert presets.available("image")["photo"][0] == "saved"


def test_unknown_preset_errors():
    with pytest.raises(SystemExit, match="Unknown image preset 'nope'"):
        presets.get("image", "nope")


def test_invalid_preset_value_is_a_usage_error(png, monkeypatch, capsys, config_home):
    config_home.mkdir(parents=True)
    (config_home / "presets.json").write_text(json.dumps({"image": {"bad": {"mode": "sparkles"}}}))
    with pytest.raises(SystemExit) as e:
        _image(monkeypatch, capsys, png, "--preset", "bad")
    assert e.value.code == 2
    assert "not one of" in capsys.readouterr().err


def test_preset_key_for_another_command_rejected(png, monkeypatch, capsys, config_home):
    config_home.mkdir(parents=True)
    (config_home / "presets.json").write_text(json.dumps({"image": {"x": {"matrix_gamma": 2}}}))
    with pytest.raises(SystemExit):
        _image(monkeypatch, capsys, png, "--preset", "x")
    assert "not a image setting" in capsys.readouterr().err


def test_every_builtin_preset_is_valid_for_its_command():
    from asciimagic import colorize_ascii, image_to_ascii, text_to_ascii, video

    parsers = {"image": image_to_ascii, "colorize": colorize_ascii, "video": video, "text": text_to_ascii}
    for command, entries in presets.BUILTIN.items():
        parser = parsers[command].build_arg_parser()
        for name in entries:
            presets._validate(parser, command, name, presets.get(command, name))


@pytest.mark.parametrize("command,argv", [
    ("video", lambda clip, out: [str(clip), str(out), "-c", "12"]),
])
def test_video_and_colorize_accept_presets(command, argv, tmp_path, png, monkeypatch, capsys):
    from asciimagic import colorize_ascii
    from asciimagic import video as video_mod

    frames = [Image.new("RGB", (32, 24), (i * 60, 40, 90)) for i in range(3)]
    clip = tmp_path / "c.gif"
    frames[0].save(clip, save_all=True, append_images=frames[1:], duration=100)
    assert video_mod.main(argv(clip, tmp_path / "v.frames") + ["--preset", "ssh-safe"]) == 0

    art = tmp_path / "a.txt"
    art.write_text("##\n##\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["colorize-ascii", str(png), str(art), "-", "--preset", "ssh-safe"])
    colorize_ascii.main()
    assert "\x1b[38;5;" in capsys.readouterr().out


def test_presets_command_list_show_delete(capsys):
    presets.save("image", "mine", {"cols": 50})
    assert presets.main(["list", "image"]) == 0
    out = capsys.readouterr().out
    assert "mine" in out and "[saved]" in out and "photo" in out and "[built-in]" in out
    assert presets.main(["show", "image", "mine"]) == 0
    assert json.loads(capsys.readouterr().out) == {"cols": 50}
    assert presets.main(["delete", "image", "mine"]) == 0
    assert presets.main(["delete", "image", "photo"]) == 1
    assert "built in" in capsys.readouterr().err


def test_corrupt_presets_file_is_reported(config_home):
    config_home.mkdir(parents=True)
    (config_home / "presets.json").write_text("{not json")
    with pytest.raises(SystemExit, match="Could not read presets file"):
        presets.available("image")
