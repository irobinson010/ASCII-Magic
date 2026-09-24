"""Redirected output must be UTF-8 even where the platform default is not
(Windows pipes/files use the ANSI code page, e.g. cp1252)."""

import os
import subprocess
import sys

import pytest
from PIL import Image


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "t.png"
    Image.new("RGB", (40, 30), (20, 20, 20)).save(p)
    return p


@pytest.mark.parametrize("module,args", [
    ("asciimagic.unified_cli", ["image", "{png}", "--mode", "braille", "-c", "12"]),
    ("asciimagic.image_to_ascii", ["{png}", "--mode", "braille", "-c", "12"]),
    ("asciimagic.unified_cli", ["text", "Hi", "-s", "box"]),
    ("asciimagic.unified_cli", ["compose", "--text", "Hi", "--style", "box"]),
])
def test_unicode_art_survives_a_cp1252_pipe(module, args, png):
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    proc = subprocess.run(
        [sys.executable, "-m", module, *[a.format(png=png) for a in args]],
        capture_output=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    text = proc.stdout.decode("utf-8")  # must be valid UTF-8
    assert any(ord(ch) > 127 for ch in text)


def test_utf8_stdout_leaves_utf8_streams_alone(monkeypatch):
    import io

    from asciimagic.console import utf8_stdout

    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", stream)
    utf8_stdout()
    assert sys.stdout is stream and stream.encoding == "utf-8"


def test_utf8_stdout_switches_other_encodings(monkeypatch):
    import io

    from asciimagic.console import utf8_stdout

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    utf8_stdout()
    stream.write("⣿")
    stream.flush()
    assert raw.getvalue() == "⣿".encode("utf-8")
