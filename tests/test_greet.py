import io
import os
import shlex

import pytest

from asciimagic import greet

# greet writes POSIX shell rc blocks; install is gated off on Windows.
pytestmark = pytest.mark.skipif(os.name == "nt", reason="greet targets POSIX shells")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("SHELL", "/bin/bash")
    return tmp_path


def _art(tmp_path, name="cat.ans", content="\x1b[38;2;0;255;0mCAT\x1b[0m\n"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_install_copies_file_and_hooks_rc(home, tmp_path, capsys):
    art = _art(tmp_path)
    assert greet.main(["install", str(art)]) == 0

    target = home / ".config" / "ascii-magic" / "greeting.ans"
    assert target.exists()

    rc = (home / ".bashrc").read_text(encoding="utf-8")
    assert greet.MARK_BEGIN in rc
    assert greet.MARK_END in rc
    assert f"cat {shlex.quote(str(target))}" in rc
    assert f"[ -r {shlex.quote(str(target))} ]" in rc  # skip quietly if file vanishes
    assert 'case "$-" in *i*)' in rc  # interactive shells only
    assert "[ -t 1 ]" in rc
    assert "ASCII_MAGIC_NO_GREETING" in rc


def test_reinstall_replaces_block_not_duplicates(home, tmp_path):
    art = _art(tmp_path)
    greet.main(["install", str(art)])
    greet.main(["install", str(art)])
    rc = (home / ".bashrc").read_text(encoding="utf-8")
    assert rc.count(greet.MARK_BEGIN) == 1


def test_install_preserves_existing_rc_content(home, tmp_path):
    (home / ".bashrc").write_text("export FOO=bar\n")
    greet.main(["install", str(_art(tmp_path))])
    rc = (home / ".bashrc").read_text(encoding="utf-8")
    assert rc.startswith("export FOO=bar\n")
    assert greet.MARK_BEGIN in rc


def test_remove_cleans_rc_and_files(home, tmp_path):
    (home / ".bashrc").write_text("export FOO=bar\n")
    greet.main(["install", str(_art(tmp_path))])
    assert greet.main(["remove"]) == 0
    rc = (home / ".bashrc").read_text(encoding="utf-8")
    assert greet.MARK_BEGIN not in rc
    assert "export FOO=bar" in rc
    assert not (home / ".config" / "ascii-magic" / "greeting.ans").exists()


def test_show_prints_static_greeting(home, tmp_path, capsys):
    greet.main(["install", str(_art(tmp_path, content="MEOW\n"))])
    capsys.readouterr()
    assert greet.main(["show"]) == 0
    assert "MEOW" in capsys.readouterr().out


def test_show_without_install_errors(home, capsys):
    assert greet.main(["show"]) == 1


def test_install_rejects_unknown_extension(home, tmp_path, capsys):
    bad = tmp_path / "art.gif"
    bad.write_bytes(b"GIF89a")
    assert greet.main(["install", str(bad)]) == 1


def test_status_reports(home, tmp_path, capsys):
    greet.main(["install", str(_art(tmp_path))])
    capsys.readouterr()
    greet.main(["status"])
    out = capsys.readouterr().out
    assert "greeting.ans" in out
    assert "yes" in out


def test_frames_roundtrip_and_play(home, tmp_path):
    frames = ["\x1b[38;2;0;200;0mA\x1b[0m", "\x1b[38;2;0;220;0mB\x1b[0m"]
    p = tmp_path / "rain.frames"
    greet.write_frames_file(p, frames, fps=1000.0, loops=2)

    got, fps, loops = greet.read_frames_file(p)
    assert got == frames
    assert fps == 1000.0
    assert loops == 2

    buf = io.StringIO()
    greet.play_frames(got, fps, loops, out=buf)
    out = buf.getvalue()
    assert out.count("\x1b[H") == 4  # 2 frames x 2 loops
    assert out.endswith("\x1b[0m\x1b[?25h\n")


def test_install_frames_uses_show_hook(home, tmp_path):
    p = tmp_path / "rain.frames"
    greet.write_frames_file(p, ["X"], fps=10, loops=1)
    greet.main(["install", str(p)])
    rc = (home / ".bashrc").read_text(encoding="utf-8")
    assert "ascii-magic-greet show" in rc
    assert (home / ".config" / "ascii-magic" / "greeting.frames").exists()


def test_greeting_block_quotes_hostile_paths():
    from pathlib import Path

    evil = Path('/tmp/a"b$(rm -rf ~)/greeting.ans')
    block = greet._greeting_block(evil)
    assert shlex.quote(str(evil)) in block
    assert f'cat "{evil}"' not in block  # the old injectable form


def test_damaged_markers_refuse_to_rewrite(home, tmp_path, capsys):
    art = _art(tmp_path)
    greet.main(["install", str(art)])
    rc_path = home / ".bashrc"
    damaged = rc_path.read_text(encoding="utf-8").replace(greet.MARK_END, "# gone")
    rc_path.write_text(damaged)

    assert greet.main(["install", str(art)]) == 1
    assert "damaged" in capsys.readouterr().err
    assert rc_path.read_text(encoding="utf-8") == damaged  # untouched

    assert greet.main(["remove"]) == 1
    assert rc_path.read_text(encoding="utf-8") == damaged


def test_install_refuses_fish_rc(home, tmp_path, capsys):
    art = _art(tmp_path)
    rc = home / "config.fish"
    assert greet.main(["install", str(art), "--rc", str(rc)]) == 1
    assert "POSIX" in capsys.readouterr().err


def test_install_refuses_symlinked_rc(home, tmp_path, capsys):
    art = _art(tmp_path)
    real = home / ".bashrc_real"
    real.write_text("")
    link = home / ".bashrc"
    link.symlink_to(real)
    assert greet.main(["install", str(art)]) == 1
    assert "symlink" in capsys.readouterr().err


def test_install_frames_then_static_leaves_one_greeting(home, tmp_path):
    p = tmp_path / "rain.frames"
    greet.write_frames_file(p, ["X"], fps=10, loops=1)
    greet.main(["install", str(p)])
    greet.main(["install", str(_art(tmp_path))])
    d = home / ".config" / "ascii-magic"
    assert (d / "greeting.ans").exists()
    assert not (d / "greeting.frames").exists()
