import shutil
import subprocess

import pytest

from asciimagic.completion import bash_script, main as completion_main, zsh_script


def test_bash_script_contents():
    script = bash_script()
    # umbrella subcommands
    for cmd in ("colorize", "image", "text", "greet", "web", "video", "completion"):
        assert cmd in script
    # flags harvested from the parsers
    for flag in ("--matrix-color", "--caption-style", "--dither", "--rotate", "--reveal"):
        assert flag in script
    # choice values for enum flags
    assert "glyph braille" in script or "braille glyph" in script
    # standalone script names bound to the same function
    for name in ("image-to-ascii", "colorize-ascii", "ascii-magic-greet", "ascii-magic-video"):
        assert name in script
    # greet's nested subcommands
    assert "install play preview remove show status" in script


def test_zsh_script_wraps_bash():
    z = zsh_script()
    assert "bashcompinit" in z
    assert "_asciimagic()" in z


def test_completion_command(capsys):
    assert completion_main(["bash"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# ascii-magic shell completion")


def test_completion_dispatch_via_unified_cli(capsys):
    from asciimagic.unified_cli import main as cli_main

    assert cli_main(["completion", "bash"]) == 0
    assert "complete -o default -F _asciimagic" in capsys.readouterr().out


BASH = shutil.which("bash")


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize(
    "words,expected,not_expected",
    [
        (["ascii-magic", ""], "video", None),                 # subcommand names
        (["ascii-magic", "co"], "colorize", "greet"),         # prefix filter
        (["ascii-magic", "image", "--mo"], "--mode", None),   # flag completion
        (["ascii-magic", "image", "--mode", ""], "braille", "--cols"),  # choice values
        (["ascii-magic", "greet", ""], "install", None),      # nested subcommands
        (["ascii-magic", "completion", ""], "zsh", None),
        (["image-to-ascii", "--qual"], "--quality", None),    # standalone script
        (["ascii-magic", "video", "--matrix-c"], "--matrix-color", None),
    ],
)
def test_bash_completion_behaves(tmp_path, words, expected, not_expected):
    script_file = tmp_path / "completion.bash"
    script_file.write_text(bash_script(), encoding="utf-8")

    comp_words = " ".join(f'"{w}"' for w in words)
    harness = f"""
source "{script_file}"
COMP_WORDS=({comp_words})
COMP_CWORD={len(words) - 1}
_asciimagic
printf '%s\\n' "${{COMPREPLY[@]}}"
"""
    r = subprocess.run([BASH, "-c", harness], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    replies = r.stdout.split()
    assert expected in replies, f"{expected!r} not in {replies!r}"
    if not_expected is not None:
        assert not_expected not in replies
