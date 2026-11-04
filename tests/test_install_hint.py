"""Install hints point where releases actually live (GitHub, not PyPI)."""

import sys

import pytest

from asciimagic import SOURCE, install_hint


def test_hint_names_the_extra_and_the_github_source():
    assert install_hint("video") == f'pip install "ascii-magic-tools[video] @ {SOURCE}"'
    assert SOURCE.startswith("git+https://github.com/")


def test_missing_extras_point_at_github(monkeypatch):
    from asciimagic import translate as tr

    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    with pytest.raises(tr.TranslationError, match=r"\[translate\] @ git\+https://github\.com/"):
        tr._require_engine()
