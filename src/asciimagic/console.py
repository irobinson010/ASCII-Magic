"""Console setup shared by the command-line entry points."""

from __future__ import annotations

import sys


def utf8_stdout() -> None:
    """Write stdout as UTF-8 even when the platform default is not.

    On Windows, redirected or piped output (``ascii-magic image x.png >
    art.txt``) uses the ANSI code page (e.g. cp1252), which cannot encode
    braille, box-drawing, or block characters, so conversion crashed with
    UnicodeEncodeError. Files written with ``-o`` are already UTF-8; this
    makes stdout match. Interactive consoles and UTF-8 locales are left alone.
    """
    stream = sys.stdout
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    if encoding == "utf8" or not hasattr(stream, "reconfigure"):
        return
    try:
        stream.reconfigure(encoding="utf-8")
    except (ValueError, OSError):  # detached or already-used stream: keep it
        pass
