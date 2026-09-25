"""Terminal display width for text that is not plain ASCII.

Japanese, Chinese, and Korean characters (and most emoji) occupy two columns
in a terminal; combining accents and zero-width characters occupy none. Code
that pads or aligns with ``len()`` misplaces everything after such text --
boxes come out too short, captions off-center. These helpers measure what
the terminal will actually show, using only the standard library's Unicode
tables ("ambiguous" characters such as box-drawing lines count as one
column, as terminals do by default).
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import List

# Continuation marker: the second column of a double-width character in a
# per-column cell grid. It renders as nothing.
CONT = ""


@lru_cache(maxsize=4096)
def char_width(ch: str) -> int:
    if not ch:
        return 0
    if ch in "​‌‍﻿" or unicodedata.combining(ch):
        return 0
    if unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 1


def str_width(s: str) -> int:
    return sum(char_width(ch) for ch in s)


def is_wide(s: str) -> bool:
    """True if any character takes other than one column."""
    return any(char_width(ch) != 1 for ch in s)


def ljust(s: str, width: int, fill: str = " ") -> str:
    return s + fill * max(0, width - str_width(s))


def truncate(s: str, width: int) -> str:
    """Longest prefix that fits in `width` columns (never splits a wide char)."""
    out, used = [], 0
    for ch in s:
        w = char_width(ch)
        if used + w > width:
            break
        out.append(ch)
        used += w
    return "".join(out)


def fit(s: str, width: int) -> str:
    """Exactly `width` columns: truncated, then padded (a wide char that
    would straddle the edge becomes a space)."""
    return ljust(truncate(s, width), width)


def to_cells(s: str) -> List[str]:
    """One entry per terminal column: a wide char is followed by CONT;
    zero-width characters attach to the preceding cell."""
    cells: List[str] = []
    for ch in s:
        w = char_width(ch)
        if w == 0:
            if cells:
                cells[-1] += ch
            else:
                cells.append(ch)
        elif w == 2:
            cells.extend((ch, CONT))
        else:
            cells.append(ch)
    return cells


def from_cells(cells: List[str]) -> str:
    """Join cells back into text, repairing half-cut wide characters: a lead
    without its continuation, or a continuation without its lead, becomes a
    space so every row keeps its column count."""
    out = []
    n = len(cells)
    for i, c in enumerate(cells):
        if c == CONT:
            if i == 0 or not _is_wide_lead(cells[i - 1]):
                out.append(" ")
            continue
        if _is_wide_lead(c) and (i + 1 >= n or cells[i + 1] != CONT):
            out.append(" ")
            continue
        out.append(c)
    return "".join(out)


def _is_wide_lead(c: str) -> bool:
    return bool(c) and char_width(c[0]) == 2


def scale_lines(lines: List[str], rows: int, cols: int) -> List[str]:
    """Nearest-neighbor scale a block of text to rows x cols terminal
    columns, treating a double-width character as two columns."""
    grid = [to_cells(ln) for ln in lines]
    src_w = max((len(r) for r in grid), default=0)
    src_h = len(grid)
    if not src_w or not src_h:
        return [" " * cols for _ in range(rows)]
    grid = [r + [" "] * (src_w - len(r)) for r in grid]
    out = []
    for y in range(rows):
        row = grid[int(y * src_h / rows)]
        out.append(from_cells([row[int(x * src_w / cols)] for x in range(cols)]))
    return out
