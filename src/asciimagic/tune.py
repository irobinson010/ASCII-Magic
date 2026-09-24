"""Find the settings that make an image look best, side by side.

``ascii-magic tune photo.png`` renders the same picture several ways (braille
vs glyph matching, dithering, inversion, contrast, gamma, character set) and
labels each with the exact flags that reproduce it. Keep the one you like:

    ascii-magic tune photo.png                         # numbered variants in the terminal
    ascii-magic tune photo.png -o sheet.html           # contact sheet in the browser
    ascii-magic tune photo.png --pick 4 --save-preset my-photos
    ascii-magic image other.png --preset my-photos

``--aspect-chart`` helps pick ``--cell-aspect`` for your terminal: it draws
squares for several cell shapes; the one that looks square is your value.
"""

from __future__ import annotations

import argparse
import html
import sys
from typing import Any, Dict, List, Optional, Tuple

Variant = Tuple[str, Dict[str, Any]]

VARIANTS: List[Variant] = [
    ("braille", {"mode": "braille"}),
    ("braille, dithered", {"mode": "braille", "dither": True}),
    ("braille, inverted (dark terminals)", {"mode": "braille", "invert": True}),
    ("braille, inverted + dithered", {"mode": "braille", "invert": True, "dither": True}),
    ("braille, autocontrast + dithered", {"mode": "braille", "autocontrast": True, "dither": True}),
    ("braille, lighter midtones", {"mode": "braille", "gamma": 0.7, "dither": True}),
    ("braille, darker midtones", {"mode": "braille", "gamma": 1.4, "dither": True}),
    ("glyph", {"mode": "glyph"}),
    ("glyph, best quality", {"mode": "glyph", "quality": "best"}),
    ("glyph, inverted", {"mode": "glyph", "quality": "best", "invert": True}),
    ("glyph, plain ASCII only", {"mode": "glyph", "quality": "best", "ascii": "printable"}),
    ("glyph, autocontrast", {"mode": "glyph", "quality": "best", "autocontrast": True}),
]

ASPECTS = (0.40, 0.45, 0.50, 0.55, 0.60)


def flags_for(settings: Dict[str, Any]) -> str:
    """The `ascii-magic image` flags that reproduce a variant."""
    out = []
    for k, v in settings.items():
        flag = "-c" if k == "cols" else "--" + k.replace("_", "-")
        out.append(flag if v is True else f"{flag} {v}")
    return " ".join(out)


def render_variant(img, settings: Dict[str, Any], cols: int, cell_aspect: float) -> str:
    from .image_to_ascii import (
        apply_cell_aspect,
        image_to_braille_from_image,
        image_to_text_glyph_from_image,
        make_charset,
    )

    img = apply_cell_aspect(img, cell_aspect)
    common = dict(
        autocontrast=settings.get("autocontrast", False),
        gamma=settings.get("gamma", 1.0),
        invert=settings.get("invert", False),
    )
    if settings.get("mode") == "glyph":
        return image_to_text_glyph_from_image(
            img=img, cols=cols, cell_w=8, cell_h=16,
            charset=make_charset(unicode_mode="off", ascii_preset=settings.get("ascii", "dense")),
            quality=settings.get("quality", "balanced"), font_path=None, font_size=None,
            topk=24, **common,
        )
    return image_to_braille_from_image(
        img, cols=cols, threshold=0.5, dither=settings.get("dither", False), **common,
    )


def aspect_chart(rows: int = 8) -> str:
    """Squares drawn for several cell aspects; the one that looks square
    on screen is the terminal's --cell-aspect."""
    boxes = []
    for a in ASPECTS:
        w = max(4, round(rows / a))
        label = f"{a:.2f}".center(w)
        top = "┌" + "─" * (w - 2) + "┐"
        mid = ["│" + " " * (w - 2) + "│" for _ in range(rows - 2)]
        mid[len(mid) // 2] = "│" + label[1:-1] + "│"
        bot = "└" + "─" * (w - 2) + "┘"
        boxes.append([top] + mid + [bot])
    lines = ["  ".join(b[i] for b in boxes) for i in range(rows)]
    return (
        "Which box looks square? Pass its number as --cell-aspect "
        "(to image, tune, or a preset).\n\n" + "\n".join(lines) + "\n"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    from .ansi import add_depth_arg

    ap = argparse.ArgumentParser(
        prog="ascii-magic tune",
        description="Render an image several ways side by side to find the best settings.",
    )
    ap.add_argument("input", nargs="?", help="Image to tune")
    ap.add_argument("-c", "--cols", type=int, default=60, help="Width of each variant (default: 60)")
    ap.add_argument("-o", "--output", default=None,
                    help="Write a contact sheet: .html (side by side) or .txt; default: terminal")
    ap.add_argument("--color", action="store_true", help="Colorize every variant from the image")
    add_depth_arg(ap)
    ap.add_argument("--cell-aspect", type=float, default=0.5, metavar="W/H",
                    help="Terminal cell width/height used for every variant (default 0.5)")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0)
    ap.add_argument("--only", default=None, metavar="N,N,...",
                    help="Render just these variant numbers")
    ap.add_argument("--pick", type=int, default=None, metavar="N",
                    help="Choose variant N: print its command (and save it with --save-preset)")
    ap.add_argument("--save-preset", default=None, metavar="NAME",
                    help="With --pick: save the variant (plus -c/--cell-aspect) as an image preset")
    ap.add_argument("--aspect-chart", action="store_true",
                    help="Print squares for several cell aspects to find your terminal's --cell-aspect")
    return ap


def _chosen(args, n_variants: int) -> List[int]:
    if not args.only:
        return list(range(1, n_variants + 1))
    try:
        picked = [int(x) for x in args.only.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--only expects numbers like 1,4,9, got {args.only!r}")
    bad = [n for n in picked if not 1 <= n <= n_variants]
    if bad:
        raise SystemExit(f"--only: no variant {bad[0]} (there are {n_variants})")
    return picked


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if args.aspect_chart:
        sys.stdout.write(aspect_chart())
        return 0
    if not args.input:
        ap.error("an image is required (or use --aspect-chart)")
    if args.cols < 4 or args.cols > 400:
        ap.error("--cols must be between 4 and 400")
    if not 0.2 <= args.cell_aspect <= 1.2:
        ap.error("--cell-aspect must be between 0.2 and 1.2")

    def full_settings(n: int) -> Dict[str, Any]:
        s = dict(VARIANTS[n - 1][1])
        s["cols"] = args.cols
        if args.cell_aspect != 0.5:
            s["cell_aspect"] = args.cell_aspect
        return s

    if args.pick is not None:
        if not 1 <= args.pick <= len(VARIANTS):
            ap.error(f"--pick: no variant {args.pick} (there are {len(VARIANTS)})")
        settings = full_settings(args.pick)
        print(f"ascii-magic image {args.input} {flags_for(settings)}")
        if args.save_preset:
            from .presets import save

            path = save("image", args.save_preset, settings, about=f"tune pick {args.pick}: {VARIANTS[args.pick - 1][0]}")
            print(f"Saved image preset {args.save_preset!r} to {path}. Use: --preset {args.save_preset}",
                  file=sys.stderr)
        return 0
    if args.save_preset:
        ap.error("--save-preset needs --pick N")

    from .image_to_ascii import open_oriented, rotate_cw

    try:
        img = rotate_cw(open_oriented(args.input, "RGB"), args.rotate)
    except OSError as e:
        raise SystemExit(f"Could not open {args.input}: {e}")

    out_path = args.output
    as_html = bool(out_path and out_path.lower().endswith((".html", ".htm")))
    colorize_fmt = ("html" if as_html else "ansi") if args.color else None

    cards = []
    for n in _chosen(args, len(VARIANTS)):
        label, settings = VARIANTS[n - 1]
        art = render_variant(img, settings, args.cols, args.cell_aspect)
        if colorize_fmt:
            from .colorize_ascii import Options, colorize_ascii_text

            art = colorize_ascii_text(img, art, opt=Options(out_format=colorize_fmt))
        cards.append((n, label, flags_for(full_settings(n)), art))

    if as_html:
        doc = _html_sheet(args.input, cards, colored=bool(colorize_fmt))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc)
        print(f"Wrote {out_path} ({len(cards)} variants). Then: ascii-magic tune {args.input} --pick N --save-preset NAME",
              file=sys.stderr)
        return 0

    text = _text_sheet(cards)
    if colorize_fmt:
        from .ansi import downsample, resolve_depth

        text = downsample(text, resolve_depth(args.color_depth, to_terminal=not out_path))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    print(f"Keep one: ascii-magic tune {args.input} --pick N --save-preset NAME", file=sys.stderr)
    return 0


def _text_sheet(cards) -> str:
    parts = []
    for n, label, flags, art in cards:
        parts.append(f"\n[{n}] {label}\n    {flags}\n")
        parts.append(art if art.endswith("\n") else art + "\n")
    return "".join(parts)


def _html_sheet(title: str, cards, colored: bool) -> str:
    def body(art: str) -> str:
        if not colored:
            return html.escape(art)
        # colorize_ascii_text returned a full HTML page; keep its <pre> content.
        start = art.find("<pre>")
        end = art.rfind("</pre>")
        return art[start + 5:end] if start >= 0 and end > start else html.escape(art)

    items = "\n".join(
        f'<figure><figcaption><b>[{n}]</b> {html.escape(label)}<br><code>{html.escape(flags)}</code>'
        f"</figcaption><pre>{body(art)}</pre></figure>"
        for n, label, flags, art in cards
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>tune: {html.escape(title)}</title>
<style>
  body {{ margin: 0; padding: 16px; background: #0b0f14; color: #d7e2ec;
         font-family: "Cascadia Mono", "DejaVu Sans Mono", Consolas, monospace; }}
  h1 {{ font-size: 15px; font-weight: normal; color: #8496a8; }}
  main {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  figure {{ margin: 0; padding: 10px; background: #121821; border: 1px solid #243044; border-radius: 6px; }}
  figcaption {{ font-size: 12px; margin-bottom: 6px; }}
  code {{ color: #37e07a; }}
  pre {{ margin: 0; font-size: 8px; line-height: 8px; color: #e0e0e0; }}
</style></head><body>
<h1>{html.escape(title)} &mdash; keep one with: ascii-magic tune {html.escape(title)} --pick N --save-preset NAME</h1>
<main>
{items}
</main></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
