"""FastAPI web GUI for ASCII Magic.

Thin HTTP wrapper over the in-memory pipeline: upload an image (or type
text), send knob values as a JSON blob, get back the raw ASCII plus ANSI
and HTML renders in one response. The single-page GUI in ``static/`` is
served from the same app.

Run locally:  ascii-magic-web  (or: uvicorn ascii_magic.webapp:app)
"""

from __future__ import annotations

import base64
import io
import json
import random
import time
from html import escape as html_escape
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import colorize_ascii as colorize_mod
from .pipeline import AsciiPipelineContext, animate as pipeline_animate, colorize, image_to_ascii, text_to_ascii

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ASCII Magic")


def _build_options(o: dict[str, Any], out_format: str) -> colorize_mod.Options:
    opt = colorize_mod.Options()
    opt.out_format = out_format
    opt.keep_top = int(o.get("keep_top") or 0)
    opt.color_top = bool(o.get("color_top"))

    size = opt.size
    for src_key, attr in (
        ("out_rows", "rows"),
        ("out_cols", "cols"),
        ("max_rows", "max_rows"),
        ("max_cols", "max_cols"),
    ):
        v = o.get(src_key)
        if v not in (None, "", 0):
            setattr(size, attr, int(v))

    h = opt.html
    h.font_size_px = int(o.get("html_font_size") or 12)
    lh = o.get("html_line_height")
    h.line_height_px = int(lh) if lh not in (None, "", 0) else None
    h.fill_spaces = bool(o.get("html_fill_spaces"))

    m = opt.matrix
    m.enabled = bool(o.get("matrix"))
    if m.enabled:
        if o.get("matrix_color"):
            m.tint = colorize_mod.parse_matrix_color(o["matrix_color"])
        m.top = bool(o.get("matrix_top"))
        m.seed = int(o["matrix_seed"]) if o.get("matrix_seed") not in (None, "") else None
        m.gamma = float(o.get("matrix_gamma") or m.gamma)
        m.fg_min = int(o.get("matrix_fg_min") if o.get("matrix_fg_min") is not None else m.fg_min)
        m.fg_max = int(o.get("matrix_fg_max") if o.get("matrix_fg_max") is not None else m.fg_max)
        m.bg_min = int(o.get("matrix_bg_min") if o.get("matrix_bg_min") is not None else m.bg_min)
        m.bg_max = int(o.get("matrix_bg_max") if o.get("matrix_bg_max") is not None else m.bg_max)
        if o.get("matrix_chars"):
            m.chars = str(o["matrix_chars"])
        m.fill_spaces = bool(o.get("matrix_fill_spaces"))
        m.use_mask = bool(o.get("matrix_mask"))
        m.mask_boost = float(o.get("matrix_mask_boost") if o.get("matrix_mask_boost") is not None else m.mask_boost)
        m.mask_density_floor = float(
            o.get("matrix_mask_density_floor") if o.get("matrix_mask_density_floor") is not None else m.mask_density_floor
        )
        m.bg_dim = float(o.get("matrix_bg_dim") if o.get("matrix_bg_dim") is not None else m.bg_dim)
        m.bg_density = float(o.get("matrix_bg_density") if o.get("matrix_bg_density") is not None else m.bg_density)
    return opt


def _plain_html(ascii_text: str, o: dict[str, Any]) -> str:
    lines = [html_escape(ln) for ln in ascii_text.splitlines()]
    lh = o.get("html_line_height")
    return colorize_mod.wrap_html(
        lines,
        title="ASCII Art",
        font_size_px=int(o.get("html_font_size") or 12),
        line_height_px=int(lh) if lh not in (None, "", 0) else None,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    from . import __version__

    return {"status": "ok", "version": __version__}


@app.post("/api/render")
async def render(
    image: Optional[UploadFile] = File(None),
    options: str = Form("{}"),
) -> dict[str, Any]:
    try:
        o: dict[str, Any] = json.loads(options)
        if not isinstance(o, dict):
            raise ValueError("options must be a JSON object")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad options JSON: {e}")

    if o.get("matrix_color"):
        try:
            colorize_mod.parse_matrix_color(o["matrix_color"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    t0 = time.perf_counter()
    ctx = AsciiPipelineContext()
    warning: Optional[str] = None

    if image is not None:
        raw = await image.read()
        try:
            ctx.source_image = Image.open(io.BytesIO(raw)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    source = o.get("source", "image")
    if source == "text":
        text = (o.get("text") or "").strip("\n")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided.")
        try:
            text_to_ascii(
                ctx,
                text,
                style=o.get("text_style", "block"),
                width=int(o.get("text_width") or 80),
                font_size=int(o.get("text_font_size") or 24),
                banner_char=(o.get("banner_char") or "#")[0],
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif source == "image":
        if ctx.source_image is None:
            raise HTTPException(status_code=400, detail="No image uploaded.")
        try:
            image_to_ascii(
                ctx,
                mode=o.get("mode", "braille"),
                cols=int(o.get("cols") or 120),
                cell_w=int(o.get("cell_w") or 8),
                cell_h=int(o.get("cell_h") or 16),
                quality=o.get("quality", "balanced"),
                topk=int(o.get("topk") or 24),
                ascii_preset=o.get("ascii_preset", "dense"),
                unicode_mode=o.get("unicode_mode", "off"),
                autocontrast=bool(o.get("autocontrast")),
                gamma=float(o.get("gamma") or 1.0),
                invert=bool(o.get("invert")),
                threshold=float(o.get("threshold") if o.get("threshold") is not None else 0.5),
                dither=bool(o.get("dither")),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

    ascii_text = ctx.ascii_text or ""
    seed: Optional[int] = None
    do_colorize = o.get("colorize", True)
    do_animate = bool(o.get("animate"))
    if do_animate:
        o = {**o, "matrix": True}

    # Colorizing/animating needs a reference image; box/banner text styles
    # do not render one, so fall back to plain output instead of erroring.
    can_colorize = ctx.source_image is not None or ctx.rendered_text_image is not None
    if (do_colorize or do_animate) and not can_colorize:
        do_colorize = do_animate = False
        warning = "No reference image for colorizing this style; returning plain ASCII."

    # ANSI, HTML, and animation are rendered separately, so a random matrix
    # seed would diverge between them — pin one and echo it back.
    if o.get("matrix") and (do_colorize or do_animate):
        if o.get("matrix_seed") in (None, ""):
            seed = random.randrange(2**31)
            o = {**o, "matrix_seed": seed}
        else:
            seed = int(o["matrix_seed"])

    if do_colorize:
        ansi = colorize(ctx, opt=_build_options(o, "ansi"))
        html_doc = colorize(ctx, opt=_build_options(o, "html"))
    else:
        ansi = ascii_text + "\n"
        html_doc = _plain_html(ascii_text, o)

    gif_b64: Optional[str] = None
    if do_animate:
        from .animate import AnimationOptions

        anim_opt = AnimationOptions(
            frames=max(1, min(int(o.get("anim_frames") or 60), 240)),
            fps=max(1.0, min(float(o.get("anim_fps") or 12), 30.0)),
            tail=max(0.5, min(float(o.get("anim_tail") or 6), 40.0)),
        )
        animation = pipeline_animate(ctx, matrix=_build_options(o, "ansi").matrix, anim=anim_opt)
        html_doc = animation.to_html(font_size_px=int(o.get("html_font_size") or 12))
        gif_b64 = base64.b64encode(animation.to_gif_bytes()).decode("ascii")

    return {
        "ascii": ascii_text,
        "ansi": ansi,
        "html": html_doc,
        "gif_b64": gif_b64,
        "seed": seed,
        "warning": warning,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
    }


# Mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="ASCII Magic web GUI")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
