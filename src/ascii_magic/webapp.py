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
from PIL import Image, ImageOps, UnidentifiedImageError

from .image_to_ascii import rotate_cw

from . import colorize_ascii as colorize_mod
from .pipeline import AsciiPipelineContext, animate as pipeline_animate, colorize, image_to_ascii, text_to_ascii

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ASCII Magic")

# Hard server-side limits — the GUI enforces friendlier ones client-side,
# but nothing stops a hand-crafted request, and the Docker CMD binds 0.0.0.0.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def _ival(o: dict[str, Any], key: str, default, lo: int, hi: int):
    """Clamped int from untrusted JSON; garbage/empty falls back to default."""
    v = o.get(key)
    if v in (None, ""):
        return default
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _fval(o: dict[str, Any], key: str, default, lo: float, hi: float):
    v = o.get(key)
    if v in (None, ""):
        return default
    try:
        n = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _build_options(o: dict[str, Any], out_format: str) -> colorize_mod.Options:
    opt = colorize_mod.Options()
    opt.out_format = out_format
    opt.keep_top = _ival(o, "keep_top", 0, 0, 5000)
    opt.color_top = bool(o.get("color_top"))

    size = opt.size
    for src_key, attr in (
        ("out_rows", "rows"),
        ("out_cols", "cols"),
        ("max_rows", "max_rows"),
        ("max_cols", "max_cols"),
    ):
        v = _ival(o, src_key, None, 1, 2000)
        if v:
            setattr(size, attr, v)

    h = opt.html
    h.font_size_px = _ival(o, "html_font_size", 12, 4, 64)
    h.line_height_px = _ival(o, "html_line_height", None, 4, 96)
    h.fill_spaces = bool(o.get("html_fill_spaces"))

    if o.get("caption_text"):
        c = opt.caption
        c.text = str(o["caption_text"])[:500]
        c.position = o.get("caption_pos", "bottom")
        c.style = o.get("caption_style", "block")
        c.scale = _fval(o, "caption_scale", 0.6, 0.05, 1.0)
        c.gap = _ival(o, "caption_gap", 1, 0, 50)
        c.color = o.get("caption_color") or None
        c.align = o.get("caption_align", "center")

    m = opt.matrix
    m.enabled = bool(o.get("matrix"))
    if m.enabled:
        if o.get("matrix_color"):
            m.tint = colorize_mod.parse_matrix_color(o["matrix_color"])
        m.top = bool(o.get("matrix_top"))
        m.seed = _ival(o, "matrix_seed", None, 0, 2**31 - 1)
        m.gamma = _fval(o, "matrix_gamma", m.gamma, 0.1, 10.0)
        m.fg_min = _ival(o, "matrix_fg_min", m.fg_min, 0, 255)
        m.fg_max = _ival(o, "matrix_fg_max", m.fg_max, 0, 255)
        m.bg_min = _ival(o, "matrix_bg_min", m.bg_min, 0, 255)
        m.bg_max = _ival(o, "matrix_bg_max", m.bg_max, 0, 255)
        if o.get("matrix_chars"):
            m.chars = str(o["matrix_chars"])[:500]
        m.fill_spaces = bool(o.get("matrix_fill_spaces"))
        m.use_mask = bool(o.get("matrix_mask"))
        m.mask_boost = _fval(o, "matrix_mask_boost", m.mask_boost, 0.0, 1.0)
        m.mask_density_floor = _fval(o, "matrix_mask_density_floor", m.mask_density_floor, 0.0, 1.0)
        m.bg_dim = _fval(o, "matrix_bg_dim", m.bg_dim, 0.0, 1.0)
        m.bg_density = _fval(o, "matrix_bg_density", m.bg_density, 0.0, 1.0)
    return opt


def _plain_html(ascii_text: str, o: dict[str, Any]) -> str:
    lines = [html_escape(ln) for ln in ascii_text.splitlines()]
    return colorize_mod.wrap_html(
        lines,
        title="ASCII Art",
        font_size_px=_ival(o, "html_font_size", 12, 4, 64),
        line_height_px=_ival(o, "html_line_height", None, 4, 96),
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    from . import __version__

    return {"status": "ok", "version": __version__}


@app.post("/api/render")
def render(
    # Plain def: Starlette runs sync handlers in a threadpool, so a slow
    # NumPy/PIL render doesn't block the event loop for other requests.
    image: Optional[UploadFile] = File(None),
    options: str = Form("{}"),
) -> dict[str, Any]:
    try:
        o: dict[str, Any] = json.loads(options)
        if not isinstance(o, dict):
            raise ValueError("options must be a JSON object")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad options JSON: {e}")

    for key in ("matrix_color", "caption_color"):
        # "image"/"image-full" are caption-only sentinels: sample the picture.
        if o.get(key) and not (key == "caption_color" and o[key] in ("image", "image-full")):
            try:
                colorize_mod.parse_matrix_color(o[key])
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

    t0 = time.perf_counter()
    ctx = AsciiPipelineContext()
    warning: Optional[str] = None

    if image is not None:
        chunks = []
        total = 0
        while True:
            chunk = image.file.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            chunks.append(chunk)
        raw = b"".join(chunks)
        try:
            img = Image.open(io.BytesIO(raw))
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image exceeds {MAX_IMAGE_PIXELS:,} pixels.",
                )
            # Honor EXIF orientation (browsers show the thumbnail rotated;
            # without this the render comes out sideways) + manual rotation.
            img = ImageOps.exif_transpose(img)
            img = rotate_cw(img, _ival(o, "rotate", 0, 0, 270))
            ctx.source_image = img.convert("RGB")  # full decode happens here
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    source = o.get("source", "image")
    if source == "text":
        text = (o.get("text") or "").strip("\n")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided.")
        try:
            text_to_ascii(
                ctx,
                text[:2000],
                style=o.get("text_style", "block"),
                width=_ival(o, "text_width", 80, 1, 500),
                font_size=_ival(o, "text_font_size", 24, 4, 200),
                banner_char=(str(o.get("banner_char") or "#"))[0],
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
                cols=_ival(o, "cols", 120, 1, 500),
                cell_w=_ival(o, "cell_w", 8, 1, 64),
                cell_h=_ival(o, "cell_h", 16, 1, 128),
                quality=o.get("quality", "balanced"),
                topk=_ival(o, "topk", 24, 1, 500),
                ascii_preset=o.get("ascii_preset", "dense"),
                unicode_mode=o.get("unicode_mode", "off"),
                autocontrast=bool(o.get("autocontrast")),
                gamma=_fval(o, "gamma", 1.0, 0.05, 10.0),
                invert=bool(o.get("invert")),
                threshold=_fval(o, "threshold", 0.5, 0.0, 1.0),
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

    # Raw-text view/download includes the caption (uncolored).
    ascii_display = ascii_text
    if o.get("caption_text"):
        from .text_to_ascii import compose_caption

        ascii_display = compose_caption(
            ascii_text,
            str(o["caption_text"])[:500],
            position=o.get("caption_pos", "bottom"),
            style=o.get("caption_style", "block"),
            scale=_fval(o, "caption_scale", 0.6, 0.05, 1.0),
            gap=_ival(o, "caption_gap", 1, 0, 50),
            align=o.get("caption_align", "center"),
        )

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
        ansi = ascii_display + "\n"
        html_doc = _plain_html(ascii_display, o)

    gif_b64: Optional[str] = None
    if do_animate:
        from .animate import AnimationOptions

        anim_opt = AnimationOptions(
            frames=_ival(o, "anim_frames", 60, 1, 240),
            fps=_fval(o, "anim_fps", 12.0, 1.0, 30.0),
            tail=_fval(o, "anim_tail", 6.0, 0.5, 40.0),
            reveal=bool(o.get("anim_reveal")),
        )
        built = _build_options(o, "ansi")
        animation = pipeline_animate(ctx, matrix=built.matrix, anim=anim_opt, caption=built.caption)
        html_doc = animation.to_html(font_size_px=_ival(o, "html_font_size", 12, 4, 64))
        gif_b64 = base64.b64encode(animation.to_gif_bytes()).decode("ascii")

    return {
        "ascii": ascii_display,
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
