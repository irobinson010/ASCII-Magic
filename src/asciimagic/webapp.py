"""FastAPI web GUI for ASCII Magic.

Thin HTTP wrapper over the in-memory pipeline: upload an image (or type
text), send knob values as a JSON blob, get back the raw ASCII plus ANSI
and HTML renders in one response. The single-page GUI in ``static/`` is
served from the same app.

Run locally:  ascii-magic-web  (or: uvicorn asciimagic.webapp:app)
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import random
import threading
import time
from contextlib import contextmanager
from html import escape as html_escape
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .image_to_ascii import flatten_alpha, rotate_cw

from . import colorize_ascii as colorize_mod
from .pipeline import AsciiPipelineContext, animate as pipeline_animate, colorize, image_to_ascii, text_to_ascii

STATIC_DIR = Path(__file__).parent / "static"


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


# Hard server-side limits — the GUI enforces friendlier ones client-side,
# but nothing stops a hand-crafted request, and the Docker CMD binds 0.0.0.0.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_VIDEO_UPLOAD_BYTES = 100 * 1024 * 1024
# Whole request body (largest upload + multipart framing + options JSON).
# Enforced while the body streams in, before anything is spooled to disk.
MAX_REQUEST_BYTES = MAX_VIDEO_UPLOAD_BYTES + 2 * 1024 * 1024

# Work budgets. Render time scales with output characters (and, for glyph
# matching, the pixels per cell; for animations, the frame count). Measured
# at roughly 11 us per character-frame for animations, so the defaults keep
# a single request to ~10 s on one core. Override via environment.
MAX_CELLS = _env_int("ASCII_MAGIC_MAX_CELLS", 250_000)
MAX_GLYPH_PIXELS = _env_int("ASCII_MAGIC_MAX_GLYPH_PIXELS", 16_000_000)
MAX_ANIM_CELL_FRAMES = _env_int("ASCII_MAGIC_MAX_ANIM_CELL_FRAMES", 1_000_000)

# Renders are CPU-bound; running more at once than there are cores only makes
# every one of them slower, and an unbounded pile-up ties up every worker
# thread. Excess requests wait briefly for a slot, then get a 503.
MAX_CONCURRENT_RENDERS = _env_int("ASCII_MAGIC_MAX_CONCURRENT", min(4, os.cpu_count() or 1))
RENDER_QUEUE_TIMEOUT_S = 15.0
_render_slots = threading.BoundedSemaphore(MAX_CONCURRENT_RENDERS)


@contextmanager
def _render_slot():
    if not _render_slots.acquire(timeout=RENDER_QUEUE_TIMEOUT_S):
        raise HTTPException(
            status_code=503,
            detail="Server busy; try again in a few seconds.",
            headers={"Retry-After": "5"},
        )
    try:
        yield
    finally:
        _render_slots.release()


def _too_large() -> HTTPException:
    return HTTPException(
        status_code=413, detail=f"Request larger than {MAX_REQUEST_BYTES // (1024 * 1024)} MB."
    )


class BodySizeLimitMiddleware:
    """Reject oversized request bodies while they stream in.

    Starlette spools multipart files to disk with no size cap before the
    handler runs, so a per-file check in the handler comes too late: a
    multi-GB POST fills the disk first. A declared Content-Length over the
    limit is refused up front; chunked bodies are counted as they arrive.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = -1
                if declared > self.max_bytes or declared < 0:
                    status = 413 if declared > self.max_bytes else 400
                    detail = _too_large().detail if status == 413 else "Bad Content-Length."
                    body = json.dumps({"detail": detail}).encode()
                    await send({
                        "type": "http.response.start",
                        "status": status,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"connection", b"close")],
                    })
                    await send({"type": "http.response.body", "body": body})
                    return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # Raised inside the route's body parsing, so FastAPI's
                    # exception handling turns it into a normal 413 response.
                    raise _too_large()
            return message

        await self.app(scope, limited_receive, send)


app = FastAPI(title="ASCII Magic")
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)


def _check_budget(value: int, limit: int, what: str, hint: str) -> None:
    if value > limit:
        raise HTTPException(
            status_code=400,
            detail=f"Output too large: {what} would be {value:,} (limit {limit:,}). {hint}",
        )


def _estimated_rows(img_w: int, img_h: int, cols: int, cell_w: int, cell_h: int) -> int:
    """Rows the image converters produce for `cols` (their aspect formula)."""
    return max(1, int((img_h / max(1, img_w)) * cols * (cell_w / cell_h)))
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif")


def _ival(o: dict[str, Any], key: str, default, lo: int, hi: int):
    """Clamped int from untrusted JSON; garbage/empty falls back to default."""
    v = o.get(key)
    if v in (None, ""):
        return default
    try:
        n = int(float(v))
    except (TypeError, ValueError, OverflowError):  # OverflowError: "inf"
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
    if not math.isfinite(n):
        return default
    return max(lo, min(hi, n))


_FALSY_STRINGS = {"", "0", "false", "off", "no"}


def _bool(o: dict[str, Any], key: str, default: bool = False) -> bool:
    """Bool from untrusted JSON; the string "false" must not read as True."""
    v = o.get(key, default)
    if isinstance(v, str):
        return v.strip().lower() not in _FALSY_STRINGS
    return bool(v)


def _choice(o: dict[str, Any], key: str, default: str, allowed: tuple[str, ...]) -> str:
    """Enum option from untrusted JSON; unknown values are a 400, not a 500
    deep in the renderer (or a silent fallback)."""
    v = o.get(key)
    if v in (None, ""):
        return default
    if v not in allowed:
        raise HTTPException(
            status_code=400, detail=f"Invalid {key}: {v!r} (expected one of {', '.join(allowed)})"
        )
    return v


CAPTION_STYLES = ("block", "small", "shadow", "box", "banner", "figlet")
CAPTION_POSITIONS = ("top", "bottom")
CAPTION_ALIGNS = ("left", "center", "right")
QUALITIES = ("fast", "balanced", "best")
ASCII_PRESETS = ("dense", "printable")


def _build_options(o: dict[str, Any], out_format: str) -> colorize_mod.Options:
    opt = colorize_mod.Options()
    opt.out_format = out_format
    opt.keep_top = _ival(o, "keep_top", 0, 0, 5000)
    opt.color_top = _bool(o, "color_top")

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
    if size.rows and size.cols:
        _check_budget(size.rows * size.cols, MAX_CELLS, "characters", "Lower the exact rows/cols.")

    h = opt.html
    h.font_size_px = _ival(o, "html_font_size", 12, 4, 64)
    h.line_height_px = _ival(o, "html_line_height", None, 4, 96)
    h.fill_spaces = _bool(o, "html_fill_spaces")

    if o.get("caption_text"):
        c = opt.caption
        c.text = str(o["caption_text"])[:500]
        c.position = _choice(o, "caption_pos", "bottom", CAPTION_POSITIONS)
        c.style = _choice(o, "caption_style", "block", CAPTION_STYLES)
        c.scale = _fval(o, "caption_scale", 0.6, 0.05, 1.0)
        c.cols = _ival(o, "caption_cols", None, 2, 500)
        c.rows = _ival(o, "caption_rows", None, 1, 200)
        c.gap = _ival(o, "caption_gap", 1, 0, 50)
        c.color = o.get("caption_color") or None
        c.align = _choice(o, "caption_align", "center", CAPTION_ALIGNS)

    m = opt.matrix
    m.enabled = _bool(o, "matrix")
    if m.enabled:
        if o.get("matrix_color"):
            m.tint = colorize_mod.parse_matrix_color(o["matrix_color"])
        m.top = _bool(o, "matrix_top")
        m.seed = _ival(o, "matrix_seed", None, 0, 2**31 - 1)
        m.gamma = _fval(o, "matrix_gamma", m.gamma, 0.1, 10.0)
        m.fg_min = _ival(o, "matrix_fg_min", m.fg_min, 0, 255)
        m.fg_max = _ival(o, "matrix_fg_max", m.fg_max, 0, 255)
        m.bg_min = _ival(o, "matrix_bg_min", m.bg_min, 0, 255)
        m.bg_max = _ival(o, "matrix_bg_max", m.bg_max, 0, 255)
        if o.get("matrix_chars"):
            m.chars = str(o["matrix_chars"])[:500]
        m.fill_spaces = _bool(o, "matrix_fill_spaces")
        m.use_mask = _bool(o, "matrix_mask")
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


def _video_from_upload(upload: Optional[UploadFile], o: dict[str, Any]):
    """Validate + spool the upload, convert to AsciiVideo. Returns
    (video, tmp_path); the caller must unlink tmp_path (the mp4 sink needs
    it alive for audio muxing)."""
    import os as os_mod
    import tempfile

    if upload is None:
        raise HTTPException(status_code=400, detail="No video uploaded.")
    suffix = os_mod.path.splitext(upload.filename or "")[1].lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type; expected one of {', '.join(VIDEO_SUFFIXES)}.",
        )

    chunks = []
    total = 0
    while True:
        chunk = upload.file.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_VIDEO_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Video larger than {MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(b"".join(chunks))
        tmp.close()
        try:
            from .video import video_to_ascii

            built = _build_options(o, "ansi")
            matrix = built.matrix if _bool(o, "matrix") else None
            caption = built.caption if o.get("caption_text") else None
            mode = o.get("video_mode") if o.get("video_mode") in ("braille", "glyph") else "braille"
            v = video_to_ascii(
                tmp.name,
                cols=_ival(o, "cols", 100, 10, 240),
                sample_fps=_fval(o, "video_fps", 8.0, 1.0, 30.0),
                max_frames=_ival(o, "video_max_frames", 60, 1, 120),
                rows=_ival(o, "video_rows", None, 1, 500),
                dither=_bool(o, "dither", True),
                threshold=_fval(o, "threshold", 0.5, 0.0, 1.0),
                gamma=_fval(o, "gamma", 1.0, 0.05, 10.0),
                autocontrast=_bool(o, "autocontrast"),
                invert=_bool(o, "invert"),
                mode=mode,
                quality=o.get("quality") if o.get("quality") in ("fast", "balanced", "best") else "balanced",
                matrix=matrix,
                caption=caption,
            )
        except (RuntimeError, ValueError, OSError) as e:
            raise HTTPException(status_code=400, detail=f"Could not read the video: {e}")
    except Exception:
        os_mod.unlink(tmp.name)
        raise
    return v, tmp.name


def _render_video(upload: Optional[UploadFile], o: dict[str, Any], t0: float) -> dict[str, Any]:
    import base64 as b64mod
    import os as os_mod

    v, tmp_path = _video_from_upload(upload, o)
    os_mod.unlink(tmp_path)  # the JSON response doesn't need the source again

    from .greet import FRAME_SEP

    ansi_frames = v.frames_ansi()
    frames_text = json.dumps({"fps": v.fps, "loops": 1}) + "\n" + FRAME_SEP.join(ansi_frames)
    gif_b64 = b64mod.b64encode(v.to_gif_bytes()).decode("ascii")
    first_lines, _ = v.frames[0]

    preview = (
        "<!doctype html><html><body style=\"margin:0;background:#000;"
        "display:grid;place-items:start center\">"
        f'<img style="max-width:100%" alt="ASCII video" '
        f'src="data:image/gif;base64,{gif_b64}"></body></html>'
    )
    return {
        "ascii": "\n".join(first_lines),
        "art": {
            "cols": max((len(ln) for ln in first_lines), default=0),
            "rows": len(first_lines),
            # the GIF bakes the caption strip into every frame
            "cap_lines": len(v.caption.lines) if v.caption else 0,
            "cap_gap": v.caption.gap if v.caption else 0,
            "cap_pos": v.caption.position if v.caption else "bottom",
            "cap_style": (o.get("caption_style", "block") if v.caption else None),
            "cap_cols": max((len(ln.strip()) for ln in v.caption.lines if ln.strip()), default=0) if v.caption else 0,
            "cap_x": min((len(ln) - len(ln.lstrip()) for ln in v.caption.lines if ln.strip()), default=0) if v.caption else 0,
        },
        "ansi": ansi_frames[0],
        "html": preview,
        "gif_b64": gif_b64,
        "frames_text": frames_text,
        "video": {"frames": len(v.frames), "fps": round(v.fps, 2)},
        "seed": None,
        "warning": None,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
    }


@app.post("/api/render/mp4")
def render_mp4(
    image: Optional[UploadFile] = File(None),
    options: str = Form("{}"),
):
    """Encode-on-demand mp4 (with the source's audio). The GUI calls this
    only when the user clicks the .mp4 download, so previews stay fast."""
    with _render_slot():
        return _render_mp4(image, options)


def _render_mp4(image: Optional[UploadFile], options: str):
    import os as os_mod
    import tempfile

    from fastapi.responses import Response

    try:
        o: dict[str, Any] = json.loads(options)
        if not isinstance(o, dict):
            raise ValueError("options must be a JSON object")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad options JSON: {e}")

    v, src_path = _video_from_upload(image, o)
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out.close()
    try:
        try:
            v.write_mp4(out.name, audio_source=src_path)
        except (RuntimeError, OSError) as e:
            raise HTTPException(status_code=400, detail=f"Could not encode mp4: {e}")
        data = open(out.name, "rb").read()
    finally:
        os_mod.unlink(src_path)
        os_mod.unlink(out.name)

    return Response(
        content=data,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="ascii-video.mp4"'},
    )


@app.post("/api/render")
def render(
    # Plain def: Starlette runs sync handlers in a threadpool, so a slow
    # NumPy/PIL render doesn't block the event loop for other requests.
    image: Optional[UploadFile] = File(None),
    options: str = Form("{}"),
) -> dict[str, Any]:
    with _render_slot():
        return _render(image, options)


def _render(image: Optional[UploadFile], options: str) -> dict[str, Any]:
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

    if o.get("source") == "video":
        return _render_video(image, o, t0)

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
            ctx.source_image = flatten_alpha(img).convert("RGB")  # full decode happens here
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    source = o.get("source", "image")
    if source == "text":
        text = o.get("text") or ""
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail="text must be a string.")
        text = text.strip("\n")
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
        mode = o.get("mode", "braille")
        cols = _ival(o, "cols", 120, 1, 500)
        cell_w = _ival(o, "cell_w", 8, 1, 32)
        cell_h = _ival(o, "cell_h", 16, 1, 64)
        cw, ch = (cell_w, cell_h) if mode == "glyph" else (2, 4)  # braille: 2x4 dots
        cells = cols * _estimated_rows(*ctx.source_image.size, cols, cw, ch)
        _check_budget(cells, MAX_CELLS, "characters", "Lower the column count.")
        if mode == "glyph":
            _check_budget(
                cells * cell_w * cell_h, MAX_GLYPH_PIXELS, "glyph-matching pixels",
                "Lower the column count or the cell size.",
            )
        try:
            image_to_ascii(
                ctx,
                mode=mode,
                cols=cols,
                cell_w=cell_w,
                cell_h=cell_h,
                quality=_choice(o, "quality", "balanced", QUALITIES),
                topk=_ival(o, "topk", 24, 1, 200),
                ascii_preset=_choice(o, "ascii_preset", "dense", ASCII_PRESETS),
                unicode_mode=o.get("unicode_mode", "off"),
                autocontrast=_bool(o, "autocontrast"),
                gamma=_fval(o, "gamma", 1.0, 0.05, 10.0),
                invert=_bool(o, "invert"),
                threshold=_fval(o, "threshold", 0.5, 0.0, 1.0),
                dither=_bool(o, "dither"),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

    ascii_text = ctx.ascii_text or ""
    seed: Optional[int] = None
    do_colorize = _bool(o, "colorize", True)
    do_animate = _bool(o, "animate")
    if do_animate:
        o = {**o, "matrix": True}

    # Raw-text view/download includes the caption (uncolored).
    ascii_display = ascii_text
    if o.get("caption_text"):
        from .text_to_ascii import compose_caption

        ascii_display = compose_caption(
            ascii_text,
            str(o["caption_text"])[:500],
            position=_choice(o, "caption_pos", "bottom", CAPTION_POSITIONS),
            style=_choice(o, "caption_style", "block", CAPTION_STYLES),
            scale=_fval(o, "caption_scale", 0.6, 0.05, 1.0),
            gap=_ival(o, "caption_gap", 1, 0, 50),
            align=_choice(o, "caption_align", "center", CAPTION_ALIGNS),
        )

    # Colorizing/animating needs a reference image; box/banner text styles
    # do not render one, so fall back to plain output instead of erroring.
    can_colorize = ctx.source_image is not None or ctx.rendered_text_image is not None
    if (do_colorize or do_animate) and not can_colorize:
        do_colorize = do_animate = False
        warning = "No reference image for colorizing this style; returning plain ASCII."

    # ANSI, HTML, and animation are rendered separately, so a random matrix
    # seed would diverge between them — pin one and echo it back.
    if _bool(o, "matrix") and (do_colorize or do_animate):
        # Same parse as _build_options, so the echoed seed is the one used.
        seed = _ival(o, "matrix_seed", None, 0, 2**31 - 1)
        if seed is None:
            seed = random.randrange(2**31)
        o = {**o, "matrix_seed": seed}

    if do_colorize:
        ansi = colorize(ctx, opt=_build_options(o, "ansi"))
        html_doc = colorize(ctx, opt=_build_options(o, "html"))
    else:
        # Exact/max sizing must work with colorize off too (the GUI's resize
        # handles set it); colorize_ascii_text applies it internally on the
        # colorized path.
        size = _build_options(o, "ansi").size
        if any((size.rows, size.cols, size.max_rows, size.max_cols)):
            lines = ascii_display.splitlines()
            target_h = colorize_mod.compute_target_art_height(size.max_rows, 0, len(lines))
            ascii_display = "\n".join(colorize_mod.scale_art_block(lines, target_h, size))
        ansi = ascii_display + "\n"
        html_doc = _plain_html(ascii_display, o)

    gif_b64: Optional[str] = None
    if do_animate:
        from .animate import AnimationOptions

        anim_opt = AnimationOptions(
            frames=_ival(o, "anim_frames", 60, 1, 240),
            fps=_fval(o, "anim_fps", 12.0, 1.0, 30.0),
            tail=_fval(o, "anim_tail", 6.0, 0.5, 40.0),
            reveal=_bool(o, "anim_reveal"),
        )
        art = (ctx.ascii_text or "").splitlines()
        _check_budget(
            len(art) * max((len(ln) for ln in art), default=0) * anim_opt.frames,
            MAX_ANIM_CELL_FRAMES, "characters x frames",
            "Lower the column count or the number of frames.",
        )
        built = _build_options(o, "ansi")
        animation = pipeline_animate(ctx, matrix=built.matrix, anim=anim_opt, caption=built.caption)
        html_doc = animation.to_html(font_size_px=_ival(o, "html_font_size", 12, 4, 64))
        gif_b64 = base64.b64encode(animation.to_gif_bytes()).decode("ascii")

    # Post-scaling art grid dimensions (pre-caption) — the GUI's resize
    # handles need them to convert pixel drags into cols/rows.
    art_lines = (ctx.ascii_text or "").splitlines()
    size = _build_options(o, "ansi").size
    if art_lines and any((size.rows, size.cols, size.max_rows, size.max_cols)):
        th = colorize_mod.compute_target_art_height(size.max_rows, 0, len(art_lines))
        art_lines = colorize_mod.scale_art_block(art_lines, th, size)
    art_dims = {
        "cols": max((len(ln) for ln in art_lines), default=0),
        "rows": len(art_lines),
        "cap_lines": 0,
        "cap_gap": 0,
        "cap_pos": "bottom",
        "cap_style": None,
    }
    # Caption rows share the art's rendered block; report how many so the
    # GUI's resize ring can exclude them. The animation player renders its
    # caption in a separate element, so nothing to exclude there.
    cap = _build_options(o, "ansi").caption
    if cap.text and not do_animate and art_dims["cols"]:
        try:
            from .text_to_ascii import caption_lines as _caption_lines

            cl = _caption_lines(
                cap.text, art_dims["cols"], style=cap.style, scale=cap.scale, align=cap.align,
                cols=cap.cols, rows=cap.rows,
            )
            art_dims["cap_lines"] = len(cl)
            art_dims["cap_gap"] = max(0, int(cap.gap))
            art_dims["cap_pos"] = cap.position
            art_dims["cap_style"] = cap.style
            inked = [ln for ln in cl if ln.strip()]
            art_dims["cap_cols"] = max((len(ln.strip()) for ln in inked), default=0)
            art_dims["cap_x"] = min((len(ln) - len(ln.lstrip()) for ln in inked), default=0)
        except Exception:
            pass  # caption metrics are best-effort; the ring just wraps everything

    return {
        "ascii": ascii_display,
        "art": art_dims,
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
