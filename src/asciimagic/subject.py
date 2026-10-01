"""Subject focus and background replacement for image conversion.

A busy background competes with the subject for characters. These controls
decide what the art is about before it is converted::

    ascii-magic image cat.jpg --focus 0.2,0.1,0.6,0.8      # crop to a region first
    ascii-magic image cat.jpg --background remove          # blank everything but the subject
    ascii-magic image cat.jpg --background blur --zoom-subject --enhance-subject
    ascii-magic image me.jpg --background image --bg-image beach.jpg -c 120 --color
    ascii-magic subject install best                       # the larger, cleaner model

The subject is found by an offline segmentation model run with ONNX Runtime
(the optional ``[subject]`` extra). Models download once, are checked
against a known SHA-256, and run locally:

- ``fast``: U^2-Net-p, 4.6 MB, a fraction of a second per image.
- ``best``: IS-Net (general use), 179 MB, about a second; cleaner edges.

No model is right every time (a dark close-up of a dark cat fooled both in
testing), so the mask can be adjusted (threshold, grow/shrink, feather,
invert) or replaced with your own mask image, and the focus box needs no
model at all.
"""

from __future__ import annotations

import hashlib
import io
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

RGB = Tuple[int, int, int]
_RELEASES = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"


@dataclass(frozen=True)
class ModelSpec:
    file: str
    sha256: str
    bytes: int
    input_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    about: str

    @property
    def url(self) -> str:
        return _RELEASES + self.file


MODELS = {
    "fast": ModelSpec(
        "u2netp.onnx", "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        4_574_861, 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
        "U^2-Net-p, 4.6 MB: quick, good on clear subjects",
    ),
    "best": ModelSpec(
        "isnet-general-use.onnx", "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        178_648_008, 1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0),
        "IS-Net, 179 MB: slower, cleaner edges",
    ),
}
BACKGROUNDS = ("keep", "remove", "blur", "fade", "color", "image")

_install_lock = threading.Lock()


class SubjectError(RuntimeError):
    """Subject detection can't run: missing extra, missing model, or bad input."""


# ---- models on disk ----


def models_dir() -> Path:
    from .translate import models_dir as base

    return base() / "subject"


def model_path(name: str) -> Path:
    return models_dir() / _spec(name).file


def _spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise SubjectError(f"unknown subject model {name!r}; choose from {', '.join(MODELS)}") from None


def installed() -> List[str]:
    return [n for n, s in MODELS.items() if (models_dir() / s.file).is_file()]


def engine_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def install(name: str, progress=None) -> Path:
    """Download a model once, verifying its size and SHA-256."""
    from .translate import _urlopen

    spec = _spec(name)
    with _install_lock:
        dest = models_dir() / spec.file
        if dest.is_file():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        digest = hashlib.sha256()
        done = 0
        try:
            with _urlopen(spec.url, 60) as resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    done += len(chunk)
                    if done > spec.bytes:
                        raise SubjectError("model download is larger than expected; refusing it")
                    digest.update(chunk)
                    out.write(chunk)
                    if progress:
                        progress(done, spec.bytes)
            if done != spec.bytes or digest.hexdigest() != spec.sha256:
                raise SubjectError("downloaded model failed its checksum; try again")
            tmp.replace(dest)
        finally:
            if tmp.exists():
                tmp.unlink()
        _session.cache_clear()
        return dest


def remove(name: str) -> bool:
    p = model_path(name)
    if p.is_file():
        p.unlink()
        _session.cache_clear()
        return True
    return False


# ---- detection ----


@lru_cache(maxsize=2)
def _session(name: str):
    try:
        import onnxruntime as ort
    except ImportError:
        raise SubjectError(
            'Subject detection needs the [subject] extra:\n'
            '    pip install "ascii-magic-tools[subject]"   (or: uv sync --extra subject)'
        ) from None
    path = model_path(name)
    if not path.is_file():
        raise SubjectError(f"the {name!r} subject model isn't installed. Run: ascii-magic subject install {name}")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def ensure_model(name: str, allow_download: bool, progress=None) -> None:
    if not engine_available():
        _session(name)  # raises the install hint
    if not model_path(name).is_file():
        if not allow_download:
            raise SubjectError(f"the {name!r} subject model isn't installed. Run: ascii-magic subject install {name}")
        install(name, progress=progress)


_mask_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_mask_lock = threading.Lock()


def _run_model(img: Image.Image, name: str) -> np.ndarray:
    spec = _spec(name)
    sess = _session(name)
    size = spec.input_size
    x = np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    x = x / max(float(x.max()), 1e-6)
    x = (x - np.array(spec.mean, dtype=np.float32)) / np.array(spec.std, dtype=np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    pred = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-6)
    out = Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.float32) / 255.0


def detect(img: Image.Image, model: str = "fast", allow_download: bool = True) -> np.ndarray:
    """Subject mask for `img`: float32 (H, W), 1 = subject. Cached per image,
    so re-rendering with other settings doesn't run the model again."""
    ensure_model(model, allow_download)
    key = (model, img.size, hashlib.blake2b(img.convert("RGB").tobytes(), digest_size=16).hexdigest())
    with _mask_lock:
        hit = _mask_cache.get(key)
        if hit is not None:
            _mask_cache.move_to_end(key)
            return hit
    mask = _run_model(img, model)
    with _mask_lock:
        _mask_cache[key] = mask
        while len(_mask_cache) > 8:
            _mask_cache.popitem(last=False)
    return mask


# ---- focus ----


@dataclass
class FocusOptions:
    box: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h as fractions of the picture
    subject: Optional[str] = None       # model that finds the subject (None: "fast" when a mask is needed)
    mask: Optional[Image.Image] = None  # your own mask (white = subject), instead of detection
    background: str = "keep"
    bg_color: RGB = (0, 0, 0)
    bg_image: Optional[Image.Image] = None
    blur: float = 0.025                 # blur radius, as a share of the picture's size
    fade: float = 0.75                  # how far a faded background moves toward blank
    zoom: bool = False                  # crop to the subject (plus `margin`)
    margin: float = 0.06
    enhance: bool = False               # set contrast from the subject's own tones
    threshold: float = 0.5
    grow: float = 0.0                   # grow (+) or shrink (-) the mask, share of the picture's size
    feather: float = 0.0                # soften the mask edge, share of the picture's size
    invert_mask: bool = False           # focus on the background instead
    allow_download: bool = True

    def needs_mask(self) -> bool:
        return self.background != "keep" or self.zoom or self.enhance or self.mask is not None

    def validate(self) -> None:
        if self.background not in BACKGROUNDS:
            raise SubjectError(f"unknown background {self.background!r}; choose from {', '.join(BACKGROUNDS)}")
        if self.background == "image" and self.bg_image is None:
            raise SubjectError("background 'image' needs a background picture (--bg-image)")
        if self.subject is not None:
            _spec(self.subject)
        if self.box is not None:
            x, y, w, h = self.box
            if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
                raise SubjectError("focus box must be fractions: 0 <= x, y < 1 and 0 < width, height <= 1")
        for name, v, lo, hi in (("threshold", self.threshold, 0.01, 0.99), ("grow", self.grow, -0.2, 0.2),
                                ("feather", self.feather, 0.0, 0.2), ("blur", self.blur, 0.0, 0.2),
                                ("fade", self.fade, 0.0, 1.0), ("margin", self.margin, 0.0, 0.5)):
            if not lo <= v <= hi:
                raise SubjectError(f"{name} must be between {lo} and {hi}")


@dataclass
class FocusResult:
    image: Image.Image                 # what to convert (and colorize from)
    mask: Optional[np.ndarray]         # subject mask over `image`, 0..1
    blank_background: bool             # convert background cells to spaces
    threshold: float
    preview: Optional[np.ndarray] = None  # mask over the original, unrotated picture


def _box_pixels(box, size) -> Tuple[int, int, int, int]:
    W, H = size
    x, y, w, h = box
    left, top = int(round(x * W)), int(round(y * H))
    right, bottom = min(W, int(round((x + w) * W))), min(H, int(round((y + h) * H)))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _rank(mask: np.ndarray, px: int, grow: bool) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8))
    f = ImageFilter.MaxFilter(3) if grow else ImageFilter.MinFilter(3)
    for _ in range(px):
        img = img.filter(f)
    return np.asarray(img, dtype=np.float32) / 255.0


def refine_mask(mask: np.ndarray, opt: FocusOptions) -> np.ndarray:
    h, w = mask.shape
    side = max(w, h)
    if opt.invert_mask:
        mask = 1.0 - mask
    if opt.grow:
        # Work on a smaller copy so large grow values stay fast.
        scale = min(1.0, 256 / side)
        small = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        px = max(1, round(abs(opt.grow) * side * scale))
        small = _rank(small, px, opt.grow > 0)
        mask = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR),
                          dtype=np.float32) / 255.0
    if opt.feather:
        img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(opt.feather * side))
        mask = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(mask, 0.0, 1.0)


def _no_ink(invert: bool) -> RGB:
    # Converters treat dark as ink; Invert flips that.
    return (0, 0, 0) if invert else (255, 255, 255)


def _composite(fg: Image.Image, bg: Image.Image, mask: np.ndarray) -> Image.Image:
    m = mask[..., None]
    out = np.asarray(fg, dtype=np.float32) * m + np.asarray(bg.convert("RGB"), dtype=np.float32) * (1 - m)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _enhance(img: Image.Image, mask: np.ndarray, threshold: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sel = lum[mask >= threshold]
    if sel.size < 16:
        return img
    lo, hi = np.percentile(sel, [1, 99])
    if hi - lo < 8:
        return img
    out = (arr - lo) * (255.0 / (hi - lo))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def apply_focus(img: Image.Image, opt: FocusOptions, rotate: int = 0, invert: bool = False) -> FocusResult:
    """Crop, find the subject, and treat the background, in that order.
    `img` is the picture as uploaded (the focus box is in its coordinates);
    `rotate` (clockwise degrees) is applied after the crop."""
    from .image_to_ascii import rotate_cw

    opt.validate()
    img = img.convert("RGB")
    full_size = img.size
    box_px = _box_pixels(opt.box, full_size) if opt.box else (0, 0, *full_size)
    user_mask = None
    if opt.mask is not None:
        user_mask = opt.mask.convert("L").resize(full_size, Image.Resampling.BILINEAR).crop(box_px)
    img = rotate_cw(img.crop(box_px), rotate)

    mask = None
    if opt.needs_mask():
        if user_mask is not None:
            mask = np.asarray(rotate_cw(user_mask, rotate), dtype=np.float32) / 255.0
        else:
            mask = detect(img, opt.subject or "fast", allow_download=opt.allow_download)
        mask = refine_mask(mask, opt)

    # The mask on the original picture, for showing what was found.
    preview = None
    if mask is not None:
        unrot = Image.fromarray((mask * 255).astype(np.uint8))
        if rotate % 360:
            unrot = unrot.rotate(rotate, expand=True)  # undo the clockwise turn
        full = Image.new("L", full_size, 0)
        full.paste(unrot.resize((box_px[2] - box_px[0], box_px[3] - box_px[1])), box_px[:2])
        preview = np.asarray(full, dtype=np.float32) / 255.0

    if opt.zoom and mask is not None:
        ys, xs = np.nonzero(mask >= opt.threshold)
        if xs.size:
            w, h = img.size
            pad = round(opt.margin * max(np.ptp(xs) + 1, np.ptp(ys) + 1))
            crop = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + 1 + pad), min(h, ys.max() + 1 + pad))
            img = img.crop(crop)
            mask = mask[crop[1]:crop[3], crop[0]:crop[2]]

    if opt.enhance and mask is not None:
        img = _enhance(img, mask, opt.threshold)

    bg = None
    if opt.background == "remove":
        bg = Image.new("RGB", img.size, _no_ink(invert))
    elif opt.background == "blur":
        bg = img.filter(ImageFilter.GaussianBlur(max(1.0, opt.blur * max(img.size))))
    elif opt.background == "fade":
        blank = Image.new("RGB", img.size, _no_ink(invert))
        bg = Image.blend(img, blank, opt.fade)
    elif opt.background == "color":
        bg = Image.new("RGB", img.size, tuple(opt.bg_color))
    elif opt.background == "image":
        bg = ImageOps.fit(opt.bg_image.convert("RGB"), img.size, Image.Resampling.LANCZOS)
    if bg is not None:
        img = _composite(img, bg, mask)

    return FocusResult(img, mask, opt.background == "remove", opt.threshold, preview)


def blank_background(art: str, mask: np.ndarray, threshold: float = 0.5) -> str:
    """Replace characters whose cell is mostly background with spaces, so a
    removed background is truly empty whatever the conversion mode."""
    lines = art.split("\n")
    rows, cols = len(lines), max((len(ln) for ln in lines), default=0)
    if not rows or not cols:
        return art
    m = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).resize((cols, rows), Image.Resampling.BOX)
    keep = np.asarray(m, dtype=np.float32) / 255.0 >= threshold
    return "\n".join(
        "".join(ch if x < cols and keep[y, x] else " " for x, ch in enumerate(ln))
        for y, ln in enumerate(lines)
    )


def preview_png(mask: np.ndarray, max_side: int = 320) -> bytes:
    img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---- command line ----


def _fractions(value: str) -> Tuple[float, float, float, float]:
    import argparse

    try:
        parts = tuple(float(p) for p in value.split(","))
    except ValueError:
        parts = ()
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected X,Y,W,H as fractions of the picture, e.g. 0.25,0.1,0.5,0.8")
    return parts  # type: ignore[return-value]


def add_focus_args(parser) -> None:
    g = parser.add_argument_group("focus & background")
    g.add_argument("--focus", type=_fractions, default=None, metavar="X,Y,W,H",
                   help="Convert only this part of the picture (fractions, e.g. 0.25,0.1,0.5,0.8)")
    g.add_argument("--subject", nargs="?", const="fast", choices=list(MODELS), default=None,
                   help="Find the subject with a model: fast (4.6 MB, default) or best (179 MB). "
                        "Implied by --background/--zoom-subject/--enhance-subject")
    g.add_argument("--mask", default=None, metavar="PATH",
                   help="Your own subject mask (white = subject) instead of detection")
    g.add_argument("--background", choices=BACKGROUNDS, default=None,
                   help="What to do behind the subject: keep, remove (blank), blur, fade, color, or image")
    g.add_argument("--bg-color", default=None, metavar="COLOR", help="Background color (implies --background color)")
    g.add_argument("--bg-image", default=None, metavar="PATH", help="Replacement background picture (implies --background image)")
    g.add_argument("--zoom-subject", action="store_true", help="Crop to the subject so the art spends its characters there")
    g.add_argument("--enhance-subject", action="store_true", help="Set contrast from the subject's own tones")
    g.add_argument("--mask-threshold", type=float, default=0.5, metavar="0..1", help="Subject cutoff (default 0.5)")
    g.add_argument("--mask-grow", type=float, default=0.0, metavar="F",
                   help="Grow (+) or shrink (-) the subject, as a share of the picture size (e.g. 0.02)")
    g.add_argument("--mask-feather", type=float, default=0.0, metavar="F", help="Soften the subject's edge (e.g. 0.01)")
    g.add_argument("--mask-invert", action="store_true", help="Treat the background as the subject")


def focus_from_args(args) -> Optional[FocusOptions]:
    """FocusOptions from add_focus_args flags, or None when none were used."""
    background = args.background or ("image" if args.bg_image else "color" if args.bg_color else "keep")
    wants = (args.focus or args.subject or args.mask or background != "keep" or args.zoom_subject
             or args.enhance_subject)
    if not wants:
        return None
    color = (0, 0, 0)
    if args.bg_color:
        from .colorize_ascii import parse_matrix_color

        try:
            color = parse_matrix_color(args.bg_color)
        except ValueError as e:
            raise SubjectError(str(e)) from None
    try:
        mask = Image.open(args.mask) if args.mask else None
        bg_image = Image.open(args.bg_image) if args.bg_image else None
    except OSError as e:
        raise SubjectError(f"couldn't open {e.filename}: {e.strerror or e}") from None
    return FocusOptions(
        box=args.focus, subject=args.subject, mask=mask, background=background,
        bg_color=color, bg_image=bg_image, zoom=args.zoom_subject, enhance=args.enhance_subject,
        threshold=args.mask_threshold, grow=args.mask_grow, feather=args.mask_feather,
        invert_mask=args.mask_invert,
    )


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="ascii-magic subject",
                                 description="Manage the subject-detection models used by --background and friends.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("install", help="Download a model (fast: 4.6 MB, best: 179 MB)")
    p.add_argument("model", choices=list(MODELS))
    p = sub.add_parser("remove", help="Delete a downloaded model")
    p.add_argument("model", choices=list(MODELS))
    sub.add_parser("list", help="Show models and whether they're installed")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "install":
            def progress(done, total):
                print(f"\rDownloading {a.model}: {done * 100 // total:3d}% of {total / 1e6:.1f} MB",
                      end="", file=sys.stderr, flush=True)

            path = install(a.model, progress=progress)
            print(f"\nInstalled {a.model} in {path}", file=sys.stderr)
            if not engine_available():
                print('Note: detection also needs: pip install "ascii-magic-tools[subject]"', file=sys.stderr)
            return 0
        if a.cmd == "remove":
            ok = remove(a.model)
            print(f"Removed {a.model}" if ok else f"{a.model} is not installed")
            return 0 if ok else 1
        have = set(installed())
        for name, spec in MODELS.items():
            print(f"  {name:<5} {'installed' if name in have else 'not installed':<14} {spec.about}")
        print(f"Models folder: {models_dir()}")
        if not engine_available():
            print('Detection needs: pip install "ascii-magic-tools[subject]"')
        return 0
    except SubjectError as e:
        print(f"ascii-magic subject: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"ascii-magic subject: download failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
