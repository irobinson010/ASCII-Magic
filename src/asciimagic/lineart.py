"""Line art: trace a picture's outlines instead of shading it.

Shading maps brightness to ink, which at terminal resolution turns most
photos into soft blobs. Outlines often read better, above all for logos,
drawings, faces and anything with clear shapes. This turns a picture into
dark lines on white; the normal converters then draw it like any other
image, and colors (``--color``) still come from the original picture.

Edges are found at about twice the output resolution, not on the full-size
photo: at full size, texture (fur, fabric, sensor noise) produces thousands
of tiny edges that average into grey once shrunk to 80 characters. At the
working size only structure big enough to show survives.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# Working resolution per output pixel (see the module docstring).
OVERSAMPLE = 2
# Bounds memory (several float arrays of this size) for huge or very tall
# uploads; still far more pixels than any output grid uses.
MAX_WORK_PIXELS = 4_000_000


def line_art(img: Image.Image, out_width: int, detail: float = 0.5) -> Image.Image:
    """Dark outlines on white ("L" mode), sized for an output ``out_width``
    pixels wide. ``detail`` 0..1: low keeps only the strongest outlines,
    high keeps finer lines and some texture."""
    if not 0.0 <= detail <= 1.0:
        raise ValueError(f"line detail must be between 0 and 1, got {detail}")
    if out_width < 1:
        raise ValueError("out_width must be >= 1")
    g = img.convert("L")
    w = min(g.width, max(8, out_width * OVERSAMPLE))
    h = max(1, round(g.height * w / g.width))
    if w * h > MAX_WORK_PIXELS:
        k = (MAX_WORK_PIXELS / (w * h)) ** 0.5
        w, h = max(1, int(w * k)), max(1, int(h * k))
    if (w, h) != g.size:
        g = g.resize((w, h), Image.Resampling.LANCZOS)

    # Smooth first: Sobel on raw pixels mostly finds noise.
    sigma = 1.6 - 1.2 * detail
    a = np.asarray(g.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32) / 255.0
    p = np.pad(a, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    mag = np.hypot(gx, gy)

    # Scale by a high percentile, not the max, so one hard edge (a logo on a
    # plain background) doesn't make every other edge look faint.
    ref = float(np.percentile(mag, 98))
    if ref < 1e-3:  # a flat picture: no outlines at all
        return Image.new("L", (w, h), 255)
    m = np.clip(mag / ref, 0.0, 1.0)
    # Drop the weakest responses (texture, gradients); keep the rest graded.
    floor = 0.45 - 0.35 * detail
    m = np.clip((m - floor) / (1.0 - floor), 0.0, 1.0)
    out = Image.fromarray(((1.0 - m) * 255).round().astype(np.uint8), "L")
    # A 1 px line at this size is half a pixel once the converter halves it,
    # which lands on the ink threshold and breaks lines into dashes. Widen
    # the dark lines by a pixel so they survive the downscale whole.
    return out.filter(ImageFilter.MinFilter(3))
