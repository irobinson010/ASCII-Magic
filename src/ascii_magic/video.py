"""Video → colorized ASCII animation.

Reads any format ffmpeg understands (mp4, webm, mov, mkv, avi, gif, ...)
via the optional ``[video]`` extra (imageio + imageio-ffmpeg), samples
frames at a target fps, converts each through the braille pipeline, and
colorizes from the frame itself.

Outputs: animated GIF, a ``.frames`` file (for ``ascii-magic greet`` /
``play``), or live terminal playback.
"""

from __future__ import annotations

import argparse
import io
import sys
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .colorize_ascii import colorize_lines_ansi
from .image_to_ascii import find_default_mono_font, image_to_braille_from_image

# Braille blank: no ink to draw
_BLANKS = (" ", "⠀")


def _require_imageio():
    try:
        import imageio.v2 as iio
    except ImportError:
        raise SystemExit(
            "Video support needs the [video] extra:\n"
            '    pip install "ascii-magic-tools[video]"   (or: uv sync --extra video)'
        )
    return iio


def read_video_frames(
    path: str,
    sample_fps: float = 10.0,
    max_frames: int = 300,
) -> Tuple[List[Image.Image], float]:
    """Sample video frames as PIL images. Returns (frames, output_fps)."""
    iio = _require_imageio()
    reader = iio.get_reader(path)
    meta = reader.get_meta_data()

    src_fps = meta.get("fps")
    if not src_fps:
        # GIFs report per-frame duration (ms) instead of fps
        duration = meta.get("duration") or 100
        src_fps = 1000.0 / duration if duration else 10.0

    step = max(1, round(src_fps / max(0.1, sample_fps)))
    out_fps = src_fps / step

    frames: List[Image.Image] = []
    for i, arr in enumerate(reader):
        if i % step:
            continue
        if len(frames) >= max_frames:
            break
        arr = np.asarray(arr)
        if arr.ndim == 2:
            img = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            img = Image.fromarray(arr[:, :, :3], mode="RGB")
        frames.append(img)
    reader.close()

    if not frames:
        raise SystemExit(f"No frames could be read from {path}")
    return frames, out_fps


class AsciiVideo:
    """Sampled video frames converted to ASCII, plus the sinks."""

    def __init__(self, frames: List[Tuple[List[str], Image.Image]], fps: float):
        self.frames = frames  # per frame: (ascii lines, source frame image)
        self.fps = fps

    def frames_ansi(self) -> List[str]:
        return [
            "\n".join(colorize_lines_ansi(lines, img, color_spaces=False))
            for lines, img in self.frames
        ]

    def to_gif_bytes(self, font_path: Optional[str] = None, font_size: int = 14) -> bytes:
        font_path = font_path or find_default_mono_font()
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
            ascent, descent = font.getmetrics()
            cell_h = ascent + descent
            cell_w = max(1, round(font.getlength("M")))
        else:
            font = ImageFont.load_default()
            cell_w, cell_h = 7, 13

        cache: dict[str, np.ndarray] = {}

        def glyph_alpha(ch: str) -> np.ndarray:
            a = cache.get(ch)
            if a is None:
                img = Image.new("L", (cell_w, cell_h), 0)
                ImageDraw.Draw(img).text((0, 0), ch, fill=255, font=font)
                a = np.asarray(img, dtype=np.float32) / 255.0
                cache[ch] = a
            return a

        images = []
        for lines, frame_img in self.frames:
            h = len(lines)
            w = max(len(ln) for ln in lines)
            grid = [ln.ljust(w) for ln in lines]
            small = frame_img.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
            px = small.load()

            canvas = np.zeros((h * cell_h, w * cell_w, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    ch = grid[y][x]
                    if ch in _BLANKS:
                        continue
                    a = glyph_alpha(ch)
                    r, g, b = px[x, y]
                    block = canvas[y * cell_h:(y + 1) * cell_h, x * cell_w:(x + 1) * cell_w]
                    block[:, :, 0] = (a * r).astype(np.uint8)
                    block[:, :, 1] = (a * g).astype(np.uint8)
                    block[:, :, 2] = (a * b).astype(np.uint8)
            images.append(Image.fromarray(canvas))

        buf = io.BytesIO()
        images[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=max(20, round(1000 / self.fps)),
            loop=0,
        )
        return buf.getvalue()

    def play(self, loops: int = 1) -> None:
        from .greet import play_frames

        play_frames(self.frames_ansi(), self.fps, loops)


def video_to_ascii(
    path: str,
    cols: int = 100,
    sample_fps: float = 10.0,
    max_frames: int = 300,
    dither: bool = True,
    threshold: float = 0.5,
    gamma: float = 1.0,
    autocontrast: bool = False,
    invert: bool = False,
) -> AsciiVideo:
    frames, out_fps = read_video_frames(path, sample_fps=sample_fps, max_frames=max_frames)
    converted = []
    for img in frames:
        art = image_to_braille_from_image(
            img,
            cols=cols,
            autocontrast=autocontrast,
            gamma=gamma,
            invert=invert,
            threshold=threshold,
            dither=dither,
        )
        converted.append((art.splitlines(), img))
    return AsciiVideo(converted, out_fps)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ascii-magic video",
        description="Convert a video into a colorized ASCII animation "
        "(any format ffmpeg reads: mp4, webm, mov, mkv, avi, gif, ...).",
    )
    ap.add_argument("input", help="Video file")
    ap.add_argument("out", nargs="?", default=None,
                    help="Output: .gif or .frames (omit to play in the terminal)")
    ap.add_argument("-c", "--cols", type=int, default=100, help="Width in characters")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Target sample/playback fps (default: 10)")
    ap.add_argument("--max-frames", type=int, default=300,
                    help="Cap on sampled frames (default: 300)")
    ap.add_argument("--no-dither", action="store_true",
                    help="Disable Floyd-Steinberg dithering")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--autocontrast", action="store_true")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--loops", type=int, default=1,
                    help="Terminal playback repeats (0 = until Ctrl-C)")
    ap.add_argument("--font-size", type=int, default=14, help="GIF glyph size")
    args = ap.parse_args(argv)

    if args.out and not args.out.lower().endswith((".gif", ".frames")):
        raise SystemExit("Output must be .gif or .frames (or omitted for terminal playback)")

    video = video_to_ascii(
        args.input,
        cols=args.cols,
        sample_fps=args.fps,
        max_frames=args.max_frames,
        dither=not args.no_dither,
        threshold=args.threshold,
        gamma=args.gamma,
        autocontrast=args.autocontrast,
        invert=args.invert,
    )

    if args.out is None:
        video.play(loops=args.loops)
    elif args.out.lower().endswith(".gif"):
        with open(args.out, "wb") as f:
            f.write(video.to_gif_bytes(font_size=args.font_size))
        print(f"Wrote {args.out} ({len(video.frames)} frames @ {video.fps:.1f} fps)")
    else:
        from pathlib import Path

        from .greet import write_frames_file

        write_frames_file(Path(args.out), video.frames_ansi(), fps=video.fps, loops=args.loops)
        print(f"Wrote {args.out} ({len(video.frames)} frames @ {video.fps:.1f} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
