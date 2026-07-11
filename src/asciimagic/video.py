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
import dataclasses
import io
import random
import re
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .colorize_ascii import (
    MatrixOptions,
    colorize_lines_ansi,
    matrix_lines_ansi,
    matrix_render_cells,
    parse_matrix_color,
)
from .image_to_ascii import (
    find_default_mono_font,
    image_to_braille_from_image,
    image_to_text_glyph_from_image,
    make_charset,
)

# Braille blank: no ink to draw
_BLANKS = (" ", "⠀")

# ffmpeg camera device syntax, e.g. <video0> (first webcam)
CAMERA_RE = re.compile(r"^<video\d+>$")

ESC_CLEAR = "\x1b[2J\x1b[H"
ESC_HIDE = "\x1b[?25l"
ESC_SHOW = "\x1b[?25h"


def is_camera(source: str) -> bool:
    return bool(CAMERA_RE.match(source))


def _arr_to_image(arr) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr[:, :, :3], mode="RGB")


def _require_imageio():
    try:
        import imageio.v2 as iio
    except ImportError:
        # RuntimeError (not SystemExit) so the web server can map it to a 400.
        raise RuntimeError(
            "Video support needs the [video] extra:\n"
            '    pip install "ascii-magic-tools[video]"   (or: uv sync --extra video)'
        )
    return iio


def _has_audio_stream(path: str) -> bool:
    import subprocess

    import imageio_ffmpeg

    proc = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
        errors="replace",
    )
    # ffmpeg exits nonzero without an output file; the stream listing on
    # stderr is still complete.
    return "Audio:" in proc.stderr


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
    try:
        for i, arr in enumerate(reader):
            if i % step:
                continue
            if len(frames) >= max_frames:
                break
            frames.append(_arr_to_image(arr))
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames could be read from {path}")
    return frames, out_fps


class AsciiVideo:
    """Sampled video frames converted to ASCII, plus the sinks."""

    def __init__(
        self,
        frames: List[Tuple[List[str], Image.Image]],
        fps: float,
        matrix: Optional[MatrixOptions] = None,
        caption=None,  # animate.CaptionRender
    ):
        self.frames = frames  # per frame: (ascii lines, source frame image)
        self.fps = fps
        self.matrix = matrix if (matrix and matrix.enabled) else None
        self.caption = caption

    def _frame_matrix(self, index: int) -> MatrixOptions:
        # Advance the seed per frame: deterministic overall, flickering glyphs.
        m = self.matrix
        seed = (m.seed + index) if m.seed is not None else None
        return dataclasses.replace(m, seed=seed)

    def frames_ansi(self) -> List[str]:
        cap_rows = None
        if self.caption:
            from .animate import caption_rows_ansi, with_caption_rows

            cap_rows = caption_rows_ansi(self.caption)
        out = []
        for i, (lines, img) in enumerate(self.frames):
            if self.matrix:
                rows = matrix_lines_ansi(lines, img, self._frame_matrix(i))
            else:
                rows = colorize_lines_ansi(lines, img, color_spaces=False)
            if cap_rows is not None:
                rows = with_caption_rows(list(rows), self.caption, cap_rows)
            out.append("\n".join(rows) + "\x1b[0m")
        return out

    def _frame_arrays(self, font_path: Optional[str] = None, font_size: int = 14) -> List[np.ndarray]:
        """Every frame drawn to an RGB array (including the caption strip)."""
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

        cap_strip: Optional[np.ndarray] = None
        if self.caption:
            from .animate import caption_strip_array

            w0 = max(len(ln) for ln in self.frames[0][0])
            cap_strip = caption_strip_array(self.caption, font, cell_w, cell_h, w0 * cell_w)

        arrays = []
        for i, (lines, frame_img) in enumerate(self.frames):
            h = len(lines)
            w = max(len(ln) for ln in lines)
            grid = [ln.ljust(w) for ln in lines]

            if self.matrix:
                mi = self._frame_matrix(i)
                cells = matrix_render_cells(lines, frame_img, mi, random.Random(mi.seed))
            else:
                small = frame_img.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
                px = small.load()

            canvas = np.zeros((h * cell_h, w * cell_w, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    if self.matrix:
                        ch, color = cells[y][x]
                        if color is None:
                            continue
                        r, g, b = color
                    else:
                        ch = grid[y][x]
                        if ch in _BLANKS:
                            continue
                        r, g, b = px[x, y]
                    a = glyph_alpha(ch)
                    block = canvas[y * cell_h:(y + 1) * cell_h, x * cell_w:(x + 1) * cell_w]
                    block[:, :, 0] = (a * r).astype(np.uint8)
                    block[:, :, 1] = (a * g).astype(np.uint8)
                    block[:, :, 2] = (a * b).astype(np.uint8)

            if cap_strip is not None:
                if self.caption.position == "top":
                    canvas = np.vstack([cap_strip, canvas])
                else:
                    canvas = np.vstack([canvas, cap_strip])
            arrays.append(canvas)
        return arrays

    def to_gif_bytes(self, font_path: Optional[str] = None, font_size: int = 14) -> bytes:
        images = [Image.fromarray(a) for a in self._frame_arrays(font_path, font_size)]
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

    def write_mp4(
        self,
        out_path: str,
        audio_source: Optional[str] = None,
        font_path: Optional[str] = None,
        font_size: int = 14,
    ) -> bool:
        """Encode the frames as an mp4, muxing audio from audio_source (usually
        the original clip). Returns True if audio made it in, False if the
        source has no audio stream and the file was written silent."""
        import imageio_ffmpeg

        arrays = self._frame_arrays(font_path, font_size)
        h, w = arrays[0].shape[:2]
        # h264 requires even dimensions
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            arrays = [np.pad(a, ((0, pad_h), (0, pad_w), (0, 0))) for a in arrays]
            h, w = arrays[0].shape[:2]

        def _write(audio: Optional[str]) -> None:
            gen = imageio_ffmpeg.write_frames(
                out_path,
                (w, h),
                fps=max(1.0, self.fps),
                macro_block_size=1,
                audio_path=audio,
                audio_codec="aac" if audio else None,
                # A truncated sample (--max-frames) is shorter than the audio.
                output_params=["-shortest"] if audio else None,
            )
            try:
                gen.send(None)
                for a in arrays:
                    gen.send(np.ascontiguousarray(a))
            finally:
                gen.close()

        if audio_source is not None and not _has_audio_stream(audio_source):
            audio_source = None  # write silent; encode errors below propagate
        _write(audio_source)
        return audio_source is not None

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
    mode: str = "braille",
    quality: str = "balanced",
    matrix: Optional[MatrixOptions] = None,
    caption=None,  # colorize_ascii.CaptionOptions
    rows: Optional[int] = None,
) -> AsciiVideo:
    frames, out_fps = read_video_frames(path, sample_fps=sample_fps, max_frames=max_frames)
    converted = _convert_frames(
        frames, cols=cols, mode=mode, quality=quality, dither=dither,
        threshold=threshold, gamma=gamma, autocontrast=autocontrast, invert=invert,
        rows=rows,
    )
    cap_render = _resolve_video_caption(caption, converted, matrix)
    return AsciiVideo(converted, out_fps, matrix=matrix, caption=cap_render)


def _convert_frame(img: Image.Image, *, cols, mode, quality, charset,
                   dither, threshold, gamma, autocontrast, invert,
                   rows: Optional[int] = None) -> List[str]:
    if rows:
        # Exact output height: pre-resize the frame onto the converter's cell
        # grid (braille cells are 2x4 px, glyph cells 8x16), so the natural
        # rows formula lands exactly on `rows`. Stretch/squish like GIMP.
        cw, ch = (8, 16) if mode == "glyph" else (2, 4)
        img = img.resize((max(1, cols) * cw, max(1, rows) * ch), Image.Resampling.LANCZOS)
    if mode == "glyph":
        art = image_to_text_glyph_from_image(
            img=img, cols=cols, cell_w=8, cell_h=16, charset=charset,
            quality=quality, font_path=None, font_size=None,
            autocontrast=autocontrast, gamma=gamma, invert=invert, topk=24,
        )
    else:
        art = image_to_braille_from_image(
            img, cols=cols, autocontrast=autocontrast, gamma=gamma,
            invert=invert, threshold=threshold, dither=dither,
        )
    return art.splitlines()


def _convert_frames(frames, *, cols, mode, quality, dither, threshold,
                    gamma, autocontrast, invert, rows=None):
    charset = make_charset(unicode_mode="off", ascii_preset="dense") if mode == "glyph" else None
    return [
        (
            _convert_frame(
                img, cols=cols, mode=mode, quality=quality, charset=charset,
                dither=dither, threshold=threshold, gamma=gamma,
                autocontrast=autocontrast, invert=invert, rows=rows,
            ),
            img,
        )
        for img in frames
    ]


def _resolve_video_caption(caption, converted, matrix):
    """Image-colored captions resolve against the first frame; otherwise a
    neutral light default (or the matrix tint when matrix is on)."""
    if caption is None or not caption.text or not converted:
        return None
    from .animate import _resolve_caption

    first_lines, first_img = converted[0]
    width = max(len(ln) for ln in first_lines)
    tint = matrix.tint if (matrix and matrix.enabled) else (224, 224, 224)
    return _resolve_caption(caption, first_img, width, tint)


def live_view(
    source: str,
    cols: int = 100,
    mode: str = "braille",
    quality: str = "balanced",
    dither: bool = True,
    threshold: float = 0.5,
    gamma: float = 1.0,
    autocontrast: bool = False,
    invert: bool = False,
    matrix: Optional[MatrixOptions] = None,
    caption=None,
    mirror: bool = True,
    out=None,
    max_frames: Optional[int] = None,
) -> int:
    """Stream a camera (or any source) to the terminal as live ASCII until
    Ctrl-C. Returns the number of frames shown. `max_frames`/`out` exist for
    testing; interactive use leaves them unset."""
    iio = _require_imageio()
    out = out or sys.stdout
    matrix = matrix if (matrix and matrix.enabled) else None
    charset = make_charset(unicode_mode="off", ascii_preset="dense") if mode == "glyph" else None

    reader = iio.get_reader(source)
    shown = 0
    cap_rows = None
    cap_render = None
    try:
        out.write(f"{ESC_CLEAR}{ESC_HIDE}")
        for arr in reader:
            img = _arr_to_image(arr)
            if mirror:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            lines = _convert_frame(
                img, cols=cols, mode=mode, quality=quality, charset=charset,
                dither=dither, threshold=threshold, gamma=gamma,
                autocontrast=autocontrast, invert=invert,
            )
            if caption is not None and cap_render is None and caption.text:
                cap_render = _resolve_video_caption(caption, [(lines, img)], matrix)
                if cap_render is not None:
                    from .animate import caption_rows_ansi

                    cap_rows = caption_rows_ansi(cap_render)
            if matrix:
                mi = dataclasses.replace(
                    matrix, seed=(matrix.seed + shown) if matrix.seed is not None else None
                )
                rows = matrix_lines_ansi(lines, img, mi)
            else:
                rows = colorize_lines_ansi(lines, img, color_spaces=False)
            if cap_rows is not None:
                from .animate import with_caption_rows

                rows = with_caption_rows(list(rows), cap_render, cap_rows)
            out.write("\x1b[H" + "\n".join(rows) + "\x1b[0m")
            out.flush()
            shown += 1
            if max_frames is not None and shown >= max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        try:
            out.write(f"\x1b[0m{ESC_SHOW}\n")
            out.flush()
        except BrokenPipeError:
            pass
    return shown


def record_camera(
    source: str,
    seconds: float = 5.0,
    mirror: bool = True,
    **convert_kwargs,
) -> AsciiVideo:
    """Capture a camera for `seconds` of wall time, then convert like a file."""
    iio = _require_imageio()
    matrix = convert_kwargs.pop("matrix", None)
    caption = convert_kwargs.pop("caption", None)

    reader = iio.get_reader(source)
    frames: List[Image.Image] = []
    t0 = time.monotonic()
    try:
        for arr in reader:
            img = _arr_to_image(arr)
            if mirror:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            frames.append(img)
            # wall-clock bound, plus a hard cap for absurdly fast sources
            if time.monotonic() - t0 >= seconds or len(frames) >= 1800:
                break
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames could be read from {source}")

    elapsed = max(time.monotonic() - t0, 1e-3)
    fps = max(1.0, len(frames) / elapsed)
    converted = _convert_frames(frames, **convert_kwargs)
    cap_render = _resolve_video_caption(caption, converted, matrix)
    return AsciiVideo(converted, fps, matrix=matrix, caption=cap_render)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ascii-magic video",
        description="Convert a video into a colorized ASCII animation "
        "(any format ffmpeg reads: mp4, webm, mov, mkv, avi, gif, ...).",
    )
    ap.add_argument("input", help="Video file, or a camera like '<video0>' for a live mirror")
    ap.add_argument("out", nargs="?", default=None,
                    help="Output: .gif, .mp4 (keeps the source audio), or .frames "
                    "(omit to play in the terminal; with a camera, omit to mirror live)")
    ap.add_argument("-c", "--cols", type=int, default=100, help="Width in characters")
    ap.add_argument("--rows", type=int, default=None, metavar="N",
                    help="Exact output height in rows (stretches/squishes the frame)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Target sample/playback fps (default: 10)")
    ap.add_argument("--max-frames", type=int, default=300,
                    help="Cap on sampled frames (default: 300)")
    ap.add_argument("--mode", choices=["braille", "glyph"], default="braille",
                    help="Per-frame conversion (default: braille)")
    ap.add_argument("--quality", choices=["fast", "balanced", "best"], default="balanced",
                    help="Glyph mode matching quality")
    ap.add_argument("--matrix", action="store_true",
                    help="Render frames as matrix glyphs driven by each frame")
    ap.add_argument("--matrix-color", default="green", metavar="COLOR",
                    help="Matrix tint: theme name or #RRGGBB")
    ap.add_argument("--matrix-seed", type=int, default=None, metavar="N",
                    help="Deterministic glyph placement (advances per frame)")
    ap.add_argument("--matrix-gamma", type=float, default=2.0, metavar="F")
    ap.add_argument("--matrix-mask", action="store_true",
                    help="Bias matrix glyphs toward inked ASCII cells")
    ap.add_argument("--no-audio", action="store_true",
                    help=".mp4 output: skip muxing the source audio")
    ap.add_argument("--seconds", type=float, default=5.0, metavar="F",
                    help="Camera sources with an output file: how long to record (default: 5)")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                    help="Camera sources: flip horizontally like a mirror (default: on)")
    ap.add_argument("--caption", default=None, metavar="TEXT",
                    help="Render TEXT as ASCII and stitch it onto every frame")
    ap.add_argument("--caption-pos", choices=["top", "bottom"], default="bottom")
    ap.add_argument("--caption-style",
                    choices=["block", "small", "shadow", "box", "banner", "figlet"],
                    default="figlet")
    ap.add_argument("--caption-cols", type=int, default=None, metavar="N",
                    help="Exact caption width in chars (free transform)")
    ap.add_argument("--caption-rows", type=int, default=None, metavar="N",
                    help="Exact caption height in rows")
    ap.add_argument("--caption-scale", type=float, default=0.6, metavar="F")
    ap.add_argument("--caption-gap", type=int, default=1, metavar="N")
    ap.add_argument("--caption-color", default=None, metavar="COLOR",
                    help="theme, #RRGGBB, image, or image-full (first frame)")
    ap.add_argument("--caption-align", choices=["left", "center", "right"], default="center")
    ap.add_argument("--no-dither", action="store_true",
                    help="Disable Floyd-Steinberg dithering")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--autocontrast", action="store_true")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--loops", type=int, default=1,
                    help="Terminal playback repeats (0 = until Ctrl-C)")
    ap.add_argument("--font-size", type=int, default=14, help="GIF glyph size")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.out and not args.out.lower().endswith((".gif", ".frames", ".mp4")):
        raise SystemExit("Output must be .gif, .mp4, or .frames (or omitted for terminal playback)")

    matrix = None
    if args.matrix:
        matrix = MatrixOptions(
            enabled=True,
            seed=args.matrix_seed,
            gamma=args.matrix_gamma,
            tint=parse_matrix_color(args.matrix_color),
            use_mask=args.matrix_mask,
        )

    caption = None
    if args.caption:
        from .colorize_ascii import CaptionOptions

        caption = CaptionOptions(
            text=args.caption,
            position=args.caption_pos,
            style=args.caption_style,
            scale=args.caption_scale,
            cols=args.caption_cols,
            rows=args.caption_rows,
            gap=args.caption_gap,
            color=args.caption_color,
            align=args.caption_align,
        )

    try:
        if is_camera(args.input):
            if args.out is None:
                live_view(
                    args.input, cols=args.cols, mode=args.mode, quality=args.quality,
                    dither=not args.no_dither, threshold=args.threshold,
                    gamma=args.gamma, autocontrast=args.autocontrast,
                    invert=args.invert, matrix=matrix, caption=caption,
                    mirror=args.mirror,
                )
                return 0
            video = record_camera(
                args.input, seconds=args.seconds, mirror=args.mirror,
                cols=args.cols, mode=args.mode, quality=args.quality,
                dither=not args.no_dither, threshold=args.threshold,
                gamma=args.gamma, autocontrast=args.autocontrast,
                invert=args.invert, matrix=matrix, caption=caption,
                rows=args.rows,
            )
        else:
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
                mode=args.mode,
                quality=args.quality,
                matrix=matrix,
                caption=caption,
                rows=args.rows,
            )
    except RuntimeError as e:
        raise SystemExit(str(e))

    if args.out is None:
        video.play(loops=args.loops)
    elif args.out.lower().endswith(".gif"):
        with open(args.out, "wb") as f:
            f.write(video.to_gif_bytes(font_size=args.font_size))
        print(f"Wrote {args.out} ({len(video.frames)} frames @ {video.fps:.1f} fps)")
    elif args.out.lower().endswith(".mp4"):
        with_audio = video.write_mp4(
            args.out,
            # cameras have no muxable audio path; record silent
            audio_source=None if (args.no_audio or is_camera(args.input)) else args.input,
            font_size=args.font_size,
        )
        note = "with source audio" if with_audio else "silent (no usable audio in source)"
        print(f"Wrote {args.out} ({len(video.frames)} frames @ {video.fps:.1f} fps, {note})")
    else:
        from pathlib import Path

        from .greet import write_frames_file

        write_frames_file(Path(args.out), video.frames_ansi(), fps=video.fps, loops=args.loops)
        print(f"Wrote {args.out} ({len(video.frames)} frames @ {video.fps:.1f} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
