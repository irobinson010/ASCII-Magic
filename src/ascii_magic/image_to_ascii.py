#!/usr/bin/env python3
import argparse
import math
import os
import platform
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps

# -----------------------------
# Charsets
# -----------------------------
def charset_ascii_printable():
    # ASCII printable range
    return "".join(chr(i) for i in range(32, 127))


def charset_ascii_dense():
    # A popular dense ASCII ramp (dark->light) plus some extras.
    return (
        "@%#*+=-:. "
        + "$&0QMWNBDHKPAqwmZOCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    )


def charset_unicode_blocks():
    # Useful Unicode shading + block fragments + a few line/box shapes
    # (Safe-ish; looks great with a font that supports them.)
    return " ░▒▓█" "▁▂▃▄▅▆▇█" "▏▎▍▌▋▊▉█" "▖▗▘▙▚▛▜▝▞▟" "─━│┃┌┐└┘├┤┬┴┼" "/\\|_-"


def charset_unicode_big():
    # A larger set: ASCII printable + blocks/shading + some extra symbols.
    # Keep it moderate so rendering/matching stays reasonable.
    extra = (
        " ·•"  # light dots
        "°º"  # small circles
        "×÷"  # operators
        "≡≈"  # lines
        "□■"  # squares
        "○●"  # circles
        "△▲▽▼"  # triangles
    )
    return charset_ascii_printable() + charset_unicode_blocks() + extra


def make_charset(unicode_mode: str, ascii_preset: str):
    if ascii_preset == "printable":
        base = charset_ascii_printable()
    else:
        base = charset_ascii_dense()

    if unicode_mode == "off":
        return base
    if unicode_mode == "blocks":
        return base + charset_unicode_blocks()
    if unicode_mode == "big":
        return charset_unicode_big()
    raise ValueError(f"Unknown unicode mode: {unicode_mode}")


def find_default_mono_font():
    system = platform.system().lower()

    candidates = []
    if "darwin" in system:  # macOS
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/Library/Fonts/Courier New.ttf",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
        ]
    elif "windows" in system:
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates = [
            os.path.join(windir, "Fonts", "CASCADIAMONO.TTF"),
            os.path.join(windir, "Fonts", "CONSOLA.TTF"),
            os.path.join(windir, "Fonts", "LUCON.TTF"),
        ]
    else:  # Linux and others
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        ]

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# -----------------------------
# Fast Sobel (vectorized)
# -----------------------------
def sobel_gradients(gray01: np.ndarray):
    """
    gray01: HxW float32 in [0,1]
    returns (mag, ang)
    """
    g = np.pad(gray01, ((1, 1), (1, 1)), mode="edge")

    # Sobel X
    gx = (
        -1 * g[:-2, :-2]
        + 1 * g[:-2, 2:]
        + -2 * g[1:-1, :-2]
        + 2 * g[1:-1, 2:]
        + -1 * g[2:, :-2]
        + 1 * g[2:, 2:]
    )

    # Sobel Y
    gy = (
        -1 * g[:-2, :-2]
        + -2 * g[:-2, 1:-1]
        + -1 * g[:-2, 2:]
        + 1 * g[2:, :-2]
        + 2 * g[2:, 1:-1]
        + 1 * g[2:, 2:]
    )

    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)
    return mag.astype(np.float32), ang.astype(np.float32)


# -----------------------------
# Preprocess
# -----------------------------
def preprocess_image(img: Image.Image, autocontrast: bool, gamma: float, invert: bool):
    if autocontrast:
        img = ImageOps.autocontrast(img)
    if gamma and abs(gamma - 1.0) > 1e-6:
        # gamma correction: out = in^(1/gamma)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr, 0, 1) ** (1.0 / gamma)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode=img.mode)
    if invert:
        img = ImageOps.invert(img)
    return img


# -----------------------------
# Glyph library (render chars to bitmap cells)
# -----------------------------
def render_glyphs(
    charset: str, cell_w: int, cell_h: int, font_path: str | None, font_size: int | None
):
    if font_size is None:
        # heuristic
        font_size = cell_h

    # Try provided font_path first; otherwise fall back to a default mono font if available;
    # otherwise use PIL's built-in default font.
    if not font_path:
        font_path = find_default_mono_font()

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    glyph_imgs = []
    glyph_feats = []
    chars = []

    for ch in charset:
        img = Image.new("L", (cell_w, cell_h), color=255)  # white
        draw = ImageDraw.Draw(img)

        # Center text via bbox
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
        except Exception:
            # Some chars may not be supported by the font -> skip
            continue

        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (cell_w - tw) // 2 - bbox[0]
        y = (cell_h - th) // 2 - bbox[1]
        draw.text((x, y), ch, fill=0, font=font)

        arr = np.asarray(img, dtype=np.float32) / 255.0  # 0=black, 1=white
        ink = 1.0 - arr  # 1=ink/dark

        mean_ink = float(np.mean(ink))
        mag, ang = sobel_gradients(ink)
        mean_mag = float(np.mean(mag))
        vx = float(np.mean(np.cos(ang) * mag))
        vy = float(np.mean(np.sin(ang) * mag))

        feat = np.array([mean_ink, mean_mag, vx, vy], dtype=np.float32)

        glyph_imgs.append(ink.reshape(-1))
        glyph_feats.append(feat)
        chars.append(ch)

    if not chars:
        raise RuntimeError(
            "No glyphs could be rendered. Try providing a font with --font."
        )

    glyph_imgs = np.stack(glyph_imgs, axis=0)  # N x P
    glyph_feats = np.stack(glyph_feats, axis=0)  # N x 4

    # Normalize features
    mu = glyph_feats.mean(axis=0, keepdims=True)
    sd = glyph_feats.std(axis=0, keepdims=True) + 1e-6
    glyph_feats_n = (glyph_feats - mu) / sd

    return glyph_imgs, glyph_feats_n, chars, (mu.reshape(-1), sd.reshape(-1))


# -----------------------------
# Matching strategies
# -----------------------------
def pick_char_fast(cell_feat_n: np.ndarray, glyph_feats_n: np.ndarray) -> int:
    d = np.sum((glyph_feats_n - cell_feat_n) ** 2, axis=1)
    return int(np.argmin(d))


def pick_char_best(cell_vec: np.ndarray, glyph_imgs: np.ndarray) -> int:
    # MSE over all glyphs (slowest, best)
    d = np.mean((glyph_imgs - cell_vec) ** 2, axis=1)
    return int(np.argmin(d))


def pick_char_balanced(
    cell_feat_n: np.ndarray,
    cell_vec: np.ndarray,
    glyph_feats_n: np.ndarray,
    glyph_imgs: np.ndarray,
    topk: int,
) -> int:
    # Feature shortlist -> MSE refine
    d_feat = np.sum((glyph_feats_n - cell_feat_n) ** 2, axis=1)
    k = min(topk, d_feat.shape[0])
    idx = np.argpartition(d_feat, k - 1)[:k]
    sub = glyph_imgs[idx]
    d = np.mean((sub - cell_vec) ** 2, axis=1)
    return int(idx[int(np.argmin(d))])


# -----------------------------
# ASCII/Unicode glyph mode
# -----------------------------
def image_to_text_glyph_mode(
    image_path: str,
    cols: int,
    cell_w: int,
    cell_h: int,
    charset: str,
    quality: str,
    font_path: str | None,
    font_size: int | None,
    autocontrast: bool,
    gamma: float,
    invert: bool,
    topk: int,
):
    img = Image.open(image_path).convert("L")
    img = preprocess_image(img, autocontrast=autocontrast, gamma=gamma, invert=invert)

    W, H = img.size
    rows = max(1, int((H / W) * cols * (cell_w / cell_h)))

    target_w = cols * cell_w
    target_h = rows * cell_h
    img = img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)

    gray = np.asarray(img, dtype=np.float32) / 255.0  # 0..1
    ink = 1.0 - gray  # 1 = dark
    mag, ang = sobel_gradients(ink)

    glyph_imgs, glyph_feats_n, chars, (mu, sd) = render_glyphs(
        charset=charset,
        cell_w=cell_w,
        cell_h=cell_h,
        font_path=font_path,
        font_size=font_size,
    )

    out_lines = []
    for r in range(rows):
        line = []
        y0, y1 = r * cell_h, (r + 1) * cell_h
        for c in range(cols):
            x0, x1 = c * cell_w, (c + 1) * cell_w

            cell_ink = ink[y0:y1, x0:x1]
            cell_mag = mag[y0:y1, x0:x1]
            cell_ang = ang[y0:y1, x0:x1]

            mean_ink = float(np.mean(cell_ink))
            mean_mag = float(np.mean(cell_mag))
            vx = float(np.mean(np.cos(cell_ang) * cell_mag))
            vy = float(np.mean(np.sin(cell_ang) * cell_mag))

            cell_feat = np.array([mean_ink, mean_mag, vx, vy], dtype=np.float32)
            cell_feat_n = (cell_feat - mu) / sd

            if quality == "fast":
                best = pick_char_fast(cell_feat_n, glyph_feats_n)
            else:
                cell_vec = cell_ink.reshape(-1).astype(np.float32)
                if quality == "balanced":
                    best = pick_char_balanced(
                        cell_feat_n, cell_vec, glyph_feats_n, glyph_imgs, topk=topk
                    )
                else:  # "best"
                    best = pick_char_best(cell_vec, glyph_imgs)

            line.append(chars[best])
        out_lines.append("".join(line))

    return "\n".join(out_lines)


# -----------------------------
# Braille mode (usually awesome)
# -----------------------------
# Braille dot mapping:
# (x=0,y=0)->dot1, (0,1)->dot2, (0,2)->dot3, (0,3)->dot7
# (x=1,y=0)->dot4, (1,1)->dot5, (1,2)->dot6, (1,3)->dot8
_BRAILLE_BITS = {
    (0, 0): 0,  # 1
    (0, 1): 1,  # 2
    (0, 2): 2,  # 3
    (1, 0): 3,  # 4
    (1, 1): 4,  # 5
    (1, 2): 5,  # 6
    (0, 3): 6,  # 7
    (1, 3): 7,  # 8
}


def image_to_braille(
    image_path: str,
    cols: int,
    autocontrast: bool,
    gamma: float,
    invert: bool,
    threshold: float,
):
    img = Image.open(image_path).convert("L")
    img = preprocess_image(img, autocontrast=autocontrast, gamma=gamma, invert=invert)

    W, H = img.size
    # Each braille character encodes a 2x4 grid
    cell_w, cell_h = 2, 4
    rows = max(1, int((H / W) * cols * (cell_w / cell_h)))  # same logic as glyph mode

    target_w = cols * cell_w
    target_h = rows * cell_h
    img = img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)

    gray = np.asarray(img, dtype=np.float32) / 255.0
    ink = 1.0 - gray  # 1=dark

    out_lines = []
    for r in range(rows):
        line = []
        y0 = r * cell_h
        for c in range(cols):
            x0 = c * cell_w
            bits = 0
            # decide each dot based on threshold
            for dy in range(4):
                for dx in range(2):
                    v = ink[y0 + dy, x0 + dx]
                    if v >= threshold:
                        bits |= 1 << _BRAILLE_BITS[(dx, dy)]
            line.append(chr(0x2800 + bits))
        out_lines.append("".join(line))
    return "\n".join(out_lines)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Local high-quality ASCII/Unicode art (glyph match + optional braille mode)"
    )
    ap.add_argument("input", help="Input image path")
    ap.add_argument(
        "-o", "--output", default=None, help="Output text file (default: stdout)"
    )

    ap.add_argument(
        "--mode",
        choices=["glyph", "braille"],
        default="glyph",
        help="glyph = match characters; braille = Unicode braille pixels (often best)",
    )
    ap.add_argument(
        "-c", "--cols", type=int, default=120, help="Output columns (characters wide)"
    )

    # Glyph mode knobs
    ap.add_argument(
        "--cell-w", type=int, default=8, help="Glyph mode: cell width in pixels"
    )
    ap.add_argument(
        "--cell-h", type=int, default=16, help="Glyph mode: cell height in pixels"
    )
    ap.add_argument(
        "--quality",
        choices=["fast", "balanced", "best"],
        default="balanced",
        help="fast=feature match; balanced=feature shortlist + MSE; best=full MSE",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=24,
        help="balanced: shortlist size (higher=better/slower)",
    )

    ap.add_argument(
        "--ascii",
        choices=["dense", "printable"],
        default="dense",
        help="ASCII base charset preset",
    )
    ap.add_argument(
        "--unicode",
        choices=["off", "blocks", "big"],
        default="off",
        help="Allow Unicode chars in glyph mode (requires font/terminal support)",
    )

    ap.add_argument(
        "--charset-file",
        default=None,
        help="Optional: file containing characters to use (overrides --ascii/--unicode).",
    )

    ap.add_argument(
        "--font", default=None, help="Path to .ttf font (recommended for unicode)"
    )
    ap.add_argument(
        "--font-size", type=int, default=None, help="Font size used for glyph rendering"
    )

    # Preprocess
    ap.add_argument("--autocontrast", action="store_true", help="Apply autocontrast")
    ap.add_argument(
        "--gamma", type=float, default=1.0, help="Gamma correction (e.g. 1.2 or 0.8)"
    )
    ap.add_argument(
        "--invert",
        action="store_true",
        help="Invert image (useful for dark backgrounds)",
    )

    # Braille knobs
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Braille mode: dot threshold in [0,1] (higher = fewer dots)",
    )

    args = ap.parse_args()
    resolved_font = args.font or find_default_mono_font()

    if args.charset_file:
        with open(args.charset_file, "r", encoding="utf-8") as f:
            charset = "".join(ch for ch in f.read())
        # Remove newlines etc, keep unique order
        seen = set()
        charset = "".join(
            [
                ch
                for ch in charset
                if (ch not in seen and not seen.add(ch) and ch not in "\r\n")
            ]
        )
    else:
        charset = make_charset(unicode_mode=args.unicode, ascii_preset=args.ascii)

    if args.mode == "braille":
        art = image_to_braille(
            image_path=args.input,
            cols=args.cols,
            autocontrast=args.autocontrast,
            gamma=args.gamma,
            invert=args.invert,
            threshold=float(np.clip(args.threshold, 0.0, 1.0)),
        )
    else:
        art = image_to_text_glyph_mode(
            image_path=args.input,
            cols=args.cols,
            cell_w=args.cell_w,
            cell_h=args.cell_h,
            charset=charset,
            quality=args.quality,
            font_path=args.font,
            font_size=args.font_size,
            autocontrast=args.autocontrast,
            gamma=args.gamma,
            invert=args.invert,
            topk=args.topk,
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(art + "\n")
    else:
        print(art)


if __name__ == "__main__":
    main()
