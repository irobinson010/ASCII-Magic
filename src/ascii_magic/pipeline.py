"""Shared context and orchestration helpers for ASCII Magic modules."""

from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image

from . import colorize_ascii as colorize_mod
from . import image_to_ascii as image_mod
from . import text_to_ascii as text_mod


@dataclass
class AsciiPipelineContext:
    """Holds shared state across multi-step ASCII processing."""

    source_image: Optional[Image.Image] = None
    source_image_path: Optional[str] = None
    source_text: Optional[str] = None
    ascii_text: Optional[str] = None
    rendered_text_image: Optional[Image.Image] = None
    colorized_output: Optional[str] = None
    colorized_format: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_image(ctx: AsciiPipelineContext, image_path: str) -> AsciiPipelineContext:
    ctx.source_image_path = image_path
    ctx.source_image = Image.open(image_path).convert("RGB")
    return ctx


def text_to_ascii(
    ctx: AsciiPipelineContext,
    text: str,
    style: str = "block",
    width: int = 80,
    font_size: int = 24,
    font_path: str | None = None,
    banner_char: str = "#",
) -> str:
    ctx.source_text = text
    ctx.metadata["text_style"] = style

    if style == "box":
        ascii_out = text_mod.text_to_box(text, width=width)
    elif style == "banner":
        ascii_out = text_mod.text_to_banner(text, char=banner_char)
    else:
        ascii_out = text_mod.text_to_ascii_art(
            text=text,
            style=style,
            width=width,
            font_size=font_size,
            font_path=font_path,
        )
        ctx.rendered_text_image = text_mod.render_text_to_image(
            text=text,
            font_size=font_size,
            font_path=font_path,
        )

    ctx.ascii_text = ascii_out
    return ascii_out


def image_to_ascii(
    ctx: AsciiPipelineContext,
    mode: str = "glyph",
    cols: int = 120,
    cell_w: int = 8,
    cell_h: int = 16,
    quality: str = "balanced",
    topk: int = 24,
    ascii_preset: str = "dense",
    unicode_mode: str = "off",
    charset_file: str | None = None,
    font_path: str | None = None,
    font_size: int | None = None,
    autocontrast: bool = False,
    gamma: float = 1.0,
    invert: bool = False,
    threshold: float = 0.5,
) -> str:
    img = ctx.source_image
    if img is None and ctx.source_image_path:
        img = Image.open(ctx.source_image_path).convert("RGB")
        ctx.source_image = img
    if img is None:
        raise ValueError("No source image in context. Call load_image(...) first.")

    if charset_file:
        with open(charset_file, "r", encoding="utf-8") as f:
            charset_raw = f.read()
        seen = set()
        charset = "".join(
            [ch for ch in charset_raw if (ch not in seen and not seen.add(ch) and ch not in "\r\n")]
        )
    else:
        charset = image_mod.make_charset(unicode_mode=unicode_mode, ascii_preset=ascii_preset)

    if mode == "braille":
        ascii_out = image_mod.image_to_braille_from_image(
            img=img,
            cols=cols,
            autocontrast=autocontrast,
            gamma=gamma,
            invert=invert,
            threshold=threshold,
        )
    else:
        ascii_out = image_mod.image_to_text_glyph_from_image(
            img=img,
            cols=cols,
            cell_w=cell_w,
            cell_h=cell_h,
            charset=charset,
            quality=quality,
            font_path=font_path,
            font_size=font_size,
            autocontrast=autocontrast,
            gamma=gamma,
            invert=invert,
            topk=topk,
        )

    ctx.ascii_text = ascii_out
    ctx.metadata["image_mode"] = mode
    return ascii_out


def colorize(
    ctx: AsciiPipelineContext,
    opt: Optional[colorize_mod.Options] = None,
) -> str:
    if ctx.source_image is None:
        if ctx.rendered_text_image is not None:
            ctx.source_image = ctx.rendered_text_image.convert("RGB")
        elif ctx.source_image_path:
            ctx.source_image = Image.open(ctx.source_image_path).convert("RGB")
        else:
            raise ValueError("No source image in context. Call load_image(...) first.")

    if not ctx.ascii_text:
        raise ValueError("No ASCII text in context. Run text_to_ascii(...) or image_to_ascii(...) first.")

    out = colorize_mod.colorize_ascii_text(ctx.source_image, ctx.ascii_text, opt=opt)
    ctx.colorized_output = out
    ctx.colorized_format = (opt.out_format if opt and opt.out_format else "ansi")
    return out
