import pytest
from PIL import Image

from ascii_magic.colorize_ascii import Options
from ascii_magic.pipeline import AsciiPipelineContext, colorize, image_to_ascii, load_image, text_to_ascii


def test_context_text_to_ascii_then_colorize_ansi():
    ctx = AsciiPipelineContext()
    ctx.source_image = Image.new("RGB", (40, 20), color=(20, 200, 40))

    ascii_out = text_to_ascii(ctx, "Hi", style="box", width=20)
    assert ascii_out
    assert ctx.ascii_text == ascii_out

    out = colorize(ctx, opt=Options(out_format="ansi"))
    assert "\x1b[" in out
    assert ctx.colorized_format == "ansi"


def test_context_image_to_ascii_and_colorize_html(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(120, 60, 200)).save(image_path)

    ctx = AsciiPipelineContext()
    load_image(ctx, str(image_path))

    ascii_out = image_to_ascii(ctx, mode="braille", cols=12)
    assert ascii_out
    assert ctx.ascii_text == ascii_out

    out = colorize(ctx, opt=Options(out_format="html"))
    assert "<!doctype html>" in out.lower()
    assert ctx.colorized_format == "html"


def test_context_text_to_ascii_uses_rendered_image_for_colorize():
    ctx = AsciiPipelineContext()
    ascii_out = text_to_ascii(ctx, "Hi", style="block", width=20)
    assert ascii_out
    assert ctx.rendered_text_image is not None

    out = colorize(ctx, opt=Options(out_format="ansi"))
    assert "\x1b[" in out


def test_context_text_to_ascii_box_clears_rendered_image():
    ctx = AsciiPipelineContext()
    text_to_ascii(ctx, "Hello", style="block", width=20)
    assert ctx.rendered_text_image is not None

    text_to_ascii(ctx, "Hello", style="box", width=20)
    assert ctx.rendered_text_image is None

    with pytest.raises(ValueError, match="No source image in context"):
        colorize(ctx, opt=Options(out_format="ansi"))


def test_context_image_to_ascii_rejects_invalid_mode():
    ctx = AsciiPipelineContext(source_image=Image.new("RGB", (32, 32), color=(20, 30, 40)))

    with pytest.raises(ValueError, match="Unsupported mode"):
        image_to_ascii(ctx, mode="invalid", cols=12)
