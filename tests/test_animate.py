import io

import numpy as np
import pytest
from PIL import Image

from ascii_magic.animate import AnimationOptions, generate
from ascii_magic.colorize_ascii import MatrixOptions
from ascii_magic.pipeline import AsciiPipelineContext, animate


def _image():
    img = Image.new("RGB", (40, 24))
    px = img.load()
    for y in range(24):
        for x in range(40):
            px[x, y] = (x * 6, 128, y * 10)
    return img


ASCII = "\n".join("#" * 30 for _ in range(12))


def _gen(seed=7, **kw):
    return generate(ASCII, _image(), m=MatrixOptions(enabled=True, seed=seed), a=AnimationOptions(**kw))


def test_deterministic_for_seed():
    a = _gen(frames=8)
    b = _gen(frames=8)
    for (ia, ga, ha), (ib, gb, hb) in zip(a.frames, b.frames):
        assert np.array_equal(ia, ib)
        assert np.array_equal(ga, gb)
        assert np.array_equal(ha, hb)


def test_different_seeds_differ():
    a = _gen(seed=1, frames=4)
    b = _gen(seed=2, frames=4)
    assert any(not np.array_equal(x[0], y[0]) for x, y in zip(a.frames, b.frames))


def test_frame_count_and_shape():
    anim = _gen(frames=10)
    assert len(anim.frames) == 10
    assert anim.size == (30, 12)


def test_frames_change_over_time():
    anim = _gen(frames=10)
    assert any(
        not np.array_equal(anim.frames[0][0], f[0]) for f in anim.frames[1:]
    )


def test_ansi_frames():
    frames = _gen(frames=4).frames_ansi()
    assert len(frames) == 4
    assert all("\x1b[38;2;" in f for f in frames)
    assert all(len(f.split("\n")) == 12 for f in frames)


def test_gif_bytes_roundtrip():
    anim = _gen(frames=5)
    data = anim.to_gif_bytes()
    assert data[:4] == b"GIF8"
    gif = Image.open(io.BytesIO(data))
    assert getattr(gif, "n_frames", 1) == 5


def test_html_player():
    doc = _gen(frames=4).to_html()
    assert "<pre" in doc
    assert "setInterval" in doc
    assert "FRAMES" in doc


def test_play_writes_frames_and_restores_cursor():
    anim = _gen(frames=3, fps=1000.0)
    buf = io.StringIO()
    anim.play(loops=2, out=buf)
    out = buf.getvalue()
    assert out.count("\x1b[H") == 6  # 3 frames x 2 loops
    assert out.startswith("\x1b[2J\x1b[?25l")
    assert out.endswith("\x1b[0m\x1b[?25h\n")


def test_empty_ascii_rejected():
    for bad in ("", "\n", "\n\n"):
        with pytest.raises(ValueError, match="Empty ASCII"):
            generate(bad, _image())


def test_empty_matrix_chars_rejected():
    with pytest.raises(ValueError, match="chars"):
        generate(ASCII, _image(), m=MatrixOptions(enabled=True, chars=""))


def test_default_matrix_chars_single_backslash():
    assert MatrixOptions().chars.count("\\") == 1


def test_ansi_frames_end_with_reset():
    for f in _gen(frames=3).frames_ansi():
        assert f.endswith("\x1b[0m")


def test_pipeline_animate():
    ctx = AsciiPipelineContext(source_image=_image())
    ctx.ascii_text = ASCII
    anim = animate(ctx, matrix=MatrixOptions(enabled=True, seed=3), anim=AnimationOptions(frames=4))
    assert len(anim.frames) == 4
    assert ctx.metadata["animated"] is True


def test_caption_in_all_animation_sinks():
    from ascii_magic.colorize_ascii import CaptionOptions

    cap = CaptionOptions(text="Cat", style="box", position="bottom")
    anim = generate(
        ASCII, _image(), m=MatrixOptions(enabled=True, seed=7),
        a=AnimationOptions(frames=3), caption=cap,
    )

    frames = anim.frames_ansi()
    import re

    plain = re.sub(r"\x1b\[[0-9;]*m", "", frames[0])
    assert "Cat" in plain
    # caption rows appear in every frame, after the art + gap
    assert all("Cat" in re.sub(r"\x1b\[[0-9;]*m", "", f) for f in frames)

    html = anim.to_html()
    assert '<pre id="cap">' in html
    assert "Cat" in html

    # GIF grows taller by the caption strip
    bare = generate(
        ASCII, _image(), m=MatrixOptions(enabled=True, seed=7), a=AnimationOptions(frames=3)
    )
    import io as _io

    g_cap = Image.open(_io.BytesIO(anim.to_gif_bytes()))
    g_bare = Image.open(_io.BytesIO(bare.to_gif_bytes()))
    assert g_cap.size[1] > g_bare.size[1]
    assert g_cap.size[0] == g_bare.size[0]


def test_caption_image_colors_in_animation():
    from ascii_magic.colorize_ascii import CaptionOptions

    cap = CaptionOptions(text="Cat", style="box", color="image")
    anim = generate(
        ASCII, _image(), m=MatrixOptions(enabled=True, seed=7),
        a=AnimationOptions(frames=2), caption=cap,
    )
    assert anim.caption.colors is not None
    assert "\x1b[38;2;" in anim._caption_rows_ansi()[0]


def test_reveal_alpha_accumulates_to_full():
    anim = _gen(frames=40, reveal=True)
    assert anim.reveal_alpha is not None and len(anim.reveal_alpha) == 40
    assert anim.reveal_alpha[0].mean() < anim.reveal_alpha[-1].mean()
    # a full column cycle passes every cell: final frame is (nearly) fully lit
    assert anim.reveal_alpha[-1].mean() > 200


def test_reveal_final_frame_shows_art_chars():
    import re

    anim = _gen(frames=40, reveal=True)
    last = re.sub(r"\x1b\[[0-9;]*m", "", anim.frames_ansi()[-1])
    # source art is '#', which is not in the default rain charset
    assert "#" in last
    first = re.sub(r"\x1b\[[0-9;]*m", "", anim.frames_ansi()[0])
    assert first.count("#") < last.count("#")


def test_reveal_sinks_build():
    anim = _gen(frames=6, reveal=True)
    gif = anim.to_gif_bytes()
    assert gif[:4] == b"GIF8"
    html = anim.to_html()
    assert "rgb(" in html  # inline revealed-art spans
    assert "setInterval" in html


def test_no_reveal_has_no_alpha():
    assert _gen(frames=3).reveal_alpha is None


def test_pipeline_animate_requires_ascii():
    ctx = AsciiPipelineContext(source_image=_image())
    with pytest.raises(ValueError, match="No ASCII text"):
        animate(ctx)
