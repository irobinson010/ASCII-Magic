import pytest
from PIL import Image, ImageDraw

pytest.importorskip("imageio")

from asciimagic import video as video_mod
from asciimagic.greet import read_frames_file


@pytest.fixture
def clip(tmp_path):
    """A small animated GIF: a bright ball moving across a dark frame."""
    frames = []
    for i in range(8):
        img = Image.new("RGB", (64, 48), (10, 10, 30))
        d = ImageDraw.Draw(img)
        x = 4 + i * 6
        d.ellipse([x, 14, x + 16, 30], fill=(240, 160, 60))
        frames.append(img)
    path = tmp_path / "clip.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return path


def test_read_video_frames(clip):
    frames, fps = video_mod.read_video_frames(str(clip), sample_fps=10.0)
    assert len(frames) == 8
    assert fps > 0
    assert frames[0].mode == "RGB"


def test_read_video_frames_sampling_and_cap(clip):
    frames, _ = video_mod.read_video_frames(str(clip), sample_fps=5.0)
    assert len(frames) == 4  # every 2nd frame of 10fps source
    frames, _ = video_mod.read_video_frames(str(clip), sample_fps=10.0, max_frames=3)
    assert len(frames) == 3


def test_video_to_ascii_moves(clip):
    v = video_mod.video_to_ascii(str(clip), cols=24, sample_fps=10.0)
    assert len(v.frames) == 8
    first_lines, _ = v.frames[0]
    last_lines, _ = v.frames[-1]
    assert first_lines != last_lines  # the ball moved


def test_frames_ansi_colorized(clip):
    v = video_mod.video_to_ascii(str(clip), cols=24)
    ansi = v.frames_ansi()
    assert len(ansi) == 8
    assert all("\x1b[38;2;" in f for f in ansi)


def test_gif_output(clip):
    v = video_mod.video_to_ascii(str(clip), cols=20, max_frames=4)
    data = v.to_gif_bytes()
    assert data[:4] == b"GIF8"
    import io

    g = Image.open(io.BytesIO(data))
    assert g.n_frames == 4


def test_cli_writes_frames_file(clip, tmp_path):
    out = tmp_path / "clip.frames"
    rc = video_mod.main([str(clip), str(out), "-c", "20", "--max-frames", "4"])
    assert rc == 0
    frames, fps, loops = read_frames_file(out)
    assert len(frames) == 4
    assert "\x1b[38;2;" in frames[0]


def test_cli_writes_gif(clip, tmp_path):
    out = tmp_path / "out.gif"
    rc = video_mod.main([str(clip), str(out), "-c", "20", "--max-frames", "3"])
    assert rc == 0
    assert out.read_bytes()[:4] == b"GIF8"


def test_cli_rejects_bad_extension(clip, tmp_path):
    with pytest.raises(SystemExit):
        video_mod.main([str(clip), str(tmp_path / "out.html")])


def test_video_glyph_mode(clip):
    v = video_mod.video_to_ascii(str(clip), cols=20, max_frames=3, mode="glyph")
    lines, _ = v.frames[0]
    # glyph mode emits dense-charset characters, not braille
    assert not any("⠀" <= ch <= "⣿" for ch in "".join(lines))


def test_video_matrix_render_deterministic_and_tinted(clip):
    from asciimagic.colorize_ascii import MatrixOptions, parse_matrix_color

    m = MatrixOptions(enabled=True, seed=7, tint=parse_matrix_color("amber"))
    a = video_mod.video_to_ascii(str(clip), cols=20, max_frames=3, matrix=m)
    b = video_mod.video_to_ascii(str(clip), cols=20, max_frames=3, matrix=m)
    fa, fb = a.frames_ansi(), b.frames_ansi()
    assert fa == fb                      # seeded => deterministic
    assert fa[0] != fa[1]                # seed advances per frame => flicker
    reds = [int(c.split(";")[0]) for c in "".join(fa).split("\x1b[38;2;")[1:]]
    assert any(r > 0 for r in reds)      # amber tint reaches the output

    gif = a.to_gif_bytes()
    assert gif[:4] == b"GIF8"


def test_video_caption_in_sinks(clip):
    from asciimagic.colorize_ascii import CaptionOptions

    cap = CaptionOptions(text="Cat", style="box", position="bottom")
    v = video_mod.video_to_ascii(str(clip), cols=24, max_frames=3, caption=cap)
    bare = video_mod.video_to_ascii(str(clip), cols=24, max_frames=3)

    import re

    frames = v.frames_ansi()
    assert all("Cat" in re.sub(r"\x1b\[[0-9;]*m", "", f) for f in frames)

    import io as _io

    g_cap = Image.open(_io.BytesIO(v.to_gif_bytes()))
    g_bare = Image.open(_io.BytesIO(bare.to_gif_bytes()))
    assert g_cap.size[1] > g_bare.size[1]  # caption strip adds height
    assert g_cap.size[0] == g_bare.size[0]


def test_video_mp4_output(clip, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    out = tmp_path / "out.mp4"
    rc = video_mod.main([str(clip), str(out), "-c", "20", "--max-frames", "3",
                         "--caption", "Hi", "--caption-style", "box"])
    assert rc == 0
    data = out.read_bytes()
    assert len(data) > 500
    assert b"ftyp" in data[:64]  # mp4 container signature


class _FakeReader:
    """Stands in for imageio's camera reader: yields moving-dot frames."""

    def __init__(self, n=40):
        self.n = n

    def __iter__(self):
        import numpy as np

        for i in range(self.n):
            f = np.full((32, 48, 3), 15, dtype=np.uint8)
            f[10:20, (i * 4) % 40:(i * 4) % 40 + 8] = (240, 160, 60)
            yield f

    def get_meta_data(self):
        return {"fps": 30}

    def close(self):
        pass


@pytest.fixture
def fake_camera(monkeypatch):
    class _IIO:
        @staticmethod
        def get_reader(source):
            return _FakeReader()

    monkeypatch.setattr(video_mod, "_require_imageio", lambda: _IIO)


def test_is_camera():
    assert video_mod.is_camera("<video0>")
    assert video_mod.is_camera("<video12>")
    assert not video_mod.is_camera("clip.mp4")
    assert not video_mod.is_camera("<videoX>")


def test_live_view_streams_and_restores(fake_camera):
    import io as _io

    buf = _io.StringIO()
    shown = video_mod.live_view("<video0>", cols=16, out=buf, max_frames=4)
    assert shown == 4
    out = buf.getvalue()
    assert out.startswith(video_mod.ESC_CLEAR + video_mod.ESC_HIDE)
    assert out.count("\x1b[H") >= 4  # cursor-home per frame
    assert out.endswith(video_mod.ESC_SHOW + "\n")
    assert "\x1b[38;2;" in out  # colorized


def test_live_view_mirror_flips(fake_camera):
    import io as _io
    import re

    strip = lambda s: re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", s)
    a = _io.StringIO()
    b = _io.StringIO()
    video_mod.live_view("<video0>", cols=16, out=a, max_frames=1, mirror=False)
    video_mod.live_view("<video0>", cols=16, out=b, max_frames=1, mirror=True)
    assert strip(a.getvalue()) != strip(b.getvalue())


def test_record_camera(fake_camera):
    v = video_mod.record_camera(
        "<video0>", seconds=0.05, cols=16, mode="braille", quality="balanced",
        dither=False, threshold=0.5, gamma=1.0, autocontrast=False, invert=False,
    )
    assert len(v.frames) >= 1
    assert v.fps >= 1.0
    assert v.to_gif_bytes()[:4] == b"GIF8"


def test_camera_cli_records_to_gif(fake_camera, tmp_path):
    out = tmp_path / "cam.gif"
    rc = video_mod.main(["<video0>", str(out), "-c", "16", "--seconds", "0.05"])
    assert rc == 0
    assert out.read_bytes()[:4] == b"GIF8"


def test_video_matrix_cli_flags(clip, tmp_path):
    out = tmp_path / "m.frames"
    rc = video_mod.main([
        str(clip), str(out), "-c", "20", "--max-frames", "3",
        "--matrix", "--matrix-seed", "5", "--matrix-color", "cyan",
    ])
    assert rc == 0
    from asciimagic.greet import read_frames_file

    frames, _, _ = read_frames_file(out)
    assert len(frames) == 3
    assert "\x1b[38;2;" in frames[0]
