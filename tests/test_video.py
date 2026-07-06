import pytest
from PIL import Image, ImageDraw

pytest.importorskip("imageio")

from ascii_magic import video as video_mod
from ascii_magic.greet import read_frames_file


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
