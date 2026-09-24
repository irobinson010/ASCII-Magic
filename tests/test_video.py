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


def test_video_exact_rows(clip):
    v = video_mod.video_to_ascii(str(clip), cols=20, max_frames=2, rows=6)
    for lines, _ in v.frames:
        assert len(lines) == 6
        assert max(len(ln) for ln in lines) == 20


def test_video_rows_cli_flag(clip, tmp_path):
    out = tmp_path / "sq.frames"
    rc = video_mod.main([str(clip), str(out), "-c", "20", "--rows", "5", "--max-frames", "2"])
    assert rc == 0
    from asciimagic.greet import read_frames_file
    import re

    frames, _, _ = read_frames_file(out)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", frames[0])
    assert len(plain.splitlines()) == 5


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


def test_cli_rejects_zero_fps(clip, tmp_path, capsys):
    from asciimagic.video import main as video_main

    with pytest.raises(SystemExit):
        video_main([str(clip), "-o", str(tmp_path / "o.gif"), "--fps", "0"])
    assert "--fps" in capsys.readouterr().err


# ---- bounded decoding ----

@pytest.fixture
def big_clip(tmp_path):
    frames = [Image.new("RGB", (800, 600), (i * 20, 40, 90)) for i in range(6)]
    path = tmp_path / "big.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return path


def test_read_frames_downscaled_to_max_width(big_clip):
    frames, _ = video_mod.read_video_frames(str(big_clip), max_width=100)
    assert all(f.size == (100, 75) for f in frames)


def test_video_to_ascii_keeps_frames_near_grid_size(big_clip):
    v = video_mod.video_to_ascii(str(big_clip), cols=20)
    assert all(img.width <= 20 * 4 for _, img in v.frames)
    assert max(len(ln) for ln in v.frames[0][0]) == 20


def test_read_frames_rejects_oversized_frames(big_clip):
    with pytest.raises(video_mod.VideoTooLarge):
        video_mod.read_video_frames(str(big_clip), max_pixels=100_000)


def test_read_frames_bounds_decoding(big_clip):
    frames, _ = video_mod.read_video_frames(str(big_clip), sample_fps=10.0, max_decoded=3)
    assert len(frames) == 3


def test_video_to_ascii_cell_frame_budget(big_clip):
    with pytest.raises(video_mod.VideoTooLarge, match="characters x frames"):
        video_mod.video_to_ascii(str(big_clip), cols=40, max_cell_frames=100)


@pytest.mark.parametrize("meta,expected", [
    ({"fps": 25.0}, 25.0),
    ({"fps": 1e9}, 10.0),
    ({"fps": float("nan")}, 10.0),
    ({"fps": -5}, 10.0),
    ({"fps": "junk"}, 10.0),
    ({"duration": 50}, 20.0),
])
def test_source_fps_sanitizes_metadata(meta, expected):
    assert video_mod._source_fps(meta) == expected


# ---- untrusted input (web uploads) ----

@pytest.fixture
def av_clip(tmp_path):
    """1 s mp4 with an aac audio track, made by the bundled ffmpeg."""
    import subprocess

    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "av.mp4"
    proc = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440",
         "-t", "1", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)],
        capture_output=True, timeout=60,
    )
    if proc.returncode != 0:  # a build without lavfi/libx264: nothing to test with
        pytest.skip(f"bundled ffmpeg cannot synthesize a test clip: {proc.stderr[-200:]!r}")
    return path


def test_untrusted_read_passes_ffmpeg_whitelists(monkeypatch, tmp_path):
    seen = {}

    class _IIO:
        @staticmethod
        def get_reader(path, **kw):
            seen.update(kw)
            return _FakeReader(n=2)

    monkeypatch.setattr(video_mod, "_require_imageio", lambda: _IIO)
    video_mod.read_video_frames(str(tmp_path / "x.mp4"), untrusted=True)
    params = seen["input_params"]
    assert params[params.index("-protocol_whitelist") + 1] == "file,pipe"
    assert "hls" not in params[params.index("-format_whitelist") + 1]
    assert "concat" not in params[params.index("-format_whitelist") + 1]


def test_untrusted_read_decodes_real_mp4(av_clip):
    frames, fps = video_mod.read_video_frames(str(av_clip), untrusted=True)
    assert len(frames) == 10 and fps == 10.0


def test_untrusted_read_refuses_concat_script(av_clip, tmp_path):
    evil = tmp_path / "evil.mkv"
    evil.write_text(f"ffconcat version 1.0\nfile {av_clip}\n")
    with pytest.raises(OSError):
        video_mod.read_video_frames(str(evil), untrusted=True)
    assert video_mod._has_audio_stream(str(evil), untrusted=True) is False


def test_audio_probe_times_out_quietly(monkeypatch, av_clip):
    import subprocess

    def hang(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=kw.get("timeout"))

    monkeypatch.setattr(subprocess, "run", hang)
    assert video_mod._has_audio_stream(str(av_clip), untrusted=True) is False


def test_mp4_keeps_source_audio(av_clip, tmp_path):
    v = video_mod.video_to_ascii(str(av_clip), cols=16, max_frames=5, untrusted=True)
    out = tmp_path / "o.mp4"
    assert v.write_mp4(str(out), audio_source=str(av_clip), untrusted_source=True) is True
    assert b"mp4a" in out.read_bytes()
