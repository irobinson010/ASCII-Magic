import io
import json

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from asciimagic.webapp import app

client = TestClient(app)


def _png_bytes(size=(32, 32), color=(200, 80, 40)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _render(options, image=True):
    files = {"image": ("t.png", _png_bytes(), "image/png")} if image else None
    return client.post(
        "/api/render", files=files, data={"options": json.dumps(options)}
    )


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "ASCII" in r.text


def test_render_image_braille():
    r = _render({"source": "image", "mode": "braille", "cols": 16})
    assert r.status_code == 200
    body = r.json()
    assert body["ascii"]
    assert "\x1b[" in body["ansi"]
    assert "<!doctype html>" in body["html"].lower()
    assert body["seed"] is None


def test_render_text_block():
    r = _render({"source": "text", "text": "Hi", "text_style": "block"}, image=False)
    assert r.status_code == 200
    body = r.json()
    assert body["ascii"]
    assert "\x1b[" in body["ansi"]


def test_render_text_box_without_image_warns_and_stays_plain():
    r = _render({"source": "text", "text": "Hi", "text_style": "box"}, image=False)
    assert r.status_code == 200
    body = r.json()
    assert body["warning"]
    assert "\x1b[" not in body["ansi"]
    assert body["ascii"] in body["ansi"]


def test_render_matrix_assigns_and_reuses_seed():
    r = _render({"source": "image", "mode": "braille", "cols": 16, "matrix": True})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["seed"], int)

    r2 = _render(
        {
            "source": "image",
            "mode": "braille",
            "cols": 16,
            "matrix": True,
            "matrix_seed": body["seed"],
        }
    )
    assert r2.json()["ansi"] == body["ansi"]


def test_render_animate_returns_gif_and_player():
    import base64

    r = _render(
        {
            "source": "image",
            "mode": "braille",
            "cols": 12,
            "matrix": True,
            "animate": True,
            "anim_frames": 4,
            "anim_fps": 10,
        }
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["seed"], int)
    assert "setInterval" in body["html"]
    gif = base64.b64decode(body["gif_b64"])
    assert gif[:4] == b"GIF8"


def test_render_animate_same_seed_same_gif():
    r1 = _render(
        {"source": "image", "cols": 12, "animate": True, "anim_frames": 3, "matrix_seed": 5}
    )
    r2 = _render(
        {"source": "image", "cols": 12, "animate": True, "anim_frames": 3, "matrix_seed": 5}
    )
    assert r1.json()["gif_b64"] == r2.json()["gif_b64"]


def test_render_matrix_color_theme_and_hex():
    for color in ("amber", "#ff00ff"):
        r = _render(
            {"source": "image", "cols": 12, "matrix": True, "matrix_seed": 3, "matrix_color": color}
        )
        assert r.status_code == 200
        assert "\x1b[38;2;" in r.json()["ansi"]


def test_render_caption_in_all_outputs():
    r = _render(
        {
            "source": "image",
            "mode": "braille",
            "cols": 16,
            "caption_text": "Whiskers",
            "caption_style": "box",
        }
    )
    assert r.status_code == 200
    body = r.json()
    assert "Whiskers" in body["ascii"]
    assert "Whiskers" in body["html"]
    import re

    assert "Whiskers" in re.sub(r"\x1b\[[0-9;]*m", "", body["ansi"])


def test_render_caption_plain_when_colorize_off():
    r = _render(
        {
            "source": "image",
            "cols": 16,
            "colorize": False,
            "caption_text": "Cat",
            "caption_style": "box",
        }
    )
    body = r.json()
    assert "Cat" in body["ascii"]
    assert "Cat" in body["ansi"]


def test_render_bad_caption_color_400():
    r = _render(
        {"source": "image", "cols": 16, "caption_text": "Cat", "caption_color": "plaid"}
    )
    assert r.status_code == 400


def test_render_bad_matrix_color_400():
    r = _render({"source": "image", "cols": 12, "matrix": True, "matrix_color": "plaid"})
    assert r.status_code == 400


def test_art_dims_in_response():
    r = _render({"source": "image", "mode": "braille", "cols": 16})
    body = r.json()
    assert body["art"]["cols"] == 16
    assert body["art"]["rows"] == len(body["ascii"].splitlines())


def test_art_dims_report_caption_rows():
    r = _render(
        {"source": "image", "mode": "braille", "cols": 20,
         "caption_text": "Cat", "caption_style": "box", "caption_gap": 2,
         "caption_pos": "top"}
    )
    art = r.json()["art"]
    assert art["cap_lines"] == 3  # box is 3 lines tall
    assert art["cap_gap"] == 2
    assert art["cap_pos"] == "top"
    assert art["cap_style"] == "box"

    # no caption -> zero
    r2 = _render({"source": "image", "mode": "braille", "cols": 20})
    assert r2.json()["art"]["cap_lines"] == 0


def test_caption_exact_dims_via_api():
    r = _render(
        {"source": "image", "mode": "braille", "cols": 40,
         "caption_text": "Hi", "caption_style": "figlet",
         "caption_cols": 20, "caption_rows": 4}
    )
    art = r.json()["art"]
    assert art["cap_lines"] == 4
    assert 18 <= art["cap_cols"] <= 22


def test_animation_caption_not_counted_in_art_block():
    # The player renders its caption in a separate element
    r = _render(
        {"source": "image", "cols": 12, "matrix": True, "animate": True,
         "anim_frames": 2, "caption_text": "Cat", "caption_style": "box"}
    )
    assert r.json()["art"]["cap_lines"] == 0


def test_video_art_dims_report_caption_rows():
    pytest.importorskip("imageio")
    r = client.post(
        "/api/render",
        files={"image": ("clip.gif", _gif_clip_bytes(), "image/gif")},
        data={"options": json.dumps({"source": "video", "cols": 24, "video_max_frames": 2,
                                     "caption_text": "Cat", "caption_style": "box"})},
    )
    art = r.json()["art"]
    assert art["cap_lines"] == 3
    assert art["cap_gap"] == 1
    assert art["cap_pos"] == "bottom"


def test_exact_sizing_without_colorize():
    # The GUI resize handles set out_rows/out_cols; must work with colorize off
    r = _render(
        {"source": "image", "mode": "braille", "cols": 16, "colorize": False,
         "out_rows": 5, "out_cols": 20}
    )
    body = r.json()
    lines = body["ascii"].splitlines()
    assert len(lines) == 5
    assert max(len(ln) for ln in lines) == 20
    assert body["art"]["cols"] == 20
    assert body["art"]["rows"] == 5


def test_video_rows_stretch():
    pytest.importorskip("imageio")
    r = client.post(
        "/api/render",
        files={"image": ("clip.gif", _gif_clip_bytes(), "image/gif")},
        data={"options": json.dumps({"source": "video", "cols": 20,
                                     "video_rows": 7, "video_max_frames": 2})},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["art"]["cols"] == 20
    assert body["art"]["rows"] == 7


def test_render_colorize_off_returns_plain():
    r = _render({"source": "image", "mode": "braille", "cols": 16, "colorize": False})
    body = r.json()
    assert "\x1b[" not in body["ansi"]


def test_render_missing_image_400():
    r = _render({"source": "image"}, image=False)
    assert r.status_code == 400


def test_render_bad_image_400():
    r = client.post(
        "/api/render",
        files={"image": ("t.png", b"not an image", "image/png")},
        data={"options": json.dumps({"source": "image"})},
    )
    assert r.status_code == 400


def _art_dims(body):
    lines = body["ascii"].splitlines()
    return max(len(ln) for ln in lines), len(lines)


def test_exif_orientation_honored():
    # 60x20 landscape tagged "rotate 90 CW to display" -> renders as 20x60 portrait
    buf = io.BytesIO()
    img = Image.new("RGB", (60, 20), (200, 80, 40))
    exif = Image.Exif()
    exif[274] = 6  # Orientation tag
    img.save(buf, format="JPEG", exif=exif.tobytes())

    r = client.post(
        "/api/render",
        files={"image": ("t.jpg", buf.getvalue(), "image/jpeg")},
        data={"options": json.dumps({"source": "image", "mode": "braille", "cols": 10,
                                     "colorize": False})},
    )
    assert r.status_code == 200
    w, h = _art_dims(r.json())
    assert h > w  # portrait after orientation is applied


def test_manual_rotate_option():
    opts = {"source": "image", "mode": "braille", "cols": 10, "colorize": False}
    flat = _render({**opts})  # 32x32 source is square; use a wide image instead
    buf = io.BytesIO()
    Image.new("RGB", (60, 20), (10, 200, 40)).save(buf, format="PNG")

    def dims(rotate):
        r = client.post(
            "/api/render",
            files={"image": ("t.png", buf.getvalue(), "image/png")},
            data={"options": json.dumps({**opts, "rotate": rotate})},
        )
        assert r.status_code == 200
        return _art_dims(r.json())

    w0, h0 = dims(0)
    w90, h90 = dims(90)
    assert w0 > h0      # landscape
    assert h90 > w90    # rotated to portrait
    assert flat.status_code == 200


def _gif_clip_bytes(n_frames=6):
    frames = []
    for i in range(n_frames):
        img = Image.new("RGB", (48, 32), (10, 10, 30))
        for x in range(8):
            img.putpixel((i * 6 + x, 16), (250, 160, 60))
        frames.append(img)
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=100, loop=0)
    return buf.getvalue()


def test_render_video_source():
    import base64

    pytest.importorskip("imageio")
    r = client.post(
        "/api/render",
        files={"image": ("clip.gif", _gif_clip_bytes(), "image/gif")},
        data={"options": json.dumps({"source": "video", "cols": 24,
                                     "video_fps": 10, "video_max_frames": 4})},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["video"]["frames"] == 4
    assert base64.b64decode(body["gif_b64"])[:4] == b"GIF8"
    assert body["frames_text"].startswith('{"fps"')
    assert "\x1b[38;2;" in body["ansi"]
    assert "data:image/gif;base64," in body["html"]


def test_render_video_matrix_and_glyph_mode():
    pytest.importorskip("imageio")
    r = client.post(
        "/api/render",
        files={"image": ("clip.gif", _gif_clip_bytes(), "image/gif")},
        data={"options": json.dumps({
            "source": "video", "cols": 24, "video_max_frames": 3,
            "video_mode": "glyph", "matrix": True, "matrix_seed": 5,
            "matrix_color": "amber",
        })},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["video"]["frames"] == 3
    assert "\x1b[38;2;" in body["ansi"]


def test_render_video_missing_file_400():
    r = client.post("/api/render", data={"options": json.dumps({"source": "video"})})
    assert r.status_code == 400


def test_render_mp4_on_demand():
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    r = client.post(
        "/api/render/mp4",
        files={"image": ("clip.gif", _gif_clip_bytes(), "image/gif")},
        data={"options": json.dumps({"source": "video", "cols": 20, "video_max_frames": 3})},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert b"ftyp" in r.content[:64]


def test_render_mp4_missing_file_400():
    r = client.post("/api/render/mp4", data={"options": "{}"})
    assert r.status_code == 400


def test_render_video_bad_suffix_400():
    r = client.post(
        "/api/render",
        files={"image": ("notes.txt", b"hello", "text/plain")},
        data={"options": json.dumps({"source": "video"})},
    )
    assert r.status_code == 400


def test_render_video_size_cap_413(monkeypatch):
    from asciimagic import webapp

    pytest.importorskip("imageio")
    monkeypatch.setattr(webapp, "MAX_VIDEO_UPLOAD_BYTES", 100)
    r = client.post(
        "/api/render",
        files={"image": ("clip.gif", _gif_clip_bytes(), "image/gif")},
        data={"options": json.dumps({"source": "video"})},
    )
    assert r.status_code == 413


def test_upload_size_cap_413(monkeypatch):
    from asciimagic import webapp

    monkeypatch.setattr(webapp, "MAX_UPLOAD_BYTES", 100)
    r = _render({"source": "image", "cols": 10})
    assert r.status_code == 413


def test_pixel_cap_400(monkeypatch):
    from asciimagic import webapp

    monkeypatch.setattr(webapp, "MAX_IMAGE_PIXELS", 500)  # 32x32 png = 1024 px
    r = _render({"source": "image", "cols": 10})
    assert r.status_code == 400


def test_huge_knobs_are_clamped_server_side():
    r = _render({"source": "image", "mode": "braille", "cols": 999999})
    assert r.status_code == 200
    width = max(len(ln) for ln in r.json()["ascii"].splitlines())
    assert width <= 500


def test_garbage_numeric_options_do_not_500():
    r = _render(
        {
            "source": "image",
            "cols": "abc",
            "gamma": "",
            "matrix": True,
            "matrix_fg_min": "nope",
            "matrix_seed": "1",
            "html_font_size": [],
            "keep_top": None,
        }
    )
    assert r.status_code == 200


def test_render_bad_options_400():
    r = client.post("/api/render", data={"options": "not json"})
    assert r.status_code == 400


def test_render_invalid_mode_400():
    r = _render({"source": "image", "mode": "nope"})
    assert r.status_code == 400


# ---- untrusted option values: 400 or a safe default, never a 500 ----

@pytest.mark.parametrize(
    "key,value,extra",
    [
        ("caption_style", "nope", {"caption_text": "Hi"}),
        ("caption_style", 5, {"caption_text": "Hi"}),
        ("caption_pos", "middle", {"caption_text": "Hi"}),
        ("caption_align", "up", {"caption_text": "Hi"}),
        ("quality", "nope", {"mode": "glyph"}),
        ("ascii_preset", "nope", {"mode": "glyph"}),
    ],
)
def test_render_rejects_unknown_enum_values(key, value, extra):
    r = _render({"source": "image", "mode": "braille", "cols": 16, **extra, key: value})
    assert r.status_code == 400
    assert key in r.json()["detail"]


@pytest.mark.parametrize("text", [["hi"], 5, {"a": 1}])
def test_render_rejects_non_string_text(text):
    r = _render({"source": "text", "text": text}, image=False)
    assert r.status_code == 400


@pytest.mark.parametrize("key", ["cols", "rotate", "gamma", "threshold"])
@pytest.mark.parametrize("value", ["inf", "-inf", "nan", 1e400])
def test_render_non_finite_numbers_fall_back(key, value):
    r = _render({"source": "image", "mode": "braille", "cols": 16, key: value})
    assert r.status_code == 200


@pytest.mark.parametrize("raw,expected", [("abc", None), ("1.5", 1), (-5, 0), ([1], None)])
def test_render_matrix_seed_garbage_is_sanitized_and_echoed(raw, expected):
    r = _render({"source": "image", "mode": "braille", "cols": 16, "matrix": True, "matrix_seed": raw})
    assert r.status_code == 200
    seed = r.json()["seed"]
    assert isinstance(seed, int) and 0 <= seed < 2**31
    if expected is not None:
        assert seed == expected


@pytest.mark.parametrize("value", ["false", "0", "off", "no", False])
def test_render_colorize_false_strings(value):
    r = _render({"source": "image", "mode": "braille", "cols": 16, "colorize": value})
    assert r.status_code == 200
    assert "\x1b[38;2;" not in r.json()["ansi"]


def test_render_colorize_true_string():
    r = _render({"source": "image", "mode": "braille", "cols": 16, "threshold": 1, "colorize": "true"})
    assert "\x1b[38;2;" in r.json()["ansi"]


# ---- resource limits ----

def _limited_app(max_bytes):
    from fastapi import FastAPI, File, UploadFile

    from asciimagic.webapp import BodySizeLimitMiddleware

    inner = FastAPI()

    @inner.post("/up")
    def up(f: UploadFile = File(...)):
        return {"n": len(f.file.read())}

    inner.add_middleware(BodySizeLimitMiddleware, max_bytes=max_bytes)
    return TestClient(inner)


def test_body_limit_allows_small_uploads():
    r = _limited_app(10_000).post("/up", files={"f": ("a.bin", b"x" * 100)})
    assert r.status_code == 200 and r.json() == {"n": 100}


def test_body_limit_rejects_declared_content_length():
    r = _limited_app(1_000).post("/up", files={"f": ("a.bin", b"x" * 5_000)})
    assert r.status_code == 413


def test_body_limit_rejects_chunked_body_without_content_length():
    boundary = "b0undary"
    payload = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"f\"; filename=\"a.bin\"\r\n\r\n"
    ).encode() + b"x" * 5_000 + f"\r\n--{boundary}--\r\n".encode()

    def chunks():
        for i in range(0, len(payload), 512):
            yield payload[i:i + 512]

    r = _limited_app(1_000).post(
        "/up", content=chunks(), headers={"content-type": f"multipart/form-data; boundary={boundary}"}
    )
    assert r.status_code == 413


def test_render_returns_503_when_all_slots_busy(monkeypatch):
    import threading

    from asciimagic import webapp

    busy = threading.BoundedSemaphore(1)
    busy.acquire()
    monkeypatch.setattr(webapp, "_render_slots", busy)
    monkeypatch.setattr(webapp, "RENDER_QUEUE_TIMEOUT_S", 0.01)
    r = _render({"source": "image", "mode": "braille", "cols": 16})
    assert r.status_code == 503
    assert r.headers.get("retry-after")


def test_render_slot_released_after_error(monkeypatch):
    import threading

    from asciimagic import webapp

    monkeypatch.setattr(webapp, "_render_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(webapp, "RENDER_QUEUE_TIMEOUT_S", 0.01)
    assert _render({"source": "image", "mode": "nope"}).status_code == 400
    assert _render({"source": "image", "mode": "braille", "cols": 16}).status_code == 200


def _render_img(options, size):
    files = {"image": ("t.png", _png_bytes(size=size), "image/png")}
    return client.post("/api/render", files=files, data={"options": json.dumps(options)})


def test_budget_rejects_huge_cell_count():
    # 1:50 portrait at 500 cols -> ~12,500 braille rows
    r = _render_img({"source": "image", "mode": "braille", "cols": 500}, size=(20, 1000))
    assert r.status_code == 400
    assert "Output too large" in r.json()["detail"]


def test_budget_rejects_huge_glyph_pixels():
    r = _render_img(
        {"source": "image", "mode": "glyph", "cols": 500, "cell_w": 32, "cell_h": 64}, size=(400, 400)
    )
    assert r.status_code == 400
    assert "glyph-matching pixels" in r.json()["detail"]


def test_budget_rejects_exact_size_over_limit():
    r = _render({"source": "image", "mode": "braille", "cols": 16, "out_rows": 2000, "out_cols": 2000})
    assert r.status_code == 400


def test_budget_rejects_long_animation():
    r = _render_img(
        {"source": "image", "mode": "braille", "cols": 240, "matrix": True, "animate": True,
         "anim_frames": 240},
        size=(400, 400),
    )
    assert r.status_code == 400
    assert "frames" in r.json()["detail"]


def test_budget_allows_gui_defaults_on_portrait_photo():
    # GUI defaults: 100 cols braille, 60-frame animation; 3:4 portrait photo.
    r = _render_img(
        {"source": "image", "mode": "braille", "cols": 100, "matrix": True, "animate": True,
         "anim_frames": 60},
        size=(300, 400),
    )
    assert r.status_code == 200


def _gif_bytes(size, n=4):
    frames = [Image.new("RGB", size, (i * 40, 60, 90)) for i in range(n)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return buf.getvalue()


def _render_video(options, data):
    return client.post(
        "/api/render",
        files={"image": ("v.gif", data, "image/gif")},
        data={"options": json.dumps({"source": "video", **options})},
    )


def test_video_render_ok_with_defaults():
    r = _render_video({"cols": 40}, _gif_bytes((64, 48)))
    assert r.status_code == 200
    assert r.json()["video"]["frames"] == 4


def test_video_budget_rejected(monkeypatch):
    from asciimagic import webapp

    monkeypatch.setattr(webapp, "MAX_ANIM_CELL_FRAMES", 100)
    r = _render_video({"cols": 40}, _gif_bytes((64, 48)))
    assert r.status_code == 400
    assert "characters x frames" in r.json()["detail"]


def test_video_oversized_frames_rejected(monkeypatch):
    from asciimagic import webapp

    monkeypatch.setattr(webapp, "MAX_IMAGE_PIXELS", 1000)
    r = _render_video({"cols": 40}, _gif_bytes((64, 48)))
    assert r.status_code == 400
    assert "pixels" in r.json()["detail"]
