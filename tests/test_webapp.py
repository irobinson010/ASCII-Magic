import io
import json

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ascii_magic.webapp import app

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


def test_render_bad_matrix_color_400():
    r = _render({"source": "image", "cols": 12, "matrix": True, "matrix_color": "plaid"})
    assert r.status_code == 400


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


def test_render_bad_options_400():
    r = client.post("/api/render", data={"options": "not json"})
    assert r.status_code == 400


def test_render_invalid_mode_400():
    r = _render({"source": "image", "mode": "nope"})
    assert r.status_code == 400
