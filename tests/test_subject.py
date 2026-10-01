"""Subject focus and background replacement (issue #50). Detection is faked
so CI never downloads a model; one test uses a real model if installed."""

import hashlib
import io
import sys

import numpy as np
import pytest
from PIL import Image

from asciimagic import subject as sj
from asciimagic.subject import FocusOptions, SubjectError, apply_focus, blank_background, refine_mask


def photo(w=80, h=60):
    """Bright left half, dark right half, a mid-gray square subject."""
    a = np.zeros((h, w, 3), np.uint8)
    a[:, : w // 2] = 230
    a[:, w // 2:] = 30
    a[20:40, 30:50] = 128
    return Image.fromarray(a)


def square_mask(img):
    m = np.zeros((img.height, img.width), np.float32)
    h, w = img.height, img.width
    m[int(h / 3):int(2 * h / 3), int(3 * w / 8):int(5 * w / 8)] = 1.0
    return m


@pytest.fixture
def fake_detect(monkeypatch):
    calls = []

    def det(img, model="fast", allow_download=True):
        calls.append((img.size, model))
        return square_mask(img)

    monkeypatch.setattr(sj, "detect", det)
    return calls


# ---- mask helpers ----

def test_blank_background_keeps_subject_cells():
    art = "\n".join(["#" * 8] * 4)
    mask = np.zeros((40, 80), np.float32)
    mask[:, 40:] = 1.0  # right half is subject
    out = blank_background(art, mask).split("\n")
    assert all(row == "    ####" for row in out)


def test_refine_invert_grow_shrink_feather():
    m = np.zeros((100, 100), np.float32)
    m[40:60, 40:60] = 1
    assert refine_mask(m, FocusOptions(invert_mask=True))[0, 0] == 1.0
    grown = refine_mask(m, FocusOptions(grow=0.05))
    shrunk = refine_mask(m, FocusOptions(grow=-0.05))
    assert (grown > 0.5).sum() > (m > 0.5).sum() > (shrunk > 0.5).sum()
    soft = refine_mask(m, FocusOptions(feather=0.03))
    assert 0 < soft[50, 38] < 1  # the edge is now soft


# ---- apply_focus ----

def test_focus_box_crops_without_any_model(fake_detect):
    r = apply_focus(photo(), FocusOptions(box=(0.25, 0.0, 0.5, 0.5)))
    assert r.image.size == (40, 30) and r.mask is None and not fake_detect


def test_remove_fills_background_with_no_ink(fake_detect):
    r = apply_focus(photo(), FocusOptions(background="remove"))
    a = np.asarray(r.image)
    assert r.blank_background and tuple(a[0, 79]) == (255, 255, 255)  # dark corner made blank
    assert tuple(a[30, 40]) == (128, 128, 128)                       # subject untouched
    inv = np.asarray(apply_focus(photo(), FocusOptions(background="remove"), invert=True).image)
    assert tuple(inv[0, 0]) == (0, 0, 0)  # with Invert, black is "no ink"


@pytest.mark.parametrize("bg,check", [
    ("color", lambda a: tuple(a[0, 0]) == (10, 20, 30)),
    ("blur", lambda a: 60 < a[5, 40, 0] < 200),  # light and dark halves mix at the seam
    ("fade", lambda a: a[0, 79, 0] > 150),  # dark corner pushed toward blank
])
def test_background_modes(fake_detect, bg, check):
    r = apply_focus(photo(), FocusOptions(background=bg, bg_color=(10, 20, 30)))
    a = np.asarray(r.image)
    assert check(a) and tuple(a[30, 40]) == (128, 128, 128) and not r.blank_background


def test_replace_background_with_picture(fake_detect):
    bg = Image.new("RGB", (300, 100), (200, 0, 0))
    r = apply_focus(photo(), FocusOptions(background="image", bg_image=bg))
    a = np.asarray(r.image)
    assert tuple(a[0, 0]) == (200, 0, 0) and tuple(a[30, 40]) == (128, 128, 128)


def test_zoom_crops_to_subject_with_margin(fake_detect):
    r = apply_focus(photo(), FocusOptions(zoom=True, margin=0.0))
    assert r.image.size == (20, 20)
    assert np.asarray(r.image)[10, 10, 0] == 128


def test_enhance_stretches_subject_tones(fake_detect):
    img = photo()
    a = np.asarray(img).copy()
    a[20:40, 30:40] = 100
    a[20:40, 40:50] = 140
    r = apply_focus(Image.fromarray(a), FocusOptions(enhance=True))
    out = np.asarray(r.image)
    assert out[30, 35, 0] < 30 and out[30, 45, 0] > 225


def test_rotation_happens_after_the_box_and_preview_maps_back(fake_detect):
    img = photo(80, 60)
    r = apply_focus(img, FocusOptions(box=(0.0, 0.0, 0.5, 1.0), background="remove"), rotate=90)
    assert r.image.size == (60, 40)  # 40x60 crop, then turned
    assert r.preview.shape == (60, 80)  # over the original, unrotated picture
    assert r.preview[:, 40:].max() == 0  # nothing outside the box


def test_user_mask_needs_no_model(fake_detect):
    m = Image.new("L", (160, 120), 0)
    m.paste(255, (0, 0, 80, 120))  # left half
    r = apply_focus(photo(), FocusOptions(mask=m, background="color", bg_color=(1, 2, 3)))
    a = np.asarray(r.image)
    assert tuple(a[0, 79]) == (1, 2, 3) and tuple(a[0, 0]) == (230, 230, 230) and not fake_detect


@pytest.mark.parametrize("opt,msg", [
    (FocusOptions(background="lava"), "unknown background"),
    (FocusOptions(background="image"), "background picture"),
    (FocusOptions(box=(0.5, 0.5, 0, 0.2)), "fractions"),
    (FocusOptions(subject="huge"), "unknown subject model"),
    (FocusOptions(grow=0.9), "grow"),
])
def test_validation(opt, msg):
    with pytest.raises(SubjectError, match=msg):
        apply_focus(photo(), opt)


# ---- detection plumbing ----

def test_detect_caches_per_image(monkeypatch):
    calls = []
    monkeypatch.setattr(sj, "ensure_model", lambda *a, **k: None)
    monkeypatch.setattr(sj, "_run_model", lambda img, name: calls.append(name) or square_mask(img))
    sj._mask_cache.clear()
    img = photo()
    sj.detect(img, "fast")
    sj.detect(img.copy(), "fast")
    sj.detect(img, "best")
    sj.detect(photo(81, 60), "fast")
    assert calls == ["fast", "best", "fast"]


def test_missing_engine_gives_install_hint(monkeypatch):
    monkeypatch.setattr(sj, "engine_available", lambda: False)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    sj._session.cache_clear()
    with pytest.raises(SubjectError, match=r"\[subject\]"):
        sj.ensure_model("fast", allow_download=True)


def test_missing_model_without_download(monkeypatch, tmp_path):
    monkeypatch.setenv("ASCII_MAGIC_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(sj, "engine_available", lambda: True)
    with pytest.raises(SubjectError, match="subject install fast"):
        sj.ensure_model("fast", allow_download=False)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_model(monkeypatch, tmp_path, payload, declared=None, sha=None):
    monkeypatch.setenv("ASCII_MAGIC_MODELS_DIR", str(tmp_path))
    spec = sj.ModelSpec("t.onnx", sha or hashlib.sha256(payload).hexdigest(), declared or len(payload),
                        320, (0.5,) * 3, (1.0,) * 3, "test")
    monkeypatch.setitem(sj.MODELS, "fast", spec)
    import asciimagic.translate as tr

    monkeypatch.setattr(tr, "_urlopen", lambda url, timeout: _Resp(payload))


def test_install_verifies_checksum(monkeypatch, tmp_path):
    _fake_model(monkeypatch, tmp_path, b"model-bytes")
    path = sj.install("fast")
    assert path.read_bytes() == b"model-bytes" and sj.installed() == ["fast"]
    assert sj.remove("fast") and sj.installed() == []


def test_install_rejects_bad_checksum_and_oversize(monkeypatch, tmp_path):
    _fake_model(monkeypatch, tmp_path, b"model-bytes", sha="0" * 64)
    with pytest.raises(SubjectError, match="checksum"):
        sj.install("fast")
    _fake_model(monkeypatch, tmp_path, b"model-bytes-too-long", declared=5)
    with pytest.raises(SubjectError, match="larger than expected"):
        sj.install("fast")
    assert not list(tmp_path.rglob("*.onnx")) and not list(tmp_path.rglob("*.part"))


# ---- CLI ----

def test_image_cli_remove_background(monkeypatch, tmp_path, capsys, fake_detect):
    from asciimagic.image_to_ascii import main

    src = tmp_path / "p.png"
    photo().save(src)
    monkeypatch.setattr(sys, "argv", ["image-to-ascii", str(src), "-c", "40", "--background", "remove"])
    main()
    out = capsys.readouterr().out.rstrip("\n").split("\n")
    assert out[0].strip() == "" and out[-1].strip() == ""  # top/bottom are background
    assert any(ln.strip() for ln in out)
    assert fake_detect and fake_detect[0][1] == "fast"


def test_image_cli_focus_and_errors(monkeypatch, tmp_path, capsys):
    from asciimagic.image_to_ascii import main

    src = tmp_path / "p.png"
    photo().save(src)
    monkeypatch.setattr(sys, "argv", ["image-to-ascii", str(src), "-c", "20", "--focus", "0,0,0.5,1"])
    main()
    assert capsys.readouterr().out.strip()
    monkeypatch.setattr(sys, "argv", ["image-to-ascii", str(src), "--focus", "1,2"])
    with pytest.raises(SystemExit):
        main()
    monkeypatch.setattr(sys, "argv", ["image-to-ascii", str(src), "--bg-image", str(tmp_path / "nope.png")])
    with pytest.raises(SystemExit, match="couldn't open"):
        main()


def test_subject_cli_list(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ASCII_MAGIC_MODELS_DIR", str(tmp_path))
    assert sj.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "fast" in out and "best" in out and "not installed" in out


# ---- real model (only where it's installed) ----

@pytest.mark.skipif(not (sj.engine_available() and "fast" in sj.installed()), reason="fast model not installed")
def test_real_model_finds_a_centered_subject():
    sj._mask_cache.clear()
    img = Image.new("RGB", (320, 320), (240, 240, 240))
    from PIL import ImageDraw

    ImageDraw.Draw(img).ellipse((100, 80, 220, 240), fill=(150, 40, 40))
    m = sj.detect(img, "fast", allow_download=False)
    assert m[160, 160] > 0.5 and m[10, 10] < 0.5
