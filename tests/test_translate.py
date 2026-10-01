"""Translation (issue #38). The model engine is faked so CI never downloads
a 100+ MB model; one optional test uses a real installed model if present."""

import io
import zipfile
from pathlib import Path

import pytest

from asciimagic import translate as tr


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCII_MAGIC_MODELS_DIR", str(tmp_path / "models"))
    tr._load.cache_clear()
    return tmp_path / "models"


def _fake_pair(root: Path, src: str, dst: str) -> None:
    d = root / f"{src}_{dst}"
    (d / "model").mkdir(parents=True)
    (d / "model" / "model.bin").write_bytes(b"x")
    (d / "sentencepiece.model").write_bytes(b"x")


@pytest.fixture
def fake_engine(monkeypatch):
    calls = []

    def run(src, dst, lines):
        calls.append((src, dst, list(lines)))
        return [f"<{dst}:{ln}>" for ln in lines]

    monkeypatch.setattr(tr, "_run", run)
    return calls


# ---- phrasebook / translate() ----

@pytest.mark.parametrize("text,out", [
    ("Good night", "おやすみなさい"),
    ("hello!", "こんにちは！"),
    ("  Thank you.  ", "ありがとう"),
    ("Happy Birthday?", "お誕生日おめでとう？"),
])
def test_phrasebook(models, fake_engine, text, out):
    assert tr.translate(text, "ja") == out
    assert fake_engine == []  # no model needed


def test_translate_uses_model_for_sentences(models, fake_engine):
    _fake_pair(models, "en", "ja")
    out = tr.translate("Good night\n\nThe cat sleeps", "ja")
    assert out == "おやすみなさい\n\n<ja:The cat sleeps>"
    assert fake_engine == [("en", "ja", ["The cat sleeps"])]


def test_translate_pivots_through_english(models, fake_engine):
    _fake_pair(models, "de", "en")
    _fake_pair(models, "en", "ja")
    assert tr.translate("Hallo Welt", "ja", source="de") == "<ja:<en:Hallo Welt>>"


def test_translate_missing_model(models, fake_engine):
    with pytest.raises(tr.TranslationError, match="translate install en ja"):
        tr.translate("The cat sleeps", "ja")


def test_translate_noop_and_bad_codes(models):
    assert tr.translate("same", "en") == "same"
    assert tr.translate("   ", "ja") == "   "
    with pytest.raises(tr.TranslationError):
        tr.translate("x", "../etc")


def test_installed_lists_complete_pairs_only(models):
    _fake_pair(models, "en", "ja")
    (models / "en_fr").mkdir()  # incomplete
    assert tr.installed() == [("en", "ja")]


# ---- install ----

def _archive(tmp_path, names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in names:
            zf.writestr(n, b"data")
    p = tmp_path / "pkg.zip"
    p.write_bytes(buf.getvalue())
    return p


INDEX = [{"from_code": "en", "to_code": "ja", "from_name": "English", "to_name": "Japanese",
          "links": ["https://example.invalid/en_ja.argosmodel"]}]


def _patch_download(monkeypatch, src_zip):
    monkeypatch.setattr(tr, "_download", lambda url, dest, progress=None: dest.write_bytes(src_zip.read_bytes()))


def test_install_unpacks_model(models, tmp_path, monkeypatch):
    _patch_download(monkeypatch, _archive(tmp_path, [
        "translate-en_ja-1_1/model/model.bin", "translate-en_ja-1_1/sentencepiece.model",
        "translate-en_ja-1_1/metadata.json",
    ]))
    path = tr.install("en", "ja", index=INDEX)
    assert (path / "model" / "model.bin").is_file()
    assert tr.installed() == [("en", "ja")]
    assert tr.remove("en", "ja") and tr.installed() == []


def test_install_rejects_zip_slip(models, tmp_path, monkeypatch):
    _patch_download(monkeypatch, _archive(tmp_path, ["../../evil.txt"]))
    with pytest.raises(tr.TranslationError, match="unsafe path"):
        tr.install("en", "ja", index=INDEX)
    assert not (tmp_path / "evil.txt").exists()


def test_install_rejects_incomplete_archive(models, tmp_path, monkeypatch):
    _patch_download(monkeypatch, _archive(tmp_path, ["pkg/README"]))
    with pytest.raises(tr.TranslationError, match="missing"):
        tr.install("en", "ja", index=INDEX)


def test_install_unknown_pair_and_non_https(models):
    with pytest.raises(tr.TranslationError, match="no published model"):
        tr.install("en", "xx", index=INDEX)
    bad = [dict(INDEX[0], links=["http://example.invalid/x"])]
    with pytest.raises(tr.TranslationError, match="https"):
        tr.install("en", "ja", index=bad)


def test_user_agent_is_not_python_urllib(monkeypatch):
    seen = {}

    def fake(req, timeout):
        seen["ua"] = req.get_header("User-agent")
        raise OSError("offline")

    monkeypatch.setattr(tr.urllib.request, "urlopen", fake)
    with pytest.raises(OSError):
        tr.fetch_index()
    assert seen["ua"].startswith("ascii-magic/")


# ---- CLI ----

def test_cli_translate(models, fake_engine, capsys):
    assert tr.main(["Good morning", "--to", "ja"]) == 0
    assert capsys.readouterr().out.strip() == "おはようございます"


def test_cli_missing_model_exits_1(models, fake_engine, capsys):
    assert tr.main(["The cat sleeps", "--to", "ja"]) == 1
    assert "translate install en ja" in capsys.readouterr().err


def test_cli_list(models, capsys):
    _fake_pair(models, "en", "ja")
    assert tr.main(["list"]) == 0
    assert "en->ja" in capsys.readouterr().out


def test_text_to_ascii_translate_flag(models, fake_engine, capsys, monkeypatch):
    from asciimagic.text_to_ascii import main

    monkeypatch.setattr("sys.argv", ["text-to-ascii", "Good night", "-s", "box"])
    main()
    plain = capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["text-to-ascii", "Good night", "--translate", "ja", "-s", "box"])
    main()
    out = capsys.readouterr().out
    assert "おやすみなさい" in out and "Good night" in plain


def test_compose_layer_translate(models, fake_engine):
    from asciimagic.compose import Canvas, Layer, Scene, compose

    scene = Scene(canvas=Canvas(cols=30, rows=3), layers=[Layer(type="text", text="Thank you", translate="ja", style="box")])
    assert "ありがとう" in compose(scene).to_text()


def test_compose_layer_translate_validates():
    from asciimagic.compose import Layer

    with pytest.raises(ValueError):
        Layer(type="text", text="x", translate="Japanese!").validate()


# ---- real model (only if installed locally) ----

@pytest.mark.skipif(not (tr.engine_available() and ("en", "ja") in tr.installed()),
                    reason="en->ja model not installed")
def test_real_model_sentence():
    tr._load.cache_clear()
    out = tr.translate("The cat is sleeping on the sofa.", "ja")
    assert any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in out)


# ---- decoding guards (single words, runaway repetition) ----

class _FakeSP:
    def encode(self, text, out_type=str):
        return list(text)

    def decode(self, toks):
        return "".join(toks)


class _FakeTranslator:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def translate_batch(self, batch, **kw):
        self.calls.append(([("".join(t)) for t in batch], kw))
        return [type("R", (), {"hypotheses": [list(self.reply(("".join(t))))]})() for t in batch]


def _fake_model(monkeypatch, reply):
    fake = _FakeTranslator(reply)
    monkeypatch.setattr(tr, "_load", lambda src, dst: (fake, _FakeSP()))
    return fake


def test_single_word_gets_a_period_hint_that_is_removed(monkeypatch):
    fake = _fake_model(monkeypatch, lambda s: "水。" if s == "water." else "?")
    assert tr._run("en", "ja", ["water", "WATER", "The water is cold"]) == ["水", "水", "?"]
    inputs, kw = fake.calls[0]
    assert inputs == ["water.", "water.", "The water is cold"]  # phrases get no hint


def test_decoding_is_guarded_against_loops(monkeypatch):
    fake = _fake_model(monkeypatch, lambda s: "x")
    tr._run("en", "ja", ["water"])
    _, kw = fake.calls[0]
    assert kw["no_repeat_ngram_size"] == 3 and kw["repetition_penalty"] > 1
    assert kw["max_decoding_length"] < 64  # proportional to a one-word input, not 256


def test_runaway_repetition_is_collapsed(monkeypatch):
    _fake_model(monkeypatch, lambda s: "水,水,水,水,水,水,水,水,水,水")
    assert tr._run("en", "ja", ["water,"]) == ["水"]


@pytest.mark.parametrize("out,src,want", [
    ("水,水,水,水", "water", "水"),
    ("水水水水", "water", "水"),
    ("猫はソファーで寝ています。", "The cat sleeps", "猫はソファーで寝ています。"),
    ("ハ・ハ・ハ・ハ", "Ha ha ha", "ハ・ハ・ハ・ハ"),  # the source repeats, so the output may
])
def test_collapse_runaway(out, src, want):
    assert tr._collapse_runaway(out, src) == want


# ---- subtitle junk from some models (en->zh) ----

@pytest.mark.parametrize("raw,single,want", [
    ("▁{\\fn黑体\\fs22\\bord1\\shad0}天空", True, "天空"),      # ASS/SSA subtitle tags
    ("⁇ 水", True, "水"),                                  # SentencePiece unknown marker
    ("边", True, "边"),                                   # private-use character
    ("天空( T)", True, "天空"),                                 # trailing gloss on a single word
    ("猫(猫咪)在睡觉", False, "猫(猫咪)在睡觉"),                  # sentences keep their parentheses
])
def test_clean(raw, single, want):
    assert tr._clean(raw, single) == want


class _TokTranslator(_FakeTranslator):
    def translate_batch(self, batch, **kw):
        self.calls.append(([("".join(t)) for t in batch], kw))
        return [type("R", (), {"hypotheses": [["<unk>", "水"]]})() for _ in batch]


def test_unknown_tokens_are_dropped_and_hint_is_japanese_only(monkeypatch):
    fake = _TokTranslator(lambda s: "")
    monkeypatch.setattr(tr, "_load", lambda src, dst: (fake, _FakeSP()))
    assert tr._run("en", "zh", ["water"]) == ["水"]   # no "⁇" from <unk>
    assert fake.calls[-1][0] == ["water"]               # Chinese gets no period hint
    tr._run("en", "ja", ["water"])
    assert fake.calls[-1][0] == ["water."]


def test_chinese_phrasebook(models, fake_engine):
    assert tr.translate("Good night", "zh") == "晚安"
    assert tr.translate("Water", "zh") == "水"
    assert fake_engine == []
