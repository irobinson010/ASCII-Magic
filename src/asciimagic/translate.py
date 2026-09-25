"""Free, offline translation for captions and text art.

Uses Argos Translate's open models, run directly with CTranslate2 and
SentencePiece (the optional ``[translate]`` extra). That skips the
``argostranslate`` package, whose released versions pull in PyTorch. Models
download once from the official Argos package index and run locally: no API
key, no per-use cost, and the text never leaves the machine::

    ascii-magic translate install en ja          # one-time, ~120 MB
    ascii-magic translate "Good morning" --to ja
    ascii-magic text "Happy birthday" --translate ja -s block

Machine translation is weakest on one- or two-word phrases ("Hi", "Sorry"),
which are exactly what captions often are, so a small phrasebook of common
short phrases is consulted first for Japanese. Always proofread the result;
the web GUI puts it in an editable box for that reason.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

INDEX_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
MAX_MODEL_BYTES = 600 * 1024 * 1024
_install_lock = threading.Lock()


class TranslationError(RuntimeError):
    """Translation can't run: missing extra, missing model, or bad input."""


# ---- where models live ----


def models_dir() -> Path:
    """ASCII_MAGIC_MODELS_DIR, else the platform's per-user data directory."""
    env = os.environ.get("ASCII_MAGIC_MODELS_DIR")
    if env:
        return Path(env)
    if sys.platform.startswith("win") and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "ascii-magic" / "models"
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "ascii-magic" / "models"


def _pair_dir(src: str, dst: str) -> Path:
    return models_dir() / f"{src}_{dst}"


def _check_code(code: str) -> str:
    if not re.fullmatch(r"[a-z]{2,3}(?:_[A-Za-z]{2,4})?", code or ""):
        raise TranslationError(f"invalid language code {code!r} (expected e.g. 'en', 'ja')")
    return code


def installed() -> List[Tuple[str, str]]:
    root = models_dir()
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if (d / "model" / "model.bin").is_file() and (d / "sentencepiece.model").is_file() and "_" in d.name:
            src, dst = d.name.split("_", 1)
            out.append((src, dst))
    return out


def engine_available() -> bool:
    try:
        import ctranslate2  # noqa: F401
        import sentencepiece  # noqa: F401
    except ImportError:
        return False
    return True


def _require_engine():
    try:
        import ctranslate2
        import sentencepiece
    except ImportError:
        raise TranslationError(
            "Translation needs the [translate] extra:\n"
            '    pip install "ascii-magic-tools[translate]"   (or: uv sync --extra translate)'
        ) from None
    return ctranslate2, sentencepiece


# ---- package index / install ----


def _urlopen(url: str, timeout: float):
    """The model host rejects Python's default "Python-urllib" user agent
    (HTTP 403), so identify the app honestly instead."""
    from . import __version__

    req = urllib.request.Request(url, headers={
        "User-Agent": f"ascii-magic/{__version__} (+https://github.com/irobinson010/ASCII-Magic)",
    })
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_index(url: str = INDEX_URL, timeout: float = 30) -> List[dict]:
    with _urlopen(url, timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, list):
        raise TranslationError("unexpected package index format")
    return data


def available(index: Optional[List[dict]] = None) -> List[Tuple[str, str, str, str]]:
    """(from_code, to_code, from_name, to_name) for every published model."""
    index = fetch_index() if index is None else index
    return sorted(
        (p.get("from_code", ""), p.get("to_code", ""), p.get("from_name", ""), p.get("to_name", ""))
        for p in index if p.get("from_code") and p.get("to_code")
    )


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if root != target and root not in target.parents:
            raise TranslationError(f"refusing unsafe path in model archive: {member.filename!r}")
    zf.extractall(dest)


def install(src: str, dst: str, index: Optional[List[dict]] = None, progress=None) -> Path:
    """Download and unpack the src->dst model from the Argos index."""
    _check_code(src)
    _check_code(dst)
    with _install_lock:
        target = _pair_dir(src, dst)
        if (target / "model" / "model.bin").is_file():
            return target
        index = fetch_index() if index is None else index
        pkg = next((p for p in index if p.get("from_code") == src and p.get("to_code") == dst), None)
        if pkg is None:
            raise TranslationError(f"no published model for {src} -> {dst}; see `ascii-magic translate list --available`")
        links = [u for u in pkg.get("links", []) if isinstance(u, str) and u.startswith("https://")]
        if not links:
            raise TranslationError(f"model {src} -> {dst} has no https download link")

        models_dir().mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=models_dir()) as tmp:
            archive = Path(tmp) / "model.zip"
            _download(links[0], archive, progress)
            try:
                with zipfile.ZipFile(archive) as zf:
                    _safe_extract(zf, Path(tmp) / "x")
            except zipfile.BadZipFile:
                raise TranslationError("downloaded model is not a valid archive") from None
            found = next((p.parent for p in (Path(tmp) / "x").rglob("sentencepiece.model")
                          if (p.parent / "model" / "model.bin").is_file()), None)
            if found is None:
                raise TranslationError("model archive is missing model/ or sentencepiece.model")
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(found), str(target))
        _load.cache_clear()
        return target


def _download(url: str, dest: Path, progress=None) -> None:
    with _urlopen(url, 60) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        if total > MAX_MODEL_BYTES:
            raise TranslationError(f"model is {total // 2**20} MB, over the {MAX_MODEL_BYTES // 2**20} MB limit")
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            done += len(chunk)
            if done > MAX_MODEL_BYTES:
                raise TranslationError("model download exceeded the size limit")
            out.write(chunk)
            if progress:
                progress(done, total)


def remove(src: str, dst: str) -> bool:
    d = _pair_dir(_check_code(src), _check_code(dst))
    if d.is_dir():
        shutil.rmtree(d)
        _load.cache_clear()
        return True
    return False


# ---- translation ----


@lru_cache(maxsize=4)
def _load(src: str, dst: str):
    ct2, spm = _require_engine()
    d = _pair_dir(src, dst)
    if not (d / "model" / "model.bin").is_file():
        raise TranslationError(
            f"the {src} -> {dst} model isn't installed. Run: ascii-magic translate install {src} {dst}"
        )
    translator = ct2.Translator(str(d / "model"), device="cpu", compute_type="int8")
    sp = spm.SentencePieceProcessor(model_file=str(d / "sentencepiece.model"))
    return translator, sp


def _run(src: str, dst: str, lines: List[str]) -> List[str]:
    translator, sp = _load(src, dst)
    todo = [ln for ln in lines if ln.strip()]
    if not todo:
        return lines
    results = translator.translate_batch(
        [sp.encode(ln, out_type=str) for ln in todo], beam_size=4, max_decoding_length=256,
    )
    it = iter(sp.decode(r.hypotheses[0]) for r in results)
    return [next(it) if ln.strip() else ln for ln in lines]


# Machine translation is weakest on the short, context-free phrases that
# captions are made of ("Hi" -> "hearing", "Sorry" -> "new information"),
# so common ones are looked up first. Keys are lowercase, punctuation-free.
PHRASEBOOK: Dict[str, Dict[str, str]] = {
    "ja": {
        "hello": "こんにちは", "hi": "やあ", "hey": "ねえ",
        "good morning": "おはようございます", "good afternoon": "こんにちは",
        "good evening": "こんばんは", "good night": "おやすみなさい",
        "goodbye": "さようなら", "bye": "バイバイ", "see you": "またね",
        "see you later": "また後で", "see you tomorrow": "また明日",
        "thank you": "ありがとう", "thanks": "ありがとう", "thank you very much": "どうもありがとうございます",
        "welcome": "ようこそ", "welcome home": "おかえりなさい", "welcome back": "おかえりなさい",
        "yes": "はい", "no": "いいえ", "sorry": "ごめんなさい", "excuse me": "すみません",
        "congratulations": "おめでとうございます", "happy birthday": "お誕生日おめでとう",
        "happy new year": "明けましておめでとうございます", "merry christmas": "メリークリスマス",
        "good luck": "頑張って", "cheers": "乾杯", "love": "愛", "i love you": "愛してる",
        "friend": "友達", "friends": "友達", "peace": "平和", "home": "家", "hope": "希望",
        "dream": "夢", "dreams": "夢", "family": "家族", "happiness": "幸せ", "courage": "勇気",
        "hello world": "ハローワールド", "game over": "ゲームオーバー", "the end": "おわり",
    },
}

_TRAILING = re.compile(r"[\s!?.。！？、,]+$")


def _phrase(line: str, dst: str) -> Optional[str]:
    book = PHRASEBOOK.get(dst)
    if not book:
        return None
    m = _TRAILING.search(line)
    tail = m.group(0).strip() if m else ""
    key = _TRAILING.sub("", line).strip().lower()
    hit = book.get(key)
    if hit is None:
        return None
    tail = tail.replace("!", "！").replace("?", "？") if dst in ("ja", "zh") else tail
    return hit + (tail if tail not in (".",) else "")


def translate(text: str, to: str, source: str = "en") -> str:
    """Translate text line by line (blank lines kept). Uses a direct model,
    or pivots through English when both halves are installed."""
    _check_code(to)
    _check_code(source)
    if to == source or not text.strip():
        return text
    lines = text.split("\n")

    phrased = [_phrase(ln, to) if source == "en" else None for ln in lines]
    need = [ln for ln, p in zip(lines, phrased) if p is None and ln.strip()]
    translated: Dict[str, str] = {}
    if need:
        have = set(installed())
        if (source, to) in have:
            outs = _run(source, to, need)
        elif source != "en" and to != "en" and {(source, "en"), ("en", to)} <= have:
            outs = _run("en", to, _run(source, "en", need))
        else:
            raise TranslationError(
                f"the {source} -> {to} model isn't installed. Run: ascii-magic translate install {source} {to}"
            )
        translated = dict(zip(need, outs))
    return "\n".join(
        p if p is not None else (translated.get(ln, ln) if ln.strip() else ln)
        for ln, p in zip(lines, phrased)
    )


def translate_or_exit(text: str, to: Optional[str], source: str = "en", prog: str = "ascii-magic") -> str:
    """For CLI flags like --translate: translate, or exit with the reason."""
    if not to:
        return text
    try:
        return translate(text, to, source)
    except TranslationError as e:
        raise SystemExit(f"{prog}: translation failed: {e}")


def add_translate_arg(parser, flag: str = "--translate", dest: str = "translate", what: str = "the text") -> None:
    parser.add_argument(
        flag, dest=dest, default=None, metavar="LANG",
        help=f"Translate {what} from English to LANG (e.g. ja) before rendering; "
             "needs `ascii-magic translate install en LANG` once",
    )


# ---- CLI ----


def _usage() -> str:
    return (
        "usage: ascii-magic translate TEXT --to LANG [--from LANG]\n"
        "       ascii-magic translate install FROM TO\n"
        "       ascii-magic translate remove FROM TO\n"
        "       ascii-magic translate list [--available]\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    from .console import utf8_stdout

    utf8_stdout()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_usage() + "\nFree offline translation (Argos models). Machine translation: proofread it.\n")
        return 0

    try:
        if argv[0] in ("install", "remove"):
            ap = argparse.ArgumentParser(prog=f"ascii-magic translate {argv[0]}")
            ap.add_argument("src")
            ap.add_argument("dst")
            a = ap.parse_args(argv[1:])
            if argv[0] == "remove":
                ok = remove(a.src, a.dst)
                print(f"Removed {a.src} -> {a.dst}" if ok else f"{a.src} -> {a.dst} is not installed")
                return 0 if ok else 1

            def progress(done, total):
                if total:
                    print(f"\rDownloading {a.src} -> {a.dst}: {done * 100 // total:3d}% "
                          f"of {total // 2**20} MB", end="", file=sys.stderr, flush=True)

            path = install(a.src, a.dst, progress=progress)
            print(f"\nInstalled {a.src} -> {a.dst} in {path}", file=sys.stderr)
            if not engine_available():
                print('Note: translating also needs: pip install "ascii-magic-tools[translate]"', file=sys.stderr)
            return 0

        if argv[0] == "list":
            ap = argparse.ArgumentParser(prog="ascii-magic translate list")
            ap.add_argument("--available", action="store_true", help="also list downloadable models")
            a = ap.parse_args(argv[1:])
            pairs = installed()
            print("Installed: " + (", ".join(f"{s}->{d}" for s, d in pairs) or "(none)"))
            print(f"Models folder: {models_dir()}")
            if a.available:
                for s, d, sn, dn in available():
                    print(f"  {s}->{d}  {sn} to {dn}")
            return 0

        ap = argparse.ArgumentParser(prog="ascii-magic translate", usage=_usage())
        ap.add_argument("text")
        ap.add_argument("--to", required=True)
        ap.add_argument("--from", dest="source", default="en")
        a = ap.parse_args(argv)
        print(translate(a.text, a.to, a.source))
        return 0
    except TranslationError as e:
        print(f"ascii-magic translate: {e}", file=sys.stderr)
        return 1
    except OSError as e:  # network errors while downloading
        print(f"ascii-magic translate: download failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
