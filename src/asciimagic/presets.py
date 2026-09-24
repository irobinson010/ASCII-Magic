"""Named, reusable settings for the CLI commands.

A preset is a saved set of flag values for one command (image, colorize,
video, text). Loading one sets those values as defaults, so any flag given
on the command line still wins::

    ascii-magic image cat.png --mode braille --dither --invert --save-preset cat-look
    ascii-magic image dog.png --preset cat-look            # same look, new picture
    ascii-magic image dog.png --preset cat-look --cols 60  # ...but narrower

Presets live in ``<config dir>/presets.json`` (``~/.config/ascii-magic`` or
``$XDG_CONFIG_HOME/ascii-magic``), next to the login greeting. A few
built-in presets ship with the package; a saved preset with the same name
takes precedence over the built-in one. ``ascii-magic presets`` lists,
shows, and deletes them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Inputs, outputs, and bookkeeping flags never belong in a preset: a preset
# is "how it looks", not "which file".
_NEVER_SAVED = frozenset({
    "input", "output", "out", "img_path", "ascii_path", "text", "stdin", "caption",
    "preset", "save_preset", "help", "debug", "log", "log_path", "log_level",
})

BUILTIN: Dict[str, Dict[str, Dict[str, Any]]] = {
    "image": {
        "photo": {
            "_about": "Photos on a light background: braille dots, dithered, contrast-stretched",
            "mode": "braille", "dither": True, "autocontrast": True,
        },
        "photo-dark-terminal": {
            "_about": "Photos for a dark terminal: bright areas become ink",
            "mode": "braille", "dither": True, "autocontrast": True, "invert": True,
        },
        "line-art": {
            "_about": "Cartoons, logos, line art: glyph matching at best quality",
            "mode": "glyph", "quality": "best",
        },
        "ssh-safe": {
            "_about": "Plain printable ASCII (no braille/Unicode) and 256 colors: shows up on any server",
            "mode": "glyph", "ascii": "printable", "unicode": "off", "color_depth": "256",
        },
    },
    "colorize": {
        "ssh-safe": {"_about": "256 colors: safe over SSH", "color_depth": "256"},
    },
    "video": {
        "ssh-safe": {"_about": "256 colors: safe over SSH", "color_depth": "256"},
    },
    "text": {},
}

COMMANDS = tuple(BUILTIN)


def presets_path() -> Path:
    from .greet import config_dir

    return config_dir() / "presets.json"


def _load_user() -> Dict[str, Dict[str, Dict[str, Any]]]:
    p = presets_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"Could not read presets file {p}: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"Presets file {p} is not a JSON object")
    return data


def _save_user(data: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    p = presets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def available(command: str) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """name -> (source, settings); user presets shadow built-ins."""
    out = {name: ("built-in", dict(s)) for name, s in BUILTIN.get(command, {}).items()}
    for name, s in (_load_user().get(command) or {}).items():
        if isinstance(s, dict):
            out[name] = ("saved", dict(s))
    return out


def get(command: str, name: str) -> Dict[str, Any]:
    found = available(command)
    if name not in found:
        names = ", ".join(sorted(found)) or "(none)"
        raise SystemExit(f"Unknown {command} preset {name!r}. Available: {names}")
    return {k: v for k, v in found[name][1].items() if not k.startswith("_")}


def save(command: str, name: str, settings: Dict[str, Any], about: Optional[str] = None) -> Path:
    if not name or name.startswith("_") or "/" in name:
        raise SystemExit(f"Invalid preset name {name!r}")
    data = _load_user()
    entry = dict(settings)
    if about:
        entry["_about"] = about
    data.setdefault(command, {})[name] = entry
    _save_user(data)
    return presets_path()


def delete(command: str, name: str) -> bool:
    data = _load_user()
    if name in (data.get(command) or {}):
        del data[command][name]
        _save_user(data)
        return True
    return False


# ---- argparse integration ----


def add_preset_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("presets")
    g.add_argument("--preset", default=None, metavar="NAME",
                   help="Start from a saved or built-in preset; flags you pass still win "
                        "(list them: ascii-magic presets)")
    g.add_argument("--save-preset", default=None, metavar="NAME",
                   help="Save this run's look (every non-default flag except input/output) as NAME")


def _savable_actions(parser: argparse.ArgumentParser) -> Dict[str, argparse.Action]:
    return {
        a.dest: a for a in parser._actions
        if a.option_strings and a.dest not in _NEVER_SAVED and a.dest != argparse.SUPPRESS
    }


def _validate(parser: argparse.ArgumentParser, command: str, name: str, settings: Dict[str, Any]) -> None:
    actions = _savable_actions(parser)
    for key, value in settings.items():
        action = actions.get(key)
        if action is None:
            parser.error(f"preset {name!r}: {key!r} is not a {command} setting")
        if action.choices is not None and value is not None and value not in action.choices:
            parser.error(f"preset {name!r}: {key}={value!r} is not one of {list(action.choices)}")


def parse_args(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]], command: str) -> argparse.Namespace:
    """parse_args with --preset/--save-preset support (flags beat the preset)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset")
    known, _ = pre.parse_known_args(argv)

    defaults = {a.dest: a.default for a in _savable_actions(parser).values()}
    if known.preset:
        settings = get(command, known.preset)
        _validate(parser, command, known.preset, settings)
        parser.set_defaults(**settings)

    ns = parser.parse_args(argv)

    if getattr(ns, "save_preset", None):
        chosen = {
            dest: getattr(ns, dest)
            for dest in _savable_actions(parser)
            if hasattr(ns, dest) and getattr(ns, dest) != defaults.get(dest)
        }
        about = f"from {known.preset}" if known.preset else None
        path = save(command, ns.save_preset, chosen, about)
        shown = " ".join(f"{k}={v}" for k, v in sorted(chosen.items())) or "(all defaults)"
        print(f"Saved {command} preset {ns.save_preset!r} to {path}: {shown}", file=sys.stderr)
    return ns


# ---- `ascii-magic presets` ----


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ascii-magic presets", description="List, show, or delete presets.")
    sub = ap.add_subparsers(dest="action")
    p = sub.add_parser("list", help="List presets (default)")
    p.add_argument("command", nargs="?", choices=COMMANDS)
    p = sub.add_parser("show", help="Show a preset's settings")
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("name")
    p = sub.add_parser("delete", help="Delete a saved preset")
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("name")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    from .console import utf8_stdout

    utf8_stdout()
    args = build_arg_parser().parse_args(argv)
    action = args.action or "list"
    if action == "list":
        for command in ([args.command] if getattr(args, "command", None) else COMMANDS):
            found = available(command)
            if not found:
                continue
            print(f"{command}:")
            for name in sorted(found):
                source, s = found[name]
                about = s.get("_about") or " ".join(
                    f"--{k.replace('_', '-')}" + ("" if v is True else f" {v}")
                    for k, v in s.items() if not k.startswith("_")
                )
                print(f"  {name:<22} [{source}] {about}")
        print(f"\nUse: ascii-magic <command> ... --preset NAME   (saved presets: {presets_path()})")
        return 0
    if action == "show":
        print(json.dumps(get(args.command, args.name), indent=2, sort_keys=True))
        return 0
    if delete(args.command, args.name):
        print(f"Deleted {args.command} preset {args.name!r}")
        return 0
    if args.name in BUILTIN.get(args.command, {}):
        print(f"{args.name!r} is built in and can't be deleted (save one with the same name to override it)",
              file=sys.stderr)
    else:
        print(f"No saved {args.command} preset {args.name!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
