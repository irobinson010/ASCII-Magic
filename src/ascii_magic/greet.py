"""Install ASCII art as a shell login greeting.

`ascii-magic-greet install art.ans` copies the art into the user's config
directory and appends a guarded, marker-delimited block to their shell rc
file so it prints on interactive logins (the classic "cat greets you when
you SSH in"). Animated greetings use the `.frames` format written by
`colorize-ascii --animate out.frames`: a JSON header line, then ANSI frames
separated by the FS control character.

Subcommands: install / remove / show / preview / status / play.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ESC = "\x1b"
FRAME_SEP = "\x1c"
MARK_BEGIN = "# >>> ascii-magic greeting >>>"
MARK_END = "# <<< ascii-magic greeting <<<"


def _home() -> Path:
    # Path.home() reads USERPROFILE on Windows and ignores a HOME override;
    # honor HOME explicitly so tests and unusual setups behave predictably.
    env = os.environ.get("HOME")
    return Path(env) if env else Path.home()


def config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else _home() / ".config"
    return base / "ascii-magic"


def default_rc_path() -> Path:
    shell = os.path.basename(os.environ.get("SHELL", "bash"))
    rc = ".zshrc" if shell == "zsh" else ".bashrc"
    return _home() / rc


def _strip_block(text: str) -> str:
    """Remove the greeting block. Raises ValueError on unbalanced markers so
    a hand-damaged block never causes silent loss of the rc content below it."""
    lines = text.splitlines()
    out: List[str] = []
    inside = False
    for ln in lines:
        if ln.strip() == MARK_BEGIN:
            if inside:
                raise ValueError("nested begin marker")
            inside = True
            continue
        if ln.strip() == MARK_END:
            if not inside:
                raise ValueError("end marker without begin marker")
            inside = False
            continue
        if not inside:
            out.append(ln)
    if inside:
        raise ValueError("begin marker without matching end marker")
    # Drop trailing blank lines left behind by a removed block
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + ("\n" if out else "")


def _write_rc(rc: Path, content: str) -> None:
    """Atomic replace so a crash mid-write can never truncate the rc file."""
    tmp = rc.parent / (rc.name + ".ascii-magic-tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, rc)


def _rewrite_rc(rc: Path, make_content) -> bool:
    """Strip our block (validated) and write the result atomically.
    Returns False (after printing) if the block markers are damaged."""
    existing = rc.read_text(encoding="utf-8") if rc.exists() else ""
    try:
        cleaned = _strip_block(existing)
    except ValueError as e:
        print(
            f"error: greeting block in {rc} looks damaged ({e}); "
            "fix or remove the marker lines by hand, then retry.",
            file=sys.stderr,
        )
        return False
    _write_rc(rc, make_content(cleaned))
    return True


def _greeting_block(target: Path) -> str:
    if target.suffix == ".frames":
        cmd = (
            'command -v ascii-magic-greet >/dev/null 2>&1 '
            "&& ascii-magic-greet show"
        )
    else:
        q = shlex.quote(str(target))
        cmd = f"[ -r {q} ] && cat {q}"
    return (
        f"{MARK_BEGIN}\n"
        'case "$-" in *i*)\n'
        f'    if [ -t 1 ] && [ -z "$ASCII_MAGIC_NO_GREETING" ]; then\n'
        f"        {cmd}\n"
        "    fi\n"
        ";;\n"
        "esac\n"
        f"{MARK_END}\n"
    )


def _rc_supported(rc: Path) -> bool:
    """The block is POSIX-shell syntax; refuse targets it would break."""
    name = rc.name.lower()
    if "fish" in name or "csh" in name:
        print(f"error: {rc.name} is not a POSIX shell rc file; only bash/zsh-style rc files are supported.", file=sys.stderr)
        return False
    if rc.is_symlink():
        print(f"error: {rc} is a symlink; pass the real rc file with --rc.", file=sys.stderr)
        return False
    return True


def installed_greeting() -> Optional[Path]:
    d = config_dir()
    for name in ("greeting.frames", "greeting.ans"):
        p = d / name
        if p.exists():
            return p
    return None


# ---- frames format ----

def read_frames_file(path: Path) -> Tuple[List[str], float, int]:
    """Returns (frames, fps, loops)."""
    raw = path.read_text(encoding="utf-8")
    head, _, body = raw.partition("\n")
    meta = json.loads(head)
    frames = body.split(FRAME_SEP)
    return frames, float(meta.get("fps", 12)), int(meta.get("loops", 1))


def write_frames_file(path: Path, frames: List[str], fps: float, loops: int = 1) -> None:
    path.write_text(
        json.dumps({"fps": fps, "loops": loops}) + "\n" + FRAME_SEP.join(frames),
        encoding="utf-8",
    )


def play_frames(frames: List[str], fps: float, loops: int, out=None) -> None:
    out = out or sys.stdout
    delay = 1.0 / max(fps, 0.1)
    try:
        out.write(f"{ESC}[2J{ESC}[?25l")
        n = 0
        while loops == 0 or n < loops:
            for f in frames:
                out.write(f"{ESC}[H" + f)
                out.flush()
                time.sleep(delay)
            n += 1
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        try:
            out.write(f"{ESC}[0m{ESC}[?25h\n")
            out.flush()
        except BrokenPipeError:
            # Stop the interpreter-shutdown flush from complaining too.
            if out is sys.stdout:
                fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(fd, sys.stdout.fileno())
                os.close(fd)


# ---- commands ----

def cmd_install(args) -> int:
    if os.name == "nt":
        print(
            "error: greet install targets POSIX shells (.bashrc/.zshrc) and is "
            "not supported on Windows.",
            file=sys.stderr,
        )
        return 1
    src = Path(args.file)
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 1
    if src.suffix not in (".ans", ".txt", ".frames"):
        print("error: greeting must be a .ans, .txt, or .frames file", file=sys.stderr)
        return 1

    rc = Path(args.rc) if args.rc else default_rc_path()
    if not _rc_supported(rc):
        return 1

    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    # Only one greeting at a time — clear both variants.
    for name in ("greeting.ans", "greeting.frames"):
        (d / name).unlink(missing_ok=True)
    ext = ".frames" if src.suffix == ".frames" else ".ans"
    target = d / f"greeting{ext}"
    shutil.copyfile(src, target)

    block = _greeting_block(target)
    ok = _rewrite_rc(
        rc,
        lambda cleaned: cleaned
        + ("\n" if cleaned and not cleaned.endswith("\n\n") else "")
        + block,
    )
    if not ok:
        return 1

    print(f"Installed greeting: {target}")
    print(f"Shell hook added to: {rc}")
    if ext == ".frames":
        print("Animated greeting plays via 'ascii-magic-greet show' (must be on PATH in login shells).")
    print("Set ASCII_MAGIC_NO_GREETING=1 to suppress it temporarily.")
    return 0


def cmd_remove(args) -> int:
    rc = Path(args.rc) if args.rc else default_rc_path()
    if rc.exists():
        if not _rewrite_rc(rc, lambda cleaned: cleaned):
            return 1
        print(f"Removed shell hook from: {rc}")
    removed = False
    for name in ("greeting.ans", "greeting.frames"):
        p = config_dir() / name
        if p.exists():
            p.unlink()
            removed = True
    if removed:
        print(f"Removed greeting files from: {config_dir()}")
    return 0


def cmd_show(args) -> int:
    p = installed_greeting()
    if p is None:
        print("No greeting installed.", file=sys.stderr)
        return 1
    if p.suffix == ".frames":
        frames, fps, loops = read_frames_file(p)
        play_frames(frames, fps, loops)
    else:
        sys.stdout.write(p.read_text(encoding="utf-8"))
    return 0


def cmd_status(args) -> int:
    p = installed_greeting()
    rc = Path(args.rc) if args.rc else default_rc_path()
    hooked = rc.exists() and MARK_BEGIN in rc.read_text(encoding="utf-8")
    print(f"Greeting file: {p if p else '(none)'}")
    print(f"Shell hook in {rc}: {'yes' if hooked else 'no'}")
    return 0


def cmd_play(args) -> int:
    frames, fps, loops = read_frames_file(Path(args.file))
    play_frames(frames, args.fps or fps, args.loops if args.loops is not None else loops)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ascii-magic-greet",
        description="Install ASCII art as a shell login greeting.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("install", help="Install a .ans/.txt/.frames file as the login greeting")
    p.add_argument("file")
    p.add_argument("--rc", default=None, help="Shell rc file to hook (default: ~/.bashrc or ~/.zshrc)")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("remove", help="Remove the greeting and its shell hook")
    p.add_argument("--rc", default=None)
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("show", help="Display the installed greeting (used by the shell hook)")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("preview", help="Alias for show")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("status", help="Report what is installed where")
    p.add_argument("--rc", default=None)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("play", help="Play a .frames animation file in the terminal")
    p.add_argument("file")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--loops", type=int, default=None)
    p.set_defaults(func=cmd_play)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
