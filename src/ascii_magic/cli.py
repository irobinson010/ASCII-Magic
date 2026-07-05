"""Unified ``ascii-magic`` command with subcommands.

Each subcommand dispatches to the corresponding module's ``main()``, so the
standalone console scripts (image-to-ascii, colorize-ascii, ...) and this
umbrella command stay behaviorally identical.
"""

from __future__ import annotations

import sys
from typing import List, Optional

COMMANDS = {
    "convert": ("ascii_magic.image_to_ascii", "Convert an image to ASCII art"),
    "text": ("ascii_magic.text_to_ascii", "Render text as ASCII art"),
    "colorize": ("ascii_magic.colorize_ascii", "Colorize ASCII art (ANSI/HTML/matrix/animation)"),
    "greet": ("ascii_magic.greet", "Install art as a shell login greeting"),
    "web": ("ascii_magic.webapp", "Start the web GUI"),
}


def _print_help(out=None) -> None:
    from . import __version__

    out = out or sys.stdout
    out.write(f"ascii-magic {__version__} — images and text to colorized ASCII art\n\n")
    out.write("usage: ascii-magic <command> [options]\n\ncommands:\n")
    for name, (_, desc) in COMMANDS.items():
        out.write(f"  {name:10s} {desc}\n")
    out.write("\nRun 'ascii-magic <command> --help' for command options.\n")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    if len(argv) < 2 or argv[1] in ("-h", "--help", "help"):
        _print_help()
        return 0
    if argv[1] == "--version":
        from . import __version__

        print(__version__)
        return 0

    cmd = argv[1]
    entry = COMMANDS.get(cmd)
    if entry is None:
        print(f"ascii-magic: unknown command '{cmd}'\n", file=sys.stderr)
        _print_help(out=sys.stderr)
        return 2

    module_name, _ = entry
    import importlib

    module = importlib.import_module(module_name)
    # Sub-mains read sys.argv; present them as their own program.
    sys.argv = [f"ascii-magic {cmd}"] + argv[2:]
    result = module.main()
    return int(result) if result is not None else 0


if __name__ == "__main__":
    sys.exit(main())
