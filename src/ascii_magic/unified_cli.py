# unified_cli.py
import sys
import importlib
import inspect
from typing import Sequence, List, Optional

COMMANDS = {
    "colorize": "ascii_magic.colorize_ascii",
    "image": "ascii_magic.image_to_ascii",
    "text": "ascii_magic.text_to_ascii",
}


def usage(prog: Optional[str] = None) -> None:
    prog = "ascii-magic"
    cmds = ", ".join(sorted(COMMANDS))
    print(f"Usage: {prog} <command> [args...]")
    print(f"Commands: {cmds}")


def _call_entry(entry, argv: List[str], module_prog: Optional[str] = None) -> int:
    try:
        sig = inspect.signature(entry)
        # If the callable accepts at least one parameter, pass the argv list.
        if len(sig.parameters) >= 1:
            return entry(argv)

        # Otherwise, the module expects to parse from sys.argv; temporarily set it.
        old_argv = list(sys.argv)
        try:
            sys.argv = [module_prog or old_argv[0]] + list(argv)
            return entry()
        finally:
            sys.argv = old_argv
    except SystemExit as se:
        code = se.code
        return code if isinstance(code, int) else 0
    except Exception as e:
        print(f"Error running command: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    if not argv or argv[0] in ("-h", "--help"):
        usage(sys.argv[0])
        return 0

    cmd, *args = argv
    module_path = COMMANDS.get(cmd)
    if not module_path:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        usage(sys.argv[0])
        return 2

    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        print(f"Failed to import command '{cmd}' ({module_path}): {e}", file=sys.stderr)
        return 3

    entry = getattr(module, "main", None)
    if not callable(entry):
        print(f"Command module '{module_path}' has no callable 'main'", file=sys.stderr)
        return 4

    return _call_entry(entry, args, module_prog=f"ascii-magic {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
