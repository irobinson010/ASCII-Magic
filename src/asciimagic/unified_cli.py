# unified_cli.py
import os
import sys
import importlib
import inspect
from typing import Sequence, List, Optional

COMMANDS = {
    "colorize": "asciimagic.colorize_ascii",
    "image": "asciimagic.image_to_ascii",
    "text": "asciimagic.text_to_ascii",
    "greet": "asciimagic.greet",
    "web": "asciimagic.webapp",
    "video": "asciimagic.video",
    "completion": "asciimagic.completion",
    "compose": "asciimagic.compose",
    "presets": "asciimagic.presets",
    "tune": "asciimagic.tune",
    "translate": "asciimagic.translate",
}


def usage(prog: Optional[str] = None) -> None:
    prog = "ascii-magic"
    cmds = ", ".join(sorted(COMMANDS))
    print(f"Usage: {prog} <command> [args...]")
    print(f"Commands: {cmds}")


def _is_closed_stdout(e: OSError) -> bool:
    """A write to a pipe whose reader exited: BrokenPipeError (EPIPE) on
    POSIX, but OSError EINVAL on Windows. EINVAL is ambiguous, so only count
    it when stdout itself is now unusable."""
    import errno

    if isinstance(e, BrokenPipeError) or e.errno == errno.EPIPE:
        return True
    if e.errno != errno.EINVAL:
        return False
    try:
        sys.stdout.flush()
    except OSError:
        return True
    return False


def _call_entry(entry, argv: List[str], module_prog: Optional[str] = None) -> int:
    try:
        sig = inspect.signature(entry)
        # If the callable accepts at least one parameter, pass the argv list.
        # Mains that return no exit code count as success.
        if len(sig.parameters) >= 1:
            ret = entry(argv)
            return ret if isinstance(ret, int) else 0

        # Otherwise, the module expects to parse from sys.argv; temporarily set it.
        old_argv = list(sys.argv)
        try:
            sys.argv = [module_prog or old_argv[0]] + list(argv)
            ret = entry()
            return ret if isinstance(ret, int) else 0
        finally:
            sys.argv = old_argv
    except SystemExit as se:
        code = se.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        # SystemExit("message") is how commands report fatal errors; Python
        # itself prints the message and exits 1 -- do the same, instead of
        # swallowing it as success.
        print(code, file=sys.stderr)
        return 1
    except OSError as e:
        if not _is_closed_stdout(e):
            if os.environ.get("ASCII_MAGIC_DEBUG"):
                raise
            print(f"Error running command: {e} (set ASCII_MAGIC_DEBUG=1 for a traceback)", file=sys.stderr)
            return 1
        # The reader went away (e.g. `ascii-magic image x.png | head`): not an
        # error. Point stdout at devnull so the shutdown flush stays quiet.
        try:
            fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(fd, sys.stdout.fileno())
            os.close(fd)
        except (OSError, ValueError):
            pass
        return 0
    except Exception as e:
        if os.environ.get("ASCII_MAGIC_DEBUG"):
            raise
        print(f"Error running command: {e} (set ASCII_MAGIC_DEBUG=1 for a traceback)", file=sys.stderr)
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .console import utf8_stdout

    utf8_stdout()
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    if not argv or argv[0] in ("-h", "--help"):
        usage(sys.argv[0])
        return 0

    if argv[0] in ("-V", "--version"):
        from asciimagic import __version__

        print(f"ascii-magic {__version__}")
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
