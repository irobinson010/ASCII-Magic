"""ASCII Magic - Convert images to ASCII art."""

__version__ = "0.1.6"
__author__ = "Ian Robinson"

"""
Expose lightweight lazy wrappers to avoid importing submodules at package
import time. Importing submodules in `__init__` causes `runpy` to warn when
executing a module with `-m` because the submodule may already appear in
`sys.modules` before execution. Wrappers import on-demand.
"""


def colorize_main(*args, **kwargs):
    from .colorize_ascii import main as _m

    return _m(*args, **kwargs)


def image_to_ascii_main(*args, **kwargs):
    from .image_to_ascii import main as _m

    return _m(*args, **kwargs)


def text_to_ascii_main(*args, **kwargs):
    from .text_to_ascii import main as _m

    return _m(*args, **kwargs)


__all__ = [
    "colorize_main",
    "image_to_ascii_main",
    "text_to_ascii_main",
]
