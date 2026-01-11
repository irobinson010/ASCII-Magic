"""ASCII Magic - Convert images to ASCII art."""

__version__ = "0.1.6"
__author__ = "Ian Robinson"

from .colorize_ascii import main as colorize_main
from .image_to_ascii import main as image_to_ascii_main
from .text_to_ascii import (
    main as text_to_ascii_main,
    text_to_ascii_art,
    text_to_box,
    text_to_banner,
)

__all__ = [
    "colorize_main",
    "image_to_ascii_main",
    "text_to_ascii_main",
    "text_to_ascii_art",
    "text_to_box",
    "text_to_banner",
]
