"""Tests for text_to_ascii module."""

import pytest
from PIL import Image
from ascii_magic.text_to_ascii import (
    text_to_box,
    text_to_banner,
    text_to_ascii_art,
    render_text_to_image,
)

# --- Fixtures ---


@pytest.fixture
def sample_text():
    return "Hello World"


# --- Tests ---


class TestTextToBox:
    def test_simple_text(self, sample_text):
        result = text_to_box(sample_text)
        lines = result.splitlines()
        assert sample_text in result
        assert len(lines) == 3
        # Ensure all lines are the same width (perfect rectangle)
        assert len(set(len(line) for line in lines)) == 1

    def test_multiline_text(self):
        result = text_to_box("Line 1\nLine 2")
        assert len(result.splitlines()) == 4

    def test_width_constraint(self):
        width = 20
        result = text_to_box("This is a long sentence", width=width)
        for line in result.splitlines():
            assert len(line) <= width


class TestTextToBanner:
    @pytest.mark.parametrize("char", ["#", "*", "@"])
    def test_custom_chars(self, char):
        result = text_to_banner("Test", char=char)
        assert char in result
        assert len(result.splitlines()) == 3

    def test_empty_input(self):
        """Edge case: empty string."""
        result = text_to_banner("")
        assert len(result.splitlines()) == 3


class TestRenderTextToImage:
    def test_basic_render(self, sample_text):
        img = render_text_to_image(sample_text)
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0

    def test_font_size_scaling(self):
        small = render_text_to_image("Size", font_size=10)
        large = render_text_to_image("Size", font_size=40)
        assert large.size[0] > small.size[0]
        assert large.size[1] > small.size[1]


class TestTextToAsciiArt:
    @pytest.mark.parametrize("style", ["block", "small", "shadow"])
    def test_styles(self, style):
        result = text_to_ascii_art("Hi", style=style)
        assert len(result) > 0
        assert isinstance(result, str)

    def test_invalid_style_raises_error(self):
        """Negative test: ensure bad styles are caught."""
        with pytest.raises(ValueError):
            text_to_ascii_art("Fail", style="comic-sans")

    def test_width_scaling(self):
        narrow = text_to_ascii_art("Scaling", width=20)
        wide = text_to_ascii_art("Scaling", width=100)

        def get_max_w(txt):
            return max(len(l) for l in txt.splitlines())

        assert get_max_w(wide) > get_max_w(narrow)


class TestIntegration:
    @pytest.mark.parametrize("style", ["box", "banner", "block"])
    def test_all_output_modes(self, style):
        """Verify unified interface or helper functions."""
        if style == "box":
            res = text_to_box("Test")
        elif style == "banner":
            res = text_to_banner("Test")
        else:
            res = text_to_ascii_art("Test", style=style)

        assert len(res) > 0
