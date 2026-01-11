"""Tests for text_to_ascii module."""

import pytest
from ascii_magic.text_to_ascii import (
    text_to_box,
    text_to_banner,
    text_to_ascii_art,
    render_text_to_image,
)


class TestTextToBox:
    """Test text_to_box function."""

    def test_simple_text(self):
        """Test box creation with simple text."""
        result = text_to_box("Hello")
        assert "Hello" in result
        assert "┌" in result
        assert "└" in result
        lines = result.split("\n")
        assert len(lines) == 3  # top, content, bottom

    def test_multiline_text(self):
        """Test box with multiple lines."""
        result = text_to_box("Hello\nWorld")
        lines = result.split("\n")
        assert len(lines) == 4  # top, 2 content, bottom
        assert "Hello" in result
        assert "World" in result

    def test_width_constraint(self):
        """Test that box respects width parameter."""
        result = text_to_box("Hello", width=20)
        lines = result.split("\n")
        max_line_len = max(len(line) for line in lines)
        assert max_line_len <= 20


class TestTextToBanner:
    """Test text_to_banner function."""

    def test_default_char(self):
        """Test banner with default character."""
        result = text_to_banner("Test")
        assert "#" in result
        assert "Test" in result

    def test_custom_char(self):
        """Test banner with custom character."""
        result = text_to_banner("Test", char="*")
        assert "*" in result
        assert "#" not in result

    def test_structure(self):
        """Test banner structure."""
        result = text_to_banner("X")
        lines = result.split("\n")
        assert len(lines) == 3  # top border, content, bottom border


class TestRenderTextToImage:
    """Test text rendering to image."""

    def test_basic_render(self):
        """Test basic text rendering."""
        img = render_text_to_image("Test")
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_custom_colors(self):
        """Test rendering with custom colors."""
        img = render_text_to_image("Test", bg_color="black", text_color="white")
        assert img.size[0] > 0

    def test_font_size_affects_output(self):
        """Test that font size affects image size."""
        img_small = render_text_to_image("Test", font_size=12)
        img_large = render_text_to_image("Test", font_size=36)
        # Larger font should produce larger image
        assert img_large.size[0] > img_small.size[0]
        assert img_large.size[1] > img_small.size[1]

    def test_padding(self):
        """Test that padding is applied."""
        img = render_text_to_image("X", padding=50)
        # Image should be larger due to padding
        assert img.size[0] > 100


class TestTextToAsciiArt:
    """Test text_to_ascii_art function."""

    def test_basic_conversion(self):
        """Test basic text to ASCII conversion."""
        result = text_to_ascii_art("A", style="block", width=20)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_different_styles(self):
        """Test different style options."""
        text = "Hi"
        block = text_to_ascii_art(text, style="block", width=20)
        small = text_to_ascii_art(text, style="small", width=20)
        shadow = text_to_ascii_art(text, style="shadow", width=20)

        # All should produce output
        assert len(block) > 0
        assert len(small) > 0
        assert len(shadow) > 0

    def test_width_parameter(self):
        """Test that width parameter affects output."""
        text = "Test"
        narrow = text_to_ascii_art(text, width=20)
        wide = text_to_ascii_art(text, width=80)

        narrow_width = max(len(line) for line in narrow.split("\n"))
        wide_width = max(len(line) for line in wide.split("\n"))

        # Narrow should be narrower than wide
        assert narrow_width < wide_width

    def test_output_format(self):
        """Test that output is properly formatted."""
        result = text_to_ascii_art("Test", width=30)
        lines = result.split("\n")
        assert len(lines) > 0
        # Each line should not exceed width significantly
        assert all(len(line) <= 35 for line in lines)


class TestIntegration:
    """Integration tests."""

    def test_cli_styles_exist(self):
        """Verify all CLI styles are functional."""
        styles = ["box", "banner", "block", "small", "shadow"]
        text = "TEST"

        for style in styles:
            if style == "box":
                result = text_to_box(text)
            elif style == "banner":
                result = text_to_banner(text)
            else:
                result = text_to_ascii_art(text, style=style, width=30)

            assert len(result) > 0
