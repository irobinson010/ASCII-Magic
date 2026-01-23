"""Tests for unified_cli module."""

import sys
from unittest.mock import Mock, patch, MagicMock
import pytest
from ascii_magic.unified_cli import main, usage, _call_entry, COMMANDS

# --- Fixtures ---


@pytest.fixture
def mock_stdout(monkeypatch):
    """Capture stdout output."""
    mock = MagicMock()
    monkeypatch.setattr(sys, "stdout", mock)
    return mock


@pytest.fixture
def mock_stderr(monkeypatch):
    """Capture stderr output."""
    mock = MagicMock()
    monkeypatch.setattr(sys, "stderr", mock)
    return mock


@pytest.fixture
def mock_module_with_main():
    """Create a mock module with a main function."""
    module = Mock()
    module.main = Mock(return_value=0)
    return module


# --- Tests for usage() ---


class TestUsage:
    def test_usage_output(self, capsys):
        """Test that usage prints the expected help message."""
        usage()
        captured = capsys.readouterr()
        assert "Usage: ascii-magic <command> [args...]" in captured.out
        assert "Commands:" in captured.out
        # Check all commands are listed
        for cmd in COMMANDS.keys():
            assert cmd in captured.out

    def test_usage_with_prog_name(self, capsys):
        """Test usage with custom program name."""
        usage(prog="custom-prog")
        captured = capsys.readouterr()
        assert "Usage: ascii-magic <command> [args...]" in captured.out


# --- Tests for _call_entry() ---


class TestCallEntry:
    def test_call_entry_with_argv_param(self):
        """Test calling entry function that accepts argv parameter."""
        mock_entry = Mock(return_value=0)
        # Create a signature with one parameter
        mock_entry.__signature__ = Mock()
        mock_entry.__signature__.parameters = {"argv": None}

        with patch("ascii_magic.unified_cli.inspect.signature") as mock_sig:
            mock_sig.return_value.parameters = {"argv": None}
            result = _call_entry(mock_entry, ["arg1", "arg2"])

        assert result == 0
        mock_entry.assert_called_once_with(["arg1", "arg2"])

    def test_call_entry_without_argv_param(self):
        """Test calling entry function that doesn't accept argv parameter."""
        mock_entry = Mock(return_value=0)

        with patch("ascii_magic.unified_cli.inspect.signature") as mock_sig:
            mock_sig.return_value.parameters = {}
            original_argv = sys.argv[:]
            result = _call_entry(mock_entry, ["arg1", "arg2"], module_prog="test-prog")

        assert result == 0
        mock_entry.assert_called_once()
        # sys.argv should be restored
        assert sys.argv == original_argv

    def test_call_entry_with_system_exit(self):
        """Test handling SystemExit exception."""
        mock_entry = Mock(side_effect=SystemExit(5))

        with patch("ascii_magic.unified_cli.inspect.signature") as mock_sig:
            mock_sig.return_value.parameters = {"argv": None}
            result = _call_entry(mock_entry, [])

        assert result == 5

    def test_call_entry_with_system_exit_none(self):
        """Test handling SystemExit with None code."""
        mock_entry = Mock(side_effect=SystemExit(None))

        with patch("ascii_magic.unified_cli.inspect.signature") as mock_sig:
            mock_sig.return_value.parameters = {"argv": None}
            result = _call_entry(mock_entry, [])

        assert result == 0

    def test_call_entry_with_exception(self, capsys):
        """Test handling general exceptions."""
        mock_entry = Mock(side_effect=RuntimeError("Test error"))

        with patch("ascii_magic.unified_cli.inspect.signature") as mock_sig:
            mock_sig.return_value.parameters = {"argv": None}
            result = _call_entry(mock_entry, [])

        assert result == 1
        captured = capsys.readouterr()
        assert "Error running command: Test error" in captured.err


# --- Tests for main() ---


class TestMain:
    def test_main_no_arguments(self, capsys):
        """Test main with no arguments shows usage."""
        result = main([])
        captured = capsys.readouterr()

        assert result == 0
        assert "Usage: ascii-magic" in captured.out

    def test_main_help_flag(self, capsys):
        """Test main with help flags."""
        for flag in ["-h", "--help"]:
            result = main([flag])
            captured = capsys.readouterr()

            assert result == 0
            assert "Usage: ascii-magic" in captured.out

    def test_main_unknown_command(self, capsys):
        """Test main with unknown command."""
        result = main(["unknown"])
        captured = capsys.readouterr()

        assert result == 2
        assert "Unknown command: unknown" in captured.err
        assert "Usage: ascii-magic" in captured.out

    def test_main_valid_command_colorize(self, mock_module_with_main):
        """Test main with valid 'colorize' command."""
        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            mock_import.return_value = mock_module_with_main
            result = main(["colorize", "arg1", "arg2"])

        assert result == 0
        mock_import.assert_called_once_with("ascii_magic.colorize_ascii")

    def test_main_valid_command_image(self, mock_module_with_main):
        """Test main with valid 'image' command."""
        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            mock_import.return_value = mock_module_with_main
            result = main(["image", "file.jpg"])

        assert result == 0
        mock_import.assert_called_once_with("ascii_magic.image_to_ascii")

    def test_main_valid_command_text(self, mock_module_with_main):
        """Test main with valid 'text' command."""
        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            mock_import.return_value = mock_module_with_main
            result = main(["text", "Hello"])

        assert result == 0
        mock_import.assert_called_once_with("ascii_magic.text_to_ascii")

    def test_main_import_error(self, capsys):
        """Test main when module import fails."""
        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            result = main(["colorize"])

        captured = capsys.readouterr()
        assert result == 3
        assert "Failed to import command 'colorize'" in captured.err
        assert "Module not found" in captured.err

    def test_main_no_main_function(self, capsys):
        """Test main when module has no main function."""
        mock_module = Mock(spec=[])  # Module without 'main'

        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            mock_import.return_value = mock_module
            result = main(["colorize"])

        captured = capsys.readouterr()
        assert result == 4
        assert "has no callable 'main'" in captured.err

    def test_main_non_callable_main(self, capsys):
        """Test main when module's main is not callable."""
        mock_module = Mock()
        mock_module.main = "not callable"

        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            mock_import.return_value = mock_module
            result = main(["colorize"])

        captured = capsys.readouterr()
        assert result == 4
        assert "has no callable 'main'" in captured.err

    def test_main_none_argv_uses_sys_argv(self, mock_module_with_main):
        """Test that main(None) uses sys.argv."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["ascii-magic", "colorize", "test"]

            with patch(
                "ascii_magic.unified_cli.importlib.import_module"
            ) as mock_import:
                mock_import.return_value = mock_module_with_main
                result = main(None)

            assert result == 0
        finally:
            sys.argv = original_argv

    def test_main_passes_args_to_subcommand(self):
        """Test that arguments are correctly passed to subcommand."""
        mock_module = Mock()
        mock_entry = Mock(return_value=0)
        mock_module.main = mock_entry

        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            with patch("ascii_magic.unified_cli.inspect.signature") as mock_sig:
                mock_import.return_value = mock_module
                mock_sig.return_value.parameters = {"argv": None}

                result = main(["image", "file.jpg", "--width", "80"])

        assert result == 0
        # Check that args were passed (excluding the command name)
        assert mock_entry.call_count == 1

    def test_main_subcommand_returns_error_code(self):
        """Test that main returns error code from subcommand."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=42)

        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            with patch("ascii_magic.unified_cli.inspect.signature") as mock_sig:
                mock_import.return_value = mock_module
                mock_sig.return_value.parameters = {"argv": None}
                result = main(["colorize", "test"])

        assert result == 42

    def test_main_with_sequence_type(self, mock_module_with_main):
        """Test that main accepts different sequence types."""
        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            mock_import.return_value = mock_module_with_main

            # Test with tuple
            result = main(("colorize", "arg"))
            assert result == 0

            # Test with list
            result = main(["colorize", "arg"])
            assert result == 0


# --- Integration-style Tests ---


class TestCommandsIntegration:
    """Test that actual commands are defined correctly."""

    def test_commands_dict_has_expected_entries(self):
        """Verify COMMANDS dictionary has expected structure."""
        assert "colorize" in COMMANDS
        assert "image" in COMMANDS
        assert "text" in COMMANDS
        assert COMMANDS["colorize"] == "ascii_magic.colorize_ascii"
        assert COMMANDS["image"] == "ascii_magic.image_to_ascii"
        assert COMMANDS["text"] == "ascii_magic.text_to_ascii"

    def test_all_commands_are_importable(self):
        """Test that all command modules can be imported."""
        import importlib

        for cmd, module_path in COMMANDS.items():
            try:
                module = importlib.import_module(module_path)
                assert hasattr(
                    module, "main"
                ), f"{module_path} should have a main function"
                assert callable(
                    getattr(module, "main")
                ), f"{module_path}.main should be callable"
            except ImportError as e:
                pytest.fail(f"Failed to import {module_path}: {e}")


# --- Edge Cases ---


class TestEdgeCases:
    def test_empty_string_command(self, capsys):
        """Test handling of empty string as command."""
        result = main([""])
        captured = capsys.readouterr()

        assert result == 2
        assert "Unknown command:" in captured.err

    def test_command_with_many_args(self, mock_module_with_main):
        """Test command with many arguments."""
        args = ["arg" + str(i) for i in range(100)]

        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            mock_import.return_value = mock_module_with_main
            result = main(["colorize"] + args)

        assert result == 0

    def test_command_with_special_characters(self, mock_module_with_main):
        """Test command arguments with special characters."""
        special_args = ["--file=test.jpg", "arg with spaces", "unicode→test"]

        with patch("ascii_magic.unified_cli.importlib.import_module") as mock_import:
            mock_import.return_value = mock_module_with_main
            result = main(["colorize"] + special_args)

        assert result == 0
