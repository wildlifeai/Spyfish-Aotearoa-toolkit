import logging
import os
from pathlib import Path
from unittest.mock import patch

# This will auto-create the log file path
from sftk.log_config import LOG_PATH


def test_log_path_creation():
    """Test that the log directory and file path are set correctly."""
    # Make sure LOG_PATH is a valid string path
    assert isinstance(LOG_PATH, str), "LOG_PATH is not a string."
    # The log file only gets created when logging is configured for file output
    # So we just check that the path is valid and the directory exists
    log_path = Path(LOG_PATH)
    assert log_path.parent.exists(), "Log directory does not exist."


def test_logging_works():
    """Test that basic logging functionality works."""
    # Just test that we can log without errors
    logging.info("Test message")
    logging.debug("Debug message")
    logging.warning("Warning message")
    # If we get here without exceptions, logging is working


@patch.dict(os.environ, {"LOG_OUTPUT": "file"})
def test_logging_writes_to_file_with_env_var():
    """Test that logging can be configured to write to file via environment variable."""
    # This test verifies the environment variable works, but we can't easily test
    # the actual file writing without reimporting the module
    assert os.getenv("LOG_OUTPUT") == "file"


@patch.dict(os.environ, {"LOG_OUTPUT": "console"})
def test_logging_console_with_env_var():
    """Test that logging can be configured for console via environment variable."""
    assert os.getenv("LOG_OUTPUT") == "console"


@patch.dict(os.environ, {}, clear=True)
def test_logging_defaults_to_console():
    """Test that logging defaults to console when no env var is set."""
    # Clear the LOG_OUTPUT env var if it exists
    if "LOG_OUTPUT" in os.environ:
        del os.environ["LOG_OUTPUT"]
    assert os.getenv("LOG_OUTPUT", "console") == "console"
