import os
import unittest
import logging

# This will auto-create the log file path
from sftk.log_config import set_log_level, get_log_level, convert_log_level, LOG_PATH

class TestLogging(unittest.TestCase):
    def test_log_path_creation(self):
        """Test that the log directory and file path are set correctly."""
        # Make sure LOG_PATH exists
        self.assertIsInstance(LOG_PATH, str, "LOG_PATH is not a string.")
        self.assertTrue(os.path.exists(LOG_PATH), "LOG_PATH does not exist.")

    def test_get_log_level(self):
        """Test that log levels are correctly converted from string to logging constants."""
        self.assertEqual(convert_log_level("DEBUG"), logging.DEBUG)
        self.assertEqual(convert_log_level("INFO"), logging.INFO)
        self.assertEqual(convert_log_level("WARNING"), logging.WARNING)
        self.assertEqual(convert_log_level("ERROR"), logging.ERROR)
        self.assertEqual(convert_log_level("CRITICAL"), logging.CRITICAL)
        self.assertEqual(convert_log_level("INVALID"), logging.INFO)

    def test_set_log_level(self):
        """Test that the logging level is set correctly."""
        set_log_level("DEBUG")
        self.assertEqual(get_log_level(), "DEBUG")

        # Test invalid log level
        with self.assertRaises(ValueError):
            set_log_level("INVALID")

    def test_logging_writes_to_file(self):
        """Test that logging messages are correctly written to the log file."""
        log_message = "This is a test log entry."
        logging.info(log_message)

        # Make sure the log file contains the log message
        with open(LOG_PATH, "r") as log_file:
            self.assertIn(log_message, log_file.read(), "Log message not found in log file.")
