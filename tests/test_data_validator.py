"""
Tests for the refactored DataValidator class.
"""

from unittest.mock import Mock, patch

import pandas as pd

from sftk.data_validator import DataValidator
from sftk.validation_strategies import ValidationConfig


def test_data_validator_uses_validation_config():
    """DataValidator should accept ValidationConfig instead of individual parameters."""
    # Mock S3Handler to avoid actual S3 calls
    with patch("sftk.data_validator.S3Handler") as mock_s3:
        mock_s3.return_value.read_df_from_s3_csv.return_value = pd.DataFrame()

        validator = DataValidator()
        config = ValidationConfig()
        config.required = True
        config.file_presence = False  # Disable file presence to avoid S3 calls

        # This should work without throwing an error
        result = validator.validate_with_config(config)
        assert isinstance(result, pd.DataFrame)


def test_data_validator_uses_strategy_registry():
    """DataValidator should use ValidationStrategyRegistry internally."""
    with patch("sftk.data_validator.S3Handler") as mock_s3:
        mock_s3.return_value.read_df_from_s3_csv.return_value = pd.DataFrame()

        validator = DataValidator()

        # Check that validator has a strategy registry
        assert hasattr(validator, "strategy_registry")
        assert validator.strategy_registry is not None


def test_data_validator_processes_datasets_with_strategies():
    """DataValidator should process each dataset using the appropriate strategies."""
    # Create mock data that doesn't have required columns
    test_df = pd.DataFrame({"name": ["Alice", None, "Bob"], "id": [1, 2, 3]})

    with patch("sftk.data_validator.S3Handler") as mock_s3:
        mock_s3.return_value.read_df_from_s3_csv.return_value = test_df

        validator = DataValidator()
        config = ValidationConfig()
        config.required = True
        config.file_presence = False  # Disable file presence to avoid S3 calls

        result = validator.validate_with_config(config)

        # Should find validation errors (missing required columns)
        assert len(result) > 0
        assert any(
            "Missing column for required check" in error_info
            for error_info in result["error_info"]
        )


def test_data_validator_with_file_presence():
    """DataValidator should handle file presence validation using the new strategy."""
    # Mock S3Handler for file presence validation
    mock_s3_handler = Mock()
    mock_s3_handler.read_df_from_s3_csv.return_value = pd.DataFrame()
    mock_s3_handler.get_paths_from_csv.return_value = {
        "all": {"file1.mp4", "file2.mp4"},  # all files
        "filtered": {"file1.mp4"},  # filtered files
    }
    mock_s3_handler.get_paths_from_s3.return_value = {"file2.mp4"}  # S3 files

    with patch("sftk.data_validator.S3Handler", return_value=mock_s3_handler):
        with patch(
            "sftk.data_validator.FILE_PRESENCE_RULES",
            {
                "file_presence": {
                    "csv_filename": "test.csv",
                    "csv_column_to_extract": "video_path",
                    "column_filter": "status",
                    "column_value": "active",
                    "valid_extensions": ["mp4"],
                    "path_prefix": "videos/",
                    "s3_sharepoint_path": "sharepoint",
                    "bucket": "test-bucket",
                }
            },
        ):
            validator = DataValidator()
            config = ValidationConfig()
            config.file_presence = True
            config.required = False  # Disable other validations
            config.unique = False
            config.foreign_keys = False
            config.formats = False
            config.column_relationships = False

            result = validator.validate_with_config(config)

            # Should find file presence errors
            assert len(result) > 0
            file_presence_errors = result[
                result["error_source"] == "file_presence_check"
            ]
            assert len(file_presence_errors) > 0


def test_data_validator_export_file_differences():
    """DataValidator should export file differences using existing validator."""
    import tempfile
    from unittest.mock import Mock, patch

    # Mock S3Handler
    mock_s3_handler = Mock()
    mock_s3_handler.read_df_from_s3_csv.return_value = pd.DataFrame()

    # Mock FilePresenceValidator
    mock_file_validator = Mock()
    mock_file_validator.get_file_differences.return_value = (
        {"missing_file.mp4"},
        {"extra_file.mp4"},
    )

    with patch("sftk.data_validator.S3Handler", return_value=mock_s3_handler):
        validator = DataValidator()
        # Replace the file presence validator in the strategy registry
        validator.strategy_registry.strategies["file_presence"] = mock_file_validator

        # Create temporary files for output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as missing_file:
            missing_path = missing_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as extra_file:
            extra_path = extra_file.name

        try:
            # Test the export method
            validator.export_file_differences(missing_path, extra_path)

            # Verify the method was called
            mock_file_validator.get_file_differences.assert_called_once()

            # Verify files were created and contain expected content
            with open(missing_path, "r") as f:
                missing_content = f.read().strip()
            with open(extra_path, "r") as f:
                extra_content = f.read().strip()

            assert missing_content == "missing_file.mp4"
            assert extra_content == "extra_file.mp4"

        finally:
            # Clean up
            import os

            if os.path.exists(missing_path):
                os.remove(missing_path)
            if os.path.exists(extra_path):
                os.remove(extra_path)
