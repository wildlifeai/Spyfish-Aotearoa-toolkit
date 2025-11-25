"""
Tests for the refactored DataValidator class.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

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


@pytest.mark.skip(reason="Temporarily disabled")
def test_data_validator_with_file_presence():
    """DataValidator should handle file presence validation using the new strategy."""
    # Mock S3Handler for file presence validation
    mock_s3_handler = Mock()
    mock_s3_handler.read_df_from_s3_csv.return_value = pd.DataFrame()
    # CSV has file1.mp4 (filtered) and file2.mp4 (all)
    # S3 has file2.mp4 only
    # So file1.mp4 is missing (in CSV filtered but not in S3)
    mock_s3_handler.get_paths_from_csv.return_value = {
        "all": {"file1.mp4", "file2.mp4"},  # all files from CSV
        "filtered": {"file1.mp4"},  # filtered files (active status only)
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
            # Replace the s3_handler to use our mock
            validator.s3_handler = mock_s3_handler
            validator.strategy_registry = validator.strategy_registry.__class__(
                validator.validation_rules,
                validator.patterns,
                mock_s3_handler,
            )

            config = ValidationConfig()
            config.file_presence = True
            config.required = False  # Disable other validations
            config.unique = False
            config.foreign_keys = False
            config.formats = False
            config.column_relationships = False

            result = validator.validate_with_config(config)

            # Should find file presence errors (file1.mp4 is missing from S3)
            assert len(result) > 0
            file_presence_errors = result[
                result["error_source"] == "file_presence_check"
            ]
            assert len(file_presence_errors) > 0


def test_data_validator_export_file_differences():
    """DataValidator should export file differences using existing validator."""
    import os
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

        # Create a temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set FOLDER_PATH to the temporary directory
            validator.FOLDER_PATH = temp_dir

            # Define filenames
            missing_filename = "missing_files_in_aws.txt"
            extra_filename = "extra_files_in_aws.txt"

            # Test the export method with custom filenames
            validator.export_file_differences(
                missing_file_name=missing_filename, extra_file_name=extra_filename
            )

            # Verify the method was called
            mock_file_validator.get_file_differences.assert_called_once()

            # Construct full paths to the files that were written
            missing_path = os.path.join(temp_dir, missing_filename)
            extra_path = os.path.join(temp_dir, extra_filename)

            # Verify files were created and contain expected content
            assert os.path.exists(
                missing_path
            ), f"Missing file not created at {missing_path}"
            assert os.path.exists(extra_path), f"Extra file not created at {extra_path}"

            with open(missing_path, "r") as f:
                missing_content = f.read().strip()
            with open(extra_path, "r") as f:
                extra_content = f.read().strip()

            assert missing_content == "missing_file.mp4"
            assert extra_content == "extra_file.mp4"


def test_data_validator_run_validation_and_export():
    """DataValidator.run_validation should run validation, populate errors_df, and export to CSV."""
    import os
    import tempfile

    test_df = pd.DataFrame({"name": ["Alice", None, "Bob"], "id": [1, 2, 3]})

    with patch("sftk.data_validator.S3Handler") as mock_s3:
        mock_s3.return_value.read_df_from_s3_csv.return_value = test_df

        with patch("sftk.data_validator.EXPORT_LOCAL", True):
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("sftk.data_validator.LOCAL_DATA_FOLDER_PATH", temp_dir):
                    validator = DataValidator()
                    validator.FOLDER_PATH = temp_dir

                    # Run validation (which calls export_to_csv internally)
                    validator.run_validation(required=True, file_presence=False)

                    # Check that errors_df was populated
                    assert validator.errors_df is not None
                    assert len(validator.errors_df) > 0

                    # Check that CSV was created by run_validation
                    csv_path = os.path.join(temp_dir, "validation_errors_cleaned.csv")
                    assert os.path.exists(csv_path)

                    # Verify CSV content matches errors_df
                    imported_df = pd.read_csv(csv_path)
                    assert len(imported_df) == len(validator.errors_df)
                    assert "error_info" in imported_df.columns
