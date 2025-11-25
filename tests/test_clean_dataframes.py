"""Tests for clean dataframe extraction functionality."""

from unittest.mock import patch

import pandas as pd

from sftk.data_validator import DataValidator
from sftk.validation_strategies import (
    CleanRowTracker,
    RequiredValidator,
    ValidationConfig,
    ValidationStrategyRegistry,
)


def test_clean_row_tracker():
    """CleanRowTracker should initialize, track, and retrieve clean row indices."""
    tracker = CleanRowTracker()

    # Test initialization
    assert tracker.clean_row_indices == {}

    # Test initializing datasets
    df1 = pd.DataFrame({"name": ["Alice", "Bob"]})
    df2 = pd.DataFrame({"name": ["Charlie", "David", "Eve"]})
    tracker.initialize_dataset("dataset1", df1)
    tracker.initialize_dataset("dataset2", df2)

    assert tracker.clean_row_indices["dataset1"] == {0, 1}
    assert tracker.clean_row_indices["dataset2"] == {0, 1, 2}

    # Test marking rows as errors
    tracker.mark_row_as_error(1, "dataset1")
    tracker.mark_row_as_error(0, "dataset2")
    tracker.mark_row_as_error(2, "dataset2")

    assert tracker.clean_row_indices["dataset1"] == {0}
    assert tracker.clean_row_indices["dataset2"] == {1}

    # Test getting clean indices
    assert tracker.get_clean_indices("dataset1") == {0}
    assert tracker.get_clean_indices("dataset2") == {1}
    assert tracker.get_clean_indices("nonexistent") == set()


def test_validation_config_extract_clean_dataframes():
    """ValidationConfig should have extract_clean_dataframes parameter."""
    config = ValidationConfig()

    # Should have the parameter and default to False
    assert hasattr(config, "extract_clean_dataframes")
    assert config.extract_clean_dataframes is False

    # Should be able to set it to True
    config_enabled = ValidationConfig(extract_clean_dataframes=True)
    assert config_enabled.extract_clean_dataframes is True


def test_validation_strategy_with_clean_row_tracker():
    """ValidationStrategy should integrate with CleanRowTracker to mark error rows."""
    tracker = CleanRowTracker()
    validation_rules = {
        "test_dataset": {
            "file_name": "test.csv",
            "required": ["name"],
            "info_columns": [],
        }
    }
    validator = RequiredValidator(
        validation_rules, max_errors=100, clean_row_tracker=tracker
    )

    # Test that validator has tracker
    assert validator.clean_row_tracker is tracker

    # Create test data and initialize tracker
    df = pd.DataFrame({"name": ["Alice", None, "Charlie"], "age": [25, 30, 35]})
    tracker.initialize_dataset("test_dataset", df)

    # Run validation
    rules = {"file_name": "test.csv", "required": ["name"], "info_columns": []}
    errors = validator.validate(rules, df, "test_dataset")

    # Should find one error and mark row 1 as having an error
    assert len(errors) == 1
    assert tracker.get_clean_indices("test_dataset") == {0, 2}

    # Test backward compatibility (without tracker)
    validator_no_tracker = RequiredValidator({}, max_errors=100)
    assert validator_no_tracker.clean_row_tracker is None
    errors = validator_no_tracker.validate(rules, df)
    assert len(errors) == 1


def test_validation_strategy_registry_with_clean_row_tracker():
    """ValidationStrategyRegistry should pass CleanRowTracker to all strategies."""
    tracker = CleanRowTracker()
    registry = ValidationStrategyRegistry(
        {}, {}, s3_handler=None, max_errors=100, clean_row_tracker=tracker
    )

    # All strategies should have the tracker
    assert registry.strategies["required"].clean_row_tracker is tracker
    assert registry.strategies["unique"].clean_row_tracker is tracker
    assert registry.strategies["formats"].clean_row_tracker is tracker

    # Test backward compatibility (without tracker)
    registry_no_tracker = ValidationStrategyRegistry(
        {}, {}, s3_handler=None, max_errors=100
    )
    assert registry_no_tracker.strategies["required"].clean_row_tracker is None


@patch("sftk.data_validator.S3Handler")
def test_data_validator_clean_dataframe_methods(mock_s3_handler):
    """DataValidator should have clean dataframe methods that work with and without tracker."""
    validator = DataValidator()

    # Test that methods exist
    assert hasattr(validator, "clean_row_tracker")
    assert hasattr(validator, "get_clean_dataframe")
    assert hasattr(validator, "get_all_clean_dataframes")
    assert hasattr(validator, "get_clean_summary")

    # Test behavior without tracker
    assert validator.clean_row_tracker is None
    assert validator.get_clean_dataframe("test_dataset").empty
    assert validator.get_all_clean_dataframes() == {}
    summary = validator.get_clean_summary()
    assert "message" in summary
    assert "clean dataframe extraction was not enabled" in summary["message"].lower()


@patch("sftk.data_validator.VALIDATION_RULES", {})
@patch("sftk.data_validator.S3Handler")
def test_data_validator_tracker_initialization(mock_s3_handler):
    """validate_with_config should initialize tracker based on config."""
    mock_s3_handler.return_value.read_df_from_s3_csv.return_value = pd.DataFrame()

    # Test tracker initialized when enabled
    validator = DataValidator()
    config = ValidationConfig(extract_clean_dataframes=True)
    validator.validate_with_config(config)
    assert validator.clean_row_tracker is not None

    # Test tracker not initialized when disabled
    validator2 = DataValidator()
    config2 = ValidationConfig(extract_clean_dataframes=False)
    validator2.validate_with_config(config2)
    assert validator2.clean_row_tracker is None


@patch(
    "sftk.data_validator.VALIDATION_RULES",
    {
        "test_dataset": {
            "file_name": "test_data.csv",
            "required": ["name"],
            "unique": [],
            "info_columns": [],
            "foreign_keys": {},
            "relationships": [],
        }
    },
)
@patch("sftk.data_validator.S3Handler")
def test_clean_dataframes_integration(mock_s3_handler):
    """Integration test: clean dataframes should be extracted correctly."""
    test_df = pd.DataFrame(
        {"name": ["Alice", None, "Charlie", "David"], "age": [25, 30, 35, 40]}
    )
    mock_s3_handler.return_value.read_df_from_s3_csv.return_value = test_df

    validator = DataValidator()
    config = ValidationConfig(
        required=True,
        file_presence=False,
        extract_clean_dataframes=True,
    )

    # Run validation
    errors_df = validator.validate_with_config(config)
    assert len(errors_df) == 1  # Missing name in row 1

    # Get clean dataframe
    clean_df = validator.get_clean_dataframe("test_dataset")
    assert len(clean_df) == 3
    assert list(clean_df["name"]) == ["Alice", "Charlie", "David"]

    # Check summary
    summary = validator.get_clean_summary()
    assert summary["datasets"]["test_dataset"]["total_rows"] == 4
    assert summary["datasets"]["test_dataset"]["clean_rows"] == 3
    assert summary["datasets"]["test_dataset"]["error_rows"] == 1
