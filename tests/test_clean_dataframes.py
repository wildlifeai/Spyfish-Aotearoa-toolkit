"""
Tests for clean dataframe extraction functionality.

Following TDD approach - tests written first, then implementation.
"""

from unittest.mock import patch

import pandas as pd

from sftk.data_validator import DataValidator
from sftk.validation_strategies import (
    CleanRowTracker,
    RequiredValidator,
    ValidationConfig,
    ValidationStrategyRegistry,
)


class TestCleanRowTracker:
    """Test the CleanRowTracker class."""

    def test_clean_row_tracker_initialization(self):
        """CleanRowTracker should initialize with empty clean_row_indices."""
        tracker = CleanRowTracker()
        assert tracker.clean_row_indices == {}

    def test_initialize_dataset(self):
        """Should initialize clean indices with all row indices for a dataset."""
        tracker = CleanRowTracker()
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

        tracker.initialize_dataset("test_dataset", df)

        expected_indices = set(df.index)  # {0, 1, 2}
        assert tracker.clean_row_indices["test_dataset"] == expected_indices

    def test_initialize_multiple_datasets(self):
        """Should handle multiple datasets independently."""
        tracker = CleanRowTracker()
        df1 = pd.DataFrame({"name": ["Alice", "Bob"]})
        df2 = pd.DataFrame({"name": ["Charlie", "David", "Eve"]})

        tracker.initialize_dataset("dataset1", df1)
        tracker.initialize_dataset("dataset2", df2)

        assert tracker.clean_row_indices["dataset1"] == {0, 1}
        assert tracker.clean_row_indices["dataset2"] == {0, 1, 2}

    def test_mark_row_as_error(self):
        """Should remove row index from clean set when marked as error."""
        tracker = CleanRowTracker()
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        tracker.initialize_dataset("test_dataset", df)

        # Mark row 1 as having an error
        tracker.mark_row_as_error(1, "test_dataset")

        expected_clean_indices = {0, 2}  # Row 1 should be removed
        assert tracker.clean_row_indices["test_dataset"] == expected_clean_indices

    def test_mark_multiple_rows_as_error(self):
        """Should handle multiple rows being marked as errors."""
        tracker = CleanRowTracker()
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "David"]})
        tracker.initialize_dataset("test_dataset", df)

        # Mark rows 0 and 2 as having errors
        tracker.mark_row_as_error(0, "test_dataset")
        tracker.mark_row_as_error(2, "test_dataset")

        expected_clean_indices = {1, 3}  # Only rows 1 and 3 should remain
        assert tracker.clean_row_indices["test_dataset"] == expected_clean_indices

    def test_mark_row_as_error_nonexistent_dataset(self):
        """Should handle marking error for non-existent dataset gracefully."""
        tracker = CleanRowTracker()

        # Should not raise an error
        tracker.mark_row_as_error(0, "nonexistent_dataset")

        # Should not create the dataset
        assert "nonexistent_dataset" not in tracker.clean_row_indices

    def test_mark_row_as_error_nonexistent_index(self):
        """Should handle marking non-existent row index gracefully."""
        tracker = CleanRowTracker()
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        tracker.initialize_dataset("test_dataset", df)

        # Mark non-existent row index (should not raise error)
        tracker.mark_row_as_error(999, "test_dataset")

        # Original indices should remain unchanged
        assert tracker.clean_row_indices["test_dataset"] == {0, 1}

    def test_get_clean_indices(self):
        """Should return clean indices for a dataset."""
        tracker = CleanRowTracker()
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        tracker.initialize_dataset("test_dataset", df)
        tracker.mark_row_as_error(1, "test_dataset")

        clean_indices = tracker.get_clean_indices("test_dataset")

        assert clean_indices == {0, 2}

    def test_get_clean_indices_nonexistent_dataset(self):
        """Should return empty set for non-existent dataset."""
        tracker = CleanRowTracker()

        clean_indices = tracker.get_clean_indices("nonexistent_dataset")

        assert clean_indices == set()

    def test_get_clean_indices_all_rows_clean(self):
        """Should return all indices when no errors marked."""
        tracker = CleanRowTracker()
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        tracker.initialize_dataset("test_dataset", df)

        clean_indices = tracker.get_clean_indices("test_dataset")

        assert clean_indices == {0, 1, 2}

    def test_get_clean_indices_all_rows_have_errors(self):
        """Should return empty set when all rows have errors."""
        tracker = CleanRowTracker()
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        tracker.initialize_dataset("test_dataset", df)

        # Mark all rows as having errors
        tracker.mark_row_as_error(0, "test_dataset")
        tracker.mark_row_as_error(1, "test_dataset")

        clean_indices = tracker.get_clean_indices("test_dataset")

        assert clean_indices == set()


class TestValidationConfigExtension:
    """Test the ValidationConfig extension for clean dataframe extraction."""

    def test_validation_config_has_extract_clean_dataframes_parameter(self):
        """ValidationConfig should have extract_clean_dataframes parameter."""
        config = ValidationConfig()

        # Should have the parameter
        assert hasattr(config, "extract_clean_dataframes")

        # Should default to False
        assert config.extract_clean_dataframes is False

    def test_validation_config_extract_clean_dataframes_can_be_set_true(self):
        """Should be able to set extract_clean_dataframes to True."""
        config = ValidationConfig(extract_clean_dataframes=True)

        assert config.extract_clean_dataframes is True

    def test_validation_config_extract_clean_dataframes_can_be_set_false(self):
        """Should be able to explicitly set extract_clean_dataframes to False."""
        config = ValidationConfig(extract_clean_dataframes=False)

        assert config.extract_clean_dataframes is False


class TestValidationStrategyWithCleanRowTracker:
    """Test ValidationStrategy integration with CleanRowTracker."""

    def test_validation_strategy_accepts_clean_row_tracker(self):
        """ValidationStrategy should accept clean_row_tracker parameter."""
        tracker = CleanRowTracker()
        validator = RequiredValidator({}, max_errors=100, clean_row_tracker=tracker)

        assert validator.clean_row_tracker is tracker

    def test_validation_strategy_works_without_clean_row_tracker(self):
        """ValidationStrategy should work without clean_row_tracker (backward compatibility)."""
        validator = RequiredValidator({}, max_errors=100)

        assert validator.clean_row_tracker is None

    def test_validation_strategy_marks_error_rows_in_tracker(self):
        """ValidationStrategy should mark error rows in the tracker when errors are found."""
        tracker = CleanRowTracker()
        # Provide validation rules so the validator can find dataset names
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

        # Create test data with missing required value
        df = pd.DataFrame({"name": ["Alice", None, "Charlie"], "age": [25, 30, 35]})

        # Initialize tracker with all rows as clean using dataset name
        tracker.initialize_dataset("test_dataset", df)

        # Run validation
        rules = {
            "file_name": "test.csv",
            "required": ["name"],
            "info_columns": [],
        }
        errors = validator.validate(rules, df, "test_dataset")

        # Should find one error (row 1 has missing name)
        assert len(errors) == 1

        # Tracker should mark row 1 as having an error
        clean_indices = tracker.get_clean_indices("test_dataset")
        assert clean_indices == {0, 2}  # Rows 0 and 2 should remain clean

    def test_validation_strategy_without_tracker_still_finds_errors(self):
        """ValidationStrategy should still find errors when no tracker is provided."""
        validator = RequiredValidator({}, max_errors=100)  # No tracker

        df = pd.DataFrame({"name": ["Alice", None, "Charlie"], "age": [25, 30, 35]})

        rules = {"file_name": "test.csv", "required": ["name"], "info_columns": []}
        errors = validator.validate(rules, df)  # No dataset_name - should still work

        # Should still find the error
        assert len(errors) == 1
        assert "Missing value in required column" in errors[0].error_info


class TestValidationStrategyRegistryWithCleanRowTracker:
    """Test ValidationStrategyRegistry integration with CleanRowTracker."""

    def test_validation_strategy_registry_accepts_clean_row_tracker(self):
        """ValidationStrategyRegistry should accept clean_row_tracker parameter."""
        tracker = CleanRowTracker()
        validation_rules = {}
        patterns = {}

        registry = ValidationStrategyRegistry(
            validation_rules,
            patterns,
            s3_handler=None,
            max_errors=100,
            clean_row_tracker=tracker,
        )

        # All strategies should have the tracker
        assert registry.strategies["required"].clean_row_tracker is tracker
        assert registry.strategies["unique"].clean_row_tracker is tracker
        assert registry.strategies["formats"].clean_row_tracker is tracker
        assert registry.strategies["foreign_keys"].clean_row_tracker is tracker
        assert registry.strategies["column_relationships"].clean_row_tracker is tracker

    def test_validation_strategy_registry_works_without_clean_row_tracker(self):
        """ValidationStrategyRegistry should work without clean_row_tracker (backward compatibility)."""
        validation_rules = {}
        patterns = {}

        registry = ValidationStrategyRegistry(
            validation_rules, patterns, s3_handler=None, max_errors=100
        )

        # All strategies should have None for tracker
        assert registry.strategies["required"].clean_row_tracker is None
        assert registry.strategies["unique"].clean_row_tracker is None
        assert registry.strategies["formats"].clean_row_tracker is None
        assert registry.strategies["foreign_keys"].clean_row_tracker is None
        assert registry.strategies["column_relationships"].clean_row_tracker is None


class TestDataValidatorWithCleanDataframes:
    """Test DataValidator integration with clean dataframe extraction."""

    @patch("sftk.data_validator.S3Handler")
    def test_data_validator_initializes_clean_row_tracker_attribute(
        self, mock_s3_handler
    ):
        """DataValidator should have clean_row_tracker attribute."""
        validator = DataValidator()

        assert hasattr(validator, "clean_row_tracker")
        assert validator.clean_row_tracker is None

    @patch("sftk.data_validator.S3Handler")
    def test_get_clean_dataframe_method_exists(self, mock_s3_handler):
        """DataValidator should have get_clean_dataframe method."""
        validator = DataValidator()

        assert hasattr(validator, "get_clean_dataframe")
        assert callable(validator.get_clean_dataframe)

    @patch("sftk.data_validator.S3Handler")
    def test_get_all_clean_dataframes_method_exists(self, mock_s3_handler):
        """DataValidator should have get_all_clean_dataframes method."""
        validator = DataValidator()

        assert hasattr(validator, "get_all_clean_dataframes")
        assert callable(validator.get_all_clean_dataframes)

    @patch("sftk.data_validator.S3Handler")
    def test_get_clean_summary_method_exists(self, mock_s3_handler):
        """DataValidator should have get_clean_summary method."""
        validator = DataValidator()

        assert hasattr(validator, "get_clean_summary")
        assert callable(validator.get_clean_summary)

    @patch("sftk.data_validator.S3Handler")
    def test_get_clean_dataframe_returns_empty_when_no_tracker(self, mock_s3_handler):
        """get_clean_dataframe should return empty DataFrame when no tracker is initialized."""
        validator = DataValidator()

        result = validator.get_clean_dataframe("test_dataset")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("sftk.data_validator.S3Handler")
    def test_get_all_clean_dataframes_returns_empty_when_no_tracker(
        self, mock_s3_handler
    ):
        """get_all_clean_dataframes should return empty dict when no tracker is initialized."""
        validator = DataValidator()

        result = validator.get_all_clean_dataframes()

        assert isinstance(result, dict)
        assert len(result) == 0

    @patch("sftk.data_validator.S3Handler")
    def test_get_clean_summary_returns_empty_when_no_tracker(self, mock_s3_handler):
        """get_clean_summary should return appropriate summary when no tracker is initialized."""
        validator = DataValidator()

        result = validator.get_clean_summary()

        assert isinstance(result, dict)
        assert "message" in result
        assert "clean dataframe extraction was not enabled" in result["message"].lower()


class TestDataValidatorIntegrationWithCleanDataframes:
    """Test full integration of clean dataframe extraction with validation flow."""

    @patch("sftk.data_validator.VALIDATION_RULES", {})
    @patch("sftk.data_validator.S3Handler")
    def test_validate_with_config_initializes_tracker_when_enabled(
        self, mock_s3_handler
    ):
        """validate_with_config should initialize clean_row_tracker when extract_clean_dataframes=True."""
        # Mock the S3Handler to avoid actual S3 calls
        mock_s3_handler.return_value.read_df_from_s3_csv.return_value = pd.DataFrame()

        validator = DataValidator()
        config = ValidationConfig(extract_clean_dataframes=True)

        # This should initialize the tracker
        validator.validate_with_config(config)

        assert validator.clean_row_tracker is not None
        assert hasattr(validator.clean_row_tracker, "clean_row_indices")

    @patch("sftk.data_validator.VALIDATION_RULES", {})
    @patch("sftk.data_validator.S3Handler")
    def test_validate_with_config_does_not_initialize_tracker_when_disabled(
        self, mock_s3_handler
    ):
        """validate_with_config should not initialize clean_row_tracker when extract_clean_dataframes=False."""
        # Mock the S3Handler to avoid actual S3 calls
        mock_s3_handler.return_value.read_df_from_s3_csv.return_value = pd.DataFrame()

        validator = DataValidator()
        config = ValidationConfig(extract_clean_dataframes=False)

        # This should not initialize the tracker
        validator.validate_with_config(config)

        assert validator.clean_row_tracker is None

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
    def test_clean_dataframes_are_extracted_correctly(self, mock_s3_handler):
        """Integration test: clean dataframes should be extracted correctly with dataset names."""
        # Mock the S3Handler to return test data
        test_df = pd.DataFrame(
            {"name": ["Alice", None, "Charlie", "David"], "age": [25, 30, 35, 40]}
        )
        mock_s3_handler.return_value.read_df_from_s3_csv.return_value = test_df

        validator = DataValidator()
        config = ValidationConfig(
            required=True,
            file_presence=False,  # Disable file presence
            extract_clean_dataframes=True,
        )

        # Run validation
        errors_df = validator.validate_with_config(config)

        # Should find 1 error (missing name in row 1)
        assert len(errors_df) == 1

        # Should be able to get clean dataframe using dataset name
        clean_df = validator.get_clean_dataframe("test_dataset")

        # Should have 3 clean rows (Alice, Charlie, David)
        assert len(clean_df) == 3
        assert list(clean_df["name"]) == ["Alice", "Charlie", "David"]

        # Summary should show correct statistics
        summary = validator.get_clean_summary()
        assert "test_dataset" in summary["datasets"]
        assert summary["datasets"]["test_dataset"]["total_rows"] == 4
        assert summary["datasets"]["test_dataset"]["clean_rows"] == 3
        assert summary["datasets"]["test_dataset"]["error_rows"] == 1
