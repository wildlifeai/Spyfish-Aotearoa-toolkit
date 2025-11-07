import pandas as pd


def test_validation_config_defaults():
    """ValidationConfig should have sensible defaults."""
    from sftk.validation_strategies import ValidationConfig

    config = ValidationConfig()
    assert config.required is False
    assert config.unique is False


def test_validation_config_enable_all():
    """ValidationConfig should enable all validations."""
    from sftk.validation_strategies import ValidationConfig

    config = ValidationConfig()
    config.enable_all_validators()
    assert config.required is True
    assert config.unique is True


def test_required_validator_error_message():
    """RequiredValidator should generate correct error message."""
    from sftk.validation_strategies import RequiredValidator

    validator = RequiredValidator({})
    rules = {"file_name": "test.csv", "required": ["name"], "info_columns": []}
    df = pd.DataFrame({"name": ["Alice", None, "Bob"]})

    errors = validator.validate(rules, df)

    assert len(errors) == 1
    assert "Missing value in required column 'name'" in errors[0].error_info
    assert "help_info:" in errors[0].error_info


def test_required_validator_finds_missing_values():
    """RequiredValidator should find rows with missing required values."""
    from sftk.validation_strategies import RequiredValidator

    validator = RequiredValidator({})
    rules = {"file_name": "test.csv", "required": ["name"], "info_columns": []}
    df = pd.DataFrame({"name": ["Alice", None, "Bob"]})

    errors = validator.validate(rules, df)

    assert len(errors) == 1
    assert "Missing value in required column" in errors[0].error_info


def test_unique_validator_finds_duplicates():
    """UniqueValidator should find duplicate values."""
    from sftk.validation_strategies import UniqueValidator

    validator = UniqueValidator({})
    rules = {"file_name": "test.csv", "unique": ["id"], "info_columns": []}
    df = pd.DataFrame({"id": [1, 2, 2, 3]})

    errors = validator.validate(rules, df)

    assert len(errors) == 2  # Both duplicate rows should be flagged
    assert all(
        "Duplicate value in unique column" in error.error_info for error in errors
    )


def test_format_validator_finds_invalid_formats():
    """FormatValidator should find values that don't match patterns."""
    from sftk.validation_strategies import FormatValidator

    patterns = {"email": r"^[^@]+@[^@]+\.[^@]+$"}
    validator = FormatValidator({}, patterns)
    rules = {"file_name": "test.csv", "info_columns": []}
    df = pd.DataFrame({"email": ["valid@test.com", "invalid-email", "also@valid.org"]})

    errors = validator.validate(rules, df)

    assert len(errors) == 1
    assert "does not match required format" in errors[0].error_info


def test_foreign_key_validator_finds_missing_references():
    """ForeignKeyValidator should find missing foreign key references."""
    from sftk.validation_strategies import ForeignKeyValidator

    # Setup validation rules with target dataset
    target_df = pd.DataFrame({"target_id": [1, 2, 3]})  # Same column name
    validation_rules = {"target": {"file_name": "target.csv", "dataset": target_df}}

    validator = ForeignKeyValidator(validation_rules)
    rules = {
        "file_name": "source.csv",
        "foreign_keys": {"target": "target_id"},
        "info_columns": [],
    }
    df = pd.DataFrame({"target_id": [1, 2, 4, 5]})  # 4 and 5 don't exist in target

    errors = validator.validate(rules, df)

    assert len(errors) == 2
    assert all("not found in" in error.error_info for error in errors)


def test_validation_strategy_registry():
    """ValidationStrategyRegistry should manage strategies correctly."""
    from sftk.validation_strategies import ValidationConfig, ValidationStrategyRegistry

    validation_rules = {}
    patterns = {"email": r"^[^@]+@[^@]+\.[^@]+$"}
    registry = ValidationStrategyRegistry(validation_rules, patterns)

    # Test getting individual strategies
    required_validator = registry.get_strategy("required")
    assert required_validator is not None

    # Test getting enabled strategies
    config = ValidationConfig()
    config.required = True
    config.unique = True

    enabled_strategies = registry.get_enabled_strategies(config)
    assert len(enabled_strategies) == 2


def test_performance_with_large_dataset():
    """Validators should handle large datasets efficiently."""
    import time

    from sftk.validation_strategies import RequiredValidator

    # Create a large dataset
    large_df = pd.DataFrame(
        {
            "name": ["Alice"] * 5000 + [None] * 1000 + ["Bob"] * 4000,
            "id": list(range(10000)),
        }
    )

    validator = RequiredValidator({})
    rules = {"file_name": "test.csv", "required": ["name"], "info_columns": ["id"]}

    start_time = time.time()
    errors = validator.validate(rules, large_df)
    end_time = time.time()

    # Should find 1000 missing values
    assert len(errors) == 1000
    # Should complete in reasonable time (less than 2 seconds)
    assert (end_time - start_time) < 2.0


def test_error_limit_functionality():
    """Validators should respect error limits to prevent memory issues."""
    from sftk.validation_strategies import RequiredValidator

    # Create dataset with many errors
    large_df = pd.DataFrame(
        {"name": [None] * 2000, "id": list(range(2000))}  # 2000 missing values
    )

    # Set a low error limit
    validator = RequiredValidator({}, max_errors=100)
    rules = {"file_name": "test.csv", "required": ["name"], "info_columns": ["id"]}

    errors = validator.validate(rules, large_df)

    # Should stop at the error limit
    assert len(errors) == 100


def test_file_presence_validator():
    """FilePresenceValidator should find missing and extra files."""
    from unittest.mock import Mock

    from sftk.validation_strategies import FilePresenceValidator

    # Mock S3Handler
    mock_s3_handler = Mock()
    mock_s3_handler.get_paths_from_csv.return_value = {
        "all": {"file1.mp4", "file2.mp4", "file3.mp4"},  # all files
        "filtered": {"file1.mp4", "file2.mp4"},  # filtered files
    }
    mock_s3_handler.get_paths_from_s3.return_value = {
        "file1.mp4",
        "file4.mp4",
    }  # S3 files

    validator = FilePresenceValidator({}, mock_s3_handler)
    rules = {
        "file_presence": {
            "csv_filename": "test.csv",
            "csv_column_to_extract": "video_path",
            "valid_extensions": ["mp4"],
            "path_prefix": "videos/",
            "s3_sharepoint_path": "sharepoint",
            "bucket": "test-bucket",
        }
    }

    errors = validator.validate(rules, pd.DataFrame())

    # Should find 1 missing file (file2.mp4) and 1 extra file (file4.mp4)
    assert len(errors) == 2
    error_messages = [error.error_info for error in errors]
    assert any(
        "file2.mp4" in msg and "not found in AWS" in msg for msg in error_messages
    )
    assert any(
        "file4.mp4" in msg and "found in AWS but not" in msg for msg in error_messages
    )


def test_file_presence_validator_get_file_differences():
    """FilePresenceValidator.get_file_differences should return missing and extra files as separate sets."""
    from unittest.mock import Mock

    from sftk.validation_strategies import FilePresenceValidator

    # Mock S3Handler
    mock_s3_handler = Mock()
    mock_s3_handler.get_paths_from_csv.return_value = {
        "all": {"file1.mp4", "file2.mp4", "file3.mp4"},  # all files
        "filtered": {"file1.mp4", "file2.mp4"},  # filtered files
    }
    mock_s3_handler.get_paths_from_s3.return_value = {
        "file1.mp4",
        "file4.mp4",
    }  # S3 files

    validator = FilePresenceValidator({}, mock_s3_handler)
    rules = {
        "file_presence": {
            "csv_filename": "test.csv",
            "csv_column_to_extract": "video_path",
            "valid_extensions": ["mp4"],
            "path_prefix": "videos/",
            "s3_sharepoint_path": "sharepoint",
            "bucket": "test-bucket",
        }
    }

    missing_files, extra_files = validator.get_file_differences(rules)

    # Should find 1 missing file (file2.mp4 in filtered CSV but not in S3)
    assert missing_files == {"file2.mp4"}

    # Should find 1 extra file (file4.mp4 in S3 but not in all CSV files)
    assert extra_files == {"file4.mp4"}


def test_file_presence_validator_get_file_differences_missing_config():
    """FilePresenceValidator.get_file_differences should raise ValueError for missing config."""
    from unittest.mock import Mock

    from sftk.validation_strategies import FilePresenceValidator

    mock_s3_handler = Mock()
    validator = FilePresenceValidator({}, mock_s3_handler)

    # Test with missing required configuration
    rules = {
        "file_presence": {
            "csv_filename": "test.csv",
            # Missing csv_column_to_extract and bucket
        }
    }

    try:
        validator.get_file_differences(rules)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "requires csv_filename, csv_column_to_extract, and bucket" in str(e)


def test_strategy_registry_mapping():
    """Test that strategy registry correctly maps config flags to strategies."""
    from sftk.validation_strategies import ValidationConfig, ValidationStrategyRegistry

    registry = ValidationStrategyRegistry({}, {}, None)
    config = ValidationConfig()

    # Test individual strategy mappings
    config.required = True
    enabled = registry.get_enabled_strategies(config)
    assert len(enabled) == 1
    assert enabled[0] == registry.strategies["required"]

    config.required = False
    config.unique = True
    enabled = registry.get_enabled_strategies(config)
    assert len(enabled) == 1
    assert enabled[0] == registry.strategies["unique"]


def test_validation_config_methods_consistency():
    """Test that ValidationConfig methods work consistently."""
    from sftk.validation_strategies import ValidationConfig

    config = ValidationConfig()

    # Initially, no validations should be enabled
    assert not config.any_enabled()

    # Enable all should enable everything
    config.enable_all_validators()
    assert config.any_enabled()
    assert config.required
    assert config.unique
    assert config.foreign_keys
    assert config.formats
    assert config.column_relationships
    assert config.file_presence

    # Disable one and check any_enabled still works
    config.required = False
    assert config.any_enabled()  # Others are still enabled

    # Disable all manually
    config.required = False
    config.unique = False
    config.foreign_keys = False
    config.formats = False
    config.column_relationships = False
    config.file_presence = False
    assert not config.any_enabled()


def test_relationship_validator_dropid_replicate_mismatch():
    """RelationshipValidator should provide specific error message for DropID replicate mismatch."""
    from sftk.validation_strategies import RelationshipValidator
    from sftk.common import DROPID_COLUMN, REPLICATE_COLUMN

    validator = RelationshipValidator({})

    # Test case where only the replicate part (last 2 digits) is different
    rules = {
        "file_name": "test.csv",
        "relationships": [
            {
                "column": DROPID_COLUMN,
                "rule": "equals",
                "template": "{SurveyID}_{SiteID}_{ReplicateWithinSite:02}",
            }
        ],
    }

    # Create test data where DropID ends with '01' but ReplicateWithinSite is 2 (should be '02')
    df = pd.DataFrame({
        "DropID": ["HOR_20211122_BUV_HOR_003_01"],
        "SurveyID": ["HOR_20211122_BUV"],
        "SiteID": ["HOR_003"],
        "ReplicateWithinSite": [2]  # This should make expected DropID end with '02'
    })

    errors = validator.validate(rules, df)

    assert len(errors) == 1
    error_message = errors[0].error_info
    assert "ReplicateWithinSite mismatch" in error_message
    assert "should end with '02' but ends with '01'" in error_message
    assert "HOR_20211122_BUV_HOR_003_02" in error_message  # Expected value
    assert "HOR_20211122_BUV_HOR_003_01" in error_message  # Actual value


def test_relationship_validator_dropid_full_mismatch():
    """RelationshipValidator should provide generic error message for full DropID mismatch."""
    from sftk.validation_strategies import RelationshipValidator

    validator = RelationshipValidator({})

    rules = {
        "file_name": "test.csv",
        "relationships": [
            {
                "column": DROPID_COLUMN,
                "rule": "equals",
                "template": "{SurveyID}_{SiteID}_{ReplicateWithinSite:02}",
            }
        ],
    }

    # Create test data where DropID has different SiteID (not just replicate mismatch)
    df = pd.DataFrame({
        "DropID": ["HOR_20211122_BUV_HOR_004_01"],  # Different SiteID (004 vs 003)
        "SurveyID": ["HOR_20211122_BUV"],
        "SiteID": ["HOR_003"],
        "ReplicateWithinSite": [2]
    })

    errors = validator.validate(rules, df)

    assert len(errors) == 1
    error_message = errors[0].error_info
    # Should use generic message, not replicate-specific message
    assert "ReplicateWithinSite mismatch" not in error_message
    assert "DropID should be" in error_message
    assert "HOR_20211122_BUV_HOR_003_02" in error_message  # Expected value
    assert "HOR_20211122_BUV_HOR_004_01" in error_message  # Actual value


def test_is_replicate_mismatch_only_helper():
    """Test the helper method _is_replicate_mismatch_only."""
    from sftk.validation_strategies import RelationshipValidator

    validator = RelationshipValidator({})

    # Test replicate-only mismatch (should return True)
    actual = "HOR_20211122_BUV_HOR_003_01"
    expected = "HOR_20211122_BUV_HOR_003_02"
    assert validator._is_replicate_mismatch_only(actual, expected) is True

    # Test different site ID (should return False)
    actual = "HOR_20211122_BUV_HOR_004_01"
    expected = "HOR_20211122_BUV_HOR_003_02"
    assert validator._is_replicate_mismatch_only(actual, expected) is False

    # Test different survey ID (should return False)
    actual = "TON_20211026_BUV_TON_033_01"
    expected = "HOR_20211122_BUV_TON_033_01"
    assert validator._is_replicate_mismatch_only(actual, expected) is False

    # Test same strings (should return False)
    actual = "HOR_20211122_BUV_HOR_003_01"
    expected = "HOR_20211122_BUV_HOR_003_01"
    assert validator._is_replicate_mismatch_only(actual, expected) is False

    # Test different lengths (should return False)
    actual = "HOR_20211122_BUV_HOR_003_1"
    expected = "HOR_20211122_BUV_HOR_003_02"
    assert validator._is_replicate_mismatch_only(actual, expected) is False
