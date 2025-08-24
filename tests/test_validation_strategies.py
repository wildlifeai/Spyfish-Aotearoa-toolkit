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
    config.enable_all()
    assert config.required is True
    assert config.unique is True


def test_error_message_for_required():
    """Should build correct message for required validation."""
    from sftk.validation_strategies import ErrorMessageBuilder, ValidationCheckType

    message = ErrorMessageBuilder.build_message(
        ValidationCheckType.REQUIRED, "name", "help info"
    )
    assert "Missing value in required column 'name'" in message


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
    mock_s3_handler.get_paths_from_csv.return_value = (
        {"file1.mp4", "file2.mp4", "file3.mp4"},  # all files
        {"file1.mp4", "file2.mp4"},  # filtered files
    )
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
    config.enable_all()
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
