from unittest.mock import Mock

import pandas as pd

from sftk.common import DROP_ID_COLUMN
from sftk.validation_strategies import (
    FilePresenceValidator,
    ForeignKeyValidator,
    FormatValidator,
    RelationshipValidator,
    RequiredValidator,
    UniqueValidator,
    ValidationConfig,
    ValidationStrategyRegistry,
)


def test_validation_config():
    """ValidationConfig should have defaults, enable_all, and any_enabled methods."""
    config = ValidationConfig()

    # Test defaults
    assert config.required is False
    assert config.unique is False
    assert not config.any_enabled()

    # Test enable_all
    config.enable_all_validators()
    assert config.required is True
    assert config.unique is True
    assert config.foreign_keys is True
    assert config.formats is True
    assert config.column_relationships is True
    assert config.file_presence is True
    assert config.any_enabled()


def test_required_validator():
    """RequiredValidator should find missing values and generate error messages."""
    validator = RequiredValidator({})
    rules = {"file_name": "test.csv", "required": ["name"], "info_columns": []}
    df = pd.DataFrame({"name": ["Alice", None, "Bob"]})

    errors = validator.validate(rules, df)

    assert len(errors) == 1
    assert "Missing value in required column 'name'" in errors[0].error_info
    assert "help_info:" in errors[0].error_info


def test_unique_validator():
    """UniqueValidator should find duplicate values."""
    validator = UniqueValidator({})
    rules = {"file_name": "test.csv", "unique": ["id"], "info_columns": []}
    df = pd.DataFrame({"id": [1, 2, 2, 3]})

    errors = validator.validate(rules, df)

    assert len(errors) == 2  # Both duplicate rows should be flagged
    assert all(
        "Duplicate value in unique column" in error.error_info for error in errors
    )


def test_format_validator():
    """FormatValidator should find values that don't match patterns."""
    patterns = {"email": r"^[^@]+@[^@]+\.[^@]+$"}
    validator = FormatValidator({}, patterns)
    rules = {"file_name": "test.csv", "info_columns": []}
    df = pd.DataFrame({"email": ["valid@test.com", "invalid-email", "also@valid.org"]})

    errors = validator.validate(rules, df)

    assert len(errors) == 1
    assert "does not match required format" in errors[0].error_info


def test_foreign_key_validator():
    """ForeignKeyValidator should find missing foreign key references."""
    target_df = pd.DataFrame({"target_id": [1, 2, 3]})
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
    """ValidationStrategyRegistry should manage and retrieve strategies."""
    validation_rules = {}
    patterns = {"email": r"^[^@]+@[^@]+\.[^@]+$"}
    registry = ValidationStrategyRegistry(validation_rules, patterns)

    # Test getting individual strategies
    required_validator = registry.get_strategy("required")
    assert required_validator is not None

    # Test getting enabled strategies based on config
    config = ValidationConfig()
    config.required = True
    config.unique = True

    enabled_strategies = registry.get_enabled_strategies(config)
    assert len(enabled_strategies) == 2
    assert enabled_strategies[0] == registry.strategies["required"]
    assert enabled_strategies[1] == registry.strategies["unique"]


def test_file_presence_validator():
    """FilePresenceValidator should find missing and extra files."""
    mock_s3_handler = Mock()
    mock_s3_handler.get_paths_from_csv.return_value = {
        "all": {"file1.mp4", "file2.mp4", "file3.mp4"},
        "filtered": {"file1.mp4", "file2.mp4"},
    }
    mock_s3_handler.get_paths_from_s3.return_value = {"file1.mp4", "file4.mp4"}

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


def test_relationship_validator():
    """RelationshipValidator should validate DropID relationships and provide specific error messages."""
    validator = RelationshipValidator({})

    rules = {
        "file_name": "test.csv",
        "relationships": [
            {
                "column": DROP_ID_COLUMN,
                "rule": "equals",
                "template": "{SurveyID}_{SiteID}_{ReplicateWithinSite:02}",
            }
        ],
    }

    # Test replicate-only mismatch (should get specific error message)
    df_replicate = pd.DataFrame(
        {
            "DropID": ["HOR_20211122_BUV_HOR_003_01"],
            "SurveyID": ["HOR_20211122_BUV"],
            "SiteID": ["HOR_003"],
            "ReplicateWithinSite": [2],  # Should be '02' not '01'
        }
    )

    errors = validator.validate(rules, df_replicate)
    assert len(errors) == 1
    assert "ReplicateWithinSite mismatch" in errors[0].error_info
    assert "should end with '02' but ends with '01'" in errors[0].error_info

    # Test full mismatch (should get generic error message)
    df_full = pd.DataFrame(
        {
            "DropID": ["HOR_20211122_BUV_HOR_004_01"],  # Different SiteID
            "SurveyID": ["HOR_20211122_BUV"],
            "SiteID": ["HOR_003"],
            "ReplicateWithinSite": [2],
        }
    )

    errors = validator.validate(rules, df_full)
    assert len(errors) == 1
    assert "ReplicateWithinSite mismatch" not in errors[0].error_info
    assert "DropID should be" in errors[0].error_info
