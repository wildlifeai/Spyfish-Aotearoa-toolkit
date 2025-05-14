import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sftk.utils import (
    EnvironmentVariableError,
    delete_file,
    flatten_list,
    get_env_var,
    get_unique_entries_df_column,
    is_format_match,
    read_file_to_df,
    temp_file_manager,
)

survey_pattern = r"^[A-Z]{3}_\d{8}_BUV$"


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [30, 25]})


def test_flatten_list_simple():
    simple_nested_list = [[1, 2], [3, 4], [5, 6]]
    assert (
        len(flatten_list(simple_nested_list)) == 6
    ), "Flattened list is not the correct length"
    assert flatten_list(simple_nested_list) == [
        1,
        2,
        3,
        4,
        5,
        6,
    ], "Flattened list does not contain the correct elements, or is not in the correct order"


def test_flatten_list_nested():
    # Test with a nested (recursive) list
    nested_list = [[1, 2], [3, 4], [5, [6, 7]]]
    assert (
        len(flatten_list(nested_list)) == 7
    ), "Flattened list is not the correct length"
    assert flatten_list(nested_list) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ], "Flattened list does not contain the correct elements, or is not in the correct order"


def test_is_format_match_valid():
    assert is_format_match(survey_pattern, "ABC_20250101_BUV")


def test_is_format_match_invalid():
    assert not is_format_match(survey_pattern, "AB_20250101_BUV")


def test_read_file_to_df_csv(sample_dataframe):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        sample_dataframe.to_csv(tmp.name, index=False)
        df = read_file_to_df(tmp.name)
        assert_frame_equal(df, sample_dataframe)


def test_read_file_to_df_excel_all_sheets(sample_dataframe):
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        with pd.ExcelWriter(tmp.name) as writer:
            sample_dataframe.to_excel(writer, sheet_name="Sheet1", index=False)
            sample_dataframe.to_excel(writer, sheet_name="Sheet2", index=False)
        sheets = read_file_to_df(tmp.name, sheet_name=None)
        assert isinstance(sheets, dict)
        assert "Sheet1" in sheets and "Sheet2" in sheets
        assert_frame_equal(sheets["Sheet1"], sample_dataframe)
        assert_frame_equal(sheets["Sheet2"], sample_dataframe)


# Tests for get_env_var
def test_get_env_var_exists():
    with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
        assert get_env_var("TEST_VAR") == "test_value"


def test_get_env_var_missing():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            EnvironmentVariableError, match="Environment variable 'MISSING_VAR' not set"
        ):
            get_env_var("MISSING_VAR")


# Tests for delete_file
def test_delete_file_exists():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    # Verify file exists
    assert os.path.exists(tmp_path)

    # Delete the file
    delete_file(tmp_path)

    # Verify file no longer exists
    assert not os.path.exists(tmp_path)


def test_delete_file_not_exists():
    # This should not raise an exception
    delete_file("non_existent_file.txt")


def test_delete_file_exception(caplog):
    with patch("os.path.exists", return_value=True):
        with patch("os.remove", side_effect=PermissionError("Permission denied")):
            delete_file("test.txt")
            assert "Failed to remove file 'test.txt': Permission denied" in caplog.text


# Tests for get_unique_entries_df_column
def test_get_unique_entries_df_column():
    # Create mock S3 handler
    mock_s3_handler = MagicMock()
    mock_df = pd.DataFrame(
        {"col1": ["value1", "value2", "value1", None], "col2": [1, 2, 3, 4]}
    )
    mock_s3_handler.read_df_from_s3_csv.return_value = mock_df

    # Test with drop_na=True (default)
    result = get_unique_entries_df_column(
        "test.csv", "col1", mock_s3_handler, "test-bucket"
    )
    assert result == {"value1", "value2"}

    # Test with drop_na=False
    result = get_unique_entries_df_column(
        "test.csv", "col1", mock_s3_handler, "test-bucket", drop_na=False
    )
    assert None in result
    assert len(result) == 3


# Tests for temp_file_manager
def test_temp_file_manager():
    # Create temporary files
    temp_files = []
    for _ in range(3):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_files.append(tmp.name)

    # Verify files exist
    for file in temp_files:
        assert os.path.exists(file)

    # Use temp_file_manager
    with temp_file_manager(temp_files):
        # Files should still exist within the context
        for file in temp_files:
            assert os.path.exists(file)

    # Files should be deleted after exiting the context
    for file in temp_files:
        assert not os.path.exists(file)


def test_temp_file_manager_with_exception():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_file = tmp.name

    # Verify file exists
    assert os.path.exists(temp_file)

    # Use temp_file_manager with an exception
    with pytest.raises(ValueError):
        with temp_file_manager([temp_file]):
            raise ValueError("Test exception")

    # File should still be deleted despite the exception
    assert not os.path.exists(temp_file)
