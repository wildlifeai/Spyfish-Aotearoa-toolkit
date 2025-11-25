import os
import tempfile
from unittest.mock import patch

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
    normalize_file_name,
    read_file_to_df,
    temp_file_manager,
    write_data_to_file,
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
    # Create df for filtering
    mock_df = pd.DataFrame(
        {
            "col1": ["value1", "value2", "value1", None],
            "col2": [1, 2, 3, 4],
            "IsBadDeployment": [True, False, False, False],
        }
    )

    # Test with drop_na=True (default)
    result = get_unique_entries_df_column(mock_df, "col1")
    assert result == {"value1", "value2"}

    # Test with drop_na=False
    result = get_unique_entries_df_column(mock_df, "col1", drop_na=False)
    assert None in result
    assert len(result) == 3

    # Test Filtering by column value multiple drops per value
    result = get_unique_entries_df_column(
        mock_df, "col1", column_filter="IsBadDeployment", column_value=False
    )
    assert result == {"value1", "value2"}

    mock_df = pd.DataFrame(
        {
            "col1": ["value1", "value2", "value1", None],
            "col2": [1, 2, 3, 4],
            "IsBadDeployment": [False, True, False, False],
        }
    )

    # Test Filtering by column value, row cancelled
    result = get_unique_entries_df_column(
        mock_df, "col1", column_filter="IsBadDeployment", column_value=False
    )
    assert result == {"value1"}


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


def test_write_data_to_file():
    """Test writing file paths to text file."""

    file_set = {"path/to/file1.mp4", "path/to/file2.mp4", "another/file3.mp4"}
    file_str = "\n".join(sorted(list(file_set)))

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        output_path = tmp.name

    try:
        write_data_to_file(file_str, output_path)

        # Read the file back and verify contents
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        # Should be sorted and contain all files
        expected_lines = sorted(file_set)
        assert lines == expected_lines

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


def test_normalize_file_name():
    """Test normalize_file_name function with various inputs."""
    from pathlib import Path

    # Test with string path
    assert normalize_file_name("/path/to/file.csv") == "file.csv"
    assert normalize_file_name("path/to/file.csv") == "file.csv"
    assert normalize_file_name("file.csv") == "file.csv"

    # Test with Path object
    assert normalize_file_name(Path("/path/to/file.csv")) == "file.csv"
    assert normalize_file_name(Path("file.csv")) == "file.csv"

    # Test with non-path values (should return as-is)
    assert normalize_file_name(123) == 123
    assert normalize_file_name(None) is None
    assert normalize_file_name([]) == []

    # Test with empty string
    assert normalize_file_name("") == ""


def test_str_to_bool():
    """Test str_to_bool function with various inputs."""
    from sftk.utils import str_to_bool

    # Test with "true" (case-insensitive, must be exactly "true")
    assert str_to_bool("true") is True
    assert str_to_bool("True") is True
    assert str_to_bool("TRUE") is True
    assert str_to_bool(" true ") is True  # With whitespace

    # Test with other values (should be False - only "true" returns True)
    assert str_to_bool("false") is False
    assert str_to_bool("False") is False
    assert str_to_bool("yes") is False
    assert str_to_bool("1") is False
    assert str_to_bool("") is False

    # Test with None
    assert str_to_bool(None) is False


def test_filter_file_paths_by_extension():
    """Test filter_file_paths_by_extension function."""
    from sftk.utils import filter_file_paths_by_extension

    file_paths = [
        "video1.mp4",
        "video2.mov",
        "image1.jpg",
        "image2.png",
        "document.pdf",
        "video3.mp4",
    ]

    # Test filtering for video extensions
    video_files = filter_file_paths_by_extension(file_paths, ["mp4", "mov"])
    assert len(video_files) == 3
    assert "video1.mp4" in video_files
    assert "video2.mov" in video_files
    assert "video3.mp4" in video_files

    # Test filtering for image extensions
    image_files = filter_file_paths_by_extension(file_paths, ["jpg", "png"])
    assert len(image_files) == 2
    assert "image1.jpg" in image_files
    assert "image2.png" in image_files

    # Test with empty list
    empty_result = filter_file_paths_by_extension(file_paths, [])
    assert len(empty_result) == 0

    # Test case insensitivity
    mixed_case = ["FILE.MP4", "file.mp4", "File.Mp4"]
    result = filter_file_paths_by_extension(mixed_case, ["mp4"])
    assert len(result) == 3


def test_convert_int_num_columns_to_int():
    """Test convert_int_num_columns_to_int function."""
    from sftk.utils import convert_int_num_columns_to_int

    # Create DataFrame with float columns that are whole numbers
    df = pd.DataFrame(
        {
            "int_col": [1.0, 2.0, 3.0],
            "float_col": [1.5, 2.5, 3.5],
            "mixed_col": [1.0, 2.5, 3.0],
            "string_col": ["a", "b", "c"],
        }
    )

    result = convert_int_num_columns_to_int(df)

    # int_col should be converted to Int64
    assert result["int_col"].dtype == "Int64"
    assert list(result["int_col"]) == [1, 2, 3]

    # float_col should remain float (has decimals)
    assert result["float_col"].dtype == "float64"

    # mixed_col should remain float (has decimals)
    assert result["mixed_col"].dtype == "float64"

    # string_col should remain object
    assert result["string_col"].dtype == "object"
