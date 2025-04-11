import pandas as pd
import pytest
import tempfile
from pandas.testing import assert_frame_equal
from sftk.utils import flatten_list, read_file_to_df, is_format_match


survey_pattern = r"^[A-Z]{3}_\d{8}_BUV$"

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "Name": ["Alice", "Bob"],
        "Age": [30, 25]
    })


def test_flatten_list_simple():
    l = [[1, 2], [3, 4], [5, 6]]
    assert len(flatten_list(l)) == 6, "Flattened list is not the correct length"
    assert flatten_list(l) == [1, 2, 3, 4, 5, 6], "Flattened list does not contain the correct elements, or is not in the correct order"

def test_flatten_list_nested():
    # Test with a nested (recursive) list
    l = [[1, 2], [3, 4], [5, [6, 7]]]
    assert len(flatten_list(l)) == 7, "Flattened list is not the correct length"
    assert flatten_list(l) == [1, 2, 3, 4, 5, 6, 7], "Flattened list does not contain the correct elements, or is not in the correct order"


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


