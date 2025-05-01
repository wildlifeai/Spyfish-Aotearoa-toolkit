import logging
import os
import re
from contextlib import contextmanager
from typing import Iterable, List, cast

import pandas as pd


def flatten_list(lst: list[list]) -> list:
    """
    Flatten a list of lists.

    Args:
        lst (list[list]): A list of lists.

    Returns:
        list: A flattened list.
    """
    flattened = []
    for item in lst:
        item = flatten_list(item) if isinstance(item, list) else [item]
        flattened.extend(item)
    return flattened


def read_file_to_df(
    file_path: str, sheet_name: str | int | list | None = 0
) -> pd.DataFrame | dict:
    """Reads a CSV or Excel file into a Pandas DataFrame.

    If you don't know the name of your sheet, you can set sheet_name=None which
    returns a dictionary with all the sheet names as keys and content of each
    sheet in dfs as values. Usign <output>.keys() outputs all the sheet names.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    return pd.read_excel(file_path, sheet_name=sheet_name)


def is_format_match(pattern, string):
    """Checks if a string matches a regex pattern.

    Args:
        pattern (str): The regex pattern to match against.
        string (str): The string to check.

    Returns:
        bool: True if the string matches the pattern, False otherwise.
    """
    if pd.isna(string):
        return False
    return bool(re.fullmatch(pattern, string))


class EnvironmentVariableError(Exception):
    """Custom exception for missing environment variables."""


def get_env_var(name: str) -> str:
    """
    Gets an environment variable and raises an error if not found.

    Args:
        name: The name of the environment variable.

    Returns:
        The value of the environment variable.

    Raises:
        EnvironmentVariableError: If the environment variable is not set.
    """
    value = os.getenv(name)
    if value is None:
        raise EnvironmentVariableError(f"Environment variable '{name}' not set.")
    return cast(str, value)


def delete_file(filename: str):
    try:
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:  # TODO less wide exception
        logging.error("Failed to remove (temporary) file %s: %s", filename, str(e))


def filter_file_paths_by_extension(
    file_paths_iterable: Iterable, valid_extensions: Iterable
) -> List[str]:
    """
    Filter a collection of file paths, returning only those that match the given file extensions.

    Parameters:
        file_paths_iterable Iterable: A set or list of file paths (strings) to filter.
        valid_extensions Iterable: A list or tuple of valid file extensions (e.g., ['mp4', 'mov']).

    Returns:
        list: A list of file paths that have an extension matching one of the valid extensions.
    """
    filtered_file_paths = []

    for file_path in file_paths_iterable:
        # Extract the file extension (e.g., 'mp4', 'jpg'), remove the leading dot
        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")

        # Include file path if its extension is in the valid list
        if ext in valid_extensions:
            filtered_file_paths.append(file_path)

    return filtered_file_paths


def get_unique_entries_df_column(
    csv_filename_path, column_name_to_extract, s3_handler, bucket, drop_na=True
) -> set:
    buv_deployment_df = s3_handler.read_df_from_s3_csv(csv_filename_path, bucket)
    if drop_na:
        csv_filepaths = set(
            buv_deployment_df[column_name_to_extract].dropna().astype(str)
        )
    else:
        csv_filepaths = set(buv_deployment_df[column_name_to_extract])

    return csv_filepaths


@contextmanager
def temp_file_manager(filenames: list[str]):
    """Context manager to handle temporary file cleanup."""
    try:
        yield
    finally:
        for filename in filenames:
            delete_file(filename)
