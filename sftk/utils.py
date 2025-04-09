import re
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


def read_file_to_df(file_path: str, sheet_name: str | int | list | None = 0) -> pd.DataFrame | dict:
    """Reads a CSV or Excel file into a Pandas DataFrame.
    
    If you don't know the name of your sheet, you can set sheet_name=None which
    returns a dictionary with all the sheet names as keys and content of each 
    sheet in dfs as values. Usign <output>.keys() outputs all the sheet names.
    """
    # TODO add tests
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
    return bool(re.fullmatch(pattern, string))   