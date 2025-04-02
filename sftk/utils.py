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


def read_file_to_df(file_path: str, sheet_name: str | int | list | None = 0):
    """Reads a CSV or Excel file into a Pandas DataFrame."""
    # TODO add tests
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    return pd.read_excel(file_path, sheet_name=sheet_name)
