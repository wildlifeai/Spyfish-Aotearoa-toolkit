def flatten_list(lst: list[list]) -> list:
    """
    Flatten a list of lists.

    Args:
        lst (list[list]): A list of lists.

    Returns:
        list: A flattened list.
    """
    return [item for sublist in lst for item in sublist]
