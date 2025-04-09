import os

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

def clamp_n_jobs(n_jobs):
    """
    Clamp the number of jobs to a valid range.

    Args:
        n_jobs (int): The number of jobs to run in parallel.

    Returns:
        int: The clamped number of jobs.
    """
    if n_jobs == -1:
        return os.cpu_count() or 1
    return max(1, n_jobs)
