import logging
from datetime import datetime as dt
from pathlib import Path


def get_logging_path() -> str:
    """
    Retrieves the path to the log file.

    This function creates a directory in the user's home directory called ".sftk" and a subdirectory called "logs".
    The logfile is named with the current date and time in the format "YYYY-MM-DD_HH-MM-SS.log".

    Returns:
        str: The path to the log file.
    """
    log_dir = Path.home() / ".sftk" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = dt.strftime(dt.now(), "%Y-%m-%d_%H-%M-%S") + ".log"
    log_file = log_dir / log_filename
    return str(log_file)

def get_log_level(level: str) -> int:
    """
    Converts a string log level to a corresponding logging constant.

    Args:
        level (str): The logging level as a string (e.g., "DEBUG", "INFO").

    Returns:
        int: The corresponding logging level constant.
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return levels.get(level.upper(), logging.INFO)

def set_log_level(level: str) -> None:
    """
    Dynamically sets the logging level for the root logger.

    Args:
        level (str): The logging level as a string (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if level.upper() in level_dict:
        logging.getLogger().setLevel(level_dict[level.upper()])
    else:
        raise ValueError(f"Invalid log level: {level}")


# Configure the logging module in global scope to ensure that all modules use the same configuration.
logging.basicConfig(
    filename=get_logging_path(),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
