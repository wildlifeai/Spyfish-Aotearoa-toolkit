"""
Data validation script.

This script runs comprehensive data validation using the DataValidator class
from the sftk module. It validates data according to configured rules and
exports the results either locally or to S3.
"""

from sftk.data_validator import DataValidator

if __name__ == "__main__":
    dv = DataValidator()
    dv.run_validation(
        remove_duplicates=True,
        extract_clean_dataframes=True,
        file_presence=True,
    )
