"""
Data validation script.

This script runs comprehensive data validation using the DataValidator class
from the sftk module. It validates data according to configured rules and
exports the results either locally or to S3.
"""

import logging
import os

# Import centralized logging configuration
from sftk import log_config  # noqa: F401
from sftk.common import EXPORT_LOCAL, LOCAL_DATA_FOLDER_PATH
from sftk.data_validator import DataValidator
from sftk.validation_strategies import ValidationConfig


def main():
    """Main function to run data validation."""
    # Logging is configured by importing sftk.log_config
    logging.info("Error validation started")

    # Create validator and configuration
    validator = DataValidator()
    config = ValidationConfig()
    config.enable_all()  # Enable all validation types

    # Run validation using new interface
    result_df = validator.validate_with_config(config)
    logging.info(f"Error validation completed, {result_df.shape[0]} errors found")

    # Export results
    if EXPORT_LOCAL:
        validator.export_to_csv(
            os.path.join(LOCAL_DATA_FOLDER_PATH, "validation_errors.csv")
        )
    else:
        validator.upload_to_s3()
    logging.info("Error validation process completed, files created/uploaded.")


if __name__ == "__main__":
    main()
