"""
Data validation orchestrator module.

This module contains the DataValidator class that orchestrates comprehensive
data validation using various validation strategies.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from sftk.common import (
    S3_BUCKET,
    S3_KSO_ERRORS_CSV,
    S3_SHAREPOINT_PATH,
    VALIDATION_PATTERNS,
    VALIDATION_RULES,
    VIDEOS_IN_AWS_INFO,
    VIDEOS_IN_SHAREPOINT_INFO,
)
from sftk.s3_handler import S3FileNotFoundError, S3Handler
from sftk.utils import convert_int_num_columns_to_int
from sftk.validation_strategies import (
    ErrorChecking,
    ValidationConfig,
    ValidationStrategyRegistry,
)


class DataValidator:
    """
    Main data validation orchestrator.

    This class coordinates comprehensive data validation by managing validation
    rules, datasets, and validation strategies. It provides a unified interface
    for running various types of validation checks and collecting results.
    """

    def __init__(self):
        """
        Initialize the DataValidator with default settings and validation rules.

        Sets up the validator with:
        - Empty error list for collecting validation errors
        - Validation patterns from configuration
        - S3 handler for accessing cloud storage
        - Validation rules loaded from S3 datasets
        - Strategy registry for managing validation strategies

        Attributes:
            errors (list): List to store ErrorChecking objects during validation
            errors_df (pd.DataFrame): DataFrame containing all validation errors
            patterns (dict): Regex patterns for format validation from VALIDATION_PATTERNS
            s3_handler (S3Handler): Handler for S3 operations
            bucket (str): S3 bucket name from configuration
            validation_rules (dict): Loaded validation rules with associated datasets
            strategy_registry (ValidationStrategyRegistry): Registry for validation strategies
        """
        self.errors = []
        self.errors_df = None
        self.patterns = VALIDATION_PATTERNS
        self.s3_handler = S3Handler()
        self.bucket = S3_BUCKET
        self.validation_rules = self._get_validation_rules()
        # Initialize with default max_errors, will be updated per validation run
        self.strategy_registry = ValidationStrategyRegistry(
            self.validation_rules, self.patterns, self.s3_handler
        )

    def _get_validation_rules(self) -> Dict[str, Any]:
        """
        Load validation rules and their associated datasets from S3.

        Creates a deep copy of the global VALIDATION_RULES configuration and loads
        the actual datasets from S3 for each rule set. If a dataset cannot be loaded,
        an error is recorded and an empty DataFrame is used as a fallback.

        Returns:
            dict: Validation rules dictionary with the following structure:
                {
                    'dataset_name': {
                        'file_name': str,  # S3 path to the dataset
                        'required': list,  # Required columns
                        'unique': list,    # Columns that must be unique
                        'foreign_keys': dict,  # Foreign key relationships
                        'relationships': list,  # Column relationship rules
                        'info_columns': list,  # Columns for error context
                        'dataset': pd.DataFrame  # The actual loaded dataset
                    }
                }

        Side Effects:
            - Records errors for datasets that cannot be loaded from S3
            - Uses empty DataFrames as fallbacks for failed loads
        """
        validation_rules = copy.deepcopy(VALIDATION_RULES)
        for dataset_name, rule_set in validation_rules.items():
            try:
                df = self.s3_handler.read_df_from_s3_csv(rule_set["file_name"])
            except S3FileNotFoundError as e:
                self._add_error(
                    file_name=dataset_name,
                    message=f"Dataset '{dataset_name}' could not be downloaded from '{rule_set["file_name"]}', error: {e}",
                )
                # Assign empty df on failure
                df = pd.DataFrame()
            validation_rules[dataset_name]["dataset"] = df
        return validation_rules

    def validate_with_config(self, config: ValidationConfig) -> pd.DataFrame:
        """
        Perform comprehensive data validation using ValidationConfig.

        Args:
            config (ValidationConfig): Configuration object specifying which validations to run

        Returns:
            pd.DataFrame: DataFrame containing all validation errors found
        """
        # If no validations are enabled, enable all
        if not config.any_enabled():
            config.enable_all()

        # Create a new registry with the config's max_errors setting
        strategy_registry = ValidationStrategyRegistry(
            self.validation_rules,
            self.patterns,
            self.s3_handler,
            config.max_errors_per_validator,
        )

        # Get enabled strategies
        enabled_strategies = strategy_registry.get_enabled_strategies(config)

        # Add file presence configuration to validation rules if needed
        if config.file_presence:
            file_presence_rules = {
                "file_presence": {
                    "csv_filename": VIDEOS_IN_SHAREPOINT_INFO["csv_filename"],
                    "csv_column_to_extract": VIDEOS_IN_SHAREPOINT_INFO["info_column"],
                    "column_filter": VIDEOS_IN_SHAREPOINT_INFO["column_filter"],
                    "column_value": VIDEOS_IN_SHAREPOINT_INFO["column_value"],
                    "valid_extensions": VIDEOS_IN_AWS_INFO["valid_movie_extensions"],
                    "path_prefix": VIDEOS_IN_AWS_INFO["path_prefix"],
                    "s3_sharepoint_path": S3_SHAREPOINT_PATH,
                    "bucket": self.bucket,
                }
            }
            # Add file presence rules to validation rules temporarily
            temp_validation_rules = {**self.validation_rules, **file_presence_rules}
        else:
            temp_validation_rules = self.validation_rules

        # Process each dataset
        for dataset_name, rules in temp_validation_rules.items():
            # Handle file presence validation separately (no DataFrame needed)
            if dataset_name == "file_presence":
                if config.file_presence:
                    self._validate_file_presence(rules, strategy_registry)
                continue

            file_name = Path(rules.get("file_name", "")).name
            df = rules.get("dataset", pd.DataFrame())

            if df.empty:
                self._add_error(
                    file_name=file_name, message="Dataset could not be loaded."
                )
                continue

            # Convert numerical values to integers where possible
            df = convert_int_num_columns_to_int(df)

            # Run each enabled validation strategy (except file presence)
            for strategy in enabled_strategies:
                # Skip file presence validator for regular datasets
                if strategy != strategy_registry.strategies.get("file_presence"):
                    strategy_errors = strategy.validate(rules, df)
                    # Strategies already return ErrorChecking objects, no conversion needed
                    self.errors.extend(strategy_errors)

        self._export_errors_from_list_to_df(config.remove_duplicates)
        return self.errors_df

    def _validate_file_presence(self, rules, strategy_registry):
        """
        Handle file presence validation separately from regular dataset validation.

        File presence validation compares video file paths referenced in CSV data
        with actual files available in S3 storage, identifying missing and extra files.
        This validation doesn't require a DataFrame as it works directly with S3.

        Args:
            rules (dict): File presence validation rules containing:
                - csv_filename: Path to CSV file with file references
                - csv_column_to_extract: Column containing file paths
                - column_filter: Column to filter on (optional)
                - column_value: Value to filter by (optional)
                - valid_extensions: List of valid file extensions
                - path_prefix: S3 path prefix for files
                - s3_sharepoint_path: SharePoint path in S3
                - bucket: S3 bucket name
            strategy_registry (ValidationStrategyRegistry): Registry containing
                the file presence validator strategy

        Side Effects:
            - Extends self.errors with any file presence validation errors
            - Logs validation start and completion
        """
        logging.info("Starting file presence validation.")
        file_presence_validator = strategy_registry.strategies.get("file_presence")
        if file_presence_validator:
            # The FilePresenceValidator expects rules with a 'file_presence' key
            file_presence_rules_wrapper = {"file_presence": rules}
            strategy_errors = file_presence_validator.validate(
                file_presence_rules_wrapper, pd.DataFrame()
            )
            # FilePresenceValidator already returns ErrorChecking objects
            self.errors.extend(strategy_errors)
        logging.info("File presence validation completed.")

    def _add_error(
        self,
        message,
        file_name=None,
        column_name=None,
        relevant_column_value=None,
        error_source="Sharepoint error validation",
    ):
        """
        Record a validation error in the errors list.

        Creates an ErrorChecking object with the provided error details and
        adds it to the internal errors list for later processing.

        Args:
            message (str): Detailed error message describing the validation failure
            file_name (str or Path, optional): Name or path of the file where
                the error occurred. Will be converted to filename only.
            column_name (str, optional): Name of the column that failed validation
            relevant_column_value (Any, optional): The actual value that caused
                the validation error
            error_source (str, optional): Source of the validation check.
                Defaults to "Sharepoint error validation".

        Side Effects:
            - Appends new ErrorChecking object to self.errors list
            - Converts file_name to just the filename (no path) if provided
        """
        if isinstance(file_name, (str, Path)):
            file_name = Path(file_name).name

        self.errors.append(
            ErrorChecking(
                column_name=column_name,
                relevant_column_value=relevant_column_value,
                relevant_file=file_name,
                error_info=message,
                error_source=error_source,
            )
        )

    def _export_errors_from_list_to_df(self, remove_duplicates):
        """
        Convert the errors list to a pandas DataFrame and optionally remove duplicates.

        Transforms the list of ErrorChecking objects into a DataFrame for easier
        analysis and export. Clears the errors list after conversion.

        Args:
            remove_duplicates (bool): Whether to remove duplicate error entries
                from the resulting DataFrame

        Side Effects:
            - Sets self.errors_df to a DataFrame containing all error data
            - Calls _deduplicate_errors() if remove_duplicates is True
            - Clears self.errors list after conversion

        Note:
            Uses the __dict__ attribute of ErrorChecking objects to create
            DataFrame columns matching the dataclass fields.
        """
        self.errors_df = pd.DataFrame([e.__dict__ for e in self.errors])

        if remove_duplicates:
            self._deduplicate_errors()
        self.errors = []

    def _deduplicate_errors(self):
        """
        Remove duplicate error entries from the errors DataFrame.

        Uses all fields from the ErrorChecking dataclass to identify and remove
        duplicate error entries. This helps reduce noise in validation reports
        when the same error occurs multiple times.

        Side Effects:
            - Modifies self.errors_df by removing duplicate rows
            - Resets DataFrame index after deduplication
            - Does nothing if errors_df is empty

        Note:
            Uses all ErrorChecking dataclass fields as the subset for
            duplicate detection, ensuring only truly identical errors are removed.
        """
        if self.errors_df.empty:
            return

        key_cols = list(ErrorChecking.__dataclass_fields__.keys())
        self.errors_df = self.errors_df.drop_duplicates(
            subset=key_cols, ignore_index=True
        )

    def export_to_csv(self, csv_file_name="validation_errors_cleaned.csv"):
        """
        Export validation errors to a CSV file.

        Saves the errors DataFrame to a CSV file for analysis and reporting.
        The CSV will contain all validation errors with their associated metadata.

        Args:
            csv_file_name (str, optional): Name of the output CSV file.
                Defaults to "validation_errors_cleaned.csv".

        Side Effects:
            - Creates a CSV file with the specified name
            - Logs the export operation

        Note:
            The CSV is exported without the DataFrame index to keep the
            output clean and focused on the error data.
        """
        self.errors_df.to_csv(csv_file_name, index=False)
        logging.info(f"Errors exported to csv file {csv_file_name}.")

    def upload_to_s3(self):
        """
        Upload validation errors DataFrame to S3 storage.

        Uploads the errors DataFrame to the configured S3 bucket and key location
        for centralized storage and access. Uses the S3Handler to manage the upload.

        Side Effects:
            - Uploads errors_df to S3 at the location specified by S3_KSO_ERRORS_CSV
            - Logs the upload operation

        Note:
            - Uses "errors" as the keyword for the upload operation
            - Uploads without the DataFrame index to keep the data clean
            - Relies on S3_BUCKET and S3_KSO_ERRORS_CSV configuration constants
        """
        keyword = "errors"
        self.s3_handler.upload_updated_df_to_s3(
            df=self.errors_df,
            key=S3_KSO_ERRORS_CSV,
            keyword=keyword,
            bucket=self.bucket,
            keep_df_index=False,
        )
        logging.info(f"Updated {keyword} DataFrame uploaded to S3")
