"""
Data validation orchestrator module.

This module contains the DataValidator class that orchestrates comprehensive
data validation using various validation strategies.
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from sftk.common import (
    EXPORT_LOCAL,
    FILE_PRESENCE_RULES,
    LOCAL_DATA_FOLDER_PATH,
    S3_SPYFISH_METADATA,
    VALIDATION_PATTERNS,
    VALIDATION_RULES,
)
from sftk.s3_handler import S3FileNotFoundError, S3Handler
from sftk.utils import (
    convert_int_num_columns_to_int,
    normalize_file_name,
    write_data_to_file,
)
from sftk.validation_strategies import (
    CleanRowTracker,
    ErrorChecking,
    ErrorSource,
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
        - Validation rules configured with reference datasets loaded from S3
        - Strategy registry for managing validation strategies

        Attributes:
            errors (list): List to store ErrorChecking objects during validation
            errors_df (pd.DataFrame): DataFrame containing all validation errors
            patterns (dict): Regex patterns for format validation from VALIDATION_PATTERNS
            s3_handler (S3Handler): Handler for S3 operations
            validation_rules (dict): Loaded validation rules with associated reference datasets
            strategy_registry (ValidationStrategyRegistry): Registry for validation strategies
        """
        self.errors = []
        self.errors_df = None
        self.patterns = VALIDATION_PATTERNS
        self.s3_handler = S3Handler()
        self.validation_rules = self._get_validation_rules()
        self.clean_row_tracker = None
        # Initialize with default max_errors, will be updated per validation run
        self.strategy_registry = ValidationStrategyRegistry(
            self.validation_rules, self.patterns, self.s3_handler
        )
        if EXPORT_LOCAL:
            self.FOLDER_PATH = LOCAL_DATA_FOLDER_PATH
        else:
            self.FOLDER_PATH = S3_SPYFISH_METADATA

    def run_validation(
        self,
        enable_all: bool = False,
        required: bool = False,
        unique: bool = False,
        foreign_keys: bool = False,
        formats: bool = False,
        column_relationships: bool = False,
        file_presence: bool = False,
        remove_duplicates: bool = True,
        extract_clean_dataframes: bool = False,
    ):
        """Main function to run data validation."""
        logging.info("Error validation started")
        config = ValidationConfig()

        config.remove_duplicates = remove_duplicates
        config.extract_clean_dataframes = extract_clean_dataframes
        if enable_all:
            config.enable_all_validators()  # Enable all validation types
        else:
            config.required = required
            config.unique = unique
            config.foreign_keys = foreign_keys
            config.formats = formats
            config.column_relationships = column_relationships
            config.file_presence = file_presence

        # Run validation using new interface
        result_df = self.validate_with_config(config)
        logging.info(f"Error validation completed, {result_df.shape[0]} errors found")
        self.export_to_csv()
        if config.extract_clean_dataframes:
            self.export_clean_dataframes_to_csv()
            summary = self.get_clean_summary()
            logging.info(f"Data info: {summary}")

        if config.file_presence:
            self.export_file_differences()

        logging.info("Error validation process completed, files created/uploaded.")

    def _get_validation_rules(self) -> Dict[str, Any]:
        """
        Load validation rules with their associated reference datasets from S3.

        Creates a deep copy of the global VALIDATION_RULES configuration and loads
        the actual reference datasets from S3 for each rule set. If a dataset cannot be loaded,
        an error is recorded and an empty DataFrame is used as a fallback.

        Returns:
            dict: Validation rules dictionary with the following structure:
                {
                    'dataset_name': {
                        'file_name': str,  # S3 path to the reference dataset
                        'required': list,  # Required columns
                        'unique': list,    # Columns that must be unique
                        'foreign_keys': dict,  # Foreign key relationships
                        'relationships': list,  # Column relationship rules
                        'info_columns': list,  # Columns for error context
                        'dataset': pd.DataFrame  # The actual loaded reference dataset
                    }
                }

        Side Effects:
            - Records errors for reference datasets that cannot be loaded from S3
            - Uses empty DataFrames as fallbacks for failed loads
        """
        validation_rules = copy.deepcopy(VALIDATION_RULES)
        for dataset_name, rule_set in validation_rules.items():
            try:
                df = self.s3_handler.read_df_from_s3_csv(rule_set["file_name"])
            except S3FileNotFoundError as e:
                error = self.create_error(
                    message=f"Dataset '{dataset_name}' could not be downloaded from '{rule_set["file_name"]}', error: {e}",
                    error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                    file_name=dataset_name,
                )
                self.errors.append(error)
                # Assign empty df on failure
                df = pd.DataFrame()
            validation_rules[dataset_name]["dataset"] = df
        return validation_rules

    def _process_datasets(
        self,
        temp_validation_rules: Dict[str, Any],
        config: ValidationConfig,
        enabled_strategies: list,
        strategy_registry: "ValidationStrategyRegistry",
    ) -> None:
        """
        Process each dataset with enabled validation strategies.

        Args:
            temp_validation_rules: Dictionary of validation rules including datasets
            config: Validation configuration object
            enabled_strategies: List of enabled validation strategy instances
            strategy_registry: Registry containing all validation strategies

        Side Effects:
            - Extends self.errors with validation errors from all datasets
            - Handles file presence validation separately
        """
        for dataset_name, rules in temp_validation_rules.items():
            # Handle file presence validation separately (no DataFrame needed)
            if dataset_name == "file_presence":
                if config.file_presence:
                    pass
                    # TODO will probably remove this part, as it's not part of the errors anymore
                    # self._validate_file_presence(rules, strategy_registry)
                continue

            file_name = normalize_file_name(rules.get("file_name", ""))
            df = rules.get("dataset", pd.DataFrame())

            if df.empty:
                error = self.create_error(
                    message="Dataset could not be loaded.",
                    error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                    file_name=file_name,
                )
                self.errors.append(error)
                continue

            # Convert numerical values to integers where possible
            df = convert_int_num_columns_to_int(df)

            # Initialize clean indices for this dataset if tracker is enabled
            if self.clean_row_tracker:
                self.clean_row_tracker.initialize_dataset(dataset_name, df)

            # Run validation strategies on the dataset
            self._run_validation_strategies(
                enabled_strategies, strategy_registry, rules, df, dataset_name
            )

    def _run_validation_strategies(
        self,
        enabled_strategies: list,
        strategy_registry: "ValidationStrategyRegistry",
        rules: Dict[str, Any],
        df: pd.DataFrame,
        dataset_name: str,
    ) -> None:
        """
        Run enabled validation strategies on a dataset.

        Args:
            enabled_strategies: List of enabled validation strategy instances
            strategy_registry: Registry containing all validation strategies
            rules: Validation rules for the current dataset
            df: DataFrame to validate

        Side Effects:
            - Extends self.errors with any validation errors found
        """
        for strategy in enabled_strategies:
            # Skip file presence validator for regular datasets
            if strategy != strategy_registry.strategies.get("file_presence"):
                strategy_errors = strategy.validate(rules, df, dataset_name)
                self.errors.extend(strategy_errors)

    def _prepare_validation_rules(self, config: ValidationConfig) -> Dict[str, Any]:
        """
        Prepare validation rules including file presence rules if needed.

        Args:
            config: Validation configuration object

        Returns:
            Dict[str, Any]: Complete validation rules including file presence if enabled
        """
        if config.file_presence:
            return {**self.validation_rules, **FILE_PRESENCE_RULES}
        return self.validation_rules

    def validate_with_config(self, config: ValidationConfig) -> pd.DataFrame:
        """
        Orchestrate comprehensive data validation using ValidationConfig.

        This method coordinates the entire validation process by:
        1. Ensuring at least one validation type is enabled
        2. Setting up validation strategies and registry
        3. Preparing validation rules (including file presence if needed)
        4. Processing all datasets with enabled validation strategies
        5. Exporting results to DataFrame format

        Args:
            config (ValidationConfig): Configuration object specifying which validations to run

        Returns:
            pd.DataFrame: DataFrame containing all validation errors found
        """
        # Ensure at least one validation type is enabled
        if not config.any_enabled():
            config.enable_all_validators()

        # Initialize clean row tracker if requested
        if config.extract_clean_dataframes:
            self.clean_row_tracker = CleanRowTracker()
        else:
            self.clean_row_tracker = None

        # Set up validation infrastructure
        strategy_registry = ValidationStrategyRegistry(
            self.validation_rules,
            self.patterns,
            self.s3_handler,
            config.max_errors_per_validator,
            self.clean_row_tracker,
        )
        enabled_strategies = strategy_registry.get_enabled_strategies(config)

        # Prepare validation rules with file presence if needed
        temp_validation_rules = self._prepare_validation_rules(config)

        # Execute validation on all datasets
        self._process_datasets(
            temp_validation_rules, config, enabled_strategies, strategy_registry
        )

        # Export and return results
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
            strategy_registry (ValidationStrategyRegistry):
                - Registry containing the file presence validator strategy

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
            self.errors.extend(strategy_errors)
        logging.info("File presence validation completed.")

    def create_error(
        self,
        survey_id: str = "",
        drop_id: str = "",
        message: str = "",
        error_source: str = "",
        file_name: str = "",
        column_name: str = "",
        relevant_column_value: Any = None,
    ) -> ErrorChecking:
        """
        Create an ErrorChecking object with the provided details.

        This is a utility method that creates ErrorChecking objects consistently
        across the DataValidator class and its validation strategies.

        Args:
            message: Detailed error message describing the validation failure
            error_source: Source of the validation check
            file_name: Name or path of the file where the error occurred
            column_name: Name of the column that failed validation
            relevant_column_value: The actual value that caused the validation error
            survey_id: SurveyID associated with the error (if available)
            drop_id: DropID associated with the error (if available)

        Returns:
            ErrorChecking object with the provided details
        """
        file_name = normalize_file_name(file_name)

        return ErrorChecking(
            survey_id=survey_id,
            drop_id=drop_id,
            column_name=column_name,
            relevant_column_value=relevant_column_value,
            relevant_file=file_name,
            error_info=message,
            error_source=error_source,
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
        Export validation errors to a CSV file or S3 bucket.

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

        if EXPORT_LOCAL:
            path = Path(self.FOLDER_PATH) / csv_file_name
            path.parent.mkdir(parents=True, exist_ok=True)

            self.errors_df.to_csv(path, index=False)
            logging.info(f"Errors exported locally to csv file {path}.")
        else:
            self.s3_handler.upload_updated_df_to_s3(
                df=self.errors_df,
                key=self.FOLDER_PATH,
                filename=csv_file_name,
                keep_df_index=False,
            )
            logging.info(
                f"Errors exported to S3 csv file {self.FOLDER_PATH}/{csv_file_name}."
            )

    def export_clean_dataframes_to_csv(self) -> None:
        """
        Export all clean dataframes to CSV files.

        Exports each clean dataframe (rows with no validation errors) to separate
        CSV files in the specified directory. Files are named with the pattern
        "clean_{dataset_name}.csv".

        Args:
            output_directory (str): Directory path where CSV files will be saved.
                The directory must exist.

        Side Effects:
            - Creates CSV files for each clean dataframe in the output directory
            - Logs the export operation for each dataset

        Raises:
            ValueError: If no clean dataframes are available (clean row tracking not enabled)
            OSError: If the output directory doesn't exist or isn't writable

        Note:
            - Only available when extract_clean_dataframes was enabled in ValidationConfig
            - CSV files are exported without DataFrame index
            - Empty dataframes are skipped
        """
        if not self.clean_row_tracker:
            raise ValueError(
                "Clean dataframes not available. Enable extract_clean_dataframes "
                "in ValidationConfig before running validation."
            )

        clean_dataframes = self.get_all_clean_dataframes()
        if not clean_dataframes:
            logging.info("No clean dataframes to export.")
            return

        for dataset_name, clean_df in clean_dataframes.items():
            if clean_df.empty:
                logging.info(f"Skipping empty clean dataframe for {dataset_name}")
                continue

            current_filename = f"clean_{dataset_name}.csv"

            if EXPORT_LOCAL:
                output_path = os.path.join(self.FOLDER_PATH, current_filename)
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                clean_df.to_csv(output_path, index=False)
                logging.info(
                    f"Clean {dataset_name} dataframe exported to {output_path}"
                )
            else:
                self.s3_handler.upload_updated_df_to_s3(
                    df=clean_df,
                    key=self.FOLDER_PATH,
                    filename=current_filename,
                    keep_df_index=False,
                )

        logging.info(
            f"Exported {len(clean_dataframes)} clean dataframes to {self.FOLDER_PATH}"
        )

    def export_file_differences(
        self,
        missing_file_name: str = "missing_files_in_aws.txt",
        extra_file_name: str = "extra_files_in_aws.txt",
        file_presence_rules: Dict[str, Any] = FILE_PRESENCE_RULES,
    ) -> None:
        """
        Export file differences to separate text files.

        Uses the existing S3Handler and FilePresenceValidator to identify files that are:
        1. Listed in CSV but missing from S3 (missing_files_path)
        2. Present in S3 but not listed in CSV (extra_files_path)

        Args:
            missing_files_path: Path for the missing files output file
            extra_files_path: Path for the extra files output file
            file_presence_rules: Optional dictionary containing file presence validation rules.
                If None, uses the default FILE_PRESENCE_RULES from sftk.common.

        Side Effects:
            - Creates two text files with file paths, one per line
            - Logs the operation and number of files found
        """
        try:
            # Get the FilePresenceValidator from the strategy registry
            file_presence_validator = self.strategy_registry.strategies.get(
                "file_presence"
            )
            if not file_presence_validator:
                logging.warning(
                    "FilePresenceValidator not available in strategy registry"
                )
                return

            # Get file differences using the existing validator
            missing_files_set, extra_files_set = (
                file_presence_validator.get_file_differences(file_presence_rules)
            )
            missing_files_data = "\n".join(sorted(missing_files_set))
            extra_files_data = "\n".join(sorted(extra_files_set))

            # Export file differences to separate text files
            missing_files_path = os.path.join(self.FOLDER_PATH, missing_file_name)
            extra_files_path = os.path.join(self.FOLDER_PATH, extra_file_name)

            if EXPORT_LOCAL:
                write_data_to_file(missing_files_data, missing_files_path)
                write_data_to_file(extra_files_data, extra_files_path)

            else:
                self.s3_handler.upload_data_to_s3(
                    missing_files_data, missing_files_path
                )
                self.s3_handler.upload_data_to_s3(extra_files_data, extra_files_path)

            logging.info(
                f"File differences exported: {len(missing_files_set)} missing, {len(extra_files_set)} extra"
            )

        except Exception as e:
            logging.error(f"Failed to export file differences: {e}")
            raise

    def get_clean_dataframe(self, dataset_name: str) -> pd.DataFrame:
        """
        Get clean dataframe for a specific dataset.

        Args:
            dataset_name: Name of the dataset to get clean rows for

        Returns:
            DataFrame containing only rows with no validation errors,
            empty DataFrame if no tracker or dataset not found
        """
        if not self.clean_row_tracker:
            return pd.DataFrame()

        clean_indices = self.clean_row_tracker.get_clean_indices(dataset_name)
        if not clean_indices:
            return pd.DataFrame()

        # Get the original dataset from validation rules
        dataset_rules = self.validation_rules.get(dataset_name)
        if not dataset_rules or "dataset" not in dataset_rules:
            return pd.DataFrame()

        original_df = dataset_rules["dataset"]
        if original_df.empty:
            return pd.DataFrame()

        # Filter to clean rows only
        return original_df.loc[list(clean_indices)].copy()

    def get_all_clean_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Get all clean dataframes.

        Returns:
            Dictionary mapping dataset names to clean DataFrames,
            empty dict if no tracker is initialized
        """
        if not self.clean_row_tracker:
            return {}

        clean_dataframes = {}
        for dataset_name in self.clean_row_tracker.clean_row_indices.keys():
            clean_df = self.get_clean_dataframe(dataset_name)
            if not clean_df.empty:
                clean_dataframes[dataset_name] = clean_df

        return clean_dataframes

    def get_clean_summary(self) -> Dict[str, Any]:
        """
        Get summary of clean vs error rows.

        Returns:
            Dictionary with summary statistics for all datasets
        """
        if not self.clean_row_tracker:
            return {
                "message": "Clean dataframe extraction was not enabled during validation",
                "datasets": {},
            }

        summary: Dict[str, Any] = {"datasets": {}}

        for dataset_name in self.clean_row_tracker.clean_row_indices.keys():
            dataset_rules = self.validation_rules.get(dataset_name)
            if dataset_rules and "dataset" in dataset_rules:
                original_df = dataset_rules["dataset"]
                clean_indices = self.clean_row_tracker.get_clean_indices(dataset_name)

                total_rows = len(original_df)
                clean_rows = len(clean_indices)
                error_rows = total_rows - clean_rows

                summary["datasets"][dataset_name] = {
                    "total_rows": total_rows,
                    "clean_rows": clean_rows,
                    "error_rows": error_rows,
                    "clean_percentage": (
                        (clean_rows / total_rows * 100) if total_rows > 0 else 0
                    ),
                    "error_percentage": (
                        (error_rows / total_rows * 100) if total_rows > 0 else 0
                    ),
                }

        return summary
