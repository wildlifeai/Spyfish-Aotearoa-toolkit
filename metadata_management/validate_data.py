import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from sftk.common import (
    EXPORT_LOCAL,
    LOCAL_DATA_FOLDER_PATH,
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


@dataclass
class ErrorChecking:
    column_name: Optional[str]
    relevant_column_value: Optional[str]
    relevant_file: str
    error_info: str
    error_source: str


class DataValidator:
    def __init__(self):
        """
        Initialize the DataValidator with default settings and validation rules.

        Sets up the validator with:
        - Empty error list for collecting validation errors
        - Validation patterns from configuration
        - S3 handler for accessing cloud storage
        - Validation rules loaded from S3 datasets

        Attributes:
            errors (list): List to store ErrorChecking objects during validation
            errors_df (pd.DataFrame): DataFrame containing all validation errors
            patterns (dict): Regex patterns for format validation from VALIDATION_PATTERNS
            s3_handler (S3Handler): Handler for S3 operations
            bucket (str): S3 bucket name from configuration
            validation_rules (dict): Loaded validation rules with associated datasets
        """
        self.errors = []
        self.errors_df = None
        self.patterns = VALIDATION_PATTERNS
        self.s3_handler = S3Handler()
        self.bucket = S3_BUCKET
        self.validation_rules = self._get_validation_rules()

    def _get_validation_rules(self):
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

    def validate(
        self,
        remove_duplicates=True,
        required=False,
        unique=False,
        foreign_keys=False,
        formats=False,
        column_relationships=False,
        file_presence=False,
    ):
        """
        Perform comprehensive data validation across all configured datasets.

        Runs various validation checks on datasets loaded from S3 according to the
        validation rules. If no specific validation types are enabled, all validation
        types will be performed by default.

        Args:
            remove_duplicates (bool, optional): Whether to remove duplicate errors
                from the final results. Defaults to True.
            required (bool, optional): Check for missing values in required columns.
                Defaults to False.
            unique (bool, optional): Check for duplicate values in columns that should
                be unique. Defaults to False.
            foreign_keys (bool, optional): Validate foreign key relationships between
                datasets. Defaults to False.
            formats (bool, optional): Validate data formats against regex patterns.
                Defaults to False.
            column_relationships (bool, optional): Check column value relationships
                within rows. Defaults to False.
            file_presence (bool, optional): Check for mismatched video files between
                CSV records and S3 storage. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing all validation errors found, with columns:
                - column_name: Name of the column with the error
                - relevant_column_value: Value that caused the error
                - relevant_file: File/dataset where the error was found
                - error_info: Detailed error message
                - error_source: Source of the validation check

        Note:
            If all validation parameters are False, all validation types will be
            enabled automatically.
        """

        if not any(
            [
                required,
                unique,
                foreign_keys,
                formats,
                column_relationships,
                file_presence,
            ]
        ):
            required = unique = foreign_keys = formats = column_relationships = (
                file_presence
            ) = True

        for dataset_name, rules in self.validation_rules.items():
            file_name = Path(rules.get("file_name", "")).name
            df = rules.get("dataset", pd.DataFrame())

            if df.empty:
                self._add_error(
                    file_name=file_name, message="Dataset could not be loaded."
                )
                continue

            # change numerical values into integers where possible
            df = convert_int_num_columns_to_int(df)

            if required:
                self._check_required(rules, df)
            if unique:
                self._check_unique(rules, df)
            if foreign_keys:
                self._check_foreign_keys(rules, df)
            if formats:
                self._check_formats(rules, df)
            if column_relationships:
                self._check_column_relationships(rules, df)

        if file_presence:
            logging.info("Starting mismatched_video_file_info processing.")
            self.get_mismatched_video_files_info(
                csv_filename=VIDEOS_IN_SHAREPOINT_INFO["csv_filename"],
                csv_column_to_extract=VIDEOS_IN_SHAREPOINT_INFO["info_column"],
                column_filter=VIDEOS_IN_SHAREPOINT_INFO["column_filter"],
                column_value=VIDEOS_IN_SHAREPOINT_INFO["column_value"],
                valid_extensions=VIDEOS_IN_AWS_INFO["valid_movie_extensions"],
                path_prefix=VIDEOS_IN_AWS_INFO["path_prefix"],
            )
        logging.info("mismatched_video_file_info processing completed.")

        self._export_errors_from_list_to_df(remove_duplicates)

        return self.errors_df

    def _add_relevant_error_info_for_row(
        self,
        row,
        file_name=None,
        col_name=None,
        check=None,
        fk_file_name=None,
        info_columns=None,
        pattern=None,
    ):
        """
        Add error information for a specific row that failed validation.

        This helper method processes a single row that failed validation and creates
        an appropriate error message based on the type of validation check that failed.
        It extracts relevant context information from the row to help identify the
        problematic data.

        Args:
            row (pd.Series): The DataFrame row that failed validation
            file_name (str, optional): Name of the file/dataset being validated
            col_name (str, optional): Name of the column that failed validation
            check (str, optional): Type of validation check ('required', 'duplicate',
                'missing_fk', 'invalid_format')
            fk_file_name (str, optional): Name of the foreign key target file
            info_columns (list, optional): List of column names to use for context
                when the main column value is missing
            pattern (str, optional): Regex pattern for format validation errors

        Side Effects:
            - Calls _add_error() to record the validation error
            - Skips processing if the entire row is NaN

        Note:
            If the primary column value is NaN, uses info_columns to provide
            context for error identification.
        """
        try:
            relevant_column_info = row[col_name]
        except KeyError as e:
            relevant_column_info = f"No {col_name} in row for check {check}, error: {e}"

        if row.isna().all():
            # TODO why do some lines come completely empty?
            return

        if pd.isna(relevant_column_info):
            relevant_column_info = " ".join(
                [
                    f"{info_column}: {row.get(info_column, '')}"
                    for info_column in info_columns
                ]
            )
        if check == "required":
            message = f"Missing value in required column '{col_name}', help_info: {relevant_column_info}."
        elif check == "duplicate":
            message = f"Duplicate value in unique column '{col_name}'"
        elif check == "missing_fk":
            message = f"Foreign key '{col_name}' = '{row[col_name]}' not found in '{fk_file_name}'"
        elif check == "invalid_format":
            message = f"Value {row[col_name]} does not match required format for {col_name}: expected pattern '{pattern}'"
        else:
            message = "No error type set."
        self._add_error(
            column_name=col_name,
            relevant_column_value=relevant_column_info,
            file_name=file_name,
            message=message,
        )

    def _check_required(self, rules, df):
        """
        Validate that required columns contain non-null values.

        Checks all columns marked as required in the validation rules to ensure
        they don't contain missing (NaN/null) values. Records an error for each
        row where a required column is missing a value.

        Args:
            rules (dict): Validation rules for the current dataset containing:
                - file_name: Name of the dataset file
                - required: List of column names that must not be null
                - info_columns: Columns to use for error context
            df (pd.DataFrame): The dataset to validate

        Side Effects:
            - Records errors for missing required columns
            - Records errors for each row with null values in required columns
        """
        file_name = Path(rules.get("file_name", "")).name
        required_cols = rules.get("required", [])
        info_columns = rules.get("info_columns", [])

        for col in required_cols:
            if col not in df.columns:
                self._add_error(
                    column_name=col,
                    file_name=file_name,
                    message=f"Missing column for required check: '{col}'",
                )
                continue
            na_rows = df[df[col].isna()]
            na_rows.apply(
                self._add_relevant_error_info_for_row,
                file_name=file_name,
                col_name=col,
                info_columns=info_columns,
                check="required",
                axis=1,
            )

    def _check_unique(self, rules, df):
        """
        Validate that columns marked as unique contain no duplicate values.

        Checks all columns marked as unique in the validation rules to ensure
        they don't contain duplicate values. Only non-null values are considered
        for duplication checking.

        Args:
            rules (dict): Validation rules for the current dataset containing:
                - file_name: Name of the dataset file
                - unique: List of column names that must contain unique values
                - info_columns: Columns to use for error context
            df (pd.DataFrame): The dataset to validate

        Side Effects:
            - Records errors for missing unique columns
            - Records errors for each row with duplicate values in unique columns

        Note:
            Uses pandas duplicated(keep=False) to identify all instances of
            duplicated values, not just the subsequent ones.
        """
        file_name = Path(rules.get("file_name", "")).name
        unique_cols = rules.get("unique", [])
        info_cols = rules.get("info_columns")

        for col in unique_cols:
            if col not in df.columns:
                self._add_error(
                    column_name=col,
                    relevant_column_value=None,
                    file_name=file_name,
                    message=f"Missing column for unique check: '{col}'",
                )
                continue

            # Check for Duplicates in unique columns
            duplicated = df[df[col].duplicated(keep=False) & df[col].notna()]
            duplicated.apply(
                self._add_relevant_error_info_for_row,
                info_columns=info_cols,
                file_name=file_name,
                col_name=col,
                check="duplicate",
                axis=1,
            )

    def _check_foreign_keys(self, rules, source_df):
        """
        Validate foreign key relationships between datasets.

        Checks that values in foreign key columns exist in the referenced target
        datasets. This ensures referential integrity across related datasets.

        Args:
            rules (dict): Validation rules for the source dataset containing:
                - file_name: Name of the source dataset file
                - foreign_keys: Dict mapping target dataset names to column names
                - info_columns: Columns to use for error context
            source_df (pd.DataFrame): The source dataset to validate

        Side Effects:
            - Records errors for missing target datasets
            - Records errors for empty target datasets
            - Records errors for missing foreign key columns
            - Records errors for each row with invalid foreign key values

        Note:
            Foreign key validation is skipped if the target dataset cannot be
            loaded or if required columns are missing from either dataset.
        """
        source_file = Path(rules.get("file_name", "")).name
        foreign_keys = rules.get("foreign_keys", {})
        info_columns = rules.get("info_columns", [])

        for target_name, fk_col in foreign_keys.items():

            target_rules = self.validation_rules.get(target_name)
            if not target_rules:
                self._add_error(
                    column_name=fk_col,
                    file_name=source_file,
                    message=f"Foreign key check skipped: target dataset '{target_name}' not found",
                )
                continue

            target_df = target_rules.get("dataset", pd.DataFrame())
            if target_df.empty:
                self._add_error(
                    column_name=fk_col,
                    file_name=source_file,
                    message=f"Foreign key check skipped: target dataset '{target_name}' is empty or could not be loaded",
                )
                continue

            if fk_col not in source_df.columns:
                self._add_error(
                    column_name=fk_col,
                    file_name=source_file,
                    message=f"Foreign key column '{fk_col}' not found in '{source_file}'",
                )
                continue

            if fk_col not in target_df.columns:
                self._add_error(
                    column_name=fk_col,
                    file_name=source_file,
                    message=f"Foreign key column '{fk_col}' not found in target '{target_name}'",
                )
                continue

            missing = source_df[~source_df[fk_col].isin(target_df[fk_col])]
            fk_file_name = Path(target_rules["file_name"]).name
            missing.apply(
                self._add_relevant_error_info_for_row,
                info_columns=info_columns,
                file_name=source_file,
                col_name=fk_col,
                fk_file_name=fk_file_name,
                check="missing_fk",
                axis=1,
            )

    def _check_formats(self, rules, df):
        """
        Validate data formats against predefined regex patterns.

        Checks that values in specific columns match their expected format patterns
        as defined in VALIDATION_PATTERNS. Only non-null values are validated.

        Args:
            rules (dict): Validation rules for the current dataset containing:
                - file_name: Name of the dataset file
                - info_columns: Columns to use for error context
            df (pd.DataFrame): The dataset to validate

        Side Effects:
            - Records errors for each row with values that don't match expected patterns

        Note:
            - Uses self.patterns (from VALIDATION_PATTERNS) for format definitions
            - Skips validation for columns not present in the dataset
            - Converts all values to strings before pattern matching
        """
        source_file = Path(rules.get("file_name", "")).name
        info_columns = rules.get("info_columns", [])

        for col, pattern in self.patterns.items():
            if col not in df.columns:
                continue
            invalid = df[~df[col].isna() & ~df[col].astype(str).str.match(pattern)]

            invalid.apply(
                self._add_relevant_error_info_for_row,
                file_name=source_file,
                col_name=col,
                info_columns=info_columns,
                pattern=pattern,
                check="invalid_format",
                axis=1,
            )

    def _check_row_relationship(self, row, file_name, col_name, relationships):
        """
        Validate relationships between column values within a single row.

        Checks that a column's value matches an expected pattern based on other
        column values in the same row. Uses template formatting to generate
        expected values from other columns.

        Args:
            row (pd.Series): The DataFrame row to validate
            file_name (str): Name of the dataset file
            col_name (str): Name of the column being validated
            relationships (dict): Relationship rules containing:
                - rule: Type of relationship check (e.g., "equals")
                - template: Format string using other column values
                - allowed_values: List of values that bypass validation
                - allow_null: Whether null values are permitted

        Side Effects:
            - Records errors for missing template columns
            - Records errors for values that don't match expected relationships

        Returns:
            None: Always returns None (used with DataFrame.apply)

        Note:
            - Skips validation for values in allowed_values list
            - Skips validation for null values if allow_null is True
            - Currently only supports "equals" rule type
        """
        rule = relationships["rule"]
        template = relationships["template"]
        allowed_values = relationships.get("allowed_values", [])
        actual = row[col_name]
        # skip allowed values
        is_null_allowed = relationships.get("allow_null")
        if (actual in allowed_values) or (is_null_allowed and pd.isna(actual)):
            return None

        try:
            expected = template.format(**row)
        except KeyError as e:
            self._add_error(
                column_name=col_name,
                relevant_column_value=None,
                file_name=file_name,
                message=f"Missing column {col_name} for relationship template: {str(e)}",
            )
            return None

        if rule == "equals" and str(actual) != str(expected):
            self._add_error(
                column_name=col_name,
                relevant_column_value=actual,
                file_name=file_name,
                message=f"{col_name} should be '{expected}', but is '{actual}'",
            )

    def _check_column_relationships(self, rules, df):
        """
        Validate column relationships across all rows in a dataset.

        Applies relationship validation rules to check that column values
        follow expected patterns based on other column values in the same row.
        This is useful for validating computed fields or enforcing business rules.

        Args:
            rules (dict): Validation rules for the current dataset containing:
                - file_name: Name of the dataset file
                - relationships: List of relationship rules, each containing:
                    - column: Name of the column to validate
                    - rule: Type of relationship (e.g., "equals")
                    - template: Format string for expected values
                    - allowed_values: Values that bypass validation
                    - allow_null: Whether null values are permitted
            df (pd.DataFrame): The dataset to validate

        Side Effects:
            - Records errors for missing relationship columns
            - Applies _check_row_relationship to each row for each relationship rule

        Note:
            Uses DataFrame.apply with axis=1 to process each row individually.
        """
        dataset_name = Path(rules.get("file_name", "")).name
        relationships = rules.get("relationships", [])

        for rel in relationships:
            col = rel["column"]

            if col not in df.columns:
                self._add_error(
                    column_name=col,
                    relevant_column_value=None,
                    file_name=dataset_name,
                    message=f"Missing column for relationship check: {col}",
                )
                continue
            # TODO check when is it better to use apply vs iterrows
            df.apply(
                self._check_row_relationship,
                file_name=dataset_name,
                col_name=col,
                relationships=rel,
                axis=1,
            )

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

    def get_mismatched_video_files_info(
        self,
        csv_filename: str,
        csv_column_to_extract: str,
        valid_extensions: Iterable[str],
        path_prefix: str = "",
        output_files: bool = EXPORT_LOCAL,
        column_filter: Optional[str] = None,
        column_value: Optional[Any] = None,
    ):
        """
        Compare video file paths from a CSV with video files available in S3.

        Identifies mismatches between video files referenced in a CSV dataset
        and actual video files stored in S3. Records validation errors for
        missing and extra files, and optionally creates local text files
        with the mismatch details.

        Args:
            csv_filename (str): Name of the CSV file in S3 (e.g., 'BUV Deployment.csv')
            csv_column_to_extract (str): Column in the CSV that contains video file paths
            valid_extensions (Iterable[str]): Valid video file extensions to check
                (e.g., ['mp4', 'mov'])
            path_prefix (str, optional): S3 path prefix for video files. Defaults to "".
            output_files (bool, optional): Whether to create local text files with
                mismatch details. Defaults to EXPORT_LOCAL.
            column_filter (str, optional): Column name to filter CSV rows by
            column_value (Any, optional): Value to filter for in column_filter

        Returns:
            tuple: (missing_files_in_aws, extra_files_in_aws) - sets of file paths

        Side Effects:
            - Records validation errors for missing and extra files
            - Creates 'missing_files_in_aws.txt' if output_files is True
            - Creates 'extra_files_in_aws.txt' if output_files is True
            - Logs progress and results

        Note:
            Uses S3_SHAREPOINT_PATH to construct the full CSV path in S3.
        """
        csv_s3_path = os.path.join(S3_SHAREPOINT_PATH, csv_filename)

        csv_filepaths_all, csv_filepaths_without_filtered_values = (
            self.s3_handler.get_paths_from_csv(
                csv_s3_path=csv_s3_path,
                csv_column=csv_column_to_extract,
                column_filter=column_filter,
                column_value=column_value,
                s3_bucket=self.bucket,
            )
        )

        s3_video_filepaths = self.s3_handler.get_paths_from_s3(
            path_prefix=path_prefix, valid_extensions=valid_extensions
        )

        # Find missing files in S3 (referenced in CSV but not found in S3)
        missing_files_in_aws = (
            csv_filepaths_without_filtered_values - s3_video_filepaths
        )
        logging.info(f"Missing video files in AWS: {len(missing_files_in_aws)}")
        # Find extra files in S3 (present in S3 but not referenced in CSV)
        extra_files_in_aws = s3_video_filepaths - csv_filepaths_all
        logging.info(f"Extra video files in AWS: {len(extra_files_in_aws)}.")
        for file in missing_files_in_aws:
            self._add_error(
                relevant_column_value=file,
                message=f"File {file} not found in AWS, but found in BUV Deployment.",
                error_source="file_presence_check",
            )
        for file in extra_files_in_aws:
            self._add_error(
                relevant_column_value=file,
                message=f"File {file} found in AWS but not BUV Deployment.",
                error_source="file_presence_check",
            )

        if output_files:
            logging.info(
                "Creating file missing_files_in_aws.txt, containing file names of "
                "videos referenced in CSV but not found in S3."
            )
            with open("missing_files_in_aws.txt", "w") as f:
                f.write("\n".join(sorted(missing_files_in_aws)))
            logging.info(
                "Creating file extra_files_in_aws.txt, containing file names of "
                "videos present in S3 but not referenced in the CSV."
            )
            with open("extra_files_in_aws.txt", "w") as f:
                f.write("\n".join(sorted(extra_files_in_aws)))
        return missing_files_in_aws, extra_files_in_aws


if __name__ == "__main__":
    # TODO figure out logging
    # Ensure logging is configured at the start
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,  # This will override any existing logging configuration
    )
    logging.info("Error validation started")
    # Run validation
    validator = DataValidator()
    validator.validate(
        remove_duplicates=True,
        required=True,
        unique=True,
        foreign_keys=True,
        formats=True,
        column_relationships=True,
        file_presence=True,
    )
    logging.info(
        f"Error validation completed, {validator.errors_df.shape[0]} errors found"
    )

    # Export to csv
    if EXPORT_LOCAL:
        validator.export_to_csv(
            os.path.join(LOCAL_DATA_FOLDER_PATH, "validation_errors.csv")
        )
    else:
        validator.upload_to_s3()
    logging.info("Error validation process completed, files created/uploaded.")
