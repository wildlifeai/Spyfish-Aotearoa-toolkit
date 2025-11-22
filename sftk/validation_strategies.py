"""
Validation strategy classes for data validation.

This module contains the abstract base class and concrete implementations
for different types of data validation strategies.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from sftk.common import DROPID_COLUMN, REPLICATE_COLUMN
from sftk.utils import normalize_file_name


class CleanRowTracker:
    """Tracks which rows remain clean (error-free) during validation.

    Notes:
        This assumes the datasets are not changed after initialisation.
        If validator code changes to change the datasets, review
        implementation of this class.
    """

    def __init__(self):
        """Initialize with empty clean_row_indices dictionary."""
        self.clean_row_indices: Dict[str, Set[int]] = {}

    def initialize_dataset(self, dataset_name: str, df: pd.DataFrame) -> None:
        """
        Initialize clean indices with all row indices for a dataset.

        Args:
            dataset_name: Name of the dataset
            df: DataFrame to initialize indices for
        """
        self.clean_row_indices[dataset_name] = set(df.index)

    def mark_row_as_error(self, row_index: int, dataset_name: str) -> None:
        """
        Remove a row index from the clean set.

        Args:
            row_index: Index of the row that has an error
            dataset_name: Name of the dataset
        """
        if dataset_name in self.clean_row_indices:
            self.clean_row_indices[dataset_name].discard(row_index)

    def get_clean_indices(self, dataset_name: str) -> Set[int]:
        """
        Get clean row indices for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Set of clean row indices, empty set if dataset not found
        """
        return self.clean_row_indices.get(dataset_name, set())


@dataclass
class ErrorChecking:
    """Data class for validation errors."""

    column_name: Optional[str]
    relevant_column_value: Optional[str]
    relevant_file: str
    error_info: str
    error_source: str


class ErrorSource(Enum):
    """Enumeration of error sources."""

    SHAREPOINT_VALIDATION = "sharepoint_data_error"
    FILE_PRESENCE_CHECK = "file_presence_check"


@dataclass
class ValidationConfig:
    """
    Configuration for validation operations.

    This class controls which types of validation are performed during data validation.
    It uses introspection to automatically detect validation fields, making it easy
    to add new validation types without updating multiple methods.

    Attributes:
        remove_duplicates (bool): Whether to remove duplicate errors from results
        required (bool): Enable validation of required (non-null) columns
        unique (bool): Enable validation of unique constraints
        foreign_keys (bool): Enable validation of foreign key relationships
        formats (bool): Enable validation of data format patterns
        column_relationships (bool): Enable validation of column relationships
        file_presence (bool): Enable validation of file presence in S3
        extract_clean_dataframes (bool): Enable extraction of clean (error-free) dataframes
        max_errors_per_validator (int): Maximum errors per validator to prevent memory issues
    """

    required: bool = False
    unique: bool = False
    foreign_keys: bool = False
    formats: bool = False
    column_relationships: bool = False
    file_presence: bool = False
    remove_duplicates: bool = True
    extract_clean_dataframes: bool = False
    max_errors_per_validator: int = (
        10000  # Limit errors per validator to prevent memory issues
    )

    def enable_all_validators(self) -> None:
        """Enable all validation types."""
        validation_fields = self._get_validation_fields()
        for field_name in validation_fields:
            setattr(self, field_name, True)

    def any_enabled(self) -> bool:
        """Check if any validation type is enabled."""
        validation_fields = self._get_validation_fields()
        return any(getattr(self, field_name) for field_name in validation_fields)

    def _get_validation_fields(self) -> list:
        """
        Get list of validation field names (excluding non-validation fields).

        Uses introspection to automatically detect boolean fields that represent
        validation types. This makes the class self-maintaining when new validation
        types are added.

        Returns:
            list: List of field names that represent validation types

        Note:
            Excludes 'remove_duplicates' and 'extract_clean_dataframes' as they are
            configuration options, not validation types.
        """
        # Get all boolean fields except configuration options
        excluded_fields = {"remove_duplicates", "extract_clean_dataframes"}
        return [
            field_name
            for field_name, field_type in self.__dataclass_fields__.items()
            if field_type.type == bool and field_name not in excluded_fields
        ]


class ValidationStrategy(ABC):
    """Abstract base class for validation strategies."""

    def __init__(
        self,
        validation_rules: Dict[str, Any],
        max_errors: int = 1000,
        clean_row_tracker=None,
    ):
        """
        Initialize the validation strategy.

        Args:
            validation_rules: Dictionary containing all validation rules and reference datasets
            max_errors: Maximum number of errors to collect per validation run
            clean_row_tracker: Optional CleanRowTracker instance for tracking clean rows
        """
        self.validation_rules = validation_rules
        self.max_errors = max_errors
        self.clean_row_tracker = clean_row_tracker

    @abstractmethod
    def validate(
        self,
        rules: Dict[str, Any],
        df: pd.DataFrame,
        dataset_name: Optional[str] = None,
    ) -> List[ErrorChecking]:
        """
        Perform validation on the given DataFrame.

        Args:
            rules: Validation rules for the current dataset
            df: DataFrame to validate
            dataset_name: Name of the dataset being validated (required for most validators,
                         optional only for FilePresenceValidator)

        Returns:
            List of ErrorChecking objects for any validation failures
        """
        pass

    def _should_continue_collecting_errors(self, current_error_count: int) -> bool:
        """Check if we should continue collecting errors based on the limit."""
        return current_error_count < self.max_errors

    def _extract_file_name(self, rules: Dict[str, Any]) -> str:
        """Extract file name from rules dictionary."""
        return normalize_file_name(rules.get("file_name", ""))

    def create_error(
        self,
        message: str,
        error_source: str,
        file_name: Optional[str] = None,
        column_name: Optional[str] = None,
        relevant_column_value: Any = None,
    ) -> ErrorChecking:
        """
        Create an ErrorChecking object with the provided details.

        This method creates ErrorChecking objects consistently across all validation strategies.
        It includes the same logic as DataValidator.create_error for compatibility.

        Args:
            message: Detailed error message describing the validation failure
            error_source: Source of the validation check
            file_name: Name or path of the file where the error occurred
            column_name: Name of the column that failed validation
            relevant_column_value: The actual value that caused the validation error

        Returns:
            ErrorChecking object with the provided details
        """
        file_name = normalize_file_name(file_name)

        return ErrorChecking(
            column_name=column_name,
            relevant_column_value=relevant_column_value,
            relevant_file=file_name,
            error_info=message,
            error_source=error_source,
        )

    def _create_missing_column_error(
        self, col_name: str, file_name: str, check_type: str
    ) -> ErrorChecking:
        """Create error for missing column during validation."""
        return self.create_error(
            message=f"Missing column for {check_type} check: '{col_name}'",
            error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
            column_name=col_name,
            relevant_column_value=None,
            file_name=file_name,
        )

    def _convert_row_tuple_to_series(self, row_tuple) -> pd.Series:
        """Convert named tuple back to Series for compatibility."""
        row = pd.Series(row_tuple._asdict(), name=row_tuple.Index)
        row.drop("Index", inplace=True)  # Remove the index field
        return row

    def _extract_relevant_column_info(
        self, row: pd.Series, col_name: str, info_columns: List[str], check_type: str
    ) -> str:
        """
        Extract relevant column information for error reporting.

        Args:
            row: The row being validated
            col_name: Name of the column being validated
            info_columns: List of columns to use for context when main column is missing/null
            check_type: Type of validation check (for error messages)

        Returns:
            String representation of the relevant column information
        """
        try:
            relevant_column_info = row[col_name]
        except KeyError as e:
            return f"No {col_name} in row for check {check_type}, error: {e}"

        # If the main column value is null/missing, use info columns for context
        if pd.isna(relevant_column_info):
            if info_columns:
                return " ".join(
                    [
                        f"{info_column}: {row.get(info_column, '')}"
                        for info_column in info_columns
                    ]
                )
            else:
                return "N/A"

        return relevant_column_info

    def _create_error_for_row(
        self,
        row: pd.Series,
        file_name: str,
        col_name: str,
        info_columns: List[str],
        check_type: str,
        message_template: str,
        skip_empty_rows: bool = True,
        skip_single_column_empty: bool = False,
        error_source: str = ErrorSource.SHAREPOINT_VALIDATION.value,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> Optional[ErrorChecking]:
        """
        Create error for a specific row that failed validation.

        This is a template method that handles common error creation logic
        while allowing each validator to customize the message. It combines
        the row-specific logic with error object creation.

        Args:
            row: The row that failed validation
            file_name: Name of the file being validated
            col_name: Name of the column that failed validation
            info_columns: List of columns to use for context information
            check_type: Type of validation check (for error messages)
            message_template: Template string for the error message
            skip_empty_rows: Whether to skip completely empty rows
            skip_single_column_empty: Whether to skip rows where only target column is empty (RequiredValidator specific)
            error_source: Source of the validation check
            **kwargs: Additional arguments for message formatting

        Returns:
            ErrorChecking object or None if row should be skipped
        """
        # Handle empty row logic
        if skip_empty_rows:
            if skip_single_column_empty:
                # TODO: check if this is necessary
                # RequiredValidator logic: skip completely empty rows but not rows with just target column missing
                if row.isna().all() and len(row) > 1:
                    return None
            else:
                # Other validators: skip completely empty rows
                if row.isna().all():
                    return None

        # Extract relevant column information
        relevant_column_info = self._extract_relevant_column_info(
            row, col_name, info_columns, check_type
        )

        # Format the message using the template and any additional kwargs
        message = message_template.format(
            col_name=col_name, relevant_column_info=relevant_column_info, **kwargs
        )

        error = self.create_error(
            message=message,
            error_source=error_source,
            column_name=col_name,
            relevant_column_value=relevant_column_info,
            file_name=file_name,
        )

        # Mark this row as having an error in the clean row tracker
        if error and self.clean_row_tracker and dataset_name:
            self.clean_row_tracker.mark_row_as_error(row.name, dataset_name)

        return error


class RequiredValidator(ValidationStrategy):
    """Validator for required column checks."""

    def validate(
        self,
        rules: Dict[str, Any],
        df: pd.DataFrame,
        dataset_name: Optional[str] = None,
    ) -> List[ErrorChecking]:
        """Validate that required columns contain non-null values."""
        errors = []
        file_name = self._extract_file_name(rules)
        required_cols = rules.get("required", [])
        info_columns = rules.get("info_columns", [])

        for col in required_cols:
            if col not in df.columns:
                errors.append(
                    self._create_missing_column_error(col, file_name, "required")
                )
                continue

            na_rows = df[df[col].isna()]
            for row_tuple in na_rows.itertuples():
                if not self._should_continue_collecting_errors(len(errors)):
                    break

                row = self._convert_row_tuple_to_series(row_tuple)

                error = self._create_error_for_row(
                    row=row,
                    file_name=file_name,
                    col_name=col,
                    info_columns=info_columns,
                    check_type="required",
                    message_template="Missing value in required column '{col_name}', help_info: {relevant_column_info}.",
                    skip_single_column_empty=True,
                    dataset_name=dataset_name,
                )
                if error:
                    errors.append(error)

        return errors


class UniqueValidator(ValidationStrategy):
    """Validator for unique column checks."""

    def validate(
        self,
        rules: Dict[str, Any],
        df: pd.DataFrame,
        dataset_name: Optional[str] = None,
    ) -> List[ErrorChecking]:
        """Validate that columns marked as unique contain no duplicate values."""
        errors = []
        file_name = self._extract_file_name(rules)
        unique_cols = rules.get("unique", [])
        info_columns = rules.get("info_columns", [])

        for col in unique_cols:
            if col not in df.columns:
                errors.append(
                    self._create_missing_column_error(col, file_name, "unique")
                )
                continue

            # Check for duplicates in unique columns
            duplicated = df[df[col].duplicated(keep=False) & df[col].notna()]
            for row_tuple in duplicated.itertuples():
                if not self._should_continue_collecting_errors(len(errors)):
                    break

                row = self._convert_row_tuple_to_series(row_tuple)

                error = self._create_error_for_row(
                    row=row,
                    file_name=file_name,
                    col_name=col,
                    info_columns=info_columns,
                    check_type="duplicate",
                    message_template="Duplicate value in unique column '{col_name}'",
                    dataset_name=dataset_name,
                )
                if error:
                    errors.append(error)

        return errors


class FormatValidator(ValidationStrategy):
    """Validator for format pattern checks."""

    def __init__(
        self,
        validation_rules: Dict[str, Any],
        patterns: Dict[str, str],
        max_errors: int = 1000,
        clean_row_tracker=None,
    ):
        """
        Initialize the format validator.

        Args:
            validation_rules: Dictionary containing all validation rules and datasets
            patterns: Dictionary mapping column names to regex patterns
            max_errors: Maximum number of errors to collect
            clean_row_tracker: Optional CleanRowTracker instance for tracking clean rows
        """
        super().__init__(validation_rules, max_errors, clean_row_tracker)
        self.patterns = patterns

    def validate(
        self,
        rules: Dict[str, Any],
        df: pd.DataFrame,
        dataset_name: Optional[str] = None,
    ) -> List[ErrorChecking]:
        """Validate data formats against predefined regex patterns."""
        errors = []
        file_name = self._extract_file_name(rules)
        info_columns = rules.get("info_columns", [])

        for col, pattern in self.patterns.items():
            if col not in df.columns:
                continue

            invalid = df[~df[col].isna() & ~df[col].astype(str).str.match(pattern)]
            for row_tuple in invalid.itertuples():
                row = self._convert_row_tuple_to_series(row_tuple)

                error = self._create_error_for_row(
                    row=row,
                    file_name=file_name,
                    col_name=col,
                    info_columns=info_columns,
                    check_type="invalid_format",
                    message_template="Value {actual_value} does not match required format for {col_name}: expected pattern '{pattern}'",
                    pattern=pattern,
                    actual_value=row[col],
                    dataset_name=dataset_name,
                )
                if error:
                    errors.append(error)

        return errors


class ForeignKeyValidator(ValidationStrategy):
    """Validator for foreign key relationship checks."""

    def validate(
        self,
        rules: Dict[str, Any],
        df: pd.DataFrame,
        dataset_name: Optional[str] = None,
    ) -> List[ErrorChecking]:
        """Validate foreign key relationships between datasets."""
        errors = []
        source_file = self._extract_file_name(rules)
        foreign_keys = rules.get("foreign_keys", {})
        info_columns = rules.get("info_columns", [])

        for target_name, fk_col in foreign_keys.items():
            target_rules = self.validation_rules.get(target_name)
            if not target_rules:
                errors.append(
                    self.create_error(
                        message=f"Foreign key check skipped: target dataset '{target_name}' not found",
                        error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                        column_name=fk_col,
                        file_name=source_file,
                    )
                )
                continue

            target_df = target_rules.get("dataset", pd.DataFrame())
            if target_df.empty:
                errors.append(
                    self.create_error(
                        message=f"Foreign key check skipped: target dataset '{target_name}' is empty or could not be loaded",
                        error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                        column_name=fk_col,
                        file_name=source_file,
                    )
                )
                continue

            if fk_col not in df.columns:
                errors.append(
                    self.create_error(
                        message=f"Foreign key column '{fk_col}' not found in '{source_file}'",
                        error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                        column_name=fk_col,
                        file_name=source_file,
                    )
                )
                continue

            if fk_col not in target_df.columns:
                errors.append(
                    self.create_error(
                        message=f"Foreign key column '{fk_col}' not found in target '{target_name}'",
                        error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                        column_name=fk_col,
                        file_name=source_file,
                    )
                )
                continue

            missing = df[~df[fk_col].isin(target_df[fk_col])]
            fk_file_name = normalize_file_name(target_rules["file_name"])

            for row_tuple in missing.itertuples():
                row = self._convert_row_tuple_to_series(row_tuple)

                error = self._create_error_for_row(
                    row=row,
                    file_name=source_file,
                    col_name=fk_col,
                    info_columns=info_columns,
                    check_type="missing_fk",
                    message_template="Foreign key '{col_name}' = '{actual_value}' not found in '{fk_file_name}'",
                    fk_file_name=fk_file_name,
                    actual_value=row[fk_col],
                    dataset_name=dataset_name,
                )
                if error:
                    errors.append(error)

        return errors


class RelationshipValidator(ValidationStrategy):
    """Validator for column relationship checks."""

    def validate(
        self,
        rules: Dict[str, Any],
        df: pd.DataFrame,
        dataset_name: Optional[str] = None,
    ) -> List[ErrorChecking]:
        """Validate column relationships across all rows in a dataset."""
        errors = []
        file_name = self._extract_file_name(rules)
        relationships = rules.get("relationships", [])

        for rel in relationships:
            col = rel["column"]

            if col not in df.columns:
                errors.append(
                    self._create_missing_column_error(col, file_name, "relationship")
                )
                continue

            for row_tuple in df.itertuples():
                row = self._convert_row_tuple_to_series(row_tuple)

                error = self._check_row_relationship(
                    row=row,
                    file_name=dataset_name,
                    col_name=col,
                    relationships=rel,
                )
                if error:
                    errors.append(error)

        return errors

    def _check_row_relationship(
        self,
        row: pd.Series,
        file_name: str,
        col_name: str,
        relationships: Dict[str, Any],
    ) -> Optional[ErrorChecking]:
        """Validate relationships between column values within a single row."""
        rule = relationships["rule"]
        template = relationships["template"]
        allowed_values = relationships.get("allowed_values", [])
        actual = row[col_name]
        is_null_allowed = relationships.get("allow_null")

        # Skip allowed values
        if (actual in allowed_values) or (is_null_allowed and pd.isna(actual)):
            return None

        try:
            expected = template.format(**row)
        except KeyError as e:
            return self.create_error(
                message=f"Missing column {col_name} for relationship template: {str(e)}",
                error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                column_name=col_name,
                relevant_column_value=None,
                file_name=file_name,
            )

        if rule == "equals" and str(actual) != str(expected):
            # Check for specific DropID replicate mismatch case
            if col_name == DROPID_COLUMN and self._is_replicate_mismatch_only(
                str(actual), str(expected)
            ):
                message = f"{REPLICATE_COLUMN} mismatch: {col_name} should end with '{str(expected)[-2:]}' but ends with '{str(actual)[-2:]}'. Full {col_name} should be '{expected}', but is '{actual}'"
            else:
                message = f"{col_name} should be '{expected}', but is '{actual}'"

            return self.create_error(
                message=message,
                error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                column_name=col_name,
                relevant_column_value=actual,
                file_name=file_name,
            )

        return None

    def _is_replicate_mismatch_only(self, actual: str, expected: str) -> bool:
        """
        Check if the mismatch is only in the replicate part (last 2 digits).

        Args:
            actual: The actual value
            expected: The expected value

        Returns:
            True if only the last 2 digits differ, False otherwise
        """
        # Both strings should have the same length and be long enough
        if len(actual) != len(expected) or len(actual) < 2:
            return False

        # Check if everything except the last 2 characters is the same
        return actual[:-2] == expected[:-2] and actual[-2:] != expected[-2:]


class FilePresenceValidator(ValidationStrategy):
    """Validator for file presence checks between CSV references and S3 storage."""

    def __init__(
        self,
        validation_rules: Dict[str, Any],
        s3_handler,
        max_errors: int = 1000,
        clean_row_tracker=None,
    ):
        """
        Initialize the file presence validator.

        Args:
            validation_rules: Dictionary containing all validation rules and reference datasets
            s3_handler: S3Handler instance for S3 operations
            max_errors: Maximum number of errors to collect
            clean_row_tracker: Optional CleanRowTracker instance for tracking clean rows
        """
        super().__init__(validation_rules, max_errors, clean_row_tracker)
        self.s3_handler = s3_handler

    def validate(
        self,
        rules: Dict[str, Any],
        df: pd.DataFrame,
        dataset_name: Optional[str] = None,
    ) -> List[ErrorChecking]:  # pylint: disable=unused-argument
        """
        Validate file presence between CSV references and S3 storage.

        Note: dataset_name is not used by this validator as it works directly with S3 files.

        This method compares video file paths referenced in CSV data with actual
        files available in S3 storage, identifying missing and extra files.

        Args:
            rules: Validation rules containing file presence configuration
            df: DataFrame to validate (not used directly, but kept for interface consistency)

        Returns:
            List of ErrorChecking objects for missing and extra files
        """
        errors: list[ErrorChecking] = []

        # Get file presence configuration from rules
        file_presence_config = rules.get("file_presence", {})
        if not file_presence_config:
            return errors

        csv_filename = file_presence_config.get("csv_filename")

        try:
            missing_files, extra_files = self.get_file_differences(rules)

            # Find missing files (in CSV but not in S3)
            for file_path in missing_files:
                if not self._should_continue_collecting_errors(len(errors)):
                    break
                errors.append(
                    self.create_error(
                        message=f"File {file_path} not found in AWS, but found in {csv_filename}",
                        relevant_column_value=file_path,
                        error_source=ErrorSource.FILE_PRESENCE_CHECK.value,
                    )
                )

            # Find extra files (in S3 but not in CSV)
            for file_path in extra_files:
                if not self._should_continue_collecting_errors(len(errors)):
                    break
                errors.append(
                    self.create_error(
                        message=f"File {file_path} found in AWS but not in {csv_filename}",
                        relevant_column_value=file_path,
                        error_source=ErrorSource.FILE_PRESENCE_CHECK.value,
                    )
                )

        except Exception as e:
            errors.append(
                self.create_error(
                    message=f"File presence validation failed: {str(e)}",
                    file_name=csv_filename,
                    error_source=ErrorSource.FILE_PRESENCE_CHECK.value,
                )
            )

        return errors

    def get_file_differences(self, rules: Dict[str, Any]) -> tuple[set[str], set[str]]:
        """
        Get file differences between CSV references and S3 storage.

        This method compares video file paths referenced in CSV data with actual
        files available in S3 storage, returning missing and extra files as separate sets.

        Args:
            rules: Validation rules containing file presence configuration

        Returns:
            Tuple of (missing_files, extra_files) where:
            - missing_files: Files listed in CSV but missing from S3
            - extra_files: Files in S3 not listed in the CSV

        Raises:
            ValueError: If required configuration is missing
            Exception: If file comparison fails due to S3 or CSV access issues
        """
        # Get file presence configuration from rules
        file_presence_config = rules.get("file_presence", {})
        if not file_presence_config:
            return set(), set()

        csv_filename = file_presence_config.get("csv_filename")
        csv_column_to_extract = file_presence_config.get("csv_column_to_extract")
        valid_extensions = file_presence_config.get("valid_extensions", [])
        path_prefix = file_presence_config.get("path_prefix", "")
        column_filter = file_presence_config.get("column_filter")
        column_value = file_presence_config.get("column_value")
        s3_sharepoint_path = file_presence_config.get("s3_sharepoint_path", "")
        bucket = file_presence_config.get("bucket")

        if not all([csv_filename, csv_column_to_extract, bucket]):
            raise ValueError(
                "File presence validation requires csv_filename, csv_column_to_extract, and bucket"
            )

        # Get file paths from CSV
        csv_s3_path = os.path.join(s3_sharepoint_path, csv_filename)
        csv_paths_result = self.s3_handler.get_paths_from_csv(
            csv_s3_path=csv_s3_path,
            csv_column=csv_column_to_extract,
            column_filter=column_filter,
            column_value=column_value,
        )
        csv_filepaths_all = csv_paths_result["all"]
        csv_filepaths_filtered = csv_paths_result["filtered"]

        # Get file paths from S3
        s3_video_filepaths = self.s3_handler.get_paths_from_s3(
            path_prefix=path_prefix, valid_extensions=valid_extensions
        )

        # Find missing files (in CSV but not in S3)
        missing_files = csv_filepaths_filtered - s3_video_filepaths

        # Find extra files (in S3 but not in CSV)
        extra_files = s3_video_filepaths - csv_filepaths_all

        return missing_files, extra_files


class ValidationStrategyRegistry:
    """Registry for managing validation strategies."""

    def __init__(
        self,
        validation_rules: Dict[str, Any],
        patterns: Dict[str, str],
        s3_handler=None,
        max_errors: int = 1000,
        clean_row_tracker=None,
    ):
        """
        Initialize the registry with validation strategies.

        Args:
            validation_rules: Dictionary containing all validation rules and reference datasets
            patterns: Dictionary mapping column names to regex patterns
            s3_handler: S3Handler instance for file presence validation
            max_errors: Maximum number of errors per validator
            clean_row_tracker: Optional CleanRowTracker instance for tracking clean rows
        """
        self.strategies = {
            "required": RequiredValidator(
                validation_rules, max_errors, clean_row_tracker
            ),
            "unique": UniqueValidator(validation_rules, max_errors, clean_row_tracker),
            "formats": FormatValidator(
                validation_rules, patterns, max_errors, clean_row_tracker
            ),
            "foreign_keys": ForeignKeyValidator(
                validation_rules, max_errors, clean_row_tracker
            ),
            "column_relationships": RelationshipValidator(
                validation_rules, max_errors, clean_row_tracker
            ),
        }

        # Add file presence validator if s3_handler is provided
        if s3_handler:
            self.strategies["file_presence"] = FilePresenceValidator(
                validation_rules, s3_handler, max_errors, clean_row_tracker
            )

    def get_strategy(self, strategy_name: str) -> Optional[ValidationStrategy]:
        """Get a validation strategy by name."""
        return self.strategies.get(strategy_name)

    def get_enabled_strategies(
        self, config: ValidationConfig
    ) -> List[ValidationStrategy]:
        """Get list of enabled validation strategies based on configuration."""
        # Map config attributes to strategy names
        strategy_mapping = {
            "required": config.required,
            "unique": config.unique,
            "formats": config.formats,
            "foreign_keys": config.foreign_keys,
            "column_relationships": config.column_relationships,
            "file_presence": config.file_presence,
        }

        enabled = []
        for strategy_name, is_enabled in strategy_mapping.items():
            if is_enabled and strategy_name in self.strategies:
                enabled.append(self.strategies[strategy_name])

        return enabled
