"""
Validation strategy classes for data validation.

This module contains the abstract base class and concrete implementations
for different types of data validation strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


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
        max_errors_per_validator (int): Maximum errors per validator to prevent memory issues
    """

    remove_duplicates: bool = True
    required: bool = False
    unique: bool = False
    foreign_keys: bool = False
    formats: bool = False
    column_relationships: bool = False
    file_presence: bool = False
    max_errors_per_validator: int = (
        10000  # Limit errors per validator to prevent memory issues
    )

    def enable_all(self) -> None:
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
            Excludes 'remove_duplicates' as it's a configuration option, not a validation type.
        """
        # Get all boolean fields except remove_duplicates and max_errors_per_validator
        return [
            field_name
            for field_name, field_type in self.__dataclass_fields__.items()
            if field_type.type == bool and field_name != "remove_duplicates"
        ]


class ValidationStrategy(ABC):
    """Abstract base class for validation strategies."""

    def __init__(self, validation_rules: Dict[str, Any], max_errors: int = 1000):
        """
        Initialize the validation strategy.

        Args:
            validation_rules: Dictionary containing all validation rules and datasets
            max_errors: Maximum number of errors to collect per validation run
        """
        self.validation_rules = validation_rules
        self.max_errors = max_errors

    @abstractmethod
    def validate(self, rules: Dict[str, Any], df: pd.DataFrame) -> List[ErrorChecking]:
        """
        Perform validation on the given DataFrame.

        Args:
            rules: Validation rules for the current dataset
            df: DataFrame to validate

        Returns:
            List of ErrorChecking objects for any validation failures
        """
        pass

    def _should_continue_collecting_errors(self, current_error_count: int) -> bool:
        """Check if we should continue collecting errors based on the limit."""
        return current_error_count < self.max_errors

    def _extract_file_name(self, rules: Dict[str, Any]) -> str:
        """Extract file name from rules dictionary."""
        return Path(rules.get("file_name", "")).name

    def _create_error(
        self,
        message: str,
        file_name: Optional[str] = None,
        column_name: Optional[str] = None,
        relevant_column_value: Optional[Any] = None,
        error_source: str = ErrorSource.SHAREPOINT_VALIDATION.value,
    ) -> ErrorChecking:
        """Create an ErrorChecking object with the provided details."""
        if isinstance(file_name, (str, Path)):
            file_name = Path(file_name).name

        return ErrorChecking(
            column_name=column_name,
            relevant_column_value=relevant_column_value,
            relevant_file=file_name,
            error_info=message,
            error_source=error_source,
        )


class RequiredValidator(ValidationStrategy):
    """Validator for required column checks."""

    def validate(self, rules: Dict[str, Any], df: pd.DataFrame) -> List[ErrorChecking]:
        """Validate that required columns contain non-null values."""
        errors = []
        file_name = self._extract_file_name(rules)
        required_cols = rules.get("required", [])
        info_columns = rules.get("info_columns", [])

        for col in required_cols:
            if col not in df.columns:
                errors.append(
                    self._create_error(
                        column_name=col,
                        file_name=file_name,
                        message=f"Missing column for required check: '{col}'",
                    )
                )
                continue

            na_rows = df[df[col].isna()]
            for row_tuple in na_rows.itertuples():
                if not self._should_continue_collecting_errors(len(errors)):
                    break

                # Convert named tuple back to Series for compatibility
                row = pd.Series(row_tuple._asdict(), name=row_tuple.Index)
                row.drop("Index", inplace=True)  # Remove the index field

                error = self._create_error_for_row(
                    row=row,
                    file_name=file_name,
                    col_name=col,
                    info_columns=info_columns,
                )
                if error:
                    errors.append(error)

        return errors

    def _create_error_for_row(
        self,
        row: pd.Series,
        file_name: str,
        col_name: str,
        info_columns: List[str],
        **kwargs,
    ) -> Optional[ErrorChecking]:
        """Create error for a specific row that failed validation."""
        # Skip completely empty rows (but not rows with just the target column missing)
        if row.isna().all() and len(row) > 1:
            return None

        try:
            relevant_column_info = row[col_name]
        except KeyError as e:
            relevant_column_info = (
                f"No {col_name} in row for check required, error: {e}"
            )

        # For missing values, use info columns for context
        if pd.isna(relevant_column_info) and info_columns:
            relevant_column_info = " ".join(
                [
                    f"{info_column}: {row.get(info_column, '')}"
                    for info_column in info_columns
                ]
            )
        elif pd.isna(relevant_column_info):
            relevant_column_info = "N/A"

        message = f"Missing value in required column '{col_name}', help_info: {relevant_column_info}."

        return self._create_error(
            column_name=col_name,
            relevant_column_value=relevant_column_info,
            file_name=file_name,
            message=message,
        )


class UniqueValidator(ValidationStrategy):
    """Validator for unique column checks."""

    def validate(self, rules: Dict[str, Any], df: pd.DataFrame) -> List[ErrorChecking]:
        """Validate that columns marked as unique contain no duplicate values."""
        errors = []
        file_name = self._extract_file_name(rules)
        unique_cols = rules.get("unique", [])
        info_columns = rules.get("info_columns", [])

        for col in unique_cols:
            if col not in df.columns:
                errors.append(
                    self._create_error(
                        column_name=col,
                        relevant_column_value=None,
                        file_name=file_name,
                        message=f"Missing column for unique check: '{col}'",
                    )
                )
                continue

            # Check for duplicates in unique columns
            duplicated = df[df[col].duplicated(keep=False) & df[col].notna()]
            for row_tuple in duplicated.itertuples():
                if not self._should_continue_collecting_errors(len(errors)):
                    break

                # Convert named tuple back to Series for compatibility
                row = pd.Series(row_tuple._asdict(), name=row_tuple.Index)
                row.drop("Index", inplace=True)  # Remove the index field

                error = self._create_error_for_row(
                    row=row,
                    file_name=file_name,
                    col_name=col,
                    info_columns=info_columns,
                )
                if error:
                    errors.append(error)

        return errors

    def _create_error_for_row(
        self,
        row: pd.Series,
        file_name: str,
        col_name: str,
        info_columns: List[str],
        **kwargs,
    ) -> Optional[ErrorChecking]:
        """Create error for a specific row that failed validation."""
        if row.isna().all():
            return None

        try:
            relevant_column_info = row[col_name]
        except KeyError as e:
            relevant_column_info = (
                f"No {col_name} in row for check duplicate, error: {e}"
            )

        if pd.isna(relevant_column_info):
            relevant_column_info = " ".join(
                [
                    f"{info_column}: {row.get(info_column, '')}"
                    for info_column in info_columns
                ]
            )

        message = f"Duplicate value in unique column '{col_name}'"

        return self._create_error(
            column_name=col_name,
            relevant_column_value=relevant_column_info,
            file_name=file_name,
            message=message,
        )


class FormatValidator(ValidationStrategy):
    """Validator for format pattern checks."""

    def __init__(
        self,
        validation_rules: Dict[str, Any],
        patterns: Dict[str, str],
        max_errors: int = 1000,
    ):
        """
        Initialize the format validator.

        Args:
            validation_rules: Dictionary containing all validation rules and datasets
            patterns: Dictionary mapping column names to regex patterns
            max_errors: Maximum number of errors to collect
        """
        super().__init__(validation_rules, max_errors)
        self.patterns = patterns

    def validate(self, rules: Dict[str, Any], df: pd.DataFrame) -> List[ErrorChecking]:
        """Validate data formats against predefined regex patterns."""
        errors = []
        file_name = self._extract_file_name(rules)
        info_columns = rules.get("info_columns", [])

        for col, pattern in self.patterns.items():
            if col not in df.columns:
                continue

            invalid = df[~df[col].isna() & ~df[col].astype(str).str.match(pattern)]
            for row_tuple in invalid.itertuples():
                # Convert named tuple back to Series for compatibility
                row = pd.Series(row_tuple._asdict(), name=row_tuple.Index)
                row.drop("Index", inplace=True)  # Remove the index field

                error = self._create_error_for_row(
                    row=row,
                    file_name=file_name,
                    col_name=col,
                    info_columns=info_columns,
                    pattern=pattern,
                    actual_value=row[col],
                )
                if error:
                    errors.append(error)

        return errors

    def _create_error_for_row(
        self,
        row: pd.Series,
        file_name: str,
        col_name: str,
        info_columns: List[str],
        **kwargs,
    ) -> Optional[ErrorChecking]:
        """Create error for a specific row that failed validation."""
        if row.isna().all():
            return None

        try:
            relevant_column_info = row[col_name]
        except KeyError as e:
            relevant_column_info = (
                f"No {col_name} in row for check invalid_format, error: {e}"
            )

        if pd.isna(relevant_column_info):
            relevant_column_info = " ".join(
                [
                    f"{info_column}: {row.get(info_column, '')}"
                    for info_column in info_columns
                ]
            )

        pattern = kwargs.get("pattern", "unknown")
        actual_value = kwargs.get("actual_value", "unknown")
        message = f"Value {actual_value} does not match required format for {col_name}: expected pattern '{pattern}'"

        return self._create_error(
            column_name=col_name,
            relevant_column_value=relevant_column_info,
            file_name=file_name,
            message=message,
        )


class ForeignKeyValidator(ValidationStrategy):
    """Validator for foreign key relationship checks."""

    def validate(self, rules: Dict[str, Any], df: pd.DataFrame) -> List[ErrorChecking]:
        """Validate foreign key relationships between datasets."""
        errors = []
        source_file = self._extract_file_name(rules)
        foreign_keys = rules.get("foreign_keys", {})
        info_columns = rules.get("info_columns", [])

        for target_name, fk_col in foreign_keys.items():
            target_rules = self.validation_rules.get(target_name)
            if not target_rules:
                errors.append(
                    self._create_error(
                        column_name=fk_col,
                        file_name=source_file,
                        message=f"Foreign key check skipped: target dataset '{target_name}' not found",
                    )
                )
                continue

            target_df = target_rules.get("dataset", pd.DataFrame())
            if target_df.empty:
                errors.append(
                    self._create_error(
                        column_name=fk_col,
                        file_name=source_file,
                        message=f"Foreign key check skipped: target dataset '{target_name}' is empty or could not be loaded",
                    )
                )
                continue

            if fk_col not in df.columns:
                errors.append(
                    self._create_error(
                        column_name=fk_col,
                        file_name=source_file,
                        message=f"Foreign key column '{fk_col}' not found in '{source_file}'",
                    )
                )
                continue

            if fk_col not in target_df.columns:
                errors.append(
                    self._create_error(
                        column_name=fk_col,
                        file_name=source_file,
                        message=f"Foreign key column '{fk_col}' not found in target '{target_name}'",
                    )
                )
                continue

            missing = df[~df[fk_col].isin(target_df[fk_col])]
            fk_file_name = Path(target_rules["file_name"]).name

            for row_tuple in missing.itertuples():
                # Convert named tuple back to Series for compatibility
                row = pd.Series(row_tuple._asdict(), name=row_tuple.Index)
                row.drop("Index", inplace=True)  # Remove the index field

                error = self._create_error_for_row(
                    row=row,
                    file_name=source_file,
                    col_name=fk_col,
                    info_columns=info_columns,
                    fk_file_name=fk_file_name,
                    actual_value=row[fk_col],
                )
                if error:
                    errors.append(error)

        return errors

    def _create_error_for_row(
        self,
        row: pd.Series,
        file_name: str,
        col_name: str,
        info_columns: List[str],
        **kwargs,
    ) -> Optional[ErrorChecking]:
        """Create error for a specific row that failed validation."""
        if row.isna().all():
            return None

        try:
            relevant_column_info = row[col_name]
        except KeyError as e:
            relevant_column_info = (
                f"No {col_name} in row for check missing_fk, error: {e}"
            )

        if pd.isna(relevant_column_info):
            relevant_column_info = " ".join(
                [
                    f"{info_column}: {row.get(info_column, '')}"
                    for info_column in info_columns
                ]
            )

        fk_file_name = kwargs.get("fk_file_name", "unknown")
        actual_value = kwargs.get("actual_value", "unknown")
        message = (
            f"Foreign key '{col_name}' = '{actual_value}' not found in '{fk_file_name}'"
        )

        return self._create_error(
            column_name=col_name,
            relevant_column_value=relevant_column_info,
            file_name=file_name,
            message=message,
        )


class RelationshipValidator(ValidationStrategy):
    """Validator for column relationship checks."""

    def validate(self, rules: Dict[str, Any], df: pd.DataFrame) -> List[ErrorChecking]:
        """Validate column relationships across all rows in a dataset."""
        errors = []
        dataset_name = self._extract_file_name(rules)
        relationships = rules.get("relationships", [])

        for rel in relationships:
            col = rel["column"]

            if col not in df.columns:
                errors.append(
                    self._create_error(
                        column_name=col,
                        relevant_column_value=None,
                        file_name=dataset_name,
                        message=f"Missing column for relationship check: {col}",
                    )
                )
                continue

            for row_tuple in df.itertuples():
                # Convert named tuple back to Series for compatibility
                row = pd.Series(row_tuple._asdict(), name=row_tuple.Index)
                row.drop("Index", inplace=True)  # Remove the index field

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
            return self._create_error(
                column_name=col_name,
                relevant_column_value=None,
                file_name=file_name,
                message=f"Missing column {col_name} for relationship template: {str(e)}",
            )

        if rule == "equals" and str(actual) != str(expected):
            message = f"{col_name} should be '{expected}', but is '{actual}'"

            return self._create_error(
                column_name=col_name,
                relevant_column_value=actual,
                file_name=file_name,
                message=message,
            )

        return None


class FilePresenceValidator(ValidationStrategy):
    """Validator for file presence checks between CSV references and S3 storage."""

    def __init__(
        self, validation_rules: Dict[str, Any], s3_handler, max_errors: int = 1000
    ):
        """
        Initialize the file presence validator.

        Args:
            validation_rules: Dictionary containing all validation rules and datasets
            s3_handler: S3Handler instance for S3 operations
            max_errors: Maximum number of errors to collect
        """
        super().__init__(validation_rules, max_errors)
        self.s3_handler = s3_handler

    def validate(self, rules: Dict[str, Any], df: pd.DataFrame) -> List[ErrorChecking]:
        """
        Validate file presence between CSV references and S3 storage.

        This method compares video file paths referenced in CSV data with actual
        files available in S3 storage, identifying missing and extra files.

        Args:
            rules: Validation rules containing file presence configuration
            df: DataFrame to validate (not used directly, but kept for interface consistency)

        Returns:
            List of ErrorChecking objects for missing and extra files
        """
        errors = []

        # Get file presence configuration from rules
        file_presence_config = rules.get("file_presence", {})
        if not file_presence_config:
            return errors

        csv_filename = file_presence_config.get("csv_filename")
        csv_column_to_extract = file_presence_config.get("csv_column_to_extract")
        valid_extensions = file_presence_config.get("valid_extensions", [])
        path_prefix = file_presence_config.get("path_prefix", "")
        column_filter = file_presence_config.get("column_filter")
        column_value = file_presence_config.get("column_value")
        s3_sharepoint_path = file_presence_config.get("s3_sharepoint_path", "")
        bucket = file_presence_config.get("bucket")

        if not all([csv_filename, csv_column_to_extract, bucket]):
            errors.append(
                self._create_error(
                    message="File presence validation skipped: missing required configuration",
                    file_name="file_presence_config",
                )
            )
            return errors

        try:
            # Get file paths from CSV
            csv_s3_path = f"{s3_sharepoint_path}/{csv_filename}".strip("/")
            csv_filepaths_all, csv_filepaths_filtered = (
                self.s3_handler.get_paths_from_csv(
                    csv_s3_path=csv_s3_path,
                    csv_column=csv_column_to_extract,
                    column_filter=column_filter,
                    column_value=column_value,
                    s3_bucket=bucket,
                )
            )

            # Get file paths from S3
            s3_video_filepaths = self.s3_handler.get_paths_from_s3(
                path_prefix=path_prefix, valid_extensions=valid_extensions
            )

            # Find missing files (in CSV but not in S3)
            missing_files = csv_filepaths_filtered - s3_video_filepaths
            for file_path in missing_files:
                if not self._should_continue_collecting_errors(len(errors)):
                    break
                errors.append(
                    self._create_error(
                        message=f"File {file_path} not found in AWS, but found in {csv_filename}",
                        relevant_column_value=file_path,
                        error_source=ErrorSource.FILE_PRESENCE_CHECK.value,
                    )
                )

            # Find extra files (in S3 but not in CSV)
            extra_files = s3_video_filepaths - csv_filepaths_all
            for file_path in extra_files:
                if not self._should_continue_collecting_errors(len(errors)):
                    break
                errors.append(
                    self._create_error(
                        message=f"File {file_path} found in AWS but not in {csv_filename}",
                        relevant_column_value=file_path,
                        error_source=ErrorSource.FILE_PRESENCE_CHECK.value,
                    )
                )

        except Exception as e:
            errors.append(
                self._create_error(
                    message=f"File presence validation failed: {str(e)}",
                    file_name=csv_filename,
                    error_source=ErrorSource.FILE_PRESENCE_CHECK.value,
                )
            )

        return errors


class ValidationStrategyRegistry:
    """Registry for managing validation strategies."""

    def __init__(
        self,
        validation_rules: Dict[str, Any],
        patterns: Dict[str, str],
        s3_handler=None,
        max_errors: int = 1000,
    ):
        """
        Initialize the registry with validation strategies.

        Args:
            validation_rules: Dictionary containing all validation rules and datasets
            patterns: Dictionary mapping column names to regex patterns
            s3_handler: S3Handler instance for file presence validation
            max_errors: Maximum number of errors per validator
        """
        self.strategies = {
            "required": RequiredValidator(validation_rules, max_errors),
            "unique": UniqueValidator(validation_rules, max_errors),
            "formats": FormatValidator(validation_rules, patterns, max_errors),
            "foreign_keys": ForeignKeyValidator(validation_rules, max_errors),
            "column_relationships": RelationshipValidator(validation_rules, max_errors),
        }

        # Add file presence validator if s3_handler is provided
        if s3_handler:
            self.strategies["file_presence"] = FilePresenceValidator(
                validation_rules, s3_handler, max_errors
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
