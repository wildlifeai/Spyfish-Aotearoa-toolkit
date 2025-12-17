"""
Data validation module.

Provides simple validation functions and a DatasetValidator class:
- validate_required: Check required columns have values
- validate_unique: Check uniqueness constraints
- validate_formats: Check format patterns
- validate_values: Check value ranges
- validate_relationships: Check column relationships
- validate_foreign_keys: Check foreign key references
- DatasetValidator: Orchestrates all validations for a single dataset
- FilePresenceValidator: Validates file presence in S3 storage
- CleanRowTracker: Tracks rows without validation errors
"""

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from sftk.common import DROP_ID_COLUMN, REPLICATE_COLUMN, SURVEY_ID_COLUMN
from sftk.utils import normalize_file_name

# =============================================================================
# Core Types
# =============================================================================


@dataclass
class ErrorChecking:
    """Data class for validation errors."""

    survey_id: Optional[str]
    drop_id: Optional[str]
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
    """Configuration for validation operations."""

    file_presence: bool = False
    remove_duplicates: bool = True
    extract_clean_dataframes: bool = False
    max_errors_per_validator: int = 10000


class CleanRowTracker:
    """Tracks which rows remain clean (error-free) during validation."""

    def __init__(self):
        self.clean_row_indices: Dict[str, Set[int]] = {}

    def initialize_dataset(self, dataset_name: str, df: pd.DataFrame) -> None:
        """Initialize clean indices with all row indices for a dataset."""
        self.clean_row_indices[dataset_name] = set(df.index)

    def mark_row_as_error(self, row_index: int, dataset_name: str) -> None:
        """Remove a row index from the clean set."""
        if dataset_name in self.clean_row_indices:
            self.clean_row_indices[dataset_name].discard(row_index)

    def get_clean_indices(self, dataset_name: str) -> Set[int]:
        """Get clean row indices for a dataset."""
        return self.clean_row_indices.get(dataset_name, set())


# =============================================================================
# Error Creation Helper
# =============================================================================


def create_error(
    survey_id: Optional[str] = None,
    drop_id: Optional[str] = None,
    message: str = "",
    error_source: str = "",
    file_name: str = "",
    column_name: Optional[str] = None,
    relevant_column_value: Any = None,
) -> ErrorChecking:
    """Create an ErrorChecking object - single consolidated error factory."""
    return ErrorChecking(
        survey_id=survey_id,
        drop_id=drop_id,
        column_name=column_name,
        relevant_column_value=relevant_column_value,
        relevant_file=normalize_file_name(file_name),
        error_info=message,
        error_source=error_source,
    )


# =============================================================================
# Validation Functions
# =============================================================================


def validate_unique(
    df: pd.DataFrame,
    rules: Dict[str, Any],
    tracker: Optional[CleanRowTracker] = None,
    dataset_name: Optional[str] = None,
    max_errors: int = 10000,
) -> List[ErrorChecking]:
    """Validate unique column constraints."""
    errors = []
    file_name = normalize_file_name(rules.get("file_name", ""))
    unique_cols = rules.get("unique", [])

    for col in unique_cols:
        if col not in df.columns:
            errors.append(
                create_error(
                    message=f"Missing column for unique check: '{col}'",
                    error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                    column_name=col,
                    file_name=file_name,
                )
            )
            continue

        duplicated = df[df[col].duplicated(keep=False) & df[col].notna()]
        for idx, row in duplicated.iterrows():
            if len(errors) >= max_errors:
                break
            errors.append(
                create_error(
                    survey_id=row.get(SURVEY_ID_COLUMN, ""),
                    drop_id=row.get(DROP_ID_COLUMN, ""),
                    message=f"Duplicate value in unique column '{col}'",
                    error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                    column_name=col,
                    relevant_column_value=row[col],
                    file_name=file_name,
                )
            )
            if tracker and dataset_name:
                tracker.mark_row_as_error(idx, dataset_name)

    return errors


def validate_foreign_keys(
    df: pd.DataFrame,
    rules: Dict[str, Any],
    all_validation_rules: Dict[str, Any],
    tracker: Optional[CleanRowTracker] = None,
    dataset_name: Optional[str] = None,
    max_errors: int = 10000,
) -> List[ErrorChecking]:
    """Validate foreign key relationships."""
    errors = []
    source_file = normalize_file_name(rules.get("file_name", ""))
    foreign_keys = rules.get("foreign_keys", {})

    for target_name, fk_col in foreign_keys.items():
        target_rules = all_validation_rules.get(target_name)
        if not target_rules:
            errors.append(
                create_error(
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
                create_error(
                    message=f"Foreign key check skipped: target dataset '{target_name}' is empty",
                    error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                    column_name=fk_col,
                    file_name=source_file,
                )
            )
            continue

        if fk_col not in df.columns or fk_col not in target_df.columns:
            errors.append(
                create_error(
                    message=f"Foreign key column '{fk_col}' not found",
                    error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                    column_name=fk_col,
                    file_name=source_file,
                )
            )
            continue

        missing = df[~df[fk_col].isin(target_df[fk_col])]
        fk_file_name = normalize_file_name(target_rules.get("file_name", ""))

        for idx, row in missing.iterrows():
            if len(errors) >= max_errors:
                break
            errors.append(
                create_error(
                    survey_id=row.get(SURVEY_ID_COLUMN, ""),
                    drop_id=row.get(DROP_ID_COLUMN, ""),
                    message=f"Foreign key '{fk_col}' = '{row[fk_col]}' not found in '{fk_file_name}'",
                    error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                    column_name=fk_col,
                    relevant_column_value=row[fk_col],
                    file_name=source_file,
                )
            )
            if tracker and dataset_name:
                tracker.mark_row_as_error(idx, dataset_name)

    return errors


def validate_required(
    row: pd.Series,
    rules: Dict[str, Any],
    file_name: str,
) -> List[ErrorChecking]:
    """Check required columns for a single row."""
    errors = []
    required_cols = rules.get("required", [])
    info_columns = rules.get("info_columns", [])
    survey_id = row.get(SURVEY_ID_COLUMN, "")
    drop_id = row.get(DROP_ID_COLUMN, "")

    for col in required_cols:
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value) or value == "" or value is None:
            help_info = " ".join(
                [f"{ic}: {row.get(ic, '')}" for ic in info_columns if ic in row.index]
            )
            errors.append(
                create_error(
                    survey_id=survey_id,
                    drop_id=drop_id,
                    message=f"Missing value in required column '{col}', help_info: {help_info}.",
                    error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                    column_name=col,
                    relevant_column_value=help_info,
                    file_name=file_name,
                )
            )
    return errors


def validate_formats(
    row: pd.Series,
    rules: Dict[str, Any],
    patterns: Dict[str, str],
    file_name: str,
) -> List[ErrorChecking]:
    """Check format patterns for a single row."""
    errors = []
    format_rules = rules.get("formats", {})
    survey_id = row.get(SURVEY_ID_COLUMN, "")
    drop_id = row.get(DROP_ID_COLUMN, "")

    for col, pattern_name in format_rules.items():
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value) or value == "":
            continue
        pattern = patterns.get(pattern_name)
        if not pattern:
            continue
        if not re.match(pattern, str(value)):
            errors.append(
                create_error(
                    survey_id=survey_id,
                    drop_id=drop_id,
                    message=f"Value {value} does not match required format for {col}: expected pattern '{pattern}'",
                    error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                    column_name=col,
                    relevant_column_value=value,
                    file_name=file_name,
                )
            )
    return errors


def validate_values(
    row: pd.Series,
    rules: Dict[str, Any],
    file_name: str,
) -> List[ErrorChecking]:
    """Check value ranges for a single row."""
    errors = []
    value_rules = rules.get("values", [])
    survey_id = row.get(SURVEY_ID_COLUMN, "")
    drop_id = row.get(DROP_ID_COLUMN, "")

    for value_rule in value_rules:
        col = value_rule["column"]
        if col not in row.index:
            continue
        actual = row[col]
        allowed_values = value_rule.get("allowed_values", [])
        if actual in allowed_values or pd.isna(actual):
            continue

        rule = value_rule["rule"]
        if rule == "value_range":
            value_range = value_rule.get("range")
            if not value_range or len(value_range) != 2:
                continue
            try:
                actual_numeric = float(actual)
            except (ValueError, TypeError):
                errors.append(
                    create_error(
                        survey_id=survey_id,
                        drop_id=drop_id,
                        message=f"{col} value '{actual}' is not a valid number",
                        error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                        column_name=col,
                        relevant_column_value=actual,
                        file_name=file_name,
                    )
                )
                continue
            if not (value_range[0] <= actual_numeric <= value_range[1]):
                errors.append(
                    create_error(
                        survey_id=survey_id,
                        drop_id=drop_id,
                        message=f"{col} value {actual_numeric} is outside valid range [{value_range[0]}, {value_range[1]}]",
                        error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                        column_name=col,
                        relevant_column_value=actual,
                        file_name=file_name,
                    )
                )
    return errors


def _is_replicate_mismatch_only(actual: str, expected: str) -> bool:
    """Check if mismatch is only in the replicate part (last 2 digits)."""
    if len(actual) != len(expected) or len(actual) < 2:
        return False
    return actual[:-2] == expected[:-2] and actual[-2:] != expected[-2:]


def validate_relationships(
    row: pd.Series,
    rules: Dict[str, Any],
    file_name: str,
) -> List[ErrorChecking]:
    """Check column relationships for a single row."""
    errors = []
    relationships = rules.get("relationships", [])
    survey_id = row.get(SURVEY_ID_COLUMN, "")
    drop_id = row.get(DROP_ID_COLUMN, "")

    for relationship in relationships:
        col = relationship["column"]
        if col not in row.index:
            continue
        actual = row[col]
        allowed_values = relationship.get("allowed_values", [])
        is_null_allowed = relationship.get("allow_null")
        if (actual in allowed_values) or (is_null_allowed and pd.isna(actual)):
            continue

        rule = relationship["rule"]
        if rule == "equals":
            template = relationship.get("template", "")
            try:
                expected = template.format(**row)
            except KeyError as e:
                errors.append(
                    create_error(
                        survey_id=survey_id,
                        drop_id=drop_id,
                        message=f"Missing column {col} for relationship template: {str(e)}",
                        error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                        column_name=col,
                        file_name=file_name,
                    )
                )
                continue

            if str(actual) != str(expected):
                if col == DROP_ID_COLUMN and _is_replicate_mismatch_only(
                    str(actual), str(expected)
                ):
                    message = f"{REPLICATE_COLUMN} mismatch: {col} should end with '{str(expected)[-2:]}' but ends with '{str(actual)[-2:]}'. Full {col} should be '{expected}', but is '{actual}'"
                else:
                    message = f"{col} should be '{expected}', but is '{actual}'"
                errors.append(
                    create_error(
                        survey_id=survey_id,
                        drop_id=drop_id,
                        message=message,
                        error_source=ErrorSource.SHAREPOINT_VALIDATION.value,
                        column_name=col,
                        relevant_column_value=actual,
                        file_name=file_name,
                    )
                )
    return errors


# =============================================================================
# File Presence Validator (needs S3 handler state)
# =============================================================================


class FilePresenceValidator:
    """Validates file presence in S3 storage."""

    def __init__(self, s3_handler, max_errors: int = 10000):
        self.s3_handler = s3_handler
        self.max_errors = max_errors

    def _extract_survey_drop_from_path(
        self, file_path: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract SurveyID and DropID from a file path (media/{SurveyID}/{DropID}/...)."""
        try:
            parts = file_path.split("/")
            if len(parts) >= 3 and parts[0] == "media":
                return parts[1] if parts[1] else None, parts[2] if parts[2] else None
        except Exception as e:
            logging.warning(
                f"Could not extract survey/drop ID from path '{file_path}': {e}"
            )
        return None, None

    def validate(self, rules: Dict[str, Any]) -> List[ErrorChecking]:
        """Validate file presence between CSV references and S3 storage."""
        errors: List[ErrorChecking] = []
        file_presence_config = rules.get("file_presence", {})
        if not file_presence_config:
            return errors

        csv_filename = file_presence_config.get("csv_filename")
        try:
            _, missing_files, extra_files = self.get_file_differences(rules)

            for file_path in missing_files:
                if len(errors) >= self.max_errors:
                    break
                survey_id, drop_id = self._extract_survey_drop_from_path(file_path)
                errors.append(
                    create_error(
                        survey_id=survey_id,
                        drop_id=drop_id,
                        message=f"File {file_path} not found in AWS, but found in {csv_filename}",
                        relevant_column_value=file_path,
                        error_source=ErrorSource.FILE_PRESENCE_CHECK.value,
                    )
                )

            for file_path in extra_files:
                if len(errors) >= self.max_errors:
                    break
                survey_id, drop_id = self._extract_survey_drop_from_path(file_path)
                errors.append(
                    create_error(
                        survey_id=survey_id,
                        drop_id=drop_id,
                        message=f"File {file_path} found in AWS but not in {csv_filename}",
                        relevant_column_value=file_path,
                        error_source=ErrorSource.FILE_PRESENCE_CHECK.value,
                    )
                )
        except Exception as e:
            errors.append(
                create_error(
                    message=f"File presence validation failed: {str(e)}",
                    file_name=csv_filename,
                    error_source=ErrorSource.FILE_PRESENCE_CHECK.value,
                )
            )
        return errors

    def get_file_differences(self, rules: Dict[str, Any]) -> tuple[set, set, set]:
        """Get file differences between CSV references and S3 storage."""
        file_presence_config = rules.get("file_presence", {})
        if not file_presence_config:
            return set(), set(), set()

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

        csv_s3_path = os.path.join(s3_sharepoint_path, csv_filename)
        csv_paths_result = self.s3_handler.get_paths_from_csv(
            csv_s3_path=csv_s3_path,
            csv_column=csv_column_to_extract,
            column_filter=column_filter,
            column_value=column_value,
        )
        csv_filepaths_all = csv_paths_result["all"]
        csv_filepaths_filtered = csv_paths_result["filtered"]

        s3_video_filepaths = self.s3_handler.get_paths_from_s3(
            path_prefix=path_prefix, valid_extensions=valid_extensions
        )

        missing_files = csv_filepaths_filtered - s3_video_filepaths
        extra_files = s3_video_filepaths - csv_filepaths_all

        return s3_video_filepaths, missing_files, extra_files


# =============================================================================
# Dataset Validator (orchestrates all validations for one dataset)
# =============================================================================


class DatasetValidator:
    """Validates a single dataset with all rules."""

    def __init__(
        self,
        rules: Dict[str, Any],
        patterns: Dict[str, str],
        all_validation_rules: Dict[str, Any],
        tracker: Optional[CleanRowTracker] = None,
    ):
        self.rules = rules
        self.patterns = patterns
        self.all_validation_rules = all_validation_rules
        self.tracker = tracker
        self.file_name = normalize_file_name(rules.get("file_name", ""))

    def validate(self, df: pd.DataFrame, dataset_name: str) -> List[ErrorChecking]:
        """Validate the dataset with all rules."""
        errors = []

        # Dataset-level validations
        errors.extend(validate_unique(df, self.rules, self.tracker, dataset_name))
        errors.extend(
            validate_foreign_keys(
                df, self.rules, self.all_validation_rules, self.tracker, dataset_name
            )
        )

        # Row-level validations (single pass through rows)
        for idx, row in df.iterrows():
            row_errors = self._validate_row(row)
            if row_errors:
                errors.extend(row_errors)
                if self.tracker:
                    self.tracker.mark_row_as_error(idx, dataset_name)

        return errors

    def _validate_row(self, row: pd.Series) -> List[ErrorChecking]:
        """Validate a single row with all row-level validations."""
        errors = []
        errors.extend(validate_required(row, self.rules, self.file_name))
        errors.extend(validate_formats(row, self.rules, self.patterns, self.file_name))
        errors.extend(validate_values(row, self.rules, self.file_name))
        errors.extend(validate_relationships(row, self.rules, self.file_name))
        return errors
