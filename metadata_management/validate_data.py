import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from sftk.clean_data import convert_int_num_columns_to_int
from sftk.common import (
    S3_BUCKET,
    S3_KSO_ERRORS_CSV,
    VALIDATION_PATTERNS,
    VALIDATION_RULES,
)
from sftk.s3_handler import S3FileNotFoundError, S3Handler


@dataclass
class ErrorChecking:
    column_name: Optional[str]
    relevant_columns_value: Optional[str]
    relevant_file: str
    error_info: str


class SharepointValidator:
    def __init__(self):
        self.errors = []
        self.errors_df = None
        self.patterns = VALIDATION_PATTERNS
        self.s3_handler = S3Handler()
        self.validation_rules = self._get_validation_rules()

    def _get_validation_rules(self):
        validation_rules = copy.deepcopy(VALIDATION_RULES)
        for dataset_name, rule_set in validation_rules.items():
            # df = self.s3_handler.read_df_from_s3_csv(rule_set["file_name"])
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
    ):

        if not any([required, unique, foreign_keys, formats, column_relationships]):
            required = unique = foreign_keys = formats = column_relationships = True

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

        self._export_errors_from_list_to_df(remove_duplicates)

        return self.errors_df

    def _iterate_df(
        self,
        row,
        file_name=None,
        col_name=None,
        check=None,
        fk_file_name=None,
        info_columns=None,
        pattern=None,
    ):
        try:
            relevant_column_info = row[col_name]
        except KeyError as e:
            relevant_column_info = f"No {col_name} in row for check {check}, error: {e}"

        if check == "required":
            relevant_column_info = " ".join(
                [
                    f"{info_column}: {row.get(info_column, '')}"
                    for info_column in info_columns
                ]
            )
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
                self._iterate_df,
                file_name=file_name,
                col_name=col,
                info_columns=info_columns,
                check="required",
                axis=1,
            )

    def _check_unique(self, rules, df):
        file_name = Path(rules.get("file_name", "")).name
        unique_cols = rules.get("unique", [])

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
                self._iterate_df,
                file_name=file_name,
                col_name=col,
                check="duplicate",
                axis=1,
            )

    def _check_foreign_keys(self, rules, source_df):
        source_file = Path(rules.get("file_name", "")).name
        foreign_keys = rules.get("foreign_keys", {})

        for target_name, fk_col in foreign_keys.items():
            target_rules = self.validation_rules.get(target_name)
            target_df = target_rules.get("dataset", pd.DataFrame())

            if not target_rules or target_df is None:
                self._add_error(
                    column_name=fk_col,
                    file_name=source_file,
                    message=f"Foreign key check skipped: target dataset '{target_name}' not found",
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
                self._iterate_df,
                file_name=source_file,
                col_name=fk_col,
                fk_file_name=fk_file_name,
                check="missing_fk",
                axis=1,
            )

    def _check_formats(self, rules, df):
        source_file = Path(rules.get("file_name", "")).name

        for col, pattern in self.patterns.items():
            if col not in df.columns:
                continue
            invalid = df[~df[col].isna() & ~df[col].astype(str).str.match(pattern)]

            invalid.apply(
                self._iterate_df,
                file_name=source_file,
                col_name=col,
                pattern=pattern,
                check="invalid_format",
                axis=1,
            )

    def _check_row_relationship(self, row, file_name, col_name, relationships):
        rule = relationships["rule"]
        template = relationships["template"]
        allowed_values = relationships.get("allowed_values", [])
        actual = row[col_name]
        # skip allowed values
        is_null_allowed = "NULL" in allowed_values
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

            df.apply(
                self._check_row_relationship,
                file_name=dataset_name,
                col_name=col,
                relationships=rel,
                axis=1,
            )

    def _add_error(
        self, file_name, message, column_name=None, relevant_column_value=None
    ):
        self.errors.append(
            ErrorChecking(
                column_name=column_name,
                relevant_columns_value=relevant_column_value,
                relevant_file=Path(file_name).name if file_name else "unknown_file",
                error_info=message,
            )
        )

    def _export_errors_from_list_to_df(self, remove_duplicates):
        self.errors_df = pd.DataFrame([e.__dict__ for e in self.errors])

        if remove_duplicates:
            self._deduplicate_errors()
        self.errors = []

    def _deduplicate_errors(self):
        if self.errors_df.empty:
            return

        key_cols = list(ErrorChecking.__dataclass_fields__.keys())
        self.errors_df = self.errors_df.drop_duplicates(
            subset=key_cols, ignore_index=True
        )

    def export_to_csv(self, csv_file_name="validation_errors_cleaned.csv"):
        self.errors_df["ErrorSource"] = "Sharepoint error validation"
        self.errors_df.to_csv(csv_file_name, index=False)
        logging.info(f"Errors exported to csv file {csv_file_name}.")

    def upload_to_s3(self):
        keyword = "errors"
        self.s3_handler.upload_updated_df_to_s3(
            df=self.errors_df,
            key=S3_KSO_ERRORS_CSV,
            keyword=keyword,
            bucket=S3_BUCKET,
            keep_df_index=False,
        )
        logging.info(f"Updated {keyword} DataFrame uploaded to S3")


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
    validator = SharepointValidator()
    validator.validate(
        remove_duplicates=True,
        required=True,
        unique=True,
        foreign_keys=True,
        formats=True,
        column_relationships=True,
    )
    logging.info(
        f"Error validation completed, {validator.errors_df.shape[0]} errors found"
    )
    # Export to csv
    validator.export_to_csv("validation_errors.csv")
    # validator.upload_to_s3()
    logging.info("Error validation process completed, files created/uploaded.")
