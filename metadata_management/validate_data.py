import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sftk.common import (
    S3_BUCKET,
    S3_KSO_ERRORS_CSV,
    VALIDATION_PATTERNS,
    VALIDATION_RULES,
)
from sftk.s3_handler import S3Handler


@dataclass
class ErrorChecking:
    ColumnName: str
    RelevantColumnsValue: str
    RelevantFile: str
    ErrorInfo: str


class SharepointValidator:
    def __init__(self):
        self.errors = []
        self.errors_df = None
        self.patterns = VALIDATION_PATTERNS
        self.s3_handler = S3Handler()
        self.validation_rules = self._get_validation_rules()

    def _get_validation_rules(self):

        for dataset_name, rule_set in VALIDATION_RULES.items():
            VALIDATION_RULES[dataset_name]["dataset"] = (
                self.s3_handler.read_df_from_s3_csv(rule_set["file_name"])
            )
        return VALIDATION_RULES

    def validate(
        self,
        remove_duplicates=True,
        required=False,
        unique=False,
        foreign_keys=False,
        formats=False,
        column_relationships=False,
    ):

        self.errors = []

        if not any([required, unique, foreign_keys, formats, column_relationships]):
            required = unique = foreign_keys = formats = column_relationships = True

        for dataset_name, rules in self.validation_rules.items():
            df = rules.get("dataset", pd.DataFrame())
            file_name = Path(rules.get("file_name", "")).name

            if df is None:
                self._add_error(
                    file_name=file_name, message="Dataset could not be loaded."
                )
                continue
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

        self._get_df(remove_duplicates)

        return self.errors_df

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
            for idx, row in na_rows.iterrows():
                help_info = (
                    " ".join([f"{x}: {row.get(x, '')}" for x in info_columns])
                    if info_columns
                    else None
                )
                self._add_error(
                    column_name=col,
                    relevant_column_value=help_info,
                    file_name=file_name,
                    message=f"Missing value in required column '{col}'",
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

            # Chack for Duplicates in unique columns
            duplicated = df[df[col].duplicated(keep=False) & df[col].notna()]
            for idx, row in duplicated.iterrows():
                self._add_error(
                    column_name=col,
                    relevant_column_value=row[col],
                    file_name=file_name,
                    message=f"Duplicate value in unique column '{col}'",
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
            for idx, row in missing.iterrows():
                self._add_error(
                    column_name=fk_col,
                    relevant_column_value=row[fk_col],
                    file_name=source_file,
                    message=(
                        f"Foreign key '{fk_col}' = '{row[fk_col]}' "
                        f"not found in '{Path(target_rules['file_name']).name}'"
                    ),
                )

    def _check_formats(self, rules, df):
        source_file = Path(rules.get("file_name", "")).name

        for col, pattern in self.patterns.items():
            if col not in df.columns:
                continue
            invalid = df[~df[col].isna() & ~df[col].astype(str).str.match(pattern)]
            for idx, row in invalid.iterrows():
                self._add_error(
                    column_name=col,
                    relevant_column_value=row[col],
                    file_name=source_file,
                    message=f"Value {row[col]} does not match required format for {col}: expected pattern '{pattern}'",
                )

    def _check_column_relationships(self, rules, df):
        dataset_name = Path(rules.get("file_name", "")).name
        relationships = rules.get("relationships", [])

        for rel in relationships:
            col = rel["column"]
            rule = rel["rule"]
            template = rel["template"]
            allowed_values = rel.get("allowed_values", [])

            if col not in df.columns:
                self._add_error(
                    column_name=col,
                    relevant_column_value=None,
                    file_name=dataset_name,
                    message=f"Missing column for relationship check: {col}",
                )
                continue

            for idx, row in df.iterrows():
                actual = row[col]

                # skip allowed values
                is_null_allowed = "NULL" in allowed_values
                if (actual in allowed_values) or (is_null_allowed and pd.isna(actual)):
                    continue

                try:
                    expected = template.format(**row)
                except KeyError as e:
                    self._add_error(
                        column_name=col,
                        relevant_column_value=None,
                        file_name=dataset_name,
                        message=f"Missing column for relationship template: {str(e)}",
                    )
                    continue

                if rule == "equals" and str(actual) != str(expected):
                    self._add_error(
                        column_name=col,
                        relevant_column_value=actual,
                        file_name=dataset_name,
                        message=f"{col} should be '{expected}', but is '{actual}'",
                    )

    def _add_error(
        self, column_name=None, relevant_column_value=None, file_name=None, message=None
    ):
        self.errors.append(
            ErrorChecking(
                ColumnName=column_name,
                RelevantColumnsValue=relevant_column_value,
                RelevantFile=Path(file_name).name if file_name else None,
                ErrorInfo=message,
            )
        )

    def _get_df(self, remove_duplicates):
        self.errors_df = pd.DataFrame([e.__dict__ for e in self.errors])
        if remove_duplicates:
            self._deduplicate_errors()

    def _deduplicate_errors(self):
        df = self.errors_df

        key_cols = list(ErrorChecking.__dataclass_fields__.keys())
        # Temporarily replace NaN with a string in key columns (for grouping)
        df.fillna("NULL", inplace=True)
        df_no_duplicates = df.groupby(key_cols, as_index=False).first()
        df_no_duplicates.replace("NULL", pd.NA, inplace=True)
        self.errors = [ErrorChecking(**row) for _, row in df_no_duplicates.iterrows()]
        self.errors_df = df_no_duplicates

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
        column_relationships=False,
    )
    logging.info(
        f"Error validation completed, {validator.errors_df.shape[0]} errors found"
    )
    # Export to csv
    # validator.export_to_csv("validation_errors_test.csv")
    validator.upload_to_s3()
    logging.info("Error validation process completed, files created/uploaded.")
