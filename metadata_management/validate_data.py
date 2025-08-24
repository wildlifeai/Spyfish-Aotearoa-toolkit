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
        self.errors = []
        self.errors_df = None
        self.patterns = VALIDATION_PATTERNS
        self.s3_handler = S3Handler()
        self.bucket = S3_BUCKET
        self.validation_rules = self._get_validation_rules()

    def _get_validation_rules(self):
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
        self.errors_df.to_csv(csv_file_name, index=False)
        logging.info(f"Errors exported to csv file {csv_file_name}.")

    def upload_to_s3(self):
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
        Compares video file paths from a CSV on with video files available in S3,
        identifies mismatches, and writes the results to local text files.

        Parameters:
            csv_filename (str): Name of the CSV file in S3
                (e.g., 'BUV Deployment.csv').
            csv_column (str): Column in the CSV that contains video file paths.
            valid_extensions (set): Set of valid video file extensions
                (e.g., {'mp4', 'mov'}).

        Outputs:
            - 'missing_files_in_aws.txt': Files listed in CSV but missing from S3.
            - 'extra_files_in_aws.txt': Files in S3 not listed in the CSV.
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
