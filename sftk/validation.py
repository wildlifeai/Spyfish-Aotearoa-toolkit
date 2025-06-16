import pandas as pd
import logging
from pathlib import Path
from io import BytesIO
from collections import defaultdict
from typing import Any, Iterable, Iterator, Union
from abc import ABC
from multiprocessing import Pool, get_context
from sftk.utils import flatten_list, clamp_n_jobs
from sftk.s3_handler import S3Handler
from sftk.common import S3_BUCKET, S3_SHAREPOINT_CSVS

class Validator(ABC):
    """
    Abstract base class for validators.
    """
    def __init__(self, n_jobs: int = -1):
        """Initialise the validator.

        This sets up the validation summary and the validation function (placeholder).

        Args:
            n_jobs (int): The number of jobs to run in parallel. Defaults to -1 (all available cores).
        """
        self._validation_summary = None
        self._validation_function = lambda _: {}
        self.n_jobs = clamp_n_jobs(n_jobs)

    def setup(self):
        """
        Setup the validator.

        This method should be overridden by subclasses to perform any necessary setup.
        """
        pass

    def validate(self, data: Iterable) -> dict:
        """
        Validate the given data.

        Args:
            data (iterable): The data to validate.

        Returns:
            dict: A dictionary containing the validation summary in the form {"filename": {validation_summary}}.

        Raises:
            ValueError: If the validation function is not set.
        """
        summary = {}
        if not self._validation_function:
            raise ValueError("Validation function is not set.")

        # Parellise processing using a multiprocessing pool
        # Use spawn context to avoid deadlocks
        with get_context("spawn").Pool(self.n_jobs) as pool:
            results = pool.map(self._validation_function, data)

        for res in flatten_list(results):
            summary.update(res)

        self._validation_summary = summary
        return summary

    def teardown(self):
        """
        Tear down the validator.

        This method should be overridden by subclasses to perform any necessary cleanup.
        """
        pass

    def cleanup(self):
        """
        Cleanup the validator.

        This method should be overridden by subclasses to perform any necessary cleanup.
        """
        pass

    def _get_data(self) -> Iterable:
        """
        Get the data to validate.

        This method should be overridden by subclasses to return the data to validate.
        """
        return []

    def run(self, data: Iterable = None) -> dict:
        """
        Run the validator.

        This method will call the setup, validate, teardown, and cleanup methods in order.

        Args:
            data (iterable, optional): The data to validate. If None, the data will be retrieved using _get_data.

        Returns:
            dict: A dictionary containing the validation summary.
        """
        self.setup()
        if data is None:
            data = self._get_data()
        self.validate(data)
        self.teardown()
        self.cleanup()
        return self._validation_summary

class SharepointValidator(Validator):
    """
    Validator for SharePoint data.
    """
    def __init__(self):
        """Initialise the SharePoint validator."""
        super().__init__()
        self._validation_function = self._validate_sharepoint_data
        self.s3_handler = None
        self.bucket_name = None
        self.prefixes = None

    def _validate_sharepoint_data(self, item: tuple[str, pd.DataFrame]) -> dict:
        """
        Validate SharePoint data.

        Args:
            item (tuple): A tuple containing the filename and the DataFrame to validate.

        Returns:
            dict: A dictionary containing the validation result.
        """
        validation_results = defaultdict(list)
        filename, df = item
        logging.info(f"Validating {filename}...")
        # Get the file base name
        base_name = Path(filename).stem

        # Check missing columns
        if base_name in self.expected_columns:
            missing_columns = self._get_missing_columns(df, self.expected_columns[base_name])
            if missing_columns:
                validation_results["missing_columns"].extend(missing_columns)
                logging.warning(f"Missing columns in {filename}: {missing_columns}")

        # Check for missing values
        missing_values = self._get_missing_values(df)
        if missing_values:
            validation_results["missing_values"].append(missing_values)
            logging.warning(f"Missing values in {filename}: {missing_values}")

        # Check unique constraints
        if base_name in self.validation_rules:
            res = self._check_constraints(df, self.validation_rules[base_name])
            if res:
                validation_results.update(res)
                logging.warning(f"Constraint violations in {filename}: {res}")

        return validation_results

    def _get_missing_columns(self, df: pd.DataFrame, expected_columns: list) -> list:
        """
        Get the missing columns in the DataFrame compared to the expected columns.

        Args:
            df (pd.DataFrame): The DataFrame to check.
            expected_columns (list): The list of expected columns.

        Returns:
            list: A list of missing columns.
        """
        missing_columns = [col for col in expected_columns if col not in df.columns]
        return missing_columns

    def _get_missing_values(self, df: pd.DataFrame) -> dict:
        """
        Get the missing values in the DataFrame for the specified columns.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Returns:
            dict: A dictionary with column names as keys and lists of missing values as values.
        """
        missing_values = {}
        for column in df.columns:
            missing = df[column].isnull().sum()
            if missing > 0:
                missing_values[column] = missing
        return missing_values

    def _check_constraints(self, df: pd.DataFrame, constraints: dict) -> dict:
        """
        Check constraints on the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to check.
            constraints (dict): The constraints to check.

        Returns:
            dict: A dictionary with constraint names as keys and lists of violations as values.
        """
        # Step one: Check unique columns
        duplicates = {}
        for cols in self.validation_rules.get("unique_columns", []):
            # Ensure cols is a list
            if isinstance(cols, str):
                cols = [cols]

            # Ensure all columns are present in the DataFrame
            if any(col not in df.columns for col in cols):
                continue

            # Check for duplicates
            duplicates = df.duplicated(subset=cols)
            if duplicates.any():
                key = "_".join(cols)
                duplicates[key] = df[duplicates].to_dict(orient="records")

        return duplicates


    def _get_data(self):
        """
        Get the SharePoint data to validate.

        This method will list, filter and retrieve SharePoint data.
        It returns data of the form [(filename, dataframe), ...].

        Returns:
            Iterable: An iterable of SharePoint items.
        """
        data = []
        for obj in self.list_objects(self.prefixes, ".csv"):
            key = obj["Key"]
            buffer = BytesIO()
            self.s3_handler.s3.download_fileobj(
                Bucket=self.bucket_name,
                Key=key,
                Fileobj=buffer
            )
            buffer.seek(0)
            try:
                df = pd.read_csv(buffer)
                data.append((key, df))
            except Exception as e:
                logging.error(f"Error reading CSV file {key}: {e}")
                continue

        return data

    def setup(self):
        """Setup the SharePoint validator.

        Creates an S3 handler and checks the connection to S3.
        Raises:
            ValueError: If S3 connection fails or bucket name/prefixes are not set.
        """
        super().setup()
        self.s3_handler = S3Handler()
        if not self.s3_handler.test_connection():
            raise ValueError("S3 connection failed.")
        self.bucket_name = S3_BUCKET
        self.prefixes = S3_SHAREPOINT_CSVS

        if not self.bucket_name or not self.prefixes:
            raise ValueError("S3 bucket name or prefixes are not set.")

        # Define expected columns for each CSV
        self.expected_columns = {
            'BUV Deployment': ['SurveyID', 'DropID'],
            'BUV Survey Metadata': ['SurveyID', 'SiteID'],
            'BUV Survey Sites': ['SiteID']
        }

        # Define validation rules for each CSV
        self.validation_rules = {
            'BUV Deployment': {
                'unique_columns': [['SurveyID', 'DropID']],
                # QUESTION: Can this be of the form 'SurveyID': 'BUV Survey Metadata'?
                'reference_mappings': {
                    'SurveyID': {'reference': 'BUV Survey Metadata', 'column': 'SurveyID'}
                }
            },
            'BUV Survey Metadata': {
                'unique_columns': ['SurveyID'],
                'reference_mappings': {
                    'SiteID': {'reference': 'BUV Survey Sites', 'column': 'SiteID'}
                }
            },
            'BUV Survey Sites': {
                'unique_columns': ['SiteID']
            }
        }

    def reference_integrity_validation(self, data: list[tuple[str, pd.DataFrame]]) -> dict:
        """Validates unique constraints and reference integrity.
        This ensures that primary keys are unique and that foreign keys reference valid primary keys.
        Params:
            data (iterable): The data to validate.

        Returns:
            dict: A dictionary containing the validation summary.
        """
        reference_integrity_summary = defaultdict(dict)
        for filename, df in data:
            base_name = Path(filename).stem
            rules = self.validation_rules.get(base_name, {})
            constraints = rules.get("reference_mappings", {})
            for source_col, ref_info in constraints.items():
                # Conditions where the reference integrity check should be skipped
                if source_col not in df.columns:
                    continue
                # Checks if the reference file exists in the data
                ref_df = next((df for fn, df in data if Path(fn).stem == ref_info["reference"]), None)
                if ref_df is None:
                    continue

                # Now the actual check
                if source_col in df.columns and ref_info["column"] in ref_df.columns:
                    # Check for missing references
                    missing_references = df[~df[source_col].isin(ref_df[ref_info["column"]])]
                    if not missing_references.empty:
                        reference_integrity_summary[filename][source_col] = {
                            "missing_references": missing_references[source_col].tolist()
                        }
                        logging.warning(f"Missing references in {filename}: {missing_references[source_col].tolist()}")


        return reference_integrity_summary


    def validate(self, data):
        summary = super().validate(data)
        reference_integrity_summary = self.reference_integrity_validation(data)
        summary['reference_integrity'] = reference_integrity_summary
        return summary

    def teardown(self):
        """Tear down the SharePoint validator.

        Removes references to the S3 handler.
        """
        super().teardown()
        if self.s3_handler:
            self.s3_handler = None


    def list_objects(self, prefixes: Union[str, list] = "", suffix: str = "") -> Iterator[dict]:
        """
        List objects in the S3 bucket.

        Args:
            prefixes (str | list): The prefix or list of prefixes to filter objects.
            suffix (str): The suffix to filter objects.

        Returns:
            Iterator[dict]: A generator yielding object metadata.
        """
        if isinstance(prefixes, str):
            prefixes = [prefixes]

        for prefix in prefixes:
            found = False
            for obj in self._fetch_objects(prefix, suffix):
                found = True
                yield obj

            if not found:
                logging.info(f"No objects found for prefix: {prefix} with suffix: {suffix}")

    def _fetch_objects(self, prefix: str, suffix: str) -> Iterator[dict]:
        """Fetch objects from a given S3 prefix, filtering by suffix."""
        for obj in self._fetch_pages(prefix):
            if obj["Key"].endswith(suffix):
                yield obj

    def _fetch_pages(self, prefix: str) -> Iterator[dict]:
        """Fetch paginated S3 objects for a given prefix."""
        paginator = self.s3_handler.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            yield from page.get("Contents", []) or []
