import io
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError
from tqdm import tqdm

from sftk.common import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET
from sftk.utils import delete_file


# TODO check when is this used
class S3FileNotFoundError(Exception):
    """Custom exception for S3 file not found scenarios."""

    pass


@dataclass
class S3FileConfig:
    """
    Configuration class for S3 file operations containing file paths and environment variables.

    Attributes:
        keyword (str): Identifier for the type of data (e.g., 'survey', 'site', 'movie')
        kso_env_var (str): Environment variable name for KSO file path
        sharepoint_env_var (str): Environment variable name for Sharepoint file path
        kso_filename (str): Temporary filename for KSO data
        sharepoint_filename (str): Temporary filename for Sharepoint data
    """

    keyword: str
    kso_env_var: str
    sharepoint_env_var: str
    kso_filename: str
    sharepoint_filename: str

    @classmethod
    def from_keyword(cls, keyword: str) -> "S3FileConfig":
        """
        Creates a configuration object for S3 file operations based on
        a keyword (e.g., "survey", "site").

        Args:
            keyword(str): String identifier for the type of data

        Returns:
            S3FileConfig: The configuration information for S3 handler.
        """
        upper = keyword.upper()
        return cls(
            keyword=keyword,
            kso_env_var=f"S3_KSO_{upper}_CSV",
            sharepoint_env_var=f"S3_SHAREPOINT_{upper}_CSV",
            kso_filename=f"{keyword}_kso_temp.csv",
            sharepoint_filename=f"{keyword}_sharepoint_temp.csv",
        )


class S3Handler:
    """
    Singleton class for interacting with an S3 bucket.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "S3Handler":
        """
        Create a new instance of the class if one does not already exist.

        Returns:
            S3Handler: The instance of the class.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                # TODO create user input option
                cls._instance.s3 = kwargs.get("s3_client") or boto3.client(
                    "s3",
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                )
                logging.info("Created a new instance of the S3Handler class.")

            return cls._instance

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns:
            str: The string representation of the class.
        """
        return f"S3Handler({self.s3})"

    def download_object_from_s3(
        self,
        key: str,
        filename: str,
        version_id: Optional[str] = None,
        bucket: str = S3_BUCKET,
    ) -> None:
        """
        Downloads an object from S3 with progress bar and error handling.

        Args:
            key (str): The S3 object key.
            filename (str): The local filename to save the object to.
            version_id (str, optional): The version ID of the object.
                Defaults to None.
            bucket (str): The S3 bucket name, defaults to env defined bucket.

        Raises:
            ClientError: If the S3 object cannot be accessed or downloaded.
            Exception: For other errors during download.
        """
        try:
            kwargs: Dict[str, Any] = {"Bucket": bucket, "Key": key}
            if version_id:
                kwargs["VersionId"] = version_id

            object_size = self.s3.head_object(**kwargs)["ContentLength"]

            def progress_update(bytes_transferred):
                pbar.update(bytes_transferred)

            with tqdm(
                total=object_size, unit="B", unit_scale=True, desc=filename
            ) as pbar:
                self.s3.download_file(
                    Bucket=bucket,
                    Key=key,
                    Filename=filename,
                    Callback=progress_update,
                    Config=boto3.s3.transfer.TransferConfig(use_threads=False),
                )
        except Exception as e:
            logging.error("Failed to download %s from S3: %s", key, e)
            raise S3FileNotFoundError(f"Failed to download {key} from S3: {e}")

    def download_and_read_s3_file(
        self, key: str, filename: str, bucket: str = S3_BUCKET
    ) -> pd.DataFrame:
        """
        Downloads an S3 object and reads it into a Pandas DataFrame.

        Args:
            key (str): The S3 object key.
            filename (str): The local filename to save the downloaded object.
            bucket (str): The S3 bucket name, defaults to env defined bucket.

        Returns:
            pd.DataFrame: The DataFrame read from the downloaded file.

        Raises:
            Exceptions: If other errors occur during download or reading.
        """
        try:
            self.download_object_from_s3(key=key, filename=filename, bucket=bucket)
            return pd.read_csv(filename)
        except BotoCoreError as e:
            logging.warning("Failed to process S3 file %s: %s", key, str(e))
            raise S3FileNotFoundError(f"Failed to download {key} from S3: {e}")

    def upload_updated_df_to_s3(
        self,
        df: pd.DataFrame,
        key: str,
        keyword: str,
        bucket: str = S3_BUCKET,
        keep_df_index=True,
    ) -> None:
        """
        Upload an updated DataFrame to S3 with progress bar and error handling.

        Args:
            df (pd.DataFrame): DataFrame to upload.
            key (str): S3 key for the file.
            keyword (str): String identifier for the type of data (e.g., "survey", "site").
            bucket (str): The S3 bucket name, defaults to env defined bucket.
        """
        temp_filename = f"updated_{keyword}_kso_temp.csv"
        try:
            df.to_csv(temp_filename, index=keep_df_index)
            with tqdm(
                total=os.path.getsize(temp_filename),
                unit="B",
                unit_scale=True,
                desc=f"Uploading {keyword}",
            ) as pbar:
                self.s3.upload_file(
                    Filename=temp_filename,
                    Bucket=bucket,
                    Key=key,
                    Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
                )
            logging.info("Successfully uploaded updated %s data to S3", keyword)
        except BotoCoreError as e:
            logging.error("Failed to upload updated %s data to S3: %s", keyword, str(e))
        finally:
            delete_file(temp_filename)

    def get_set_filenames_from_s3(
        self, bucket: str = S3_BUCKET, prefix: str = "", suffixes: tuple = ()
    ) -> set[str]:
        """
        Retrieve a set of all object keys (file paths) in an S3 bucket under
        a given prefix, optionally filtering by file suffixes.

        Parameters:
            bucket (str): The S3 bucket name, defaults to env defined bucket.
            prefix (str, optional): Folder path within the bucket to filter
                objects. Defaults to "" (entire bucket).
            suffixes (tuple, optional): A tuple of lowercase file suffixes to
                filter object keys (e.g., ("mp4", "jpg")). If empty, all objects
                are returned regardless of suffix. Case-insensitive.

        Returns:
            set[str]: A set of S3 object keys (strings) matching the specified
                prefix and suffixes.
        """
        s3_filepaths = set()
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                # If there are no suffixes / if the path ends with one of the given suffixes
                if not suffixes or obj["Key"].lower().endswith(tuple(suffixes)):
                    s3_filepaths.add(obj["Key"])
        return s3_filepaths

    def rename_s3_objects_from_dict(
        self,
        rename_pairs: dict,
        prefix="",
        suffixes: Iterable = {},
        bucket=S3_BUCKET,
        try_run=False,
    ):
        files_from_aws = self.get_set_filenames_from_s3(bucket, prefix, suffixes)

        for old_name, new_name in rename_pairs.items():

            if old_name in files_from_aws:
                try:
                    if not try_run:
                        # Copy
                        self.s3.copy_object(
                            Bucket=bucket,
                            CopySource={"Bucket": bucket, "Key": old_name},
                            Key=new_name,
                        )
                        # Delete
                        self.s3.delete_object(Bucket=bucket, Key=old_name)
                except BotoCoreError as e:
                    logging.warning(
                        f"Failed to rename {old_name} to {new_name}, error: {str(e)}"
                    )
                logging.info(f"Renamed: {old_name} ➜ {new_name}")
            else:
                logging.info(f"File not found in the {S3_BUCKET} bucket: {old_name}.")

        logging.info("Rename complete")

    def read_df_from_s3_csv(
        self, csv_s3_path: str, s3_bucket: str = S3_BUCKET
    ) -> pd.DataFrame:
        """
        Downloads a CSV file from S3 and loads it into a pandas DataFrame.

        Parameters:
            csv_filename (str): The name of the CSV file (e.g., 'data.csv').
            csv_path (str): The S3 key prefix/path (e.g., 'folder/subfolder').
            s3_bucket (str): The name of the S3 bucket, defaults to env defined bucket.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        # Download the object to memory
        response = self.s3.get_object(Bucket=s3_bucket, Key=csv_s3_path)

        return pd.read_csv(io.BytesIO(response["Body"].read()))
