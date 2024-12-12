"""
This module processes and transforms SharePoint CSV data for compatibility
with the KSO software, storing the results in S3.

Steps:
    - Connect to S3.
    - Download csv files with SharePoint- and KSO-formatted data
    - Check new or different data from latest sharepoint copy
    - Update KSO-formatted data with latest sharepoint copy data
"""

import os
import logging
from typing import Optional, cast
from dataclasses import dataclass
from contextlib import contextmanager
import boto3

from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Load environment variables from .env file only in local environment
if os.environ.get("GITHUB_ACTIONS") != "true":
    load_dotenv()

# S3 configuration
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

S3_SHAREPOINT_MOVIE_CSV = os.getenv("S3_SHAREPOINT_MOVIE_CSV")
S3_SHAREPOINT_SURVEY_CSV = os.getenv("S3_SHAREPOINT_SURVEY_CSV")
S3_SHAREPOINT_SITE_CSV = os.getenv("S3_SHAREPOINT_SITE_CSV")
S3_SHAREPOINT_SPECIES_CSV = os.getenv("S3_SHAREPOINT_SPECIES_CSV")
S3_KSO_MOVIE_CSV = os.getenv("S3_KSO_MOVIE_CSV")
S3_KSO_SURVEY_CSV = os.getenv("S3_KSO_SURVEY_CSV")
S3_KSO_SITE_CSV = os.getenv("S3_KSO_SITE_CSV")
S3_KSO_ANNOTATIONS_CSV = os.getenv("S3_KSO_ANNOTATIONS_CSV")
S3_KSO_SPECIES_CSV = os.getenv("S3_KSO_SPECIES_CSV")


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


class S3FileNotFoundError(Exception):
    """Custom exception for S3 file not found scenarios."""


class EnvironmentVariableError(Exception):
    """Custom exception for missing environment variables."""


@contextmanager
def temp_file_manager(filenames: list[str]):
    """Context manager to handle temporary file cleanup."""
    try:
        yield
    finally:
        for filename in filenames:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                logging.error(
                    "Failed to remove temporary file %s: %s", filename, str(e)
                )


def get_s3_client() -> boto3.client:
    """
    Creates and returns an S3 client.

    Returns:
        boto3.client: The S3 client.
    """
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def download_object_from_s3(
    client: boto3.client, bucket: str, key: str, filename: str, version_id: str = None
) -> None:
    """
    Downloads an object from S3 with progress bar and error handling.

    Args:
        client (boto3.client): The S3 client.
        bucket (str): The S3 bucket name.
        key (str): The S3 object key.
        filename (str): The local filename to save the object to.
        version_id (str, optional): The version ID of the object. Defaults to None.
    """
    try:
        kwargs = {"Bucket": bucket, "Key": key}
        if version_id:
            kwargs["VersionId"] = version_id

        object_size = client.head_object(**kwargs)["ContentLength"]

        def progress_update(bytes_transferred):
            pbar.update(bytes_transferred)

        with tqdm(total=object_size, unit="B", unit_scale=True, desc=filename) as pbar:
            client.download_file(
                Bucket=bucket,
                Key=key,
                Filename=filename,
                Callback=progress_update,
                Config=boto3.s3.transfer.TransferConfig(use_threads=False),
            )
    except Exception as e:
        logging.error("Failed to download %s from S3: %s", key, e)
        raise


def get_s3_file_config(keyword: str) -> S3FileConfig:
    """
    Creates a configuration object for S3 file operations based on
    a keyword (e.g., "survey", "site").

    Args:
        keyword(str): String identifier for the type of data

    Returns:
        S3FileConfig: The configuration information for S3 client.
    """

    kso_env_var = f"S3_KSO_{keyword.upper()}_CSV"
    sharepoint_env_var = f"S3_SHAREPOINT_{keyword.upper()}_CSV"

    kso_path = os.getenv(kso_env_var)
    sharepoint_path = os.getenv(sharepoint_env_var)

    if not kso_path:
        raise EnvironmentVariableError(f"Environment variable '{kso_env_var}' not set.")
    if not sharepoint_path:
        raise EnvironmentVariableError(
            f"Environment variable '{sharepoint_env_var}' not set."
        )

    return S3FileConfig(
        keyword=keyword,
        kso_env_var=kso_env_var,
        sharepoint_env_var=sharepoint_env_var,
        kso_filename=f"{keyword}_kso_temp.csv",
        sharepoint_filename=f"{keyword}_sharepoint_temp.csv",
    )


def get_env_var(name: str) -> str:
    """
    Gets an environment variable and raises an error if not found.

    Args:
        name: The name of the environment variable.

    Returns:
        The value of the environment variable.

    Raises:
        EnvironmentVariableError: If the environment variable is not set.
    """
    value = os.getenv(name)
    if value is None:
        raise EnvironmentVariableError(f"Environment variable '{name}' not set.")
    return cast(str, value)


def download_and_read_s3_file(
    s3_client, bucket: str, key: str, filename: str
) -> Optional[pd.DataFrame]:
    """
    Downloads an S3 object and reads it into a Pandas DataFrame.

    Args:
        s3_client: The S3 client.
        bucket: The S3 bucket name.
        key: The S3 object key.
        filename: The local filename to save the downloaded object.

    Returns:
        The DataFrame read from the downloaded file, or None if an error occurs.

    Raises:
        S3FileNotFoundError: If the S3 object is not found.
        Other exceptions: If other errors occur during download or reading.
    """

    try:
        download_object_from_s3(
            client=s3_client, bucket=bucket, key=key, filename=filename
        )
        return pd.read_csv(filename)
    except Exception as e:
        logging.warning("Failed to process S3 file %s: %s", key, str(e))
        raise S3FileNotFoundError(f"Failed to process file {key}: {str(e)}")


def process_s3_files(
    s3_client, keywords: list[str], bucket: str
) -> dict[str, tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """
    Downloads and reads KSO and SharePoint CSV files from S3 for given keywords.

    Args:
        s3_client: The S3 client.
        keywords: A list of keywords to process.
        bucket: The S3 bucket name.

    Returns:
        A dictionary mapping keywords to tuples of (KSO DataFrame, SharePoint DataFrame).
    """
    results = {}

    for keyword in keywords:
        kso_df = None
        sharepoint_df = None

        try:
            config = get_s3_file_config(keyword)

            with temp_file_manager([config.kso_filename, config.sharepoint_filename]):
                try:
                    # Download and read KSO file
                    kso_df = download_and_read_s3_file(
                        s3_client,
                        bucket,
                        get_env_var(config.kso_env_var),
                        config.kso_filename,
                    )

                    # Download and read Sharepoint file
                    sharepoint_df = download_and_read_s3_file(
                        s3_client,
                        bucket,
                        get_env_var(config.sharepoint_env_var),
                        config.sharepoint_filename,
                    )

                except S3FileNotFoundError as e:
                    logging.warning(
                        "CSV file not found in S3 for keyword %s: %s", keyword, str(e)
                    )

        except EnvironmentVariableError as e:
            logging.error(
                "Environment variable error for keyword %s: %s", keyword, str(e)
            )
        except Exception as e:
            logging.error("Unexpected error processing keyword %s: %s", keyword, str(e))

        results[keyword] = (kso_df, sharepoint_df)

    return results


def validate_and_filter_dfs(
    movie_sharepoint_df: pd.DataFrame, site_sharepoint_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Check for missing values in specified columns of movie and site DataFrames,
    log the missing values, and filter them out.

    Args:
        movie_sharepoint_df: DataFrame containing movie SharePoint data.
        site_sharepoint_df: DataFrame containing site SharePoint data.

    Returns:
        Tuple of filtered movie_sharepoint_df and site_sharepoint_df.
    """
    try:
        # Columns to validate for each DataFrame
        movie_required_columns = [
            "DropID",
            "SurveyID",
            "SiteID",
            "Latitude",
            "Longitude",
        ]
        site_required_columns = ["SiteID", "LinkToMarineReserve"]

        # Validate movie DataFrame
        if movie_sharepoint_df is not None:
            missing_movie = movie_sharepoint_df[
                movie_sharepoint_df[movie_required_columns].isna().any(axis=1)
            ]

            if not missing_movie.empty:
                logging.info(
                    f"The following {len(missing_movie)} rows with "
                    f"missing values in the {movie_required_columns} "
                    "required columns of the movie SharePoint list copy"
                    " will be dropped."
                )

                logging.info(missing_movie)
                # Filter out rows with missing values
                movie_sharepoint_df = movie_sharepoint_df.dropna(
                    subset=movie_required_columns
                )

        else:
            logging.info("No missing values found for movie SharePoint list")
        # Validate site DataFrame
        if site_sharepoint_df is not None:
            missing_site = site_sharepoint_df[
                site_sharepoint_df[site_required_columns].isna().any(axis=1)
            ]

            if not missing_site.empty:
                logging.info(
                    f"Found the following {len(missing_site)} rows with "
                    f"missing values in the {site_required_columns} "
                    "required columns of the site SharePoint list copy."
                )
                logging.info(missing_site)
                # Filter out rows with missing values
                site_sharepoint_df = site_sharepoint_df.dropna(
                    subset=site_required_columns
                )
        else:
            logging.info("No missing values found for site SharePoint list")

        return movie_sharepoint_df, site_sharepoint_df

    except Exception as e:
        logging.error(f"Error occurred during DataFrame validation: {e}", exc_info=True)
        raise


def standarise_sharepoint_to_kso(
    results: dict[str, tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]
) -> dict[str, tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """
    Standardise and integrate data from SharePoint copies to align with KSO data structure.

    This function combines latitude and longitude columns from the `movie` SharePoint DataFrame
    into the `site` SharePoint DataFrame based on the `SiteID` column.

    Args:
        results: Dictionary containing tuples of (kso_df, sharepoint_df) for each keyword.

    Returns:
        Updated dictionary with standardised SharePoint DataFrames.
    """
    try:
        # Extract relevant DataFrames
        movie_sharepoint_df = results.get("movie", (None, None))[1]
        site_sharepoint_df = results.get("site", (None, None))[1]

        if movie_sharepoint_df is None or site_sharepoint_df is None:
            logging.warning(
                "Movie or Site SharePoint DataFrame is missing. Skipping standardization."
            )
            return results

        logging.info("Checking for potential missing values from movies and sites.")
        # Filter out empty rows for critical values
        movie_sharepoint_df, site_sharepoint_df = validate_and_filter_dfs(
            movie_sharepoint_df, site_sharepoint_df
        )

        logging.info(
            "Overwritting site coordinates from site sharepoint copy with actual drop"
            " coordinates from deployment sharepoint copy."
        )

        # Ensure the relevant columns exist
        if not {"SiteID", "Latitude", "Longitude"}.issubset(
            movie_sharepoint_df.columns
        ):
            logging.warning(
                "Required columns ('SiteID', 'Latitude', 'Longitude')\
                are missing in movie DataFrame."
            )
            return results

        if "SiteID" not in site_sharepoint_df.columns:
            logging.warning("Required column 'SiteID' is missing in site DataFrame.")
            return results

        # Drop latitude and longitude from site DataFrame to avoid duplicates
        site_sharepoint_df = site_sharepoint_df.drop(columns=["Latitude", "Longitude"])

        # Rename SiteID from movie and site DataFrame to avoid duplicates
        movie_sharepoint_df, site_sharepoint_df = (
            df.rename(columns={"SiteID": "orig_SiteID"})
            for df in (movie_sharepoint_df, site_sharepoint_df)
        )

        # Extract the year from Survey
        movie_sharepoint_df["year"] = movie_sharepoint_df["SurveyID"].str[6:8]

        # Create new SiteID on movie DataFrame based on orig SiteID and year of the survey
        movie_sharepoint_df["SiteID"] = (
            movie_sharepoint_df["orig_SiteID"].astype(str)
            + "_"
            + movie_sharepoint_df["year"]
        )

        # Merge latitude and longitude into site DataFrame
        lat_long_df = movie_sharepoint_df[
            ["orig_SiteID", "SiteID", "Latitude", "Longitude"]
        ].drop_duplicates(subset="SiteID")
        site_sharepoint_df = site_sharepoint_df.merge(
            lat_long_df, on="orig_SiteID", how="right"
        )

        # Drop latitude and longitude from site DataFrame to avoid duplicates
        site_sharepoint_df = site_sharepoint_df.drop(columns=["orig_SiteID"])
        movie_sharepoint_df = movie_sharepoint_df.drop(
            columns=["orig_SiteID", "year", "Longitude", "Latitude"]
        )

        # Update the results dictionary with the modified site DataFrame
        results["site"] = (results["site"][0], site_sharepoint_df)
        results["movie"] = (results["movie"][0], movie_sharepoint_df)

        logging.info("Movie/BUV Drop coordinates have been successfully updated.")
        return results

    except Exception as e:
        logging.error(
            "Error occurred in standarise_sharepoint_to_kso: %s", str(e), exc_info=True
        )
        raise


def validate_dataframes(
    kso_df: pd.DataFrame, sharepoint_df: pd.DataFrame, unique_id_column: str
) -> list:
    """
    Validate input DataFrames and the presence and uniqueness of the specified identifier column.

    Args:
        kso_df: KSO DataFrame
        sharepoint_df: Sharepoint DataFrame
        unique_id_column: Name of the column used as a unique identifier for data records.

    Returns:
        A list of validation errors encountered, empty if no issues found.
    """
    validation_log = []

    # Check for None DataFrames
    if kso_df is None or sharepoint_df is None:
        raise ValueError("Cannot compare DataFrames: One or both DataFrames are None")

    # Validate unique identifier column presence
    if unique_id_column not in kso_df.columns:
        validation_log.append(f"Column '{unique_id_column}' is missing in kso_df.")

    if unique_id_column not in sharepoint_df.columns:
        validation_log.append(
            f"Column '{unique_id_column}' is missing in sharepoint_df."
        )

    # Check for duplicates in unique identifier column
    if kso_df[unique_id_column].duplicated().any():
        validation_log.append(
            "kso_df has duplicate values in the unique identifier column."
        )

    if sharepoint_df[unique_id_column].duplicated().any():
        validation_log.append(
            "sharepoint_df has duplicate values in the unique identifier column."
        )

    return validation_log


def align_dataframe_columns(kso_df: pd.DataFrame, sharepoint_df: pd.DataFrame) -> None:
    """
    Align and preprocess DataFrame columns by identifying and handling missing or extra columns.

    Args:
        kso_df: KSO DataFrame to modify.
        sharepoint_df: Sharepoint DataFrame to modify.
    """

    # Identify column mismatches
    missing_columns = set(kso_df.columns) - set(sharepoint_df.columns)
    extra_columns = set(sharepoint_df.columns) - set(kso_df.columns)

    # Handle extra columns
    if extra_columns:
        logging.info(
            f"Dropping extra columns from SharePoint DataFrame: {extra_columns}"
        )
        sharepoint_df.drop(columns=list(extra_columns), inplace=True)

    # Handle missing columns
    if missing_columns:
        logging.info(
            f"Adding missing columns to SharePoint DataFrame: {missing_columns}"
        )
        for col in missing_columns:
            sharepoint_df[col] = None


def upload_updated_df_to_s3(
    s3_client: boto3.client, df: pd.DataFrame, bucket: str, key: str, keyword: str
) -> None:
    """
    Upload an updated DataFrame to S3 with progress bar and error handling.

    Args:
        s3_client: Boto3 S3 client.
        df: DataFrame to upload.
        bucket: S3 bucket name.
        key: S3 key for the file.
        keyword: String identifier for the type of data (e.g., "survey", "site").
    """
    temp_filename = f"updated_{keyword}_kso_temp.csv"
    try:
        df.to_csv(temp_filename, index=True)
        with tqdm(
            total=os.path.getsize(temp_filename),
            unit="B",
            unit_scale=True,
            desc=f"Uploading {keyword}",
        ) as pbar:
            s3_client.upload_file(
                Filename=temp_filename,
                Bucket=bucket,
                Key=key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
        logging.info("Successfully uploaded updated %s data to S3", keyword)
    except Exception as e:
        logging.error("Failed to upload updated %s data to S3: %s", keyword, str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def compare_and_update_dataframes(
    kso_df: pd.DataFrame,
    sharepoint_df: pd.DataFrame,
    s3_client: boto3.client,
    bucket: str,
    keyword: str,
) -> pd.DataFrame:
    """
    Compare KSO and Sharepoint DataFrames, log differences, and update the KSO DataFrame
    based on the Sharepoint data. Uploads the updated KSO DataFrame to S3 if changes were made.

    Args:
        kso_df: DataFrame containing KSO data.
        sharepoint_df: DataFrame containing Sharepoint data.
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        keyword: String identifier for the type of data (e.g., "survey", "site").

    """

    # Configuration for different data types
    keyword_lookup = {
        "survey": "SurveyID",
        "site": "SiteID",
        "movie": "DropID",
        "species": "species_id",
    }

    column_mappings = {
        "survey": {},
        "site": {"ID": "schema_site_id"},
        "movie": {},
        "species": {"DOC_TaxonID": "species_id"},
    }

    # Validate input
    if keyword not in keyword_lookup or keyword not in column_mappings:
        raise ValueError(f"Unknown keyword: {keyword}")

    unique_id_column = keyword_lookup[keyword]

    # Rename the column names in sharepoint_df based on the dictionary
    sharepoint_df.rename(columns=column_mappings[keyword], inplace=True)

    # Log initial DataFrame information
    logging.info("Initial DataFrame sizes:")
    logging.info(f"KSO DataFrame: {kso_df.shape[0]} rows")
    logging.info(f"Sharepoint DataFrame: {sharepoint_df.shape[0]} rows")

    # Validate DataFrames
    validation_issues = validate_dataframes(kso_df, sharepoint_df, unique_id_column)
    if validation_issues:
        logging.error("Validation issues found:")
        for issue in validation_issues:
            logging.error("- %s", issue)

    # Identify and handle column mismatches
    missing_columns = set(kso_df.columns) - set(sharepoint_df.columns)
    extra_columns = set(sharepoint_df.columns) - set(kso_df.columns)

    if extra_columns:
        logging.info(
            f"Dropping extra columns from SharePoint DataFrame: {extra_columns}"
        )
        sharepoint_df.drop(columns=list(extra_columns), inplace=True)

    if missing_columns:
        logging.info(
            f"Adding missing columns to SharePoint DataFrame: {missing_columns}"
        )
        for col in missing_columns:
            sharepoint_df[col] = None

    # Align columns
    align_dataframe_columns(kso_df, sharepoint_df)

    # Set unique identifier as index
    kso_df.set_index(unique_id_column, inplace=True)
    sharepoint_df.set_index(unique_id_column, inplace=True)

    # Find rows in sharepoint_df that are not in kso_df
    new_rows = sharepoint_df.loc[~sharepoint_df.index.isin(kso_df.index)]

    if not new_rows.empty:
        logging.info(f"Adding {len(new_rows)} new rows from Sharepoint")
        kso_df = pd.concat([kso_df, new_rows])

    # Find rows in kso_df that are not in sharepoint_df
    missing_rows = kso_df.loc[~kso_df.index.isin(sharepoint_df.index)]

    if not missing_rows.empty:
        logging.info(
            f"Temporarily adding {len(new_rows)} new rows from kso_df. Consider updating the Sharepoint list"
        )
        sharepoint_df = pd.concat([sharepoint_df, missing_rows])

    # Ensure consistent column set
    common_columns = list(set(kso_df.columns) & set(sharepoint_df.columns))

    # Create a copy of KSO DataFrame to modify
    updated_kso_df = kso_df.copy()

    # Prepare DataFrames for comparison
    # 1. Use only common columns
    kso_compare = kso_df[common_columns]
    sharepoint_compare = sharepoint_df[common_columns]

    # 2. Combine indexes to create a unified index
    all_indexes = sorted(set(kso_df.index) | set(sharepoint_df.index))

    # 3. Reindex both DataFrames with the combined index
    kso_reindexed = kso_compare.reindex(all_indexes)
    sharepoint_reindexed = sharepoint_compare.reindex(all_indexes)

    dataframe_changed = False

    # Compare dataframes
    try:
        # Compare DataFrames with identical labels
        differences = kso_reindexed.compare(sharepoint_reindexed, keep_equal=False)

        # Process differences
        if not differences.empty:
            detailed_differences = []

            # Iterate through differences
            for idx in differences.index:
                for col in common_columns:
                    # Check if there's a difference in this column
                    kso_val = kso_reindexed.loc[idx, col]
                    sharepoint_val = sharepoint_reindexed.loc[idx, col]

                    # Log and track differences
                    if pd.notna(sharepoint_val) and (
                        pd.isna(kso_val) or kso_val != sharepoint_val
                    ):
                        dataframe_changed = True
                        detailed_differences.append(
                            {
                                "index": idx,
                                "column": col,
                                "old_value": kso_val,
                                "new_value": sharepoint_val,
                            }
                        )

                        # Update the value in the original DataFrame
                        updated_kso_df.loc[idx, col] = sharepoint_val

                        # Detailed logging
                        logging.info(
                            f"Difference in {keyword} DataFrame: "
                            f"Row {idx}, Column {col}: "
                            f"Changed from '{kso_val}' "
                            f"to '{sharepoint_val}'"
                        )

            # Log summary of differences
            logging.info(f"Total differences found: {len(detailed_differences)}")

    except Exception as e:
        logging.error(f"Error comparing DataFrames for {keyword}: {e}")
        return

    # Check for new rows
    if not new_rows.empty:
        dataframe_changed = True

    # Upload to S3 only if changes were made
    if dataframe_changed:
        # Upload to S3
        try:
            config = get_s3_file_config(keyword)
            upload_updated_df_to_s3(
                s3_client,
                updated_kso_df,
                bucket,
                get_env_var(config.kso_env_var),
                keyword,
            )
            logging.info(f"Updated {keyword} DataFrame uploaded to S3")
        except Exception as e:
            logging.error(f"Error uploading updated DataFrame to S3: {e}")
    else:
        logging.info(f"No differences found in {keyword} DataFrame. No updates needed.")


def main():
    """Main function to process each sharepoint list and update KSO data if needed."""
    logging.info("Starting main function")
    keywords = ["survey", "site", "movie", "species"]
    logging.info(f"Processing csv files in S3 with the following keywords: {keywords}")

    try:
        logging.info("Initializing S3 client...")
        s3_client = get_s3_client()
        logging.info("S3 client initialized successfully")

        logging.info("Download and check S3 files...")
        results = process_s3_files(s3_client, keywords, S3_BUCKET)
        logging.info("S3 files were succesfully downloaded and checked.")

        logging.info(
            "Formatting movie and site info from sharepoint copy to match KSO requirements..."
        )
        results = standarise_sharepoint_to_kso(results)
        logging.info("movie and site info has been succesfully formatted")

        for keyword, (kso_df, sharepoint_df) in results.items():
            logging.info(f"Processing {keyword} data...")
            if kso_df is None:
                logging.info("KSO DataFrame is not available.")
            if sharepoint_df is None:
                logging.info("Sharepoint DataFrame is not available.")

            if kso_df is not None and sharepoint_df is not None:
                # Compare DataFrames and update and upload if different
                compare_and_update_dataframes(
                    kso_df=kso_df,
                    sharepoint_df=sharepoint_df,
                    s3_client=s3_client,
                    bucket=S3_BUCKET,
                    keyword=keyword,
                )

            else:
                logging.warning(
                    f"Skipping comparison for {keyword}: "
                    "One or both DataFrames are None"
                )

    except Exception as e:
        logging.error(
            "An error occurred during processing of sharepoint list info: %s",
            str(e),
            exc_info=True,
        )


if __name__ == "__main__":
    # Ensure logging is configured at the start
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,  # This will override any existing logging configuration
    )
    logging.info("Script started")
    main()
    logging.info("Script completed")
