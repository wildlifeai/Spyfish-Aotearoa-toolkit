import os
import logging
from typing import Dict
import boto3
from dotenv import load_dotenv
import pandas as pd


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("s3_data_processing.log")],
)
logger = logging.getLogger(__name__)


# S3 configuration
def load_aws_credentials(env_path):
    """
    Load AWS credentials from .env file.

    Args:
        env_path (str, optional): Path to .env file.
                                  If None, uses default dotenv behavior.

    Returns:
        tuple: AWS access key and secret key
    """
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    return (
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        os.getenv("S3_BUCKET"),
    )


def create_s3_client(access_key, secret_key):
    """
    Create and return an S3 client.

    Args:
        access_key (str): AWS access key
        secret_key (str): AWS secret key

    Returns:
        boto3.client: S3 client
    """
    try:
        return boto3.client(
            "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        raise ValueError("Unable to create S3 client. Check AWS credentials.")


def read_csv_from_s3(client, bucket, key):
    """
    Read a CSV file from S3 and return as a pandas DataFrame.

    Args:
        client (boto3.client): S3 client
        bucket (str): S3 bucket name
        key (str): S3 object key

    Returns:
        pd.DataFrame: DataFrame from CSV file
    """
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"], low_memory=False)

    except Exception as e:
        logger.error(f"Failed to process S3 file {key}: {e}")
        raise IOError(f"Could not download or read file {key}")


def process_annotations_dataframe(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Process annotations dataframe with comprehensive error handling and logging.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary of dataframes

    Returns:
        pd.DataFrame: Processed annotations dataframe
    """
    try:
        # Calculate the max count per movie, annotator and species
        max_count_df = (
            dataframes["annotations"]
            .groupby(["DropID", "annotatedBy", "scientificName"], as_index=False)[
                "maxinterval"
            ]
            .max()
        )

        # Create a DataFrame with all possible combinations
        all_combinations = pd.MultiIndex.from_product(
            [
                max_count_df["DropID"].unique(),
                max_count_df["annotatedBy"].unique(),
                max_count_df["scientificName"].unique(),
            ],
            names=["DropID", "annotatedBy", "scientificName"],
        )
        all_combinations_df = pd.DataFrame(index=all_combinations).reset_index()

        # Merge the original DataFrame with all combinations
        comb_max_count_df = all_combinations_df.merge(
            max_count_df, how="left", on=["DropID", "annotatedBy", "scientificName"]
        )

        # Comprehensive merge with additional error handling
        annotations_df = (
            comb_max_count_df.merge(dataframes["movies"], on="DropID", how="left")
            .merge(dataframes["sites"], on="SiteID", how="left")
            .merge(dataframes["surveys"], on="SurveyID", how="left")
            .merge(dataframes["species"], on="scientificName", how="left")
        )

        return annotations_df
    except Exception as e:
        logger.error(f"Failed to process annotations dataframe: {e}")
        raise


def main(env_path=None):
    """
    Main function to orchestrate S3 file download and processing.
    """
    # CSV file keys
    csv_keys = {
        "annotations": "spyfish_metadata/kso_csvs/annotations_buv_doc.csv",
        "movies": "spyfish_metadata/kso_csvs/movies_buv_doc.csv",
        "sites": "spyfish_metadata/kso_csvs/sites_buv_doc.csv",
        "surveys": "spyfish_metadata/kso_csvs/surveys_buv_doc.csv",
        "species": "spyfish_metadata/kso_csvs/species_buv_doc.csv",
    }

    # Load AWS credentials and create S3 client
    try:
        access_key, secret_key, bucket_name = load_aws_credentials(env_path)
        logger.info(f"AWS Credentials loaded. Bucket: {bucket_name}")
    except Exception as cred_error:
        logger.error(f"Credential loading failed: {cred_error}")
        raise

    # Create S3 client
    try:
        s3_client = create_s3_client(access_key, secret_key)
    except Exception as client_error:
        logger.error(f"S3 client creation failed: {client_error}")
        raise

    # Read CSV files from S3 and store in a dictionary of DataFrames
    dataframes = {}
    for name, key in csv_keys.items():
        try:
            logger.info(f"Attempting to read {name} from S3")
            dataframes[name] = read_csv_from_s3(s3_client, bucket_name, key)
            logger.info(f"Successfully read {name}. Shape: {dataframes[name].shape}")
        except Exception as read_error:
            logger.error(f"Failed to read {name}: {read_error}")
            raise

    # Process the annotations dataframe using the dedicated function
    try:
        processed_annotations_df = process_annotations_dataframe(dataframes)
        logger.info(
            f"Successfully processed annotations dataframe. Shape: {processed_annotations_df.shape}"
        )

        # Export processed annotations to CSV
        processed_annotations_df.to_csv("processed_annotations.csv", index=False)
        logger.info(
            "Processed annotations dataframe saved to 'processed_annotations.csv'"
        )
        movies_df = dataframes["movies"]
        sites_df = dataframes["sites"]
        surveys_df = dataframes["surveys"]
        species_df = dataframes["species"]

        return processed_annotations_df, movies_df, sites_df, surveys_df, species_df

    except Exception as process_error:
        logger.error(f"Failed to process annotations dataframe: {process_error}")
        raise
