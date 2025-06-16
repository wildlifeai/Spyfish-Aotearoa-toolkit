import logging
import os
from typing import Any, Iterable, Optional

from sftk.common import S3_BUCKET, S3_SHAREPOINT_PATH
from sftk.s3_handler import S3Handler
from sftk.utils import filter_file_paths_by_extension, get_unique_entries_df_column


def get_mismatched_video_files_info(
    csv_filename: str,
    csv_column: str,
    valid_extensions: Iterable[str],
    output_files: bool = True,
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

    s3_handler = S3Handler()
    csv_s3_path = os.path.join(S3_SHAREPOINT_PATH, csv_filename)
    logging.info(f"Processing CSV: {csv_s3_path}.")

    # Load dataframe from AWS
    csv_df = s3_handler.read_df_from_s3_csv(csv_s3_path, S3_BUCKET)

    # Load unique file paths from the CSV column excluding the filtered values
    csv_filepaths_without_filtered_values = get_unique_entries_df_column(
        csv_df,
        csv_column,
        column_filter=column_filter,
        column_value=column_value,
    )
    # Load all unique file paths from the CSV column
    csv_filepaths_all = get_unique_entries_df_column(csv_df, csv_column)

    logging.info(f"Unique file paths from CSV: {len(csv_filepaths_all)}.")
    logging.info(
        f"Unique file paths from CSV, without filtered value {column_filter} as {column_value}: {len(csv_filepaths_without_filtered_values)}."
    )

    logging.info(f"Processing the files in the bucket: {S3_BUCKET}.")
    # Get all file paths currently in S3
    s3_filepaths = s3_handler.get_set_filenames_from_s3(bucket=S3_BUCKET)

    # Filter only video files based on their extension
    s3_video_filepaths = set(
        filter_file_paths_by_extension(s3_filepaths, valid_extensions)
    )
    logging.info(f"Video files in S3: {len(s3_video_filepaths)}")

    # Find missing files in S3 (referenced in CSV but not found in S3)
    missing_files_in_aws = csv_filepaths_without_filtered_values - s3_video_filepaths
    logging.info(f"Missing video files in AWS: {len(missing_files_in_aws)}")

    if output_files:
        logging.info(
            "Creating file missing_files_in_aws.txt, containing file names of "
            "videos referenced in CSV but not found in S3."
        )
        with open("missing_files_in_aws.txt", "w") as f:
            f.write("\n".join(sorted(missing_files_in_aws)))

    # Find extra files in S3 (present in S3 but not referenced in CSV)
    extra_files_in_aws = s3_video_filepaths - csv_filepaths_all
    logging.info(f"Extra video files in AWS: {len(extra_files_in_aws)}.")

    if output_files:
        logging.info(
            "Creating file extra_files_in_aws.txt, containing file names of "
            "videos present in S3 but not referenced in the CSV."
        )

        with open("extra_files_in_aws.txt", "w") as f:
            f.write("\n".join(sorted(extra_files_in_aws)))


if __name__ == "__main__":
    logging.info("Starting mismatched_video_file_info processing.")

    csv_filename = "BUV Deployment.csv"
    csv_column_to_extract = "LinkToVideoFile"

    valid_extensions = (
        "avi",
        "wmv",
        "mp4",
        "mov",
        "mpg",
    )  # avi and wmv not found in bucket,

    get_mismatched_video_files_info(
        csv_filename,
        csv_column_to_extract,
        valid_extensions,
        column_filter="IsBadDeployment",
        column_value=False,
    )

    logging.info("mismatched_video_file_info processing completed.")
