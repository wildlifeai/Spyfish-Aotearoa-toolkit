"""
BIIGLE API Handler for Spyfish Aotearoa project.

This module provides functionality to interact with the BIIGLE API for:
- Creating and managing volumes
- Setting up label trees with scientific names
- Exporting and processing annotations
- Integrating with S3 for video file management

Example usage:
    # Initialize handler
    biigle_handler = BiigleHandler()

    # Get projects
    projects = biigle_handler.get_projects()

    # Create volume with files from S3
    volume = biigle_handler.create_volume_from_s3_files(
        project_id=3711,
        volume_name="My Volume",
        s3_url=biigle_handler.build_s3_url("biigle_clips/my_folder/"),
        files=["video1.mp4", "video2.mp4"]
    )


    # Export annotations
    annotations_df = biigle_handler.export_annotations(volume_id=12345)
"""

import glob
import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from sftk.common import (
    BIIGLE_ANNOTATION_REPORT_TYPE,
    BIIGLE_DISK_ID,
    BIIGLE_PROJECT_ID,
    LOCAL_DATA_FOLDER_PATH,
)
from sftk.external.biigle_api import Api


class BiigleHandler:
    """Handler for BIIGLE API operations."""

    def __init__(self, email: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize BiigleHandler with API credentials.


        Raises:
            ValueError: If credentials are not provided and not found in environment.
        """
        self.email = email
        self.token = token

        # Import biigle API here to avoid import errors if not installed
        try:
            self.api = Api(self.email, self.token)
        except ImportError as e:
            raise ImportError(
                "biigle package is required. Install from https://github.com/biigle/community-resources"
            ) from e
        except Exception as e:
            raise Exception(f"Failed to initialize BIIGLE API: {e}") from e

        logging.info("BiigleHandler initialized successfully")

    def get_projects(self) -> List[Dict[str, Any]]:
        """
        Get all projects that the user can access.

        Returns:
            List[Dict[str, Any]]: List of project dictionaries with id, name, description, etc.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response = self.api.get("projects")
            projects = response.json()
            logging.info(f"Retrieved {len(projects)} projects")
            return projects
        except Exception as e:
            logging.error(f"Failed to get projects: {e}")
            raise

    def create_pending_volume(
        self, project_id: int = BIIGLE_PROJECT_ID, media_type: str = "video"
    ) -> Dict[str, Any]:
        """
        Create a pending volume in a project.

        Args:
            project_id: The ID of the project to create the volume in.
            media_type: The type of media for the volume (default: "video").

        Returns:
            Dict[str, Any]: The created pending volume information.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response = self.api.post(
                f"projects/{project_id}/pending-volumes",
                json={"media_type": media_type},
            )
            pending_volume = response.json()
            logging.info(f"Created pending volume with ID: {pending_volume['id']}")
            return pending_volume
        except Exception as e:
            logging.error(f"Failed to create pending volume: {e}")
            raise

    def setup_volume_with_files(
        self, pending_volume_id: int, volume_name: str, s3_url: str, files: List[str]
    ) -> Dict[str, Any]:
        """
        Configure a pending volume with name, S3 URL, and file list.

        Args:
            pending_volume_id: The ID of the pending volume to configure.
            volume_name: The name for the volume.
            s3_url: The S3 URL where the files are located.
            files: List of file names to include in the volume.

        Returns:
            Dict[str, Any]: The configured volume information.

        Raises:
            Exception: If the API call fails.
        """
        try:
            payload = {"name": volume_name, "url": s3_url, "files": files}
            response = self.api.put(
                f"pending-volumes/{pending_volume_id}", json=payload
            )
            volume_info = response.json()
            logging.info(f"Configured volume '{volume_name}' with {len(files)} files")
            return volume_info
        except Exception as e:
            logging.error(f"Failed to setup volume with files: {e}")
            raise

    def create_label_tree(
        self,
        csv_path: str,
        tree_name: str,
        tree_description: str,
        project_id: int = BIIGLE_PROJECT_ID,
    ) -> Dict[str, Any]:
        """
        Create a label tree and populate it with scientific names from a CSV file.

        Args:
            project_id: The ID of the project to create the label tree in.
            csv_path: Path to CSV file containing label data (name, color, source_id columns).
            tree_name: Name for the label tree.
            tree_description: Description for the label tree.

        Returns:
            Dict[str, Any]: Dictionary containing tree_id and labels information.

        Raises:
            Exception: If the API call fails or CSV processing fails.
        """
        try:
            # Create the label tree
            tree_config = {
                "name": tree_name,
                "description": tree_description,
                "visibility_id": 2,  # Private by default, can change later in the label tree settings.
                "project_id": project_id,
            }

            tree_response = self.api.post("label-trees", json=tree_config)
            tree_info = tree_response.json()
            tree_id = tree_info["id"]
            logging.info(f"Created label tree '{tree_name}' with ID: {tree_id}")

            # Read labels from CSV and add them to the tree
            labels_df = pd.read_csv(csv_path)
            created_labels = {}
            count = 0

            for idx, row in labels_df.iterrows():
                count += 1
                if count == 5:
                    break
                label_config = {
                    "name": row["name"],
                    "color": row["color"],
                    "source_id": row["source_id"],
                    "label_source_id": 999,
                }

                label_response = self.api.post(
                    f"label-trees/{tree_id}/labels", json=label_config
                )
                label_info = label_response.json()
                created_labels[row["name"]] = label_info

            logging.info(f"Added {len(created_labels)} labels to tree '{tree_name}'")

            return {
                "tree_id": tree_id,
                "tree_info": tree_info,
                "labels": created_labels,
            }

        except Exception as e:
            logging.error(f"Failed to create label tree: {e}")
            raise

    def export_annotations(
        self,
        volume_id: int,
        extract_dir: str = LOCAL_DATA_FOLDER_PATH,
        type_id: int = BIIGLE_ANNOTATION_REPORT_TYPE,
    ) -> pd.DataFrame:
        """
        Export annotations from a volume and return as DataFrame.

        Args:
            volume_id: The ID of the volume to export annotations from.
            type_id: The type ID for the report (default: 8 for annotations).
            extract_dir: Directory to extract the downloaded files to.

        Returns:
            pd.DataFrame: DataFrame containing the annotation data.

        Raises:
            Exception: If the API call fails or file processing fails.
        """
        try:
            # Create annotation report
            report_response = self.api.post(
                f"volumes/{volume_id}/reports", json={"type_id": type_id}
            )
            report_info = report_response.json()
            report_id = report_info["id"]
            logging.info(f"Created annotation report with ID: {report_id}")

            # Download the report
            download_response = self.api.get(f"reports/{report_id}")

            export_path = Path(extract_dir, "biigle_exports")
            # Create extraction directory
            os.makedirs(export_path, exist_ok=True)
            # Save and extract the ZIP file
            zip_file_path = Path(export_path, f"{volume_id}_biigle_annotations.zip")
            with open(zip_file_path, "wb") as file:
                file.write(download_response.content)

            export_path = Path(export_path, f"{volume_id}_biigle_annotations_raw")

            # Extract ZIP file
            try:
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(export_path)
                logging.info(f"Files extracted to: {export_path}")
            except zipfile.BadZipFile:
                raise Exception("The downloaded file is not a valid zip file.")

            # Find and read the CSV file
            csv_files = glob.glob(os.path.join(export_path, "*.csv"))

            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {export_path}")

            if len(csv_files) > 1:
                print(csv_files)
                print(type(csv_files))
                logging.warning(
                    f"Multiple CSV files found. These are the file names: {csv_files}"
                )

                logging.info(f"Using the first one: {csv_files[0]}")

            logging.info(f"Processing BUV Deployment: {csv_files[0]}")

            annotations_df = pd.read_csv(csv_files[0])
            logging.info(
                f"Loaded {len(annotations_df)} annotations from {csv_files[0]}"
            )

            return annotations_df

        except Exception as e:
            logging.error(f"Failed to export annotations: {e}")
            raise

    def get_volumes(self, project_id: int = BIIGLE_PROJECT_ID) -> List[Dict[str, Any]]:
        """
        Get all volumes in a project.

        Args:
            project_id: The ID of the project to get volumes from.

        Returns:
            List[Dict[str, Any]]: List of volume dictionaries.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response = self.api.get(f"projects/{project_id}/volumes")
            volumes = response.json()
            logging.info(f"Retrieved {len(volumes)} volumes from project {project_id}")
            return volumes
        except Exception as e:
            logging.error(f"Failed to get volumes for project {project_id}: {e}")
            raise

    def create_volume_from_s3_files(
        self,
        project_id: int,
        volume_name: str,
        s3_url: str,
        files: List[str],
        media_type: str = "video",
    ) -> Dict[str, Any]:
        """
        Complete workflow to create a volume with files from S3.

        This is a convenience method that combines create_pending_volume and
        setup_volume_with_files.

        Args:
            project_id: The ID of the project to create the volume in.
            volume_name: The name for the volume.
            s3_url: The S3 URL where the files are located.
            files: List of file names to include in the volume.
            media_type: The type of media for the volume (default: "video").

        Returns:
            Dict[str, Any]: The created volume information.

        Raises:
            Exception: If any step of the process fails.
        """
        try:
            # Step 1: Create pending volume
            pending_volume = self.create_pending_volume(project_id, media_type)
            pending_volume_id = pending_volume["id"]

            # Step 2: Setup volume with files
            volume_info = self.setup_volume_with_files(
                pending_volume_id, volume_name, s3_url, files
            )

            logging.info(
                f"Successfully created volume '{volume_name}' with {len(files)} files"
            )
            return volume_info

        except Exception as e:
            logging.error(f"Failed to create volume from S3 files: {e}")
            raise

    def build_s3_url(self, s3_path: str, disk_id: int = BIIGLE_DISK_ID) -> str:
        """
        Build a BIIGLE-compatible S3 URL from an S3 path.

        Args:
            s3_path: The S3 path (e.g., "biigle_clips/TON_20221205_BUV/TON_20221205_BUV_TON_044_01/")
            disk_id: The BIIGLE disk ID for the S3 bucket.

        Returns:
            str: The BIIGLE-compatible S3 URL.
        """
        # Ensure path ends with /
        if not s3_path.endswith("/"):
            s3_path += "/"

        return f"disk-{disk_id}://{s3_path}"
