import logging
import math
import re
from typing import Optional

import pandas as pd

from sftk.biigle_handler import BiigleHandler
from sftk.common import BIIGLE_API_EMAIL, BIIGLE_API_TOKEN

SCALE_BAR_LENGTH_CM = 10


class BiigleParser:
    """Handler for BIIGLE API operations."""

    def __init__(
        self,
        email: Optional[str] = BIIGLE_API_EMAIL,
        token: Optional[str] = BIIGLE_API_TOKEN,
    ):
        self.biigle_handler = BiigleHandler(email=email, token=token)

    def process_video_annotations(
        self,
        volume_id: int,
    ):
        annotations_df = self.biigle_handler.fetch_annotations_df(volume_id=volume_id)

        required_columns = [
            "label_name",
            "video_id",
            "video_filename",
            "shape_id",
            "shape_name",
            "points",
            "frames",
        ]

        annotations_df = annotations_df[required_columns]

        drop_id = re.sub(r"\.mp4.*", "", annotations_df["video_filename"].iloc[0])
        annotations_df["DropID"] = drop_id

        annotations_df = self.extract_time_values(annotations_df)

        # TODO check sorting and time, not doing it for every single df...
        # sort df based on time:
        annotations_df = annotations_df.sort_values(
            by=["video_filename", "start_seconds", "frame_seconds"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        max_n_30s_df = self.process_30s_count(annotations_df)
        max_n_df = self.process_max_count(max_n_30s_df)
        sizes_df = self.process_sizes(annotations_df)

        processed_dfs = {
            "drop_id": drop_id,
            "max_n_30s_df": max_n_30s_df,
            "max_n_df": max_n_df,
            "sizes_df": sizes_df,
        }
        logging.info(
            f"Processed annotations for volume {volume_id} with DropID {drop_id}"
        )

        return processed_dfs

    def process_30s_count(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """Processes raw Biigle annotations to count species per frame and format data."""

        # Extract relevant rows
        count_df = annotations_df[annotations_df["shape_name"] == "Rectangle"].copy()

        #  Group and count species occurrences
        grouped_df = (
            count_df.groupby(
                [
                    "video_filename",
                    "label_name",
                    "start_seconds",
                    "frame_seconds",
                    "time_of_max",
                ]
            )
            .size()
            .reset_index(name="max_count")
        )

        grouped_df = grouped_df.sort_values(
            ["start_seconds", "frame_seconds"], ascending=[True, True]
        )

        return grouped_df

    def process_max_count(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        From the whole video, select the row with the maximum count per (label_name).

        If there are ties, pick the earliest time.
        """
        # Ensure correct ordering: highest count first, earliest frame/time in case of ties
        ordered_annotations_df = annotations_df.sort_values(
            ["max_count", "start_seconds", "frame_seconds"],
            ascending=[False, True, True],
        )

        # Drop duplicates, keeping only the "first" occurrence per video/species
        result_df = ordered_annotations_df.drop_duplicates(
            subset=["label_name"], keep="first"
        )

        result_df = result_df.sort_values(
            ["start_seconds", "frame_seconds"], ascending=[True, True]
        )

        return result_df.reset_index(drop=True)

    def process_sizes(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        if annotations_df[annotations_df["label_name"] == "___scale bar"].empty:
            return pd.DataFrame()

        sizes_df = annotations_df[annotations_df["shape_name"] == "LineString"].copy()

        sizes_df["size_px"] = sizes_df["points"].apply(self.get_size)

        # In case people drew more scale bars.
        scale_size = sizes_df[sizes_df["label_name"] == "___scale bar"][
            "size_px"
        ].mean()
        sizes_df["size_cm"] = sizes_df["size_px"] * SCALE_BAR_LENGTH_CM / scale_size

        # Drop scale bar:
        sizes_df = sizes_df[sizes_df["label_name"] != "___scale bar"]

        sizes_df = sizes_df.sort_values(
            ["start_seconds", "frame_seconds"], ascending=[True, True]
        )

        return sizes_df[
            [
                "label_name",
                "video_filename",
                "start_seconds",
                "frame_seconds",
                "time_of_max",
                "size_px",
                "size_cm",
            ]
        ]

    def get_size(self, coordinates):
        points = self.parse_points(coordinates)
        return self.sum_distances(points)

    def parse_points(self, points_str):
        nums = [float(x) for x in points_str.strip("[]").split(",")]
        pairs = list(zip(nums[0::2], nums[1::2]))
        return pairs

    def sum_distances(self, points):
        return sum(
            math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(points[:-1], points[1:])
        )

    def extract_time_values(self, annotations_df):
        # Extract start seconds from filename (e.g., ..._clip_115_30... -> 115)
        annotations_df["start_seconds"] = pd.to_numeric(
            annotations_df["video_filename"].str.extract(r"_clip_(\d+)_", expand=False),
            errors="coerce",
        ).astype("Int64")

        # Extract frame seconds from the "frames" '[seconds]' string
        annotations_df["frame_seconds"] = (
            annotations_df["frames"].str.strip("[]").astype(float)
        )

        # Calculate total seconds and format to HH:MM:SS
        total_seconds = (
            annotations_df["start_seconds"] + annotations_df["frame_seconds"]
        )

        annotations_df["time_of_max"] = pd.to_datetime(
            total_seconds, unit="s"
        ).dt.strftime("%H:%M:%S")

        return annotations_df

    def format_count_annotations_output(
        self, annotations_df: pd.DataFrame
    ) -> pd.DataFrame:
        annotations_df["ScientificName"] = (
            annotations_df["label_name"]
            .str.split(" - ")
            .str[1]
            .fillna(annotations_df["label_name"])
        )

        # Rename columns to match target format
        renamed_df = annotations_df.rename(
            columns={"max_count": "MaxInterval", "time_of_max": "TimeOfMax"}
        )
        # Add constant columns (may change according to video)
        renamed_df["AnnotatedBy"] = "expert"
        renamed_df["IntervalAnnotation"] = 30
        renamed_df["ConfidenceAgreement"] = "NA"

        # Select and reorder final columns
        final_df = renamed_df[
            [
                "DropID",
                "ScientificName",
                "TimeOfMax",
                "MaxInterval",
                "AnnotatedBy",
                "IntervalAnnotation",
                "ConfidenceAgreement",
            ]
        ]

        return final_df


if __name__ == "__main__":
    biigle_parser = BiigleParser()
    processed_annotations_df = biigle_parser.process_video_annotations(25516)
