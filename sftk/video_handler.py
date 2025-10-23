import logging
import os
import shutil
from IPython.display import HTML
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import pandas as pd

from sftk.s3_handler import S3Handler
from sftk.common import S3_BUCKET

logger = logging.getLogger(__name__)


def _process_single_drop(
    drop_data_dict: dict,  # Pass dict instead of DataFrame
    download_dir: str,
    output_dir: str,
    delete_originals: bool,
    test_mode: bool,
    max_workers: int,
    sequential_download: bool,
    ffmpeg_path: str,
) -> None:
    """
    Process a single drop's worth of videos.
    
    Args:
        drop_data_dict: Dictionary containing:
            - 'keys': list of S3 keys
            - 'drop_id': str
            - 'survey_id': str
    """
    s3_handler = S3Handler()
    downloaded_files = []
    output_path = None

    def _download_videos_parallel(keys: list) -> List[Path]:
        """Download all videos for a drop in parallel."""
        downloaded_files_local = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(
                    s3_handler.download_object_from_s3,
                    key,
                    str(Path(download_dir) / Path(key).name),
                ): key
                for key in keys
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    if future.result():
                        downloaded_files_local.append(Path(download_dir) / Path(key).name)
                    else:
                        logger.error(f"Download failed for {key}")
                except Exception as e:
                    logger.error(f"Download failed for {key}: {e}")
        return sorted(downloaded_files_local)

    try:
        keys = drop_data_dict['keys']
        
        if not sequential_download:
            downloaded_files = _download_videos_parallel(keys)
        else:
            downloaded_files = []
            for key in keys:
                local_path = Path(download_dir) / Path(key).name
                if s3_handler.download_object_from_s3(key, str(local_path)):
                    downloaded_files.append(local_path)
                else:
                    logger.error(f"Sequential download failed for {key}")
            downloaded_files = sorted(downloaded_files)

        if not downloaded_files:
            raise RuntimeError("No files were successfully downloaded")

        drop_id = drop_data_dict['drop_id']
        survey_id = drop_data_dict['survey_id']
        output_path = Path(output_dir) / f"{drop_id}.mp4"
        
        temp_processor = VideoProcessor(s3_handler, test_mode=test_mode)
        if not temp_processor.concatenate_videos(downloaded_files, output_path):
            raise RuntimeError("Video concatenation failed")

        if not test_mode:
            # Create a minimal dict for upload
            upload_data = {
                'DropID': drop_id,
                'SurveyID': survey_id,
                'Key': keys
            }
            temp_processor._upload_and_cleanup_from_dict(output_path, upload_data)
    finally:
        if 'temp_processor' in locals():
            temp_processor._cleanup_local_files(downloaded_files, output_path)


class VideoProcessor:
    """
    A class to handle video processing tasks like verification, repair, and concatenation.
    """

    MOVIE_EXTENSIONS = {".wmv", ".mpg", ".mov", ".avi", ".mp4", ".MOV", ".MP4"}

    def __init__(
        self,
        s3_handler: S3Handler,
        prefix: str = "",
        gopro_prefix: str = "GX",
        delete_originals: bool = False,
        test_mode: bool = True,
        max_workers: int = 4,
        verify_videos: bool = False,
        parallel_drops: bool = True,
        sequential_download: bool = True,
    ):
        self.s3_handler = s3_handler
        self.prefix = prefix
        self.gopro_prefix = gopro_prefix
        self.delete_originals = delete_originals
        self.test_mode = test_mode
        self.max_workers = max_workers
        self.verify_videos = verify_videos
        self.parallel_drops = parallel_drops
        self.sequential_download = sequential_download

        self.download_dir = Path("downloaded_movies")
        self.output_dir = Path("concatenated_videos") 

        self.download_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Get movies and filter for go pro
        movies_df = self.get_movies_df(prefix=self.prefix)
        self.movies_df = _add_path_parts_to_df(movies_df)
        self.filtered_df = get_filtered_movies_df(
            movies_df=self.movies_df.copy(), gopro_prefix=self.gopro_prefix
        )

        self.ffmpeg_path = self._find_ffmpeg()
        if not self.ffmpeg_path:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH."
            )

    def _find_ffmpeg(self) -> Optional[str]:
        """Find ffmpeg executable."""
        if shutil.which("ffmpeg"):
            logger.info("Found ffmpeg in PATH.")
            return "ffmpeg"

        possible_paths = [
            Path.cwd() / "ffmpeg.exe",
            Path.cwd() / "bin" / "ffmpeg.exe",
        ]
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found ffmpeg at: {path}")
                return str(path)
        return None
    
    def _upload_and_cleanup_from_dict(
        self, output_path: Path, upload_data: dict
    ) -> None:
        """Upload concatenated video and delete original parts from S3 if configured."""
        drop_id = upload_data['DropID']
        survey_id = upload_data['SurveyID']
        new_key = f"media/{survey_id}/{drop_id}/{drop_id}.mp4"

        self.s3_handler.upload_file_to_s3(str(output_path), new_key)
        logger.info(f"Successfully uploaded concatenated video to {new_key}")

        if self.delete_originals:
            for key in upload_data['Key']:
                self.s3_handler.s3.delete_object(Bucket=self.s3_handler.bucket, Key=key)
                logger.info(f"Deleted original file {key}")

    def verify_video_file_deep(self, file_path: Path) -> bool:
        """Deep verification using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=True
            )
            import json

            data = json.loads(result.stdout) # type: ignore
            if "format" in data and "streams" in data:
                logger.info(f"✅ Video file verified: {file_path}")
                return True
            logger.error(f"Invalid video format for {file_path}")
            return False
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            Exception,
        ) as e:
            logger.error(f"Error verifying video file {file_path}: {e}")
            return False

    def _try_repair_video(
        self, corrupted_path: Path, temp_dir: Path
    ) -> Optional[Path]:
        """Attempts to repair a corrupted video file by re-muxing it."""
        repaired_path = temp_dir / f"repaired_{corrupted_path.name}"
        logger.warning(f"Attempting to repair corrupted video: {corrupted_path}")
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(corrupted_path),
            "-c",
            "copy",
            "-ignore_unknown",
            str(repaired_path),
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                timeout=600,
            )
            if self.verify_video_file_deep(repaired_path):
                logger.info(f"✅ Repair successful. New file is valid: {repaired_path}")
                return repaired_path
            logger.error("❌ Repair command ran, but the output file is invalid.")
            return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"❌ Failed to repair video '{corrupted_path}'. Stderr: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during repair: {e}")
            return None

    def concatenate_videos(
        self, video_paths: List[Path], output_path: Path
    ) -> bool:
        """Concatenate multiple videos, attempting to repair any corrupted files first."""
        list_file = None
        temp_dir = Path(tempfile.mkdtemp(prefix="video_repair_"))
        logger.info(f"Using temporary directory for repairs: {temp_dir}")

        try:
            videos_to_concat = []
            for path in video_paths:
                if self.verify_video_file_deep(path):
                    videos_to_concat.append(path)
                else:
                    repaired_path = self._try_repair_video(path, temp_dir)
                    if repaired_path:
                        videos_to_concat.append(repaired_path)
                    else:
                        logger.critical(
                            f"Could not repair '{path}'. Aborting concatenation."
                        )
                        return False

            if not videos_to_concat:
                raise ValueError("No valid videos available to concatenate.")

            # Store the paths of the movies in temp_dir to avoid issues with processing multiple drops in parallel
            list_file = temp_dir / "file_list.txt"

            with open(list_file, "w", encoding="utf-8") as f:
                for path in videos_to_concat:
                    f.write(f"file '{str(path.resolve()).replace(chr(92), '/')}'\n")

            cmd = [
                self.ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(output_path),
            ]

            logger.info(f"Running command: {' '.join(cmd)}")
            start_time = time.time()
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                timeout=3600,
            )
            logger.info(
                f"✅ Success! Concatenation took {time.time() - start_time:.2f} seconds."
            )

            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error("Output video file is missing or empty after concatenation.")
                return False
            return True
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"❌ Concatenation failed: {e}")
            return False
        finally:
            if list_file and list_file.exists():
                list_file.unlink()
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def process_gopro_videos(self) -> None:
        """Process GoPro videos by DropID."""
        drop_ids = self.filtered_df["DropID"].unique()
        valid_drops = []
        
        for drop_id in drop_ids:
            drop_data = self.filtered_df[self.filtered_df["DropID"] == drop_id]
            if all(str(name).startswith(self.gopro_prefix) for name in drop_data["fileName"]):
                # Convert DataFrame to dict for pickling
                drop_dict = {
                    'keys': drop_data["Key"].tolist(),
                    'drop_id': drop_data["DropID"].iloc[0],
                    'survey_id': drop_data["SurveyID"].iloc[0]
                }
                valid_drops.append((drop_id, drop_dict))
            else:
                logger.warning(
                    f"Skipping DropID {drop_id}: Not all videos start with {self.gopro_prefix}"
                )

        if self.test_mode and valid_drops:
            logger.info("Test mode is enabled. Processing only the first valid drop.")
            valid_drops = valid_drops[:1]
        
        logger.info(f"Processing {len(valid_drops)} valid drops")

        if self.parallel_drops:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        _process_single_drop,
                        drop_dict,  # Pass dict instead of DataFrame
                        str(self.download_dir),
                        str(self.output_dir),
                        self.delete_originals,
                        self.test_mode,
                        self.max_workers,
                        self.sequential_download,
                        self.ffmpeg_path,
                    ) for drop_id, drop_dict in valid_drops
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing a drop: {e}")
        else:
            for drop_id, drop_dict in valid_drops:
                try:
                    logger.info(f"Processing DropID {drop_id}...")
                    _process_single_drop(
                        drop_dict,
                        str(self.download_dir),
                        str(self.output_dir),
                        self.delete_originals,
                        self.test_mode,
                        self.max_workers,
                        self.sequential_download,
                        self.ffmpeg_path
                    )
                    logger.info(f"✅ Successfully processed DropID {drop_id}")
                except Exception as e:
                    logger.error(f"❌ Error processing DropID {drop_id}: {e}")
                    continue

    def get_movies_df(self, prefix: str = "") -> pd.DataFrame:
        """Get DataFrame of movie files in S3 bucket with their sizes."""
        objects = self.s3_handler.get_objects_from_s3(
            prefix=prefix, 
            suffixes=tuple(self.MOVIE_EXTENSIONS),
            keys_only=False  # Explicitly request full objects
        )
        # Type assertion for clarity
        assert isinstance(objects, list), "Expected list of objects"
        movie_data = [{"Key": obj["Key"], "Size": obj["Size"]} for obj in objects]
        return pd.DataFrame(movie_data)

    def _upload_and_cleanup(
        self, output_path: Path, drop_data: pd.DataFrame
    ) -> None:
        """Upload concatenated video and delete original parts from S3 if configured."""
        drop_id = drop_data["DropID"].iloc[0]
        survey_id = drop_data["SurveyID"].iloc[0]
        new_key = f"media/{survey_id}/{drop_id}/{drop_id}.mp4"

        self.s3_handler.upload_file_to_s3(str(output_path), new_key)
        logger.info(f"Successfully uploaded concatenated video to {new_key}")

        if self.delete_originals:
            for key in drop_data["Key"]:
                self.s3_handler.s3.delete_object(Bucket=self.s3_handler.bucket, Key=key)
                logger.info(f"Deleted original file {key}")

    def _cleanup_local_files(
        self, downloaded_files: List[Path], output_path: Optional[Path]
    ) -> None:
        """Clean up local files."""
        for file_path in downloaded_files:
            if file_path.exists():
                file_path.unlink()
        if output_path and output_path.exists():
            output_path.unlink()

    def find_already_concatenated_movies_df(self, size_tolerance: float = 0.01, ) -> pd.DataFrame:
        """Find individual movie files that can be removed because a concatenated version already exists."""
        concatenated_movies = self.movies_df[self.movies_df["fileNameNoExt"] == self.movies_df["DropID"]]
        gopro_movies = self.movies_df[self.movies_df.fileName.str.startswith(self.gopro_prefix, na=False)]

        files_to_remove_list = []

        for _, concatenated_movie in concatenated_movies.iterrows():
            drop_id = concatenated_movie["DropID"]
            gopro_parts = gopro_movies[gopro_movies["DropID"] == drop_id]

            if not gopro_parts.empty:
                total_parts_size = gopro_parts["Size"].sum()
                concatenated_size = concatenated_movie["Size"]

                # Check if the concatenated file size is close to the sum of the parts
                if abs(total_parts_size - concatenated_size) / concatenated_size < size_tolerance:
                    files_to_remove_list.append(gopro_parts)

        if not files_to_remove_list:
            return pd.DataFrame()

        return pd.concat(files_to_remove_list)

    def preview_movie(self, key: str, expiration: int = 26400) -> Optional[HTML]:
        """
        Generates an HTML video player for a movie in S3.

        Args:
            key (str): The S3 key of the movie file.
            expiration (int): The URL's expiration time in seconds.

        Returns:
            Optional[HTML]: An IPython HTML object for display in a notebook, or None on failure.
        """
        movie_url = self.s3_handler.generate_presigned_url(key, expiration=expiration)

        if not movie_url:
            logger.error(f"Could not generate preview URL for {key}")
            return None

        html_code = f"""
        <div style="display: flex; align-items: center; width: 100%;">
            <div style="width: 60%; padding-right: 10px;">
                <video width="100%" controls>
                    <source src="{movie_url}">
                </video>
            </div>
        </div>
        """
        return HTML(html_code)


def _add_path_parts_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adds SurveyID, DropID, fileName, and fileNameNoExt from the 'Key' column."""
    if df.empty or "Key" not in df.columns:
        return df

    parts = df["Key"].str.split("/", expand=True)
    if parts.shape[1] < 4:
        return df # Not the expected structure

    df = df.assign(SurveyID=parts[1], DropID=parts[2], fileName=parts[3])
    df["fileNameNoExt"] = df["fileName"].str.replace(".mp4", "", case=False)
    return df


def get_filtered_movies_df(
    movies_df: pd.DataFrame, gopro_prefix: str = "GX"
) -> pd.DataFrame:
    """Filter movies DataFrame to find GoPro groups that need concatenation."""
    go_pro_movies_df = movies_df[movies_df.fileName.str.startswith(gopro_prefix, na=False)].copy()

    # Find DropIDs that already have a concatenated file
    movies_df["fileNameNoExt"] = movies_df["fileName"].str.replace(".mp4", "", case=False)
    matching_dropids = movies_df[movies_df["fileNameNoExt"] == movies_df["DropID"]]["DropID"].unique()

    # Exclude groups that already have a concatenated file
    df_no_matching = go_pro_movies_df[~go_pro_movies_df["DropID"].isin(matching_dropids)]

    # Only consider groups with more than one video to concatenate
    grouped_counts = df_no_matching.groupby("DropID")["fileName"].nunique()
    filtered_dropids = grouped_counts[grouped_counts > 1].index

    return df_no_matching[df_no_matching["DropID"].isin(filtered_dropids)]
