import logging
import os
import shutil
from multiprocessing import Queue, Manager
from logging.handlers import QueueHandler, QueueListener
import json
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


def _worker_log_config(log_queue: Queue) -> None:
    """Configure logging for a worker process to send logs to a queue."""
    queue_handler = QueueHandler(log_queue)
    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)


def _should_download_file(local_path: Path, s3_size: int, tolerance: float = 0.01) -> bool:
    """
    Check if a file needs to be downloaded by comparing local and S3 file sizes.
    
    Args:
        local_path: Path to the local file
        s3_size: Size of the file in S3 (in bytes)
        tolerance: Acceptable size difference as a fraction (default 1%)
    
    Returns:
        True if file should be downloaded, False if local file is valid
    """
    if not local_path.exists():
        return True
    
    local_size = local_path.stat().st_size
    
    # Check if file is empty
    if local_size == 0:
        logger.warning(f"Local file {local_path.name} is empty, will re-download")
        return True
    
    # Check if sizes match within tolerance
    if s3_size > 0:
        size_diff = abs(local_size - s3_size) / s3_size
        if size_diff <= tolerance:
            logger.info(f"✓ File {local_path.name} already exists with correct size, skipping download")
            return False
        else:
            logger.warning(
                f"⚠ Local file {local_path.name} size mismatch "
                f"(local: {local_size:,} bytes, S3: {s3_size:,} bytes), will re-download"
            )
    
    return True


def _process_single_drop(
    drop_data_dict: dict,
    download_dir: str,
    output_dir: str,
    delete_originals: bool,
    test_mode: bool,
    max_workers: int,
    sequential_download: bool,
    log_queue: Optional[Queue] = None,
) -> None:
    """
    Process a single drop's worth of videos.
    """
    if log_queue:
        _worker_log_config(log_queue)
    
    # Get logger AFTER configuring it
    process_logger = logging.getLogger(__name__)
    
    drop_id = drop_data_dict['drop_id']
    keys = drop_data_dict['keys']
    sizes = drop_data_dict['sizes']
    survey_id = drop_data_dict['survey_id']
    
    process_logger.info(f"{'='*60}")
    process_logger.info(f"🎬 Processing DropID: {drop_id}")
    process_logger.info(f"   Files to process: {len(keys)}")
    process_logger.info(f"{'='*60}")
    
    s3_handler = S3Handler()
    downloaded_files = []
    output_path = None
    drop_download_dir: Optional[Path] = None

    def _download_videos_parallel(keys_with_sizes: list) -> List[Path]:
        """Download all videos for a drop in parallel."""
        downloaded_files_local = []
        futures = {}
        files_to_download = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if drop_download_dir is None:
                raise ValueError("drop_download_dir must be set before parallel download.")

            for key, s3_size in keys_with_sizes:
                local_path = drop_download_dir / Path(key).name
                
                # Check if download is needed
                if not _should_download_file(local_path, s3_size):
                    downloaded_files_local.append(local_path)
                    continue

                files_to_download += 1
                future = executor.submit(
                    s3_handler.download_object_from_s3, key, str(local_path)
                )
                futures[future] = (key, local_path)

            if files_to_download > 0:
                process_logger.info(f"⬇ Downloading {files_to_download} file(s) in parallel...")
            
            for future in as_completed(futures):
                key, local_path = futures[future]
                try:
                    if future.result():
                        downloaded_files_local.append(local_path)
                        process_logger.info(f"✓ Downloaded: {local_path.name}")
                    else:
                        process_logger.error(f"✗ Download failed for {key}")
                except Exception as e:
                    process_logger.error(f"✗ Download failed for {key}: {e}")
        
        return sorted(downloaded_files_local)

    try:
        # Create a drop-specific download directory
        base_download_dir = Path(download_dir)
        drop_download_dir = base_download_dir / drop_id
        drop_download_dir.mkdir(parents=True, exist_ok=True)

        # Create list of (key, size) tuples
        keys_with_sizes = list(zip(keys, sizes))

        if not sequential_download:
            downloaded_files = _download_videos_parallel(keys_with_sizes)
        else:
            downloaded_files = []
            files_to_download = sum(
                1 for key, s3_size in keys_with_sizes 
                if _should_download_file(drop_download_dir / Path(key).name, s3_size)
            )
            
            if files_to_download > 0:
                process_logger.info(f"⬇ Downloading {files_to_download} file(s) sequentially...")
            
            for key, s3_size in keys_with_sizes:
                local_path = drop_download_dir / Path(key).name
                
                # Check if download is needed
                if not _should_download_file(local_path, s3_size):
                    downloaded_files.append(local_path)
                    continue
                
                if s3_handler.download_object_from_s3(key, str(local_path)):
                    downloaded_files.append(local_path)
                    process_logger.info(f"✓ Downloaded: {local_path.name}")
                else:
                    process_logger.error(f"✗ Sequential download failed for {key}")
            
            downloaded_files = sorted(downloaded_files)

        if not downloaded_files:
            raise RuntimeError("No files were successfully downloaded or found locally")

        # Validate all downloaded files
        process_logger.info(f"🔍 Validating {len(downloaded_files)} file(s)...")
        for f in downloaded_files:
            if not f.exists():
                raise RuntimeError(f"File {f.name} does not exist after download")
            if f.stat().st_size == 0:
                raise RuntimeError(f"File {f.name} is empty (0 bytes)")

        output_path = Path(output_dir) / f"{drop_id}.mp4"
        
        # Create temporary VideoProcessor instance for this drop
        temp_processor = VideoProcessor(
            s3_handler, 
            test_mode=test_mode,
            _skip_init_logs=True  # Skip repeated initialization logs
        )
        
        process_logger.info(f"🎞️  Concatenating videos into: {output_path.name}")
        concatenation_success, concatenation_message = temp_processor.concatenate_videos(
            downloaded_files, output_path
        )
        
        if not concatenation_success:
            raise RuntimeError(f"Video concatenation failed: {concatenation_message}")

        if not test_mode:
            process_logger.info(f"☁️  Uploading concatenated video to S3...")
            upload_data = {
                'DropID': drop_id,
                'SurveyID': survey_id,
                'Key': keys
            }
            temp_processor._upload_and_cleanup_from_dict(output_path, upload_data)
        else:
            process_logger.info(f"🧪 Test mode: Skipping S3 upload for {drop_id}")
            
    except Exception as e:
        process_logger.error(f"❌ Failed to process drop {drop_id}: {e}")
        raise
    finally:
        if 'temp_processor' in locals() and drop_download_dir:
            temp_processor._cleanup_local_files(
                downloaded_files, output_path, drop_download_dir, drop_id
            )


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
        parallel_drops: bool = True,
        sequential_download: bool = True,
        _skip_init_logs: bool = False,  # Internal flag to avoid log spam in workers
    ):
        self.s3_handler = s3_handler
        self.prefix = prefix.strip()
        self.gopro_prefix = gopro_prefix
        self.delete_originals = delete_originals
        self.test_mode = test_mode
        self.max_workers = max_workers
        self.parallel_drops = parallel_drops
        self.sequential_download = sequential_download

        if not _skip_init_logs:
            logger.info(f"{'='*80}")
            logger.info(f"🎥 VideoProcessor Initialization")
            logger.info(f"{'='*80}")
            logger.info(f"   Prefix: '{prefix}'")
            logger.info(f"   GoPro prefix: '{gopro_prefix}'")
            logger.info(f"   Test mode: {test_mode}")
            logger.info(f"   Parallel drops: {parallel_drops}")
            logger.info(f"   Max workers: {max_workers}")
            logger.info(f"   Sequential download: {sequential_download}")
            logger.info(f"   Delete originals: {delete_originals}")
            logger.info(f"{'='*80}")
        
        self.download_dir = Path("downloaded_movies")
        self.output_dir = Path("concatenated_videos")

        self.download_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.ffmpeg_path = self._find_ffmpeg()
        if not self.ffmpeg_path:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH."
            )

        if not _skip_init_logs:
            # Pre-fetch and filter movies to provide upfront information
            logger.info(f"🔍 Fetching movie list from S3 with prefix: '{self.prefix}'")
            self.movies_df = self.get_movies_df(prefix=self.prefix)
            logger.info(f"📁 Found {len(self.movies_df):,} movie files with the specified prefix")
            
            movies_df_with_parts = _add_path_parts_to_df(self.movies_df)
            self.filtered_df = get_filtered_movies_df(
                movies_df=movies_df_with_parts.copy(), 
                gopro_prefix=self.gopro_prefix
            )
            num_drops = self.filtered_df['DropID'].nunique() if not self.filtered_df.empty else 0
            logger.info(f"🎬 Found {num_drops} drop(s) that require concatenation")
            logger.info(f"{'='*80}\n")

    def _find_ffmpeg(self) -> Optional[str]:
        """Find ffmpeg executable."""
        if shutil.which("ffmpeg"):
            if not hasattr(self, '_skip_init_logs') or not self._skip_init_logs:
                logger.info("✓ Found ffmpeg in PATH")
            return "ffmpeg"

        possible_paths = [
            Path.cwd() / "ffmpeg.exe",
            Path.cwd() / "bin" / "ffmpeg.exe",
        ]
        for path in possible_paths:
            if path.exists():
                logger.info(f"✓ Found ffmpeg at: {path}")
                return str(path)
        return None
    
    def _upload_and_cleanup_from_dict(
        self, output_path: Path, upload_data: dict
    ) -> None:
        """Upload concatenated video and delete original parts from S3 if configured."""
        drop_id = upload_data['DropID']
        survey_id = upload_data['SurveyID']
        new_key = f"media/{survey_id}/{drop_id}/{drop_id}.mp4"

        logger.info(f"⬆️  Uploading to S3: {new_key}")
        upload_start = time.time()
        self.s3_handler.upload_file_to_s3(str(output_path), new_key)
        upload_time = time.time() - upload_start
        
        # Get file size for logging
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Successfully uploaded {file_size_mb:.2f} MB in {upload_time:.2f}s to: {new_key}")

        if self.delete_originals:
            logger.info(f"🗑️  Deleting {len(upload_data['Key'])} original file(s) from S3...")
            deleted_count = 0
            for key in upload_data['Key']:
                try:
                    self.s3_handler.s3.delete_object(Bucket=self.s3_handler.bucket, Key=key)
                    deleted_count += 1
                    logger.info(f"   ✓ Deleted: {Path(key).name}")
                except Exception as e:
                    logger.error(f"   ✗ Failed to delete {Path(key).name}: {e}")
            logger.info(f"✅ Deleted {deleted_count}/{len(upload_data['Key'])} original files from S3")
        else:
            logger.info(f"ℹ️  Keeping original files in S3 (delete_originals=False)")

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
            data = json.loads(result.stdout)
            if "format" in data and "streams" in data:
                logger.debug(f"✓ Video file verified: {file_path.name}")
                return True
            logger.error(f"✗ Invalid video format for {file_path.name}")
            return False
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            Exception,
        ) as e:
            logger.error(f"✗ Error verifying video file {file_path.name}: {e}")
            return False

    def _try_repair_with_gpfixer(
        self, corrupted_path: Path, temp_dir: Path
    ) -> Optional[Path]:
        """Use GP-Fixer for GoPro-specific repairs."""
        repaired_path = temp_dir / f"gpfixed_{corrupted_path.name}"
        
        try:
            logger.info(f"🔧 Attempting GP-Fixer repair...")
            cmd = ["gpfixer", "-i", str(corrupted_path), "-o", str(repaired_path)]
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
            
            if repaired_path.exists() and self.verify_video_file_deep(repaired_path):
                logger.info(f"✅ GP-Fixer repair successful!")
                return repaired_path
            return None
        except Exception as e:
            logger.warning(f"⚠ GP-Fixer repair failed: {e}")
            return None
            
    def _try_repair_with_partial_recovery(
        self, corrupted_path: Path, temp_dir: Path, all_video_paths: List[Path]
    ) -> Optional[Path]:
        """Extract moov atom from reference and prepend to corrupted file."""
        repaired_path = temp_dir / f"recovered_{corrupted_path.name}"
        
        reference_video = None
        for path in all_video_paths:
            if path != corrupted_path and self.verify_video_file_deep(path):
                reference_video = path
                break
        
        if not reference_video:
            logger.warning("⚠ No reference video found for partial recovery")
            return None
        
        try:
            logger.info(f"🔧 Attempting partial recovery with MP4Box...")
            # Extract moov atom from reference using MP4Box
            cmd = [
                "MP4Box", "-raw", "1", str(reference_video),
                "-out", str(temp_dir / "moov.dat")
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=60)
            
            # Note: This is a placeholder - real implementation would be more complex
            return None
        except Exception as e:
            logger.warning(f"⚠ Partial recovery failed: {e}")
            return None
            
    def _try_repair_with_untrunc(
        self, corrupted_path: Path, temp_dir: Path, all_video_paths: List[Path]
    ) -> Optional[Path]:
        """Repair using untrunc - excellent for missing moov atoms."""
        repaired_path = temp_dir / f"untrunc_{corrupted_path.name}"
        
        # Find a valid reference video from the same drop
        reference_video = None
        for path in all_video_paths:
            if path != corrupted_path and self.verify_video_file_deep(path):
                reference_video = path
                break
        
        if not reference_video:
            logger.error("✗ No reference video found for untrunc repair")
            return None
        
        try:
            logger.info(f"🔧 Attempting untrunc repair...")
            cmd = ["untrunc", str(reference_video), str(corrupted_path)]
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
            
            # untrunc creates a file with _fixed suffix
            fixed_file = corrupted_path.parent / f"{corrupted_path.stem}_fixed{corrupted_path.suffix}"
            if fixed_file.exists():
                shutil.move(str(fixed_file), str(repaired_path))
                if self.verify_video_file_deep(repaired_path):
                    logger.info(f"✅ untrunc repair successful!")
                    return repaired_path
            
            logger.warning("⚠ untrunc ran but did not produce a valid output")
            return None
        except Exception as e:
            logger.warning(f"⚠ untrunc repair failed: {e}")
            return None
            
    def _try_repair_video(
        self, corrupted_path: Path, temp_dir: Path, all_video_paths: List[Path]
    ) -> Optional[Path]:
        """Attempts to repair a corrupted video file using multiple strategies."""
        logger.warning(f"⚠ Video file corrupted: {corrupted_path.name}")
        
        # Strategy 1: untrunc (best for missing moov atom)
        if shutil.which("untrunc"):
            repaired = self._try_repair_with_untrunc(corrupted_path, temp_dir, all_video_paths)
            if repaired:
                return repaired
        
        # Strategy 2: GP-Fixer (GoPro-specific)
        if shutil.which("gpfixer"):
            repaired = self._try_repair_with_gpfixer(corrupted_path, temp_dir)
            if repaired:
                return repaired
        
        # Strategy 3: ffmpeg re-encode (slowest, last resort)
        repaired = self._try_repair_with_partial_recovery(corrupted_path, temp_dir, all_video_paths)
        if repaired:
            return repaired
        
        logger.error(f"❌ All repair strategies failed for {corrupted_path.name}")
        return None

    def concatenate_videos(self, video_paths: List[Path], output_path: Path) -> tuple[bool, str]:  # type: ignore
        """Concatenate multiple videos, attempting to repair any corrupted files first."""
        if not video_paths:
            error_msg = "No video paths provided for concatenation."
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        list_file = None
        temp_dir = Path(tempfile.mkdtemp(prefix="video_repair_"))

        try:
            videos_to_concat = []
            needs_repair = []
            
            # First pass: identify files needing repair
            for path in video_paths:
                if not path.exists():
                    error_msg = f"Input file '{path.name}' does not exist."
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg
                    
                if path.stat().st_size == 0:
                    error_msg = f"Input file '{path.name}' is empty (0 bytes)."
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

                if self.verify_video_file_deep(path):
                    videos_to_concat.append(path)
                else:
                    needs_repair.append(path)
            
            # Second pass: attempt repairs
            if needs_repair:
                logger.warning(f"⚠ {len(needs_repair)} file(s) need repair")
                for corrupted_path in needs_repair:
                    repaired_path = self._try_repair_video(corrupted_path, temp_dir, video_paths)
                    if repaired_path:
                        videos_to_concat.append(repaired_path)
                    else:
                        error_msg = f"Could not repair '{corrupted_path.name}'. Aborting concatenation."
                        logger.critical(f"❌ {error_msg}")
                        return False, error_msg

            if not videos_to_concat:
                error_msg = "No valid videos available to concatenate after verification/repair."
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            # Store file list in temp directory
            list_file = temp_dir / "file_list.txt"
            with open(list_file, "w", encoding="utf-8") as f:
                for path in videos_to_concat:
                    line = f"file '{path.resolve().as_posix()}'\n"
                    f.write(line)

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

            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                timeout=3600,
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Concatenation successful! Took {elapsed:.2f} seconds")

            if not output_path.exists() or output_path.stat().st_size == 0:
                error_msg = f"Output file '{output_path.name}' is missing or empty after concatenation."
                logger.error(f"❌ {error_msg}")
                return False, error_msg
                
            return True, "Concatenation successful."
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            stderr = getattr(e, 'stderr', 'N/A')
            error_msg = (
                f"CONCATENATION FAILED for {output_path.name}: {e}\n"
                f"--- FFMPEG STDERR ---\n{stderr}\n--- END FFMPEG STDERR ---"
            )
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during concatenation: {e}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        finally:
            # Clean up temporary files and directory
            if list_file and list_file.exists():
                list_file.unlink()
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def process_gopro_videos(self) -> None:
        """Process GoPro videos by DropID."""
        if self.filtered_df.empty:
            logger.info("ℹ️  No GoPro videos found that require concatenation")
            return

        drop_ids = self.filtered_df["DropID"].unique()
        valid_drops = []
        
        for drop_id in drop_ids:
            drop_data = self.filtered_df[self.filtered_df["DropID"] == drop_id]
            
            # Verify all files start with gopro_prefix
            if all(str(name).startswith(self.gopro_prefix) for name in drop_data["fileName"]):  # type: ignore
                # Convert DataFrame to dict for pickling, include sizes
                drop_dict = {
                    'keys': drop_data["Key"].tolist(),
                    'sizes': drop_data["Size"].tolist(),
                    'drop_id': drop_data["DropID"].iloc[0],
                    'survey_id': drop_data["SurveyID"].iloc[0]
                }
                valid_drops.append((drop_id, drop_dict))
            else:
                logger.warning(
                    f"⚠ Skipping DropID {drop_id}: Not all videos start with {self.gopro_prefix}"
                )

        if self.test_mode and valid_drops:
            valid_drops = valid_drops[-1:]
            logger.info(f"🧪 Test mode enabled: Processing only the last drop")
        
        logger.info(f"📊 Processing {len(valid_drops)} valid drop(s)\n")

        # Setup logging for parallel execution
        log_queue = None
        log_listener = None
        manager = None
        
        if self.parallel_drops:
            manager = Manager()
            log_queue = manager.Queue(-1)
            log_listener = QueueListener(log_queue, logging.StreamHandler())
            log_listener.start()

        processing_start = time.time()
        successful = 0
        failed = 0

        try:
            if self.parallel_drops:
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures_to_drop_id = {
                        executor.submit(
                            _process_single_drop,
                            drop_dict,
                            str(self.download_dir),
                            str(self.output_dir),
                            self.delete_originals,
                            self.test_mode,
                            self.max_workers,
                            self.sequential_download,
                            log_queue,
                        ): drop_id for drop_id, drop_dict in valid_drops
                    }
                    
                    for future in as_completed(futures_to_drop_id):
                        drop_id = futures_to_drop_id[future]
                        try:
                            future.result()
                            successful += 1
                            logger.info(f"✅ Successfully processed DropID {drop_id}\n")
                        except Exception as e:
                            failed += 1
                            logger.error(f"❌ Error processing DropID {drop_id}: {e}\n")
            else:
                for drop_id, drop_dict in valid_drops:
                    try:
                        _process_single_drop(
                            drop_dict,
                            str(self.download_dir),
                            str(self.output_dir),
                            self.delete_originals,
                            self.test_mode,
                            self.max_workers,
                            self.sequential_download,
                            None,
                        )
                        successful += 1
                        logger.info(f"✅ Successfully processed DropID {drop_id}\n")
                    except Exception as e:
                        failed += 1
                        logger.error(f"❌ Error processing DropID {drop_id}: {e}\n")
                        continue
        finally:
            if log_listener:
                log_listener.stop()
            
            # Summary
            total_time = time.time() - processing_start
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 Processing Summary")
            logger.info(f"{'='*80}")
            logger.info(f"   Total drops: {len(valid_drops)}")
            logger.info(f"   ✅ Successful: {successful}")
            logger.info(f"   ❌ Failed: {failed}")
            logger.info(f"   ⏱️  Total time: {total_time:.2f} seconds")
            logger.info(f"{'='*80}\n")

    def get_movies_df(self, prefix: str = "") -> pd.DataFrame:
        """Get DataFrame of movie files in S3 bucket with their sizes."""
        objects = self.s3_handler.get_objects_from_s3(
            prefix=prefix,
            suffixes=tuple(self.MOVIE_EXTENSIONS),
            keys_only=False
        )
        assert isinstance(objects, list), "Expected list of objects"
        movie_data = [{"Key": obj["Key"], "Size": obj["Size"]} for obj in objects]
        return pd.DataFrame(movie_data)

    def _cleanup_local_files(
        self,
        downloaded_files: List[Path],
        output_path: Optional[Path],
        drop_specific_download_dir: Optional[Path] = None,
        drop_id: str = ""
    ) -> None:
        """Clean up local files after processing."""
        if not self.test_mode:
            logger.info(f"🗑️  Cleaning up local files for {drop_id}...")
            deleted_count = 0
            for file_path in downloaded_files:
                if file_path.exists():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"   ✗ Could not delete {file_path.name}: {e}")
            
            logger.info(f"   Deleted {deleted_count}/{len(downloaded_files)} local video file(s)")
            
            if drop_specific_download_dir and drop_specific_download_dir.exists():
                try:
                    drop_specific_download_dir.rmdir()
                    logger.info(f"   ✓ Removed drop directory: {drop_specific_download_dir.name}")
                except OSError:
                    logger.debug(f"   Could not remove {drop_specific_download_dir.name} (not empty)")
        else:
            logger.info(f"🧪 Test mode: Keeping {len(downloaded_files)} downloaded file(s) for {drop_id}")
            
        # Clean up output file after upload (in non-test mode)
        if output_path and output_path.exists() and not self.test_mode:
            try:
                output_path.unlink()
                logger.info(f"   ✓ Deleted local output file: {output_path.name}")
            except Exception as e:
                logger.warning(f"   ✗ Could not delete output file {output_path.name}: {e}")

    def find_already_concatenated_movies_df(
        self, size_tolerance: float = 0.01
    ) -> pd.DataFrame:
        """Find individual movie files that can be removed because concatenated version exists."""
        movies_df = self.get_movies_df(prefix=self.prefix)
        movies_df_with_parts = _add_path_parts_to_df(movies_df)

        concatenated_movies = movies_df_with_parts[
            movies_df_with_parts["fileNameNoExt"] == movies_df_with_parts["DropID"]
        ]
        gopro_movies = movies_df_with_parts[
            movies_df_with_parts.fileName.str.startswith(self.gopro_prefix, na=False)
        ]

        files_to_remove_list = []

        for _, concatenated_movie in concatenated_movies.iterrows():
            drop_id = concatenated_movie["DropID"]
            gopro_parts = gopro_movies[gopro_movies["DropID"] == drop_id]

            if not gopro_parts.empty:
                total_parts_size = gopro_parts["Size"].sum()
                concatenated_size = concatenated_movie["Size"]

                # Check if sizes match within tolerance
                if abs(total_parts_size - concatenated_size) / concatenated_size < size_tolerance:
                    files_to_remove_list.append(gopro_parts)

        if not files_to_remove_list:
            return pd.DataFrame()

        return pd.concat(files_to_remove_list)

    def preview_movie(self, key: str, expiration: int = 26400) -> Optional[HTML]:
        """
        Generates an HTML video player for a movie in S3.

        Args:
            key: The S3 key of the movie file
            expiration: The URL's expiration time in seconds

        Returns:
            IPython HTML object for display, or None on failure
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
        logger.warning("⚠ Key structure does not match expected format (media/survey/drop/file)")
        return df

    df = df.assign(SurveyID=parts[1], DropID=parts[2], fileName=parts[3])
    df["fileNameNoExt"] = df["fileName"].str.replace(".mp4", "", case=False, regex=False)
    return df


def get_filtered_movies_df(
    movies_df: pd.DataFrame, gopro_prefix: str = "GX"
) -> pd.DataFrame:
    """Filter movies DataFrame to find GoPro groups that need concatenation."""
    if movies_df.empty or "fileName" not in movies_df.columns:
        return pd.DataFrame()

    go_pro_movies_df = movies_df[
        movies_df.fileName.str.startswith(gopro_prefix, na=False)
    ].copy()

    # Find DropIDs that already have a concatenated file
    movies_df["fileNameNoExt"] = movies_df["fileName"].str.replace(
        ".mp4", "", case=False, regex=False
    )
    matching_dropids = movies_df[
        movies_df["fileNameNoExt"] == movies_df["DropID"]
    ]["DropID"].unique()

    # Exclude groups that already have a concatenated file
    df_no_matching = go_pro_movies_df[~go_pro_movies_df["DropID"].isin(matching_dropids)]

    # Only consider groups with more than one video to concatenate
    grouped_counts = df_no_matching.groupby("DropID")["fileName"].nunique()
    filtered_dropids = grouped_counts[grouped_counts > 1].index

    return df_no_matching[df_no_matching["DropID"].isin(filtered_dropids)]