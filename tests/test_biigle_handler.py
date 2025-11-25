import io
import zipfile
from unittest.mock import Mock, patch

import pandas as pd


class TestBiigleHandlerInitialization:
    """Test BiigleHandler initialization and API connection setup."""


class TestBiigleHandlerGetProjects:
    """Test BiigleHandler get_projects method."""

    @patch("sftk.external.biigle_api.Api")
    def test_get_projects_returns_list_of_projects(self, mock_api_class):
        """Test that get_projects returns a list of accessible projects."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": 1, "name": "Project 1", "description": "Test project 1"},
            {"id": 2, "name": "Project 2", "description": "Test project 2"},
        ]

        mock_api_instance = Mock()
        mock_api_instance.get.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")
        projects = handler.get_projects()

        mock_api_instance.get.assert_called_once_with("projects")
        assert len(projects) == 2
        assert projects[0]["id"] == 1
        assert projects[0]["name"] == "Project 1"
        assert projects[1]["id"] == 2
        assert projects[1]["name"] == "Project 2"


class TestBiigleHandlerGetVolumes:
    """Test BiigleHandler get_volumes method."""

    def test_get_volumes_returns_list_of_volumes(self):
        """Test that get_volumes returns a list of volumes from a project."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": 1, "name": "Volume 1"},
            {"id": 2, "name": "Volume 2"},
        ]

        mock_api_instance = Mock()
        mock_api_instance.get.return_value = mock_response
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")
            volumes = handler.get_volumes(project_id=3711)

            mock_api_instance.get.assert_called_once_with("projects/3711/volumes")
            assert len(volumes) == 2
            assert volumes[0]["id"] == 1
            assert volumes[0]["name"] == "Volume 1"


class TestBiigleHandlerCreatePendingVolume:
    """Test BiigleHandler create_pending_volume method."""

    def test_create_pending_volume_success(self):
        """Test that create_pending_volume creates a new pending volume successfully."""
        mock_response = Mock()
        mock_response.json.return_value = {"id": 12345, "name": None, "url": None}

        mock_api_instance = Mock()
        mock_api_instance.post.return_value = mock_response
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")
            pending_volume = handler.create_pending_volume(project_id=3711)

            mock_api_instance.post.assert_called_once_with(
                "projects/3711/pending-volumes", json={"media_type": "video"}
            )
            assert pending_volume["id"] == 12345

    def test_create_pending_volume_with_media_type(self):
        """Test that create_pending_volume accepts custom media type."""
        mock_response = Mock()
        mock_response.json.return_value = {"id": 12345, "name": None, "url": None}

        mock_api_instance = Mock()
        mock_api_instance.post.return_value = mock_response
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")
            pending_volume = handler.create_pending_volume(
                project_id=3711, media_type="image"
            )

            mock_api_instance.post.assert_called_once_with(
                "projects/3711/pending-volumes", json={"media_type": "image"}
            )
            assert pending_volume["id"] == 12345


class TestBiigleHandlerSetupVolumeWithFiles:
    """Test BiigleHandler setup_volume_with_files method."""

    def test_setup_volume_with_files_success(self):
        """Test that setup_volume_with_files configures pending volume successfully."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": 12345,
            "name": "test_volume",
            "url": "s3://test",
        }

        mock_api_instance = Mock()
        mock_api_instance.put.return_value = mock_response
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            pending_volume_id = 12345
            volume_name = "test_volume"
            s3_url = (
                "disk-98://biigle_clips/TON_20221205_BUV/TON_20221205_BUV_TON_044_01/"
            )
            files = ["file1.mp4", "file2.mp4"]

            result = handler.setup_volume_with_files(
                pending_volume_id, volume_name, s3_url, files
            )

            expected_payload = {"name": volume_name, "url": s3_url, "files": files}
            mock_api_instance.put.assert_called_once_with(
                f"pending-volumes/{pending_volume_id}", json=expected_payload
            )
            assert result["id"] == 12345
            assert result["name"] == "test_volume"


class TestBiigleHandlerCreateVolumeFromS3Files:
    """Test BiigleHandler create_volume_from_s3_files convenience method."""

    def test_create_volume_from_s3_files_success(self):
        """Test that create_volume_from_s3_files creates and configures volume."""
        mock_pending_response = Mock()
        mock_pending_response.json.return_value = {"id": 12345}

        mock_setup_response = Mock()
        mock_setup_response.json.return_value = {
            "id": 12345,
            "name": "test_volume",
        }

        mock_api_instance = Mock()
        mock_api_instance.post.return_value = mock_pending_response
        mock_api_instance.put.return_value = mock_setup_response
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            result = handler.create_volume_from_s3_files(
                project_id=3711,
                volume_name="test_volume",
                s3_url="disk-98://biigle_clips/test/",
                files=["file1.mp4", "file2.mp4"],
            )

            assert result["id"] == 12345
            assert result["name"] == "test_volume"
            assert mock_api_instance.post.call_count == 1
            assert mock_api_instance.put.call_count == 1


class TestBiigleHandlerCreateLabelTree:
    """Test BiigleHandler create_label_tree method."""

    def test_create_label_tree_success(self):
        """Test that create_label_tree creates label tree and adds labels successfully."""
        mock_tree_response = Mock()
        mock_tree_response.json.return_value = {"id": 456, "name": "Test Tree"}

        mock_label_response = Mock()
        mock_label_response.json.return_value = {"id": 789, "name": "Test Species"}

        mock_api_instance = Mock()
        mock_api_instance.post.side_effect = [
            mock_tree_response,
            mock_label_response,
            mock_label_response,
        ]
        mock_api_class = Mock(return_value=mock_api_instance)

        # Create actual DataFrame for CSV
        labels_df = pd.DataFrame(
            {
                "name": ["Species 1", "Species 2"],
                "color": ["FF0000", "00FF00"],
                "source_id": [123, 456],
            }
        )

        with patch("sftk.biigle_handler.Api", mock_api_class):
            with patch("pandas.read_csv", return_value=labels_df):
                from sftk.biigle_handler import BiigleHandler

                handler = BiigleHandler("test@example.com", "test_token")

                result = handler.create_label_tree(
                    csv_path="test_labels.csv",
                    tree_name="Test Tree",
                    tree_description="Test Description",
                    project_id=3711,
                )

                # Check label tree creation (visibility_id should be 2, not 1)
                expected_tree_config = {
                    "name": "Test Tree",
                    "description": "Test Description",
                    "visibility_id": 2,  # Private by default
                    "project_id": 3711,
                }
                mock_api_instance.post.assert_any_call(
                    "label-trees", json=expected_tree_config
                )

                # Check that labels were created
                assert mock_api_instance.post.call_count == 3  # 1 tree + 2 labels
                assert result["tree_id"] == 456
                assert len(result["labels"]) == 2


class TestBiigleHandlerCreateReport:
    """Test BiigleHandler create_report method."""

    def test_create_report_success(self):
        """Test that create_report creates a report and returns report_id."""
        mock_response = Mock()
        mock_response.json.return_value = {"id": 999}
        mock_response.raise_for_status = Mock()

        mock_api_instance = Mock()
        mock_api_instance.post.return_value = mock_response
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            report_id = handler.create_report("volumes", resource_id=12345, type_id=8)

            mock_api_instance.post.assert_called_once_with(
                "volumes/12345/reports", json={"type_id": 8}
            )
            assert report_id == 999


class TestBiigleHandlerDownloadReportZipBytes:
    """Test BiigleHandler download_report_zip_bytes method."""

    def test_download_report_zip_bytes_success_immediate(self):
        """Test that download_report_zip_bytes downloads ZIP when ready immediately."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake zip content"

        mock_api_instance = Mock()
        mock_api_instance.get.return_value = mock_response
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            zip_bytes = handler.download_report_zip_bytes(report_id=999)

            mock_api_instance.get.assert_called_once_with(
                "reports/999", raise_for_status=False
            )
            assert zip_bytes == b"fake zip content"

    def test_download_report_zip_bytes_with_polling(self):
        """Test that download_report_zip_bytes polls until report is ready."""
        mock_not_ready = Mock()
        mock_not_ready.status_code = 404

        mock_ready = Mock()
        mock_ready.status_code = 200
        mock_ready.content = b"fake zip content"

        mock_api_instance = Mock()
        mock_api_instance.get.side_effect = [mock_not_ready, mock_ready]
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            with patch("time.sleep"):  # Don't actually sleep in tests
                from sftk.biigle_handler import BiigleHandler

                handler = BiigleHandler("test@example.com", "test_token")

                zip_bytes = handler.download_report_zip_bytes(
                    report_id=999, max_tries=5, poll_interval=0.1
                )

                assert mock_api_instance.get.call_count == 2
                assert zip_bytes == b"fake zip content"


class TestBiigleHandlerReadCsvsFromZipBytes:
    """Test BiigleHandler read_csvs_from_zip_bytes method."""

    def test_read_csvs_from_zip_bytes_simple(self):
        """Test reading CSV files from a simple ZIP."""
        # Create a ZIP with CSV files in memory
        csv1_content = "col1,col2\n1,2\n3,4"
        csv2_content = "col1,col2\n5,6\n7,8"

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("file1.csv", csv1_content)
            zf.writestr("file2.csv", csv2_content)
        zip_bytes = zip_buffer.getvalue()

        mock_api_instance = Mock()
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            csv_dict = handler.read_csvs_from_zip_bytes(zip_bytes)

            assert len(csv_dict) == 2
            assert "file1.csv" in csv_dict
            assert "file2.csv" in csv_dict
            assert len(csv_dict["file1.csv"]) == 2
            assert len(csv_dict["file2.csv"]) == 2

    def test_read_csvs_from_zip_bytes_nested(self):
        """Test reading CSV files from nested ZIPs."""
        # Create nested ZIP: outer.zip contains inner.zip which contains file.csv
        inner_csv = "col1,col2\n1,2"
        inner_zip_buffer = io.BytesIO()
        with zipfile.ZipFile(inner_zip_buffer, "w") as zf:
            zf.writestr("inner_file.csv", inner_csv)
        inner_zip_bytes = inner_zip_buffer.getvalue()

        outer_zip_buffer = io.BytesIO()
        with zipfile.ZipFile(outer_zip_buffer, "w") as zf:
            zf.writestr("inner.zip", inner_zip_bytes)
        outer_zip_bytes = outer_zip_buffer.getvalue()

        mock_api_instance = Mock()
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            csv_dict = handler.read_csvs_from_zip_bytes(
                outer_zip_bytes, allow_nested=True
            )

            assert len(csv_dict) == 1
            assert "inner_file.csv" in csv_dict
            assert len(csv_dict["inner_file.csv"]) == 1


class TestBiigleHandlerConcatCsvDict:
    """Test BiigleHandler concat_csv_dict method."""

    def test_concat_csv_dict_success(self):
        """Test that concat_csv_dict concatenates DataFrames with source column."""
        mock_api_instance = Mock()
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            csv_dict = {
                "file1.csv": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
                "file2.csv": pd.DataFrame({"col1": [5, 6], "col2": [7, 8]}),
            }

            result = handler.concat_csv_dict(csv_dict, source_col="source_file")

            assert len(result) == 4
            assert "source_file" in result.columns
            assert set(result["source_file"].unique()) == {"file1.csv", "file2.csv"}


class TestBiigleHandlerExportReportToDf:
    """Test BiigleHandler export_report_to_df method."""

    def test_export_report_to_df_success(self):
        """Test that export_report_to_df creates report, downloads, and returns DataFrame."""
        # Mock report creation
        mock_report_response = Mock()
        mock_report_response.json.return_value = {"id": 999}
        mock_report_response.raise_for_status = Mock()

        # Mock ZIP download
        csv_content = "col1,col2\n1,2\n3,4"
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("annotations.csv", csv_content)
        zip_bytes = zip_buffer.getvalue()

        mock_download_response = Mock()
        mock_download_response.status_code = 200
        mock_download_response.content = zip_bytes

        mock_api_instance = Mock()
        mock_api_instance.post.return_value = mock_report_response
        mock_api_instance.get.return_value = mock_download_response
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            result_df = handler.export_report_to_df(
                resource="volumes", resource_id=12345, type_id=8
            )

            # Check report was created
            mock_api_instance.post.assert_called_once_with(
                "volumes/12345/reports", json={"type_id": 8}
            )

            # Check ZIP was downloaded
            mock_api_instance.get.assert_called_once_with(
                "reports/999", raise_for_status=False
            )

            # Check result
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 2
            assert "source_file" in result_df.columns


class TestBiigleHandlerBuildS3Url:
    """Test BiigleHandler build_s3_url convenience method."""

    def test_build_s3_url(self):
        """Test that build_s3_url creates correct BIIGLE S3 URL format."""
        mock_api_instance = Mock()
        mock_api_class = Mock(return_value=mock_api_instance)

        with patch("sftk.biigle_handler.Api", mock_api_class):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            # Test without trailing slash (default disk_id is 134)
            s3_url = handler.build_s3_url("biigle_clips/test_folder")
            assert s3_url == "disk-134://biigle_clips/test_folder/"

            # Test with trailing slash
            s3_url_with_slash = handler.build_s3_url("biigle_clips/test_folder/")
            assert s3_url_with_slash == "disk-134://biigle_clips/test_folder/"

            # Test with custom disk_id
            s3_url_custom = handler.build_s3_url("biigle_clips/test", disk_id=99)
            assert s3_url_custom == "disk-99://biigle_clips/test/"
