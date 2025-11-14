import sys
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

pytest.skip("Skipping this file", allow_module_level=True)


class TestBiigleHandlerInitialization:
    """Test BiigleHandler initialization and API connection setup."""

    def test_biigle_handler_initialization_with_credentials(self):
        """Test that BiigleHandler initializes correctly with API credentials."""
        # Arrange
        mock_api_instance = Mock()
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            email = "test@example.com"
            token = "test_token"

            # Act
            handler = BiigleHandler(email, token)

            # Assert
            mock_api_class.assert_called_once_with(email, token)
            assert handler.api == mock_api_instance
            assert handler.email == email
            assert handler.token == token

    @patch("sftk.common.BIIGLE_API_EMAIL", "env_email@example.com")
    @patch("sftk.common.BIIGLE_API_TOKEN", "env_token")
    def test_biigle_handler_initialization_with_env_credentials(self):
        """Test that BiigleHandler uses environment credentials when none provided."""
        # Arrange
        mock_api_instance = Mock()
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            # Act
            handler = BiigleHandler()

            # Assert
            mock_api_class.assert_called_once_with("env_email@example.com", "env_token")
            assert handler.api == mock_api_instance
            assert handler.email == "env_email@example.com"
            assert handler.token == "env_token"

    def test_biigle_handler_initialization_missing_credentials(self):
        """Test that BiigleHandler raises error when credentials are missing."""
        # Mock the biigle module
        mock_biigle_module = MagicMock()

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            with (
                patch("sftk.common.BIIGLE_API_EMAIL", None),
                patch("sftk.common.BIIGLE_API_TOKEN", None),
            ):
                from sftk.biigle_handler import BiigleHandler

                # Act & Assert
                with pytest.raises(
                    ValueError, match="BIIGLE API credentials are required"
                ):
                    BiigleHandler(None, None)

    def test_biigle_handler_initialization_api_connection_failure(self):
        """Test that BiigleHandler handles API connection failures gracefully."""
        # Arrange
        mock_api_class = Mock(side_effect=Exception("API connection failed"))

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            # Act & Assert
            with pytest.raises(Exception, match="Failed to initialize BIIGLE API"):
                BiigleHandler("test@example.com", "test_token")


class TestBiigleHandlerGetProjects:
    """Test BiigleHandler get_projects method."""

    def test_get_projects_returns_list_of_projects(self):
        """Test that get_projects returns a list of accessible projects."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": 1, "name": "Project 1", "description": "Test project 1"},
            {"id": 2, "name": "Project 2", "description": "Test project 2"},
        ]

        mock_api_instance = Mock()
        mock_api_instance.get.return_value = mock_response
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            # Act
            projects = handler.get_projects()

            # Assert
            mock_api_instance.get.assert_called_once_with("projects")
            assert len(projects) == 2
            assert projects[0]["id"] == 1
            assert projects[0]["name"] == "Project 1"
            assert projects[1]["id"] == 2
            assert projects[1]["name"] == "Project 2"

    def test_get_projects_handles_api_error(self):
        """Test that get_projects handles API errors gracefully."""
        # Arrange
        mock_api_instance = Mock()
        mock_api_instance.get.side_effect = Exception("API error")
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            # Act & Assert
            with pytest.raises(Exception, match="API error"):
                handler.get_projects()


class TestBiigleHandlerCreatePendingVolume:
    """Test BiigleHandler create_pending_volume method."""

    def test_create_pending_volume_success(self):
        """Test that create_pending_volume creates a new pending volume successfully."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"id": 12345, "name": None, "url": None}

        mock_api_instance = Mock()
        mock_api_instance.post.return_value = mock_response
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")
            project_id = 3711

            # Act
            pending_volume = handler.create_pending_volume(project_id)

            # Assert
            mock_api_instance.post.assert_called_once_with(
                f"projects/{project_id}/pending-volumes", json={"media_type": "video"}
            )
            assert pending_volume["id"] == 12345

    def test_create_pending_volume_with_media_type(self):
        """Test that create_pending_volume accepts custom media type."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"id": 12345, "name": None, "url": None}

        mock_api_instance = Mock()
        mock_api_instance.post.return_value = mock_response
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")
            project_id = 3711

            # Act
            pending_volume = handler.create_pending_volume(
                project_id, media_type="image"
            )

            # Assert
            mock_api_instance.post.assert_called_once_with(
                f"projects/{project_id}/pending-volumes", json={"media_type": "image"}
            )
            assert pending_volume["id"] == 12345

    def test_create_pending_volume_api_error(self):
        """Test that create_pending_volume handles API errors gracefully."""
        # Arrange
        mock_api_instance = Mock()
        mock_api_instance.post.side_effect = Exception("API error")
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            # Act & Assert
            with pytest.raises(Exception, match="API error"):
                handler.create_pending_volume(3711)


class TestBiigleHandlerSetupVolumeWithFiles:
    """Test BiigleHandler setup_volume_with_files method."""

    def test_setup_volume_with_files_success(self):
        """Test that setup_volume_with_files configures pending volume successfully."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": 12345,
            "name": "test_volume",
            "url": "s3://test",
        }

        mock_api_instance = Mock()
        mock_api_instance.put.return_value = mock_response
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            pending_volume_id = 12345
            volume_name = "test_volume"
            s3_url = (
                "disk-98://biigle_clips/TON_20221205_BUV/TON_20221205_BUV_TON_044_01/"
            )
            files = ["file1.mp4", "file2.mp4"]

            # Act
            result = handler.setup_volume_with_files(
                pending_volume_id, volume_name, s3_url, files
            )

            # Assert
            expected_payload = {"name": volume_name, "url": s3_url, "files": files}
            mock_api_instance.put.assert_called_once_with(
                f"pending-volumes/{pending_volume_id}", json=expected_payload
            )
            assert result["id"] == 12345
            assert result["name"] == "test_volume"

    def test_setup_volume_with_files_api_error(self):
        """Test that setup_volume_with_files handles API errors gracefully."""
        # Arrange
        mock_api_instance = Mock()
        mock_api_instance.put.side_effect = Exception("API error")
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            # Act & Assert
            with pytest.raises(Exception, match="API error"):
                handler.setup_volume_with_files(
                    12345, "test", "s3://test", ["file1.mp4"]
                )


class TestBiigleHandlerCreateLabelTree:
    """Test BiigleHandler create_label_tree method."""

    def test_create_label_tree_success(self):
        """Test that create_label_tree creates label tree and adds labels successfully."""
        # Arrange
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

        # Mock the biigle module and pandas
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        # Mock pandas DataFrame
        mock_df = Mock()
        mock_df.iterrows.return_value = [
            (0, {"name": "Species 1", "color": "FF0000", "source_id": 123}),
            (1, {"name": "Species 2", "color": "00FF00", "source_id": 456}),
        ]

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            with patch("pandas.read_csv", return_value=mock_df):
                from sftk.biigle_handler import BiigleHandler

                handler = BiigleHandler("test@example.com", "test_token")

                project_id = 3711
                csv_path = "test_labels.csv"
                tree_name = "Test Tree"
                tree_description = "Test Description"

                # Act
                result = handler.create_label_tree(
                    project_id, csv_path, tree_name, tree_description
                )

                # Assert
                # Check label tree creation
                expected_tree_config = {
                    "name": tree_name,
                    "description": tree_description,
                    "visibility_id": 1,
                    "project_id": project_id,
                }
                mock_api_instance.post.assert_any_call(
                    "label-trees", json=expected_tree_config
                )

                # Check label creation calls
                expected_label_calls = [
                    (
                        "label-trees/456/labels",
                        {
                            "json": {
                                "name": "Species 1",
                                "color": "FF0000",
                                "source_id": 123,
                            }
                        },
                    ),
                    (
                        "label-trees/456/labels",
                        {
                            "json": {
                                "name": "Species 2",
                                "color": "00FF00",
                                "source_id": 456,
                            }
                        },
                    ),
                ]

                assert mock_api_instance.post.call_count == 3  # 1 tree + 2 labels
                assert result["tree_id"] == 456
                assert len(result["labels"]) == 2

    def test_create_label_tree_api_error(self):
        """Test that create_label_tree handles API errors gracefully."""
        # Arrange
        mock_api_instance = Mock()
        mock_api_instance.post.side_effect = Exception("API error")
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):

            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            # Act & Assert
            with pytest.raises(Exception, match="API error"):
                handler.create_label_tree(3711, "test.csv", "Test", "Description")


class TestBiigleHandlerExportAnnotations:
    """Test BiigleHandler export_annotations method."""

    def test_export_annotations_success(self):
        """Test that export_annotations downloads and processes annotations successfully."""
        # Arrange
        mock_report_response = Mock()
        mock_report_response.json.return_value = {"id": 999}

        mock_download_response = Mock()
        mock_download_response.content = b"fake zip content"

        mock_api_instance = Mock()
        mock_api_instance.post.return_value = mock_report_response
        mock_api_instance.get.return_value = mock_download_response
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        # Create a simple mock DataFrame
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=5)  # Mock len() for logging

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            # Mock all the file operations after handler is created
            with patch("builtins.open", mock_open()):
                with patch("zipfile.ZipFile"):
                    with patch("os.makedirs"):
                        with patch("glob.glob", return_value=["test_annotations.csv"]):
                            with patch("pandas.read_csv", return_value=mock_df):

                                volume_id = 12345
                                type_id = 8
                                extract_dir = "test_extract"

                                # Act
                                result = handler.export_annotations(
                                    volume_id, type_id, extract_dir
                                )

                                # Assert
                                # Check report creation
                                mock_api_instance.post.assert_called_once_with(
                                    f"volumes/{volume_id}/reports",
                                    json={"type_id": type_id},
                                )

                                # Check report download
                                mock_api_instance.get.assert_called_once_with(
                                    "reports/999"
                                )

                                assert result == mock_df

    def test_export_annotations_api_error(self):
        """Test that export_annotations handles API errors gracefully."""
        # Arrange
        mock_api_instance = Mock()
        mock_api_instance.post.side_effect = Exception("API error")
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            # Act & Assert
            with pytest.raises(Exception, match="API error"):
                handler.export_annotations(12345, 8, "test_dir")


class TestBiigleHandlerIntegration:
    """Integration tests for BiigleHandler complete workflows."""

    def test_complete_volume_workflow(self):
        """Test the complete workflow from creating volume to exporting annotations."""
        # Arrange
        mock_projects_response = Mock()
        mock_projects_response.json.return_value = [
            {"id": 3711, "name": "Test Project"}
        ]

        mock_pending_volume_response = Mock()
        mock_pending_volume_response.json.return_value = {"id": 12345}

        mock_setup_response = Mock()
        mock_setup_response.json.return_value = {"id": 12345, "name": "test_volume"}

        mock_report_response = Mock()
        mock_report_response.json.return_value = {"id": 999}

        mock_download_response = Mock()
        mock_download_response.content = b"fake zip content"

        mock_api_instance = Mock()
        mock_api_instance.get.side_effect = [
            mock_projects_response,
            mock_download_response,
        ]
        mock_api_instance.post.side_effect = [
            mock_pending_volume_response,
            mock_report_response,
        ]
        mock_api_instance.put.return_value = mock_setup_response
        mock_api_class = Mock(return_value=mock_api_instance)

        # Mock the biigle module
        mock_biigle_module = MagicMock()
        mock_biigle_module.Api = mock_api_class

        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=10)

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            # Mock file operations for export_annotations
            with patch("builtins.open", mock_open()):
                with patch("zipfile.ZipFile"):
                    with patch("os.makedirs"):
                        with patch("glob.glob", return_value=["annotations.csv"]):
                            with patch("pandas.read_csv", return_value=mock_df):

                                # Act - Complete workflow
                                # 1. Get projects
                                projects = handler.get_projects()

                                # 2. Create pending volume
                                pending_volume = handler.create_pending_volume(3711)

                                # 3. Setup volume with files
                                volume_info = handler.setup_volume_with_files(
                                    12345,
                                    "test_volume",
                                    "s3://test",
                                    ["file1.mp4", "file2.mp4"],
                                )

                                # 4. Export annotations
                                annotations = handler.export_annotations(12345)

                                # Assert
                                assert len(projects) == 1
                                assert pending_volume["id"] == 12345
                                assert volume_info["name"] == "test_volume"
                                assert annotations == mock_df

                                # Verify all API calls were made
                                assert (
                                    mock_api_instance.get.call_count == 2
                                )  # projects + download
                                assert (
                                    mock_api_instance.post.call_count == 2
                                )  # pending volume + report
                                assert (
                                    mock_api_instance.put.call_count == 1
                                )  # setup volume

    def test_convenience_methods(self):
        """Test convenience methods for common workflows."""
        # Arrange
        mock_biigle_module = MagicMock()
        mock_api_class = Mock()
        mock_biigle_module.Api = mock_api_class

        with patch.dict("sys.modules", {"biigle": mock_biigle_module}):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")

            # Test build_s3_url
            s3_url = handler.build_s3_url("biigle_clips/test_folder")
            assert s3_url == "disk-98://biigle_clips/test_folder/"

            s3_url_with_slash = handler.build_s3_url("biigle_clips/test_folder/")
            assert s3_url_with_slash == "disk-98://biigle_clips/test_folder/"

            # Test with custom disk_id
            s3_url_custom = handler.build_s3_url("biigle_clips/test", disk_id=99)
            assert s3_url_custom == "disk-99://biigle_clips/test/"
