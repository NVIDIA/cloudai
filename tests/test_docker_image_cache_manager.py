import subprocess
import unittest
from unittest.mock import patch

import requests
from cloudai.util.docker_image_cache_manager import (
    DockerImageCacheManager,
    DockerImageCacheResult,
)


class TestDockerImageCacheManager(unittest.TestCase):
    @patch("os.path.isfile")
    @patch("os.path.exists")
    @patch("os.access")
    def test_ensure_docker_image_file_exists(self, mock_access, mock_exists, mock_isfile):
        manager = DockerImageCacheManager("/fake/install/path", True)
        mock_isfile.return_value = True
        mock_exists.return_value = True
        result = manager.ensure_docker_image("/tmp/existing_file.sqsh", "subdir", "docker_image.sqsh")
        assert result.success
        assert result.docker_image_path == "/tmp/existing_file.sqsh"
        assert result.message == "Docker image file path is valid."

    @patch("os.path.isfile")
    @patch("os.path.exists")
    @patch("os.access")
    def test_ensure_docker_image_url_cache_enabled(self, mock_access, mock_exists, mock_isfile):
        manager = DockerImageCacheManager("/fake/install/path", True)
        mock_isfile.return_value = False
        mock_exists.return_value = True
        mock_access.return_value = True
        with patch.object(
            manager,
            "cache_docker_image",
            return_value=DockerImageCacheResult(
                True, "/fake/install/path/subdir/docker_image.sqsh", "Docker image cached successfully."
            ),
        ):
            result = manager.ensure_docker_image("docker.io/hello-world", "subdir", "docker_image.sqsh")
            assert result.success
            assert result.docker_image_path == "/fake/install/path/subdir/docker_image.sqsh"
            assert result.message == "Docker image cached successfully."

    @patch("os.path.isfile")
    @patch("shutil.which")
    @patch("subprocess.run")
    @patch("cloudai.util.docker_image_cache_manager.DockerImageCacheManager._check_prerequisites")
    def test_cache_docker_image(self, mock_check_prerequisites, mock_run, mock_which, mock_isfile):
        manager = DockerImageCacheManager("/fake/install/path", True)

        # Test when cached file already exists
        mock_isfile.return_value = True
        result = manager.cache_docker_image("docker.io/hello-world", "subdir", "image.tar.gz")
        assert result.success, f"Expected success, but got failure: {result.message}"
        assert result.message == "Cached Docker image already exists."

        # Ensure prerequisites are always met for the following tests
        mock_check_prerequisites.return_value = DockerImageCacheResult(True, "", "All prerequisites are met.")

        # Test caching failure due to subprocess error
        mock_isfile.return_value = False
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
        result = manager.cache_docker_image("docker.io/hello-world", "subdir", "image.tar.gz")
        assert not result.success, f"Expected failure, but got success: {result.message}"
        assert "Install path" in result.message and "does not exist" in result.message

    @patch("os.path.isfile")
    @patch("os.path.exists")
    @patch("os.access")
    @patch("os.remove")
    def test_remove_cached_image(self, mock_remove, mock_access, mock_exists, mock_isfile):
        manager = DockerImageCacheManager("/fake/install/path", True)

        # Test successful removal
        mock_isfile.return_value = True
        result = manager.remove_cached_image("subdir", "image.tar.gz")
        assert result.success
        assert result.message == "Cached Docker image removed successfully."
        mock_remove.assert_called_once_with("/fake/install/path/subdir/image.tar.gz")

        # Test failed removal due to OSError
        mock_remove.side_effect = OSError("Mocked OSError")
        result = manager.remove_cached_image("subdir", "image.tar.gz")
        assert not result.success
        assert (
            result.message
            == "Failed to remove cached Docker image at /fake/install/path/subdir/image.tar.gz. Error: Mocked OSError"
        )

        # Test no file to remove
        mock_isfile.return_value = False
        result = manager.remove_cached_image("subdir", "image.tar.gz")
        assert result.success
        assert result.message == "No cached Docker image found to remove."

    @patch("os.path.isfile")
    @patch("os.path.exists")
    @patch("os.access")
    @patch("os.remove")
    def test_uninstall_cached_image(self, mock_remove, mock_access, mock_exists, mock_isfile):
        manager = DockerImageCacheManager("/fake/install/path", True)

        # Test successful uninstallation and subdirectory removal
        mock_isfile.return_value = True
        mock_exists.return_value = True
        mock_access.return_value = True
        mock_listdir = patch("os.listdir", return_value=[]).start()
        result = manager.uninstall_cached_image("subdir", "image.tar.gz")
        assert result.success
        assert (
            result.message == "Cached Docker image uninstalled successfully."
            or result.message == "Subdirectory removed successfully."
        )
        mock_remove.assert_called_once_with("/fake/install/path/subdir/image.tar.gz")

        # Test successful uninstallation but subdirectory not empty
        mock_listdir.return_value = ["otherfile"]
        result = manager.uninstall_cached_image("subdir", "image.tar.gz")
        assert result.success
        assert result.message == "Cached Docker image uninstalled successfully."
        mock_remove.assert_called_with("/fake/install/path/subdir/image.tar.gz")

        # Test failed removal due to OSError
        mock_remove.side_effect = OSError("Mocked OSError")
        result = manager.uninstall_cached_image("subdir", "image.tar.gz")
        assert not result.success
        assert (
            result.message
            == "Failed to remove cached Docker image at /fake/install/path/subdir/image.tar.gz. Error: Mocked OSError"
        )

        # Test no subdirectory to remove
        mock_isfile.return_value = False
        mock_exists.return_value = False
        result = manager.uninstall_cached_image("subdir", "image.tar.gz")
        assert result.success
        assert result.message == "Cached Docker image uninstalled successfully."

        # Cleanup
        patch.stopall()

    @patch("shutil.which")
    @patch("requests.head")
    def test_check_prerequisites(self, mock_head, mock_which):
        manager = DockerImageCacheManager("/fake/install/path", True)

        # Ensure enroot and srun are installed
        mock_which.side_effect = lambda x: x in ["enroot", "srun"]

        # Test all prerequisites met
        mock_head.return_value.status_code = 200
        result = manager._check_prerequisites("https://registry-1.docker.io/v2/library/hello-world/manifests/latest")
        assert result.success
        assert result.message == "All prerequisites are met."

        # Test enroot not installed
        mock_which.side_effect = lambda x: x != "enroot"
        result = manager._check_prerequisites("https://registry-1.docker.io/v2/library/hello-world/manifests/latest")
        assert not result.success
        assert result.message == "enroot are required for caching Docker images but are not installed."

        # Test srun not installed
        mock_which.side_effect = lambda x: x != "srun"
        result = manager._check_prerequisites("https://registry-1.docker.io/v2/library/hello-world/manifests/latest")
        assert not result.success
        assert result.message == "srun are required for caching Docker images but are not installed."

        # Ensure enroot and srun are installed again
        mock_which.side_effect = lambda x: x in ["enroot", "srun"]

        # Test Docker image URL not accessible
        mock_head.return_value.status_code = 404
        result = manager._check_prerequisites("https://registry-1.docker.io/v2/library/hello-world/manifests/latest")
        assert not result.success
        assert "Docker image URL" in result.message
        assert "not found" in result.message

        # Test unauthorized access to Docker image URL
        mock_head.return_value.status_code = 401
        result = manager._check_prerequisites("https://registry-1.docker.io/v2/library/hello-world/manifests/latest")
        assert result.success
        assert "All prerequisites are met." in result.message

        # Test exception while checking Docker image URL
        mock_head.side_effect = requests.RequestException("Mocked Exception")
        result = manager._check_prerequisites("https://registry-1.docker.io/v2/library/hello-world/manifests/latest")
        assert not result.success
        assert "Failed to check" in result.message

    @patch("requests.head")
    def test_check_docker_image_accessibility_success(self, mock_head):
        manager = DockerImageCacheManager("/fake/install/path", True)
        mock_head.return_value.status_code = 200
        result = manager._check_docker_image_accessibility(
            "registry-1.docker.io/v2/library/hello-world/manifests:latest"
        )
        self.assertTrue(result.success)
        self.assertEqual(
            result.message,
            "Docker image URL https://registry-1.docker.io/v2/library/hello-world/manifests:latest is accessible.",
        )

    @patch("requests.head")
    def test_check_docker_image_accessibility_not_found(self, mock_head):
        manager = DockerImageCacheManager("/fake/install/path", True)
        mock_head.return_value.status_code = 404
        result = manager._check_docker_image_accessibility(
            "registry-1.docker.io/v2/library/hello-world/manifests:latest"
        )
        self.assertFalse(result.success)
        self.assertIn("not found", result.message)

    @patch("requests.head")
    def test_check_docker_image_accessibility_unauthorized(self, mock_head):
        manager = DockerImageCacheManager("/fake/install/path", True)
        mock_head.return_value.status_code = 401
        result = manager._check_docker_image_accessibility(
            "registry-1.docker.io/v2/library/hello-world/manifests:latest"
        )
        self.assertTrue(result.success)
        self.assertIn("Unauthorized access", result.message)

    @patch("requests.head")
    def test_check_docker_image_accessibility_failure(self, mock_head):
        manager = DockerImageCacheManager("/fake/install/path", True)
        mock_head.return_value.status_code = 500
        result = manager._check_docker_image_accessibility(
            "registry-1.docker.io/v2/library/hello-world/manifests:latest"
        )
        self.assertFalse(result.success)
        self.assertIn("Failed to access", result.message)

    @patch("requests.head", side_effect=requests.RequestException("Test Exception"))
    def test_check_docker_image_accessibility_exception(self, mock_head):
        manager = DockerImageCacheManager("/fake/install/path", True)
        result = manager._check_docker_image_accessibility(
            "registry-1.docker.io/v2/library/hello-world/manifests:latest"
        )
        self.assertFalse(result.success)
        self.assertIn("Failed to check", result.message)


if __name__ == "__main__":
    unittest.main()
