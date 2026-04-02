# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test case: cache_docker_images_locally = false with DockerURL type.

When cache_docker_images_locally=false in system TOML:
1. cloudai install doesn't set DockerImage._installed_path
2. DockerImage.installed_path returns DockerURL(self.url)
3. slurm_command_gen_strategy.py checks isinstance(installed, Path) vs DockerURL
4. DockerURL is passed directly to pyxis (it pulls from registry)

The fix uses a custom DockerURL type to distinguish URLs from Paths.
"""

from pathlib import Path

from cloudai._core.installables import DockerImage
from cloudai._core.types import DockerURL


class TestCacheDockerImagesLocallyFalse:
    """Test cases for cache_docker_images_locally=false with DockerURL type."""

    def test_installed_path_returns_docker_url_when_not_cached(self):
        """When _installed_path is None, installed_path returns a DockerURL."""
        docker_url = "nvcr.io/nvidian/nemo:26.04.rc2"
        img = DockerImage(url=docker_url)
        
        assert img._installed_path is None, "Precondition: _installed_path should be None"
        
        result = img.installed_path
        
        # Returns DockerURL type (not plain str)
        assert isinstance(result, DockerURL), f"Expected DockerURL, got {type(result)}"
        assert result == docker_url

    def test_installed_path_returns_path_when_cached(self):
        """When _installed_path is set, installed_path returns a Path."""
        docker_url = "nvcr.io/nvidian/nemo:26.04.rc2"
        sqsh_path = Path("/install/nvcr.io_nvidian__nemo__26.04.rc2.sqsh")
        
        img = DockerImage(url=docker_url)
        img.installed_path = sqsh_path
        
        result = img.installed_path
        
        # Returns Path type (not DockerURL)
        assert isinstance(result, Path), f"Expected Path, got {type(result)}"
        assert result == sqsh_path.absolute()

    def test_docker_url_is_subclass_of_str(self):
        """DockerURL is a str subclass, so it works with string operations."""
        url = DockerURL("nvcr.io/nvidian/nemo:26.04.rc2")
        
        assert isinstance(url, str)
        assert isinstance(url, DockerURL)
        assert "nvcr.io" in url
        assert url.startswith("nvcr.io")

    def test_type_check_distinguishes_url_from_path(self):
        """isinstance() can distinguish DockerURL from Path."""
        docker_url = "nvcr.io/nvidian/nemo:26.04.rc2"
        img = DockerImage(url=docker_url)
        
        # Not cached - returns DockerURL
        result_uncached = img.installed_path
        assert isinstance(result_uncached, DockerURL)
        assert not isinstance(result_uncached, Path)
        
        # Cached - returns Path
        img.installed_path = Path("/install/image.sqsh")
        result_cached = img.installed_path
        assert isinstance(result_cached, Path)
        assert not isinstance(result_cached, DockerURL)

    def test_container_path_resolution_logic(self):
        """
        Test the correct container path resolution logic.
        
        This is what slurm_command_gen_strategy.py should do:
        - Path: call .absolute() and convert to str
        - DockerURL: pass directly as str
        """
        def resolve_container_path(installed) -> str:
            if isinstance(installed, Path):
                return str(installed.absolute())
            # DockerURL - pass directly to pyxis
            return str(installed)
        
        # DockerURL case
        url_result = resolve_container_path(DockerURL("nvcr.io/nvidian/nemo:26.04.rc2"))
        assert url_result == "nvcr.io/nvidian/nemo:26.04.rc2"
        assert not url_result.startswith("/")  # Not mangled into a local path
        
        # Path case
        path_result = resolve_container_path(Path("/install/image.sqsh"))
        assert path_result == "/install/image.sqsh"

    def test_cache_filename_generation(self):
        """Verify cache filename is correctly generated from docker URL."""
        docker_url = "nvcr.io/nvidian/nemo:26.04.rc2"
        img = DockerImage(url=docker_url)
        
        expected = "nvcr.io_nvidian__nemo__26.04.rc2.sqsh"
        assert img.cache_filename == expected
