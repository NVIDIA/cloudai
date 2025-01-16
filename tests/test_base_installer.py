# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from concurrent.futures import Future
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from cloudai import BaseInstaller, InstallStatusResult
from cloudai.installer.installables import DockerImage, GitRepo, Installable
from cloudai.systems import SlurmSystem
from cloudai.util import prepare_output_dir


def create_real_future(result):
    future = Future()
    future.set_result(result)
    return future


class MyInstaller(BaseInstaller):
    def install_one(self, item: Installable) -> InstallStatusResult:
        return InstallStatusResult(success=True)

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        return InstallStatusResult(success=True)

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        return InstallStatusResult(success=True)


@pytest.fixture
def docker_image() -> DockerImage:
    return DockerImage("fake_url/img")


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
class TestBaseInstaller:
    @pytest.fixture
    def installer(self, slurm_system: SlurmSystem):
        return MyInstaller(slurm_system)

    def test_install_success(self, mock_executor: Mock, installer: MyInstaller, docker_image: DockerImage):
        mock_executor.return_value.__enter__.return_value.submit.return_value = create_real_future(
            InstallStatusResult(success=True)
        )

        result = installer.install([docker_image])

        assert result.success
        assert result.message == "All items installed successfully."

    def test_install_failure(self, mock_executor: Mock, installer: MyInstaller, docker_image: DockerImage):
        mock_executor.return_value.__enter__.return_value.submit.return_value = create_real_future(
            InstallStatusResult(False)
        )

        result = installer.install([docker_image])

        assert not result.success
        assert result.message == "1 item(s) failed to install."

    def test_uninstall_success(self, mock_executor: Mock, installer: MyInstaller, docker_image: DockerImage):
        mock_executor.return_value.__enter__.return_value.submit.return_value = create_real_future(
            InstallStatusResult(True)
        )

        result = installer.uninstall([docker_image])

        assert result.success
        assert result.message == "All items uninstalled successfully."

    def test_uninstall_failure(self, mock_executor: Mock, installer: MyInstaller, docker_image: DockerImage):
        mock_executor.return_value.__enter__.return_value.submit.return_value = create_real_future(
            InstallStatusResult(False)
        )

        result = installer.uninstall([docker_image])

        assert not result.success
        assert result.message == "1 item(s) failed to uninstall."

    def test_installs_only_uniq(self, mock_executor: Mock, installer: MyInstaller, docker_image: DockerImage):
        mock_executor.return_value.__enter__.return_value.submit.return_value = create_real_future(0)

        installer.install([docker_image, docker_image])

        assert mock_executor.return_value.__enter__.return_value.submit.call_count == 1

    def test_uninstalls_only_uniq(self, mock_executor: Mock, installer: MyInstaller, docker_image: DockerImage):
        mock_executor.return_value.__enter__.return_value.submit.return_value = create_real_future(0)

        installer.uninstall([docker_image, docker_image])

        assert mock_executor.return_value.__enter__.return_value.submit.call_count == 1


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://fake_url/img", "fake_url__img__notag.sqsh"),
        ("nvcr.io/nvidia/pytorch:24.02-py3", "nvcr.io_nvidia__pytorch__24.02-py3.sqsh"),
        ("/local/disk/file", "file__notag.sqsh"),
    ],
)
def test_docker_cache_filename(url: str, expected: str):
    assert DockerImage(url).cache_filename == expected, f"Input: {url}"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/NVIDIA/cloudai.git", "cloudai__commit"),
        ("git@github.com:NVIDIA/cloudai.git", "cloudai__commit"),
        ("./cloudai", "cloudai__commit"),
    ],
)
def test_git_repo_name(url: str, expected: str):
    assert GitRepo(url, "commit").repo_name == expected


@pytest.fixture
def no_access_dir(tmp_path: Path) -> Generator[Path, None, None]:
    d = tmp_path / "no-access-dir"
    d.mkdir(exist_ok=True)
    d.chmod(0o000)
    yield d
    d.chmod(0o777)  # restore access so it can be deleted


class TestPrepareOutputDir:
    def test_already_exists(self, tmp_path: Path):
        assert prepare_output_dir(tmp_path) == tmp_path

    def test_not_exists(self, tmp_path: Path):
        assert prepare_output_dir(tmp_path / "new-dir") == tmp_path / "new-dir"

    def test_exists_but_file(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        p = tmp_path / "file"
        p.touch()
        assert prepare_output_dir(p) is None
        assert f"Output path {p} exists and is not a directory." in caplog.text

    def test_not_writable(self, no_access_dir: Path, caplog: pytest.LogCaptureFixture):
        assert prepare_output_dir(no_access_dir) is None
        assert f"Output path {no_access_dir} exists but is not writable." in caplog.text

    def test_parent_wo_access(self, no_access_dir: Path, caplog: pytest.LogCaptureFixture):
        subdir = no_access_dir / "subdir"
        assert prepare_output_dir(subdir) is None
        assert f"Output path {subdir} is not accessible:" in caplog.text
