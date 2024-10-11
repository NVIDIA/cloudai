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
from unittest.mock import MagicMock, Mock, patch

import pytest
from cloudai import InstallStatusResult, TestTemplate
from cloudai._core.base_installer import BaseInstaller
from cloudai._core.test import DockerImage, Installable, Test
from cloudai.installer.slurm_installer import SlurmInstaller
from cloudai.systems import SlurmSystem


@pytest.fixture
def test_success() -> Test:
    template = MagicMock(spec=TestTemplate)
    template.name = "test_template_success"
    template.install.return_value = InstallStatusResult(success=True)
    template.uninstall.return_value = InstallStatusResult(success=True)
    return template


@pytest.fixture
def test_failure() -> Test:
    template = MagicMock(spec=TestTemplate)
    template.name = "test_template_failure"
    template.install.return_value = InstallStatusResult(success=False, message="Installation failed")
    template.uninstall.return_value = InstallStatusResult(success=False, message="Uninstallation failed")
    return template


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


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_install_success(mock_executor: Mock, slurm_system: SlurmSystem, test_success: Mock):
    installer = MyInstaller(slurm_system)
    mock_future = create_real_future(test_success.install.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.install([test_success])

    assert result.success
    assert result.message == "All test templates installed successfully."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_install_failure(mock_executor: Mock, slurm_system: SlurmSystem, test_failure: Mock):
    installer = MyInstaller(slurm_system)
    mock_future = create_real_future(test_failure.install.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.install([test_failure])

    assert not result.success
    assert result.message == "Some test templates failed to install."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_uninstall_success(mock_executor: Mock, slurm_system: SlurmSystem, test_success: Mock):
    installer = MyInstaller(slurm_system)
    mock_future = create_real_future(test_success.uninstall.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.uninstall([test_success])

    assert result.success
    assert result.message == "All test templates uninstalled successfully."


@patch("cloudai._core.base_installer.ThreadPoolExecutor", autospec=True)
def test_uninstall_failure(mock_executor: Mock, slurm_system: SlurmSystem, test_failure: Mock):
    installer = MyInstaller(slurm_system)
    mock_future = create_real_future(test_failure.uninstall.return_value)
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = installer.uninstall([test_failure])

    assert not result.success
    assert result.message == "Some test templates failed to uninstall."


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


class TestInstallOneDocker:
    @pytest.fixture
    def installer(self, slurm_system: SlurmSystem):
        si = SlurmInstaller(slurm_system)
        si.install_path.mkdir()
        return si

    def test_image_is_local(self, installer: SlurmInstaller):
        cached_file = installer.system.install_path / "some_image"
        d = DockerImage(str(cached_file))
        cached_file.touch()
        res = installer._install_docker_image(d)

        assert res.success
        assert res.message == f"Docker image file path is valid: {cached_file}."
        assert d.installed_path == cached_file

    def test_image_is_already_cached(self, installer: SlurmInstaller):
        d = DockerImage("fake_url/img")
        cached_file = installer.system.install_path / d.cache_filename
        cached_file.touch()

        res = installer._install_docker_image(d)

        assert res.success
        assert res.message == f"Cached Docker image already exists at {cached_file}."
        assert d.installed_path == cached_file

    def test_uninstall_docker_image(self, installer: SlurmInstaller):
        d = DockerImage("fake_url/img")
        cached_file = installer.system.install_path / d.cache_filename
        cached_file.touch()

        res = installer._uninstall_docker_image(d)

        assert res.success
        assert res.message == f"Cached Docker image removed successfully from {cached_file}."
        assert d.installed_path == d.url

    def test_is_installed_when_cache_exists(self, installer: SlurmInstaller):
        d = DockerImage("fake_url/img")
        cached_file = installer.system.install_path / d.cache_filename
        cached_file.touch()

        res = installer.is_installed_one(d)

        assert res.success
        assert res.message == f"Cached Docker image already exists at {cached_file}."
        assert d.installed_path == cached_file
