# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from typing import cast

import pytest

from cloudai.core import DockerImage, PythonExecutable
from cloudai.systems.slurm.slurm_installer import SlurmInstaller
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nemo_launcher import NeMoLauncherCmdArgs, NeMoLauncherTestDefinition


@pytest.fixture
def installer(slurm_system: SlurmSystem):
    si = SlurmInstaller(slurm_system)
    si.system.install_path.mkdir()
    si._check_low_thread_environment = lambda threshold=None: False
    return si


class TestInstallOneDocker:
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
        assert not cached_file.exists(), "Cache file should be deleted after uninstallation"

    def test_is_installed_when_cache_exists(self, installer: SlurmInstaller):
        d = DockerImage("fake_url/img")
        cached_file = installer.system.install_path / d.cache_filename
        cached_file.touch()

        res = installer.is_installed_one(d)

        assert res.success
        assert res.message == f"Cached Docker image already exists at {cached_file}."
        assert d.installed_path == cached_file

    def test_cache_disabled(self, installer: SlurmInstaller):
        d = DockerImage("fake_url/img")
        installer.system.cache_docker_images_locally = False
        res = installer.is_installed_one(d)
        assert res.success
        assert d.installed_path == d.url


def test_mark_as_installed(slurm_system: SlurmSystem):
    tdef = NeMoLauncherTestDefinition(
        name="name", description="desc", test_template_name="tt", cmd_args=NeMoLauncherCmdArgs()
    )
    docker = cast(DockerImage, tdef.installables[0])
    py_script = cast(PythonExecutable, tdef.installables[1])

    installer = SlurmInstaller(slurm_system)
    res = installer.mark_as_installed(tdef.installables)

    assert res.success
    assert docker.installed_path == slurm_system.install_path / docker.cache_filename
    assert py_script.git_repo.installed_path == slurm_system.install_path / py_script.git_repo.repo_name


@pytest.mark.parametrize("cache", [True, False])
def test_mark_as_installed_docker_image_system_is_respected(slurm_system: SlurmSystem, cache: bool):
    slurm_system.cache_docker_images_locally = cache
    docker = DockerImage(url="fake_url/img")
    installer = SlurmInstaller(slurm_system)
    res = installer.mark_as_installed([docker])
    assert res.success
    if cache:
        assert docker.installed_path == slurm_system.install_path / docker.cache_filename
    else:
        assert docker.installed_path == docker.url


def test_mark_as_installed_local_container(slurm_system: SlurmSystem):
    """
    Test for marking a local docker image as installed.

    The issue appeared when a DockerImage with existing local path was marked as installed,
    and installed_path was overwritten with a default value.
    """
    installer = SlurmInstaller(slurm_system)
    slurm_system.install_path.mkdir(parents=True, exist_ok=True)
    local_image = slurm_system.install_path / "local_image.sqsh"
    local_image.touch()

    docker_image = DockerImage(url=str(local_image.absolute()))
    docker_image.installed_path = local_image  # simulate installation

    installer.mark_as_installed_one(docker_image)

    assert docker_image.installed_path == local_image.absolute()


def test_mark_as_installed_duplicate_local_containers_preserve_local_path(slurm_system: SlurmSystem):
    installer = SlurmInstaller(slurm_system)

    local_image = Path("/tmp/local_image.sqsh")
    first = DockerImage(url=str(local_image))
    second = DockerImage(url=str(local_image))

    result = installer.mark_as_installed([first, second])

    assert result.success
    assert first.installed_path == local_image
    assert second.installed_path == local_image
