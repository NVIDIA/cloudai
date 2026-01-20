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

import shutil
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Generator, cast
from unittest.mock import Mock, patch

import pytest

from cloudai.core import (
    BaseInstaller,
    DockerImage,
    File,
    GitRepo,
    HFModel,
    Installable,
    InstallStatusResult,
    PythonExecutable,
)
from cloudai.systems.kubernetes.kubernetes_installer import KubernetesInstaller
from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem
from cloudai.systems.slurm import SlurmInstaller, SlurmSystem
from cloudai.systems.slurm.docker_image_cache_manager import DockerImageCacheResult
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

    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult:
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

        assert mock_executor.return_value.__enter__.return_value.submit.call_count == 1 + len(
            installer.system.system_installables()
        )

    def test_uninstalls_only_uniq(self, mock_executor: Mock, installer: MyInstaller, docker_image: DockerImage):
        mock_executor.return_value.__enter__.return_value.submit.return_value = create_real_future(
            installer.uninstall_one(docker_image)
        )

        installer.uninstall([docker_image, docker_image])

        assert mock_executor.return_value.__enter__.return_value.submit.call_count == 1 + len(
            installer.system.system_installables()
        )

    def test_all_items_with_duplicates(self, _, installer: MyInstaller, docker_image: DockerImage):
        all_items = installer.all_items([docker_image, docker_image], with_duplicates=True)
        assert len(all_items) == 3
        assert all_items[:2] == [docker_image, docker_image]

    def test_all_items_without_duplicates(self, _, installer: MyInstaller, docker_image: DockerImage):
        all_items = installer.all_items([docker_image, docker_image])
        assert len(all_items) == 2
        assert docker_image in all_items


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://fake_url/img", "fake_url__img__notag.sqsh"),
        ("nvcr.io#nvidia/pytorch:24.02-py3", "nvcr.io_nvidia__pytorch__24.02-py3.sqsh"),
        ("/local/disk/file", "file__notag.sqsh"),
        ("gitlab.com#org/team/image:latest", "gitlab.com_org_team__image__latest.sqsh"),
    ],
)
def test_docker_cache_filename(url: str, expected: str):
    assert DockerImage(url).cache_filename == expected, f"Input: {url}"


def test_docker_image_installed_path():
    docker_image = DockerImage("fake_url/img")

    # Test with string path (URL)
    string_path = "fake_url/img"
    docker_image._installed_path = string_path
    assert docker_image.installed_path == "fake_url/img"

    # Test with Path object
    path_obj = Path("/another/path")
    docker_image._installed_path = path_obj
    assert isinstance(docker_image.installed_path, Path)
    assert docker_image.installed_path == path_obj.absolute()


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/NVIDIA/cloudai.git", "cloudai__commit"),
        ("git@github.com:NVIDIA/cloudai.git", "cloudai__commit"),
        ("./cloudai", "cloudai__commit"),
    ],
)
def test_git_repo_name(url: str, expected: str):
    assert GitRepo(url=url, commit="commit").repo_name == expected


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
        assert f"Output path '{p.absolute()}' exists but is not a directory." in caplog.text

    def test_not_writable(self, no_access_dir: Path, caplog: pytest.LogCaptureFixture):
        assert prepare_output_dir(no_access_dir) is None
        assert f"Output path '{no_access_dir.absolute()}' exists but is not writable." in caplog.text

    def test_parent_wo_access(self, no_access_dir: Path, caplog: pytest.LogCaptureFixture):
        subdir = no_access_dir / "subdir"
        assert prepare_output_dir(subdir) is None
        assert f"Output path '{subdir.absolute()}' is not writable." in caplog.text

    def test_invalid_path_no_stacktrace(self, caplog: pytest.LogCaptureFixture):
        invalid_path = Path("/non/existent/path")
        assert prepare_output_dir(invalid_path) is None
        assert "Traceback" not in caplog.text


def test_system_installables_are_used(slurm_system: SlurmSystem):
    installer = MyInstaller(slurm_system)
    installer.install_one = Mock(return_value=InstallStatusResult(True))
    installer.uninstall_one = Mock(return_value=InstallStatusResult(True))
    installer.is_installed_one = Mock(return_value=InstallStatusResult(True))
    installer.mark_as_installed_one = Mock(return_value=InstallStatusResult(True))
    installer._populate_successful_install = Mock()

    installer.install([])
    assert installer.install_one.call_count == len(slurm_system.system_installables())

    installer.uninstall([])
    assert installer.uninstall_one.call_count == len(slurm_system.system_installables())

    installer.is_installed([])
    assert installer.is_installed_one.call_count == len(slurm_system.system_installables())

    installer.mark_as_installed([])
    assert installer.mark_as_installed_one.call_count == len(slurm_system.system_installables())


class TestSuccessIsPopulated:
    @pytest.fixture
    def installer(self, slurm_system: SlurmSystem):
        slurm_system.install_path.mkdir(parents=True, exist_ok=True)
        installer = SlurmInstaller(slurm_system)
        installer._check_prerequisites = Mock(return_value=InstallStatusResult(True))
        # make sure system installables are installed
        for ins in installer.system.system_installables():
            item = cast(File, ins)
            shutil.copyfile(item.src, installer.system.install_path / item.src.name, follow_symlinks=False)

        return installer

    def test_both_installed(self, installer: SlurmInstaller):
        f1, f2 = File(src=Path(__file__)), File(src=Path(__file__))
        res = installer.install([f1, f2])
        assert res.success
        assert f1._installed_path is not None, "First file is not installed"
        assert f2._installed_path is not None, "Second file is not installed"
        assert f1._installed_path == f2._installed_path, "Files are installed to different paths"

    def test_both_is_installed(self, installer: SlurmInstaller):
        f = installer.system.install_path / "file"
        f.touch()
        f1, f2 = File(src=f), File(src=f)
        res = installer.is_installed([f1, f2])
        assert res.success, res.message
        assert f1._installed_path is not None, "First file is not installed"
        assert f2._installed_path is not None, "Second file is not installed"
        assert f1._installed_path == f2._installed_path, "Files are installed to different paths"

    def test_order_of_items_does_not_matter(self, installer: SlurmInstaller):
        f = installer.system.install_path / "file"
        f1, f2 = File(src=f), File(src=f)
        assert f1._installed_path is None, "First file is installed before testing"
        assert f2._installed_path is None, "Second file is installed before testing"

        installer._populate_successful_install([f1, f2], {})
        assert f1._installed_path is None, "First file was marked as installed, but should not be"
        assert f2._installed_path is None, "Second file was marked as installed, but should not be"

        installer._populate_successful_install([f1, f2], {f1: InstallStatusResult(success=True)})
        assert f1._installed_path is not None, (
            "First ('self', present in the statuses) file was not marked as installed"
        )
        assert f2._installed_path is not None, "Second file was not marked as installed"

        f1._installed_path, f2._installed_path = None, None
        installer._populate_successful_install([f2, f1], {f1: InstallStatusResult(success=True)})
        assert f1._installed_path is not None, (
            "First ('self', present in the statuses) file was not marked as installed"
        )
        assert f2._installed_path is not None, "Second file was not marked as installed"

    def test_both_mark_as_installed(self, installer: SlurmInstaller):
        f1, f2 = File(src=Path(__file__)), File(src=Path(__file__))

        assert f1 == f2, "Files should be equal"
        assert f1 is not f2, "Files should be distinct objects"
        assert f1._installed_path is None, "First file should not be installed before test"
        assert f2._installed_path is None, "Second file should not be installed before test"

        res = installer.mark_as_installed([f1, f2])

        assert res.success, "mark_as_installed should succeed"
        assert f1._installed_path is not None, "First file should have installed_path set"
        assert f2._installed_path is not None, "Second file should have installed_path set"
        assert f1._installed_path == f2._installed_path, "Files should have same installed_path"


@pytest.fixture(params=["k8s", "slurm"])
def installer(
    request: Any, k8s_system: KubernetesSystem, slurm_system: SlurmSystem
) -> KubernetesInstaller | SlurmInstaller:
    installer = KubernetesInstaller(k8s_system) if request.param == "k8s" else SlurmInstaller(slurm_system)

    installer.system.install_path.mkdir(parents=True)
    installer._check_low_thread_environment = lambda threshold=None: False
    return installer


def test_check_supported(installer: KubernetesInstaller | SlurmInstaller):
    if isinstance(installer, SlurmInstaller):
        installer._install_docker_image = lambda item: DockerImageCacheResult(True)
        installer._uninstall_docker_image = lambda item: DockerImageCacheResult(True)
        installer.docker_image_cache_manager.check_docker_image_exists = Mock(return_value=DockerImageCacheResult(True))
    installer._install_python_executable = lambda item: InstallStatusResult(True)
    installer._uninstall_python_executable = lambda item: InstallStatusResult(True)
    installer._is_python_executable_installed = lambda item: InstallStatusResult(True)
    installer.hf_model_manager = Mock()

    git = GitRepo(url="./git_url", commit="commit_hash")
    items = [DockerImage("fake_url/img"), PythonExecutable(git), HFModel("model_name"), File(Path(__file__))]
    for item in items:
        res = installer.install_one(item)
        assert res.success, f"Failed to install {item} for {installer.__class__.__name__=} {res.message=}"

        res = installer.is_installed_one(item)
        assert res.success, f"Failed to check installation of {item} for {installer.__class__.__name__=} {res.message=}"

        res = installer.uninstall_one(item)
        assert res.success, f"Failed to uninstall {item} for {installer.__class__.__name__=} {res.message=}"

        res = installer.mark_as_installed_one(item)
        assert res.success, f"Failed to mark as installed {item} for {installer.__class__.__name__=} {res.message=}"


class MyInstallable(Installable):
    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return hash("MyInstallable")


def test_check_unsupported(installer: KubernetesInstaller | SlurmInstaller):
    unsupported = MyInstallable()
    for func in [
        installer.install_one,
        installer.uninstall_one,
        installer.is_installed_one,
        installer.mark_as_installed_one,
    ]:
        res = func(unsupported)
        assert not res.success
        assert res.message == f"Unsupported item type: {type(unsupported)}"
