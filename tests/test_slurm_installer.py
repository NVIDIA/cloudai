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
from subprocess import CompletedProcess
from typing import cast
from unittest.mock import Mock, patch

import pytest

from cloudai.core import DockerImage, File, GitRepo, InstallStatusResult, PythonExecutable
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


class TestInstallOnePythonExecutable:
    @pytest.fixture
    def git(self):
        return GitRepo(url="./git_url", commit="commit_hash")

    @pytest.fixture
    def setup_repo(self, installer: SlurmInstaller, git: GitRepo):
        repo_dir = installer.system.install_path / git.repo_name
        subdir = repo_dir / "subdir"

        repo_dir.mkdir(parents=True, exist_ok=True)
        subdir.mkdir(parents=True, exist_ok=True)

        pyproject_file = subdir / "pyproject.toml"
        requirements_file = subdir / "requirements.txt"

        pyproject_file.touch()
        requirements_file.touch()

        return repo_dir, subdir, pyproject_file, requirements_file

    def test_venv_created(self, installer: SlurmInstaller, git: GitRepo):
        py = PythonExecutable(git)
        venv_path = installer.system.install_path / py.venv_name
        installer._install_dependencies = Mock(return_value=InstallStatusResult(True))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._create_venv(py)
        assert res.success
        mock_run.assert_called_once_with(["python", "-m", "venv", str(venv_path)], capture_output=True, text=True)

    @pytest.mark.parametrize("failure_on_venv_creation,reqs_install_failure", [(True, False), (False, True)])
    def test_error_creating_venv(
        self, installer: SlurmInstaller, git: GitRepo, failure_on_venv_creation: bool, reqs_install_failure: bool
    ):
        py = PythonExecutable(git)
        venv_path = installer.system.install_path / py.venv_name

        if reqs_install_failure:
            installer._install_dependencies = Mock(return_value=InstallStatusResult(False, "err"))

        def mock_run(*args, **kwargs):
            venv_path.mkdir()
            if failure_on_venv_creation and "venv" in args[0]:
                return CompletedProcess(args=args, returncode=1, stderr="err")
            return CompletedProcess(args=args, returncode=0)

        with patch("subprocess.run", side_effect=mock_run):
            res = installer._create_venv(py)
        assert not res.success
        if failure_on_venv_creation:
            assert res.message == "Failed to create venv:\nSTDOUT:\nNone\nSTDERR:\nerr"
        else:
            assert res.message == "err"
        assert not venv_path.exists(), "venv folder wasn't removed after unsuccessful installation"

    def test_venv_already_exists(self, installer: SlurmInstaller, git: GitRepo):
        py = PythonExecutable(git)
        venv_path = installer.system.install_path / py.venv_name
        venv_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._create_venv(py)
        assert mock_run.call_count == 0
        assert res.success
        assert res.message == f"Virtual environment already exists at {venv_path}."

    def test_requirements_no_file(self, installer: SlurmInstaller, git: GitRepo):
        py = PythonExecutable(git)
        venv_path = installer.system.install_path / py.venv_name
        venv_path.mkdir()
        res = installer._install_requirements(venv_path, installer.system.install_path / "requirements.txt")
        assert not res.success
        assert (
            res.message
            == f"Requirements file is invalid or does not exist: {installer.system.install_path / 'requirements.txt'}"
        )

    def test_requirements_installed(self, installer: SlurmInstaller):
        requirements_file = installer.system.install_path / "requirements.txt"
        venv_path = installer.system.install_path / "venv"
        requirements_file.touch()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._install_requirements(venv_path, requirements_file)
        assert res.success
        mock_run.assert_called_once_with(
            [str(venv_path / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
        )

    def test_requirements_not_installed(self, installer: SlurmInstaller):
        requirements_file = installer.system.install_path / "requirements.txt"
        requirements_file.touch()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._install_requirements(installer.system.install_path, requirements_file)
        assert not res.success
        assert res.message == "Failed to install dependencies from requirements.txt: err"

    def test_all_good_flow(self, installer: SlurmInstaller, git: GitRepo):
        py = PythonExecutable(git)
        py.git_repo.installed_path = installer.system.install_path / py.git_repo.repo_name

        repo_dir = py.git_repo.installed_path
        repo_dir.mkdir(parents=True, exist_ok=True)
        pyproject_file = repo_dir / "pyproject.toml"
        pyproject_file.write_text("[tool.poetry]\nname = 'dummy_project'")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._install_python_executable(py)

        assert res.success
        assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
        assert py.venv_path == installer.system.install_path / py.venv_name

    def test_is_installed_no_repo(self, installer: SlurmInstaller, git: GitRepo):
        py = PythonExecutable(git)
        res = installer._is_python_executable_installed(py)
        assert not res.success
        assert res.message == f"Git repository {py.git_repo.url} not cloned"
        assert not (installer.system.install_path / py.git_repo.repo_name).exists()
        assert not py.git_repo.installed_path
        assert not (installer.system.install_path / py.venv_name).exists()
        assert not py.venv_path

    def test_is_installed_no_venv(self, installer: SlurmInstaller, git: GitRepo):
        py = PythonExecutable(git)
        (installer.system.install_path / py.git_repo.repo_name).mkdir()
        res = installer._is_python_executable_installed(py)
        assert not res.success
        assert res.message == f"Virtual environment not created for {py.git_repo.url}"
        assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
        assert (installer.system.install_path / py.git_repo.repo_name).exists()
        assert not (installer.system.install_path / py.venv_name).exists()
        assert not py.venv_path

    def test_is_installed_ok(self, installer: SlurmInstaller, git: GitRepo):
        py = PythonExecutable(git)
        (installer.system.install_path / py.git_repo.repo_name).mkdir()
        (installer.system.install_path / py.venv_name).mkdir()
        res = installer._is_python_executable_installed(py)
        assert res.success
        assert res.message == "Python executable installed"
        assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
        assert (installer.system.install_path / py.git_repo.repo_name).exists()
        assert py.venv_path == installer.system.install_path / py.venv_name
        assert py.venv_path

    def test_uninstall_no_venv(self, installer: SlurmInstaller, git: GitRepo):
        py = PythonExecutable(git)
        py.venv_path = installer.system.install_path / py.venv_name
        res = installer._uninstall_python_executable(py)
        assert res.success
        assert res.message == f"Virtual environment {py.venv_name} is not created."

    def test_uninstall_venv_removed_ok(self, installer: SlurmInstaller, git: GitRepo):
        py = PythonExecutable(git)
        (installer.system.install_path / py.venv_name).mkdir()
        (installer.system.install_path / py.venv_name / "file").touch()
        py.venv_path = installer.system.install_path / py.venv_name
        res = installer._uninstall_python_executable(py)
        assert res.success
        assert not (installer.system.install_path / py.venv_name).exists()
        assert not py.venv_path

    def test_install_python_executable_prefers_pyproject_toml(
        self, installer: SlurmInstaller, git: GitRepo, setup_repo
    ):
        repo_dir, subdir, _, _ = setup_repo

        py = PythonExecutable(git, project_subpath=Path("subdir"), dependencies_from_pyproject=True)

        installer._install_pyproject = Mock(return_value=InstallStatusResult(True))
        installer._install_requirements = Mock(return_value=InstallStatusResult(True))

        py.git_repo.installed_path = repo_dir

        res = installer._install_dependencies(py)

        assert res.success
        installer._install_pyproject.assert_called_once_with(installer.system.install_path / py.venv_name, subdir)
        installer._install_requirements.assert_not_called()

    def test_install_python_executable_prefers_requirements_txt(
        self, installer: SlurmInstaller, git: GitRepo, setup_repo
    ):
        repo_dir, *_ = setup_repo

        py = PythonExecutable(git, project_subpath=Path("subdir"), dependencies_from_pyproject=False)

        installer._install_requirements = Mock(return_value=InstallStatusResult(True))
        installer._install_pyproject = Mock(return_value=InstallStatusResult(True))

        py.git_repo.installed_path = repo_dir

        res = installer._install_dependencies(py)

        assert res.success
        installer._install_pyproject.assert_not_called()


class TestInstallOneFile:
    @pytest.fixture
    def f(self, tmp_path: Path) -> File:
        f = tmp_path / "file"
        f.write_text("content")
        return File(f)

    def test_no_dst(self, installer: SlurmInstaller, f: File):
        res = installer.install_one(f)
        assert res.success
        assert f.installed_path == installer.system.install_path / f.src.name
        assert f.installed_path.exists()
        assert f.installed_path.read_bytes() == f.src.read_bytes()

    def test_file_exists_but_overriden(self, installer: SlurmInstaller, f: File):
        f.installed_path = installer.system.install_path / f.src.name
        f.installed_path.touch()
        res = installer.install_one(f)
        assert res.success
        assert f.src.read_bytes() == f.installed_path.read_bytes()

    def test_is_installed_checks_content(self, installer: SlurmInstaller, f: File):
        f.installed_path = installer.system.install_path / f.src.name
        f.installed_path.touch()
        f.src.write_text("new content")

        res = installer.is_installed_one(f)
        assert not res.success


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
