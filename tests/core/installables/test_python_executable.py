# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import patch

import pytest

from cloudai.core import BaseInstaller, GitRepo, InstallStatusResult, PythonExecutable


@pytest.fixture
def git() -> GitRepo:
    return GitRepo(url="./git_url", commit="commit_hash")


@pytest.fixture
def installer(slurm_system):
    installer = BaseInstaller(slurm_system)
    installer.system.install_path.mkdir(parents=True)
    installer._check_low_thread_environment = lambda threshold=None: False
    return installer


@pytest.fixture
def setup_repo(installer: BaseInstaller, git: GitRepo):
    repo_dir = installer.system.install_path / git.repo_name
    subdir = repo_dir / "subdir"

    repo_dir.mkdir(parents=True, exist_ok=True)
    subdir.mkdir(parents=True, exist_ok=True)

    pyproject_file = subdir / "pyproject.toml"
    requirements_file = subdir / "requirements.txt"

    pyproject_file.touch()
    requirements_file.touch()

    return repo_dir, subdir, pyproject_file, requirements_file


def test_explicit_python_version_overrides_repo_version(tmp_path: Path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".python-version").write_text("3.10.16\n")
    py = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash", python_version="3.11.9"))

    assert py._resolve_python_version(repo_path) == "3.11.9"

    py.git_repo.python_version = ""
    assert py._resolve_python_version(repo_path) == "3.10.16"


def test_nearest_repo_python_version_is_used(tmp_path: Path):
    repo_path = tmp_path / "repo"
    project_path = repo_path / "packages" / "project"
    project_path.mkdir(parents=True)
    (repo_path / ".python-version").write_text("3.10.16\n")
    (repo_path / "packages" / ".python-version").write_text("3.11.9\n")
    py = PythonExecutable(
        GitRepo(url="./git_url", commit="commit_hash"),
        project_subpath=Path("packages/project"),
    )

    assert py._resolve_python_version(repo_path) == "3.11.9"


def test_python_version_falls_back_to_cloudai_interpreter(tmp_path: Path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (tmp_path / ".python-version").write_text("3.11.9\n")
    py = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))

    with patch("cloudai._core.installables.python_executable.sys.executable", "/cloudai/bin/python"):
        assert py._resolve_python_version(repo_path) == "/cloudai/bin/python"


def test_venv_created(installer: BaseInstaller, git: GitRepo):
    git.python_version = "3.11.9"
    git.installed_path = installer.system.install_path / git.repo_name
    py = PythonExecutable(git)
    venv_path = installer.system.install_path / py.venv_name
    with (
        patch("cloudai._core.installables.python_executable.resolve_uv_bin", return_value="/cloudai/bin/uv"),
        patch.object(PythonExecutable, "_install_dependencies", return_value=InstallStatusResult(True)),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        res = py._create_venv(installer)
    assert res.success
    mock_run.assert_called_once_with(
        ["/cloudai/bin/uv", "venv", "--python", "3.11.9", "--seed", str(venv_path)],
        cwd=str(git.installed_path),
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("failure_on_venv_creation,reqs_install_failure", [(True, False), (False, True)])
def test_error_creating_venv(
    installer: BaseInstaller,
    git: GitRepo,
    failure_on_venv_creation: bool,
    reqs_install_failure: bool,
):
    py = PythonExecutable(git)
    venv_path = installer.system.install_path / py.venv_name

    def mock_run(*args, **kwargs):
        venv_path.mkdir()
        if failure_on_venv_creation and "venv" in args[0]:
            return CompletedProcess(args=args, returncode=1, stderr="err")
        return CompletedProcess(args=args, returncode=0)

    dependencies_result = InstallStatusResult(False, "err") if reqs_install_failure else InstallStatusResult(True)
    with (
        patch("cloudai._core.installables.python_executable.resolve_uv_bin", return_value="/cloudai/bin/uv"),
        patch.object(PythonExecutable, "_install_dependencies", return_value=dependencies_result),
        patch("subprocess.run", side_effect=mock_run),
    ):
        res = py._create_venv(installer)
    assert not res.success
    if failure_on_venv_creation:
        assert res.message == "Failed to create venv using uv:\nSTDOUT:\nNone\nSTDERR:\nerr"
    else:
        assert res.message == "err"
    assert not venv_path.exists(), "venv folder wasn't removed after unsuccessful installation"


def test_venv_already_exists(installer: BaseInstaller, git: GitRepo):
    py = PythonExecutable(git)
    venv_path = installer.system.install_path / py.venv_name
    venv_path.mkdir()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
        res = py._create_venv(installer)
    assert mock_run.call_count == 0
    assert res.success
    assert res.message == f"Virtual environment already exists at {venv_path}."


def test_requirements_no_file(installer: BaseInstaller, git: GitRepo):
    py = PythonExecutable(git)
    venv_path = installer.system.install_path / py.venv_name
    venv_path.mkdir()
    res = py._install_requirements(venv_path, installer.system.install_path / "requirements.txt")
    assert not res.success
    assert (
        res.message
        == f"Requirements file is invalid or does not exist: {installer.system.install_path / 'requirements.txt'}"
    )


def test_requirements_installed(installer: BaseInstaller):
    requirements_file = installer.system.install_path / "requirements.txt"
    venv_path = installer.system.install_path / "venv"
    requirements_file.touch()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        res = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))._install_requirements(
            venv_path, requirements_file
        )
    assert res.success
    mock_run.assert_called_once_with(
        [str(venv_path / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements_file)],
        capture_output=True,
        text=True,
    )


def test_requirements_not_installed(installer: BaseInstaller):
    requirements_file = installer.system.install_path / "requirements.txt"
    requirements_file.touch()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
        res = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))._install_requirements(
            installer.system.install_path, requirements_file
        )
    assert not res.success
    assert res.message == "Failed to install dependencies from requirements.txt: err"


def test_all_good_flow(installer: BaseInstaller, git: GitRepo):
    py = PythonExecutable(git)
    py.git_repo.installed_path = installer.system.install_path / py.git_repo.repo_name

    repo_dir = py.git_repo.installed_path
    repo_dir.mkdir(parents=True, exist_ok=True)
    pyproject_file = repo_dir / "pyproject.toml"
    pyproject_file.write_text("[tool.poetry]\nname = 'dummy_project'")

    with (
        patch("cloudai._core.installables.python_executable.resolve_uv_bin", return_value="/cloudai/bin/uv"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout=f"{git.commit}\n", stderr="")
        res = py.install(installer)

    assert res.success
    assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
    assert py.venv_path == installer.system.install_path / py.venv_name


def test_is_installed_no_repo(installer: BaseInstaller, git: GitRepo):
    py = PythonExecutable(git)
    res = py.is_installed(installer)
    assert not res.success
    assert res.message == f"Git repository {py.git_repo.url} not cloned"
    assert not (installer.system.install_path / py.git_repo.repo_name).exists()
    assert not py.git_repo.installed_path
    assert not (installer.system.install_path / py.venv_name).exists()
    assert not py.venv_path


def test_is_installed_no_venv(installer: BaseInstaller, git: GitRepo):
    py = PythonExecutable(git)
    (installer.system.install_path / py.git_repo.repo_name).mkdir()
    res = py.is_installed(installer)
    assert not res.success
    assert res.message == f"Virtual environment not created for {py.git_repo.url}"
    assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
    assert (installer.system.install_path / py.git_repo.repo_name).exists()
    assert not (installer.system.install_path / py.venv_name).exists()
    assert not py.venv_path


def test_is_installed_ok(installer: BaseInstaller, git: GitRepo):
    py = PythonExecutable(git)
    (installer.system.install_path / py.git_repo.repo_name).mkdir()
    (installer.system.install_path / py.venv_name).mkdir()
    res = py.is_installed(installer)
    assert res.success
    assert res.message == "Python executable installed"
    assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
    assert (installer.system.install_path / py.git_repo.repo_name).exists()
    assert py.venv_path == installer.system.install_path / py.venv_name
    assert py.venv_path


def test_uninstall_no_venv(installer: BaseInstaller, git: GitRepo):
    py = PythonExecutable(git)
    py.venv_path = installer.system.install_path / py.venv_name
    res = py.uninstall(installer)
    assert res.success
    assert res.message == f"Virtual environment {py.venv_name} is not created."


def test_uninstall_venv_removed_ok(installer: BaseInstaller, git: GitRepo):
    py = PythonExecutable(git)
    (installer.system.install_path / py.venv_name).mkdir()
    (installer.system.install_path / py.venv_name / "file").touch()
    py.venv_path = installer.system.install_path / py.venv_name
    res = py.uninstall(installer)
    assert res.success
    assert not (installer.system.install_path / py.venv_name).exists()
    assert not py.venv_path


def test_install_python_executable_prefers_pyproject_toml(
    installer: BaseInstaller,
    git: GitRepo,
    setup_repo,
):
    repo_dir, subdir, _, _ = setup_repo

    py = PythonExecutable(git, project_subpath=Path("subdir"), dependencies_from_pyproject=True)
    py.git_repo.installed_path = repo_dir

    with (
        patch.object(PythonExecutable, "_install_pyproject", return_value=InstallStatusResult(True)) as pyproject,
        patch.object(PythonExecutable, "_install_requirements", return_value=InstallStatusResult(True)) as reqs,
    ):
        res = py._install_dependencies(installer)

    assert res.success
    pyproject.assert_called_once_with(installer.system.install_path / py.venv_name, subdir)
    reqs.assert_not_called()


def test_install_python_executable_prefers_requirements_txt(
    installer: BaseInstaller,
    git: GitRepo,
    setup_repo,
):
    repo_dir, *_ = setup_repo

    py = PythonExecutable(git, project_subpath=Path("subdir"), dependencies_from_pyproject=False)
    py.git_repo.installed_path = repo_dir

    with (
        patch.object(PythonExecutable, "_install_requirements", return_value=InstallStatusResult(True)) as reqs,
        patch.object(PythonExecutable, "_install_pyproject", return_value=InstallStatusResult(True)) as pyproject,
    ):
        res = py._install_dependencies(installer)

    assert res.success
    pyproject.assert_not_called()
    reqs.assert_called_once()
