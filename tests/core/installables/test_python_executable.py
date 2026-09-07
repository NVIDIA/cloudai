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

import subprocess
import sys
import threading
import time
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from cloudai.core import BaseInstaller, GitRepo, InstallStatusResult, PredictorConfig, PythonExecutable


@pytest.fixture
def git() -> GitRepo:
    return GitRepo(url="./git_url", commit="commit_hash")


@pytest.fixture
def installer(slurm_system) -> BaseInstaller:
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


def _create_python_file(venv_path: Path) -> Path:
    python_path = venv_path / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.touch()
    return python_path


def test_explicit_python_version_overrides_repository_pin(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    project_dir = repo_path / "package"
    project_dir.mkdir(parents=True)
    (project_dir / ".python-version").write_text("3.10.16\n")
    py = PythonExecutable(
        GitRepo(url="./git_url", commit="commit_hash", python_version="3.11.9"),
        project_subpath=Path("package"),
    )

    assert py._resolve_python_request(repo_path) == ("3.11.9", True)


def test_nearest_python_version_is_used_from_project_subpath(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    project_dir = repo_path / "packages" / "nested" / "project"
    project_dir.mkdir(parents=True)
    (repo_path / ".python-version").write_text("3.10.16\n")
    (repo_path / "packages" / ".python-version").write_text("3.11.9\n")
    py = PythonExecutable(
        GitRepo(url="./git_url", commit="commit_hash"),
        project_subpath=Path("packages/nested/project"),
    )

    assert py._resolve_python_request(repo_path) == ("3.11.9", True)


def test_python_version_lookup_is_bounded_by_repository_root(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    project_dir = repo_path / "package"
    project_dir.mkdir(parents=True)
    (tmp_path / ".python-version").write_text("9.9.9\n")
    py = PythonExecutable(
        GitRepo(url="./git_url", commit="commit_hash"),
        project_subpath=Path("package"),
    )

    with patch("cloudai._core.installables.python_executable.sys.executable", "/cloudai/bin/python"):
        assert py._resolve_python_request(repo_path) == ("/cloudai/bin/python", False)


@pytest.mark.parametrize("filename", [".python-versions", ".tool-versions", "runtime.txt", "pyproject.toml"])
def test_unrelated_python_version_files_are_not_inspected(tmp_path: Path, filename: str) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / filename).write_text("3.11.9\n")
    py = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))

    with patch("cloudai._core.installables.python_executable.sys.executable", "/cloudai/bin/python"):
        assert py._resolve_python_request(repo_path) == ("/cloudai/bin/python", False)


def test_venv_created_with_bundled_uv_and_selected_interpreter(installer: BaseInstaller) -> None:
    git = GitRepo(url="./git_url", commit="commit_hash", python_version="3.11.9")
    py = PythonExecutable(git, project_subpath=Path("package"))
    repo_path = installer.system.install_path / git.repo_name
    project_dir = repo_path / "package"
    project_dir.mkdir(parents=True)
    git.installed_path = repo_path
    venv_path = installer.system.install_path / py.venv_name

    with (
        patch(
            "cloudai._core.installables.python_executable.resolve_uv_bin",
            return_value="/cloudai/bin/uv",
        ) as resolve_uv,
        patch.object(PythonExecutable, "_install_dependencies", return_value=InstallStatusResult(True)),
        patch("subprocess.run") as run,
    ):

        def create_venv(*args, **kwargs):
            _create_python_file(venv_path)
            return CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        run.side_effect = create_venv
        res = py._create_venv(installer)

    assert res.success
    resolve_uv.assert_called_once_with()
    run.assert_called_once_with(
        ["/cloudai/bin/uv", "venv", "--python", "3.11.9", "--seed", str(venv_path)],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
    )
    assert (venv_path / ".cloudai-python-request").read_text().strip() == "3.11.9"


@pytest.mark.parametrize("failure_stage", ["venv", "dependencies"])
def test_failed_installation_removes_partial_venv(
    installer: BaseInstaller,
    failure_stage: str,
) -> None:
    git = GitRepo(url="./git_url", commit="commit_hash", python_version="3.11.9")
    py = PythonExecutable(git)
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    git.installed_path = repo_path
    venv_path = installer.system.install_path / py.venv_name

    def create_partial_venv(*args, **kwargs):
        venv_path.mkdir(parents=True)
        return CompletedProcess(args=args, returncode=1 if failure_stage == "venv" else 0, stderr="err")

    dependencies_result = (
        InstallStatusResult(False, "dependency error") if failure_stage == "dependencies" else InstallStatusResult(True)
    )
    with (
        patch("cloudai._core.installables.python_executable.resolve_uv_bin", return_value="/cloudai/bin/uv"),
        patch.object(PythonExecutable, "_install_dependencies", return_value=dependencies_result),
        patch("subprocess.run", side_effect=create_partial_venv),
    ):
        res = py._create_venv(installer)

    assert not res.success
    assert "err" in res.message
    assert not venv_path.exists()
    assert py.venv_path is None


@pytest.mark.parametrize("marker_value", [None, "3.10.16"])
def test_stale_pinned_legacy_venv_is_recreated(
    installer: BaseInstaller,
    git: GitRepo,
    marker_value: str | None,
) -> None:
    py = PythonExecutable(git)
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    (repo_path / ".python-version").write_text("3.11.9\n")
    git.installed_path = repo_path
    venv_path = installer.system.install_path / py.venv_name
    _create_python_file(venv_path)
    stale_file = venv_path / "stale"
    stale_file.touch()
    marker = venv_path / ".cloudai-python-request"
    if marker_value is not None:
        marker.write_text(marker_value)

    def recreate_venv(*args, **kwargs):
        assert not stale_file.exists()
        _create_python_file(venv_path)
        return CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    with (
        patch("cloudai._core.installables.python_executable.resolve_uv_bin", return_value="/cloudai/bin/uv"),
        patch.object(PythonExecutable, "_install_dependencies", return_value=InstallStatusResult(True)),
        patch("subprocess.run", side_effect=recreate_venv) as run,
    ):
        res = py._create_venv(installer)

    assert res.success
    run.assert_called_once()
    assert marker.read_text().strip() == "3.11.9"
    assert not stale_file.exists()


def test_matching_marker_keeps_existing_pinned_venv(installer: BaseInstaller, git: GitRepo) -> None:
    py = PythonExecutable(git)
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    (repo_path / ".python-version").write_text("3.11.9\n")
    git.installed_path = repo_path
    venv_path = installer.system.install_path / py.venv_name
    _create_python_file(venv_path)
    (venv_path / ".cloudai-python-request").write_text("3.11.9")

    with patch("subprocess.run") as run:
        res = py._create_venv(installer)

    assert res.success
    assert res.message == f"Virtual environment already exists at {venv_path}."
    run.assert_not_called()


@pytest.mark.parametrize("marker_value", [None, "3.10.16"])
def test_is_installed_rejects_missing_or_mismatched_marker_for_pinned_environment(
    installer: BaseInstaller,
    git: GitRepo,
    marker_value: str | None,
) -> None:
    py = PythonExecutable(git)
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    (repo_path / ".python-version").write_text("3.11.9\n")
    venv_path = installer.system.install_path / py.venv_name
    _create_python_file(venv_path)
    if marker_value is not None:
        (venv_path / ".cloudai-python-request").write_text(marker_value)

    res = py.is_installed(installer)

    assert not res.success
    assert "Python interpreter request" in res.message
    assert py.venv_path is None


def test_is_installed_accepts_matching_marker_for_pinned_environment(
    installer: BaseInstaller,
    git: GitRepo,
) -> None:
    py = PythonExecutable(git)
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    (repo_path / ".python-version").write_text("3.11.9\n")
    venv_path = installer.system.install_path / py.venv_name
    _create_python_file(venv_path)
    (venv_path / ".cloudai-python-request").write_text("3.11.9")

    res = py.is_installed(installer)

    assert res.success
    assert py.venv_path == venv_path


def test_is_installed_preserves_unpinned_legacy_venv_without_marker(
    installer: BaseInstaller,
    git: GitRepo,
) -> None:
    py = PythonExecutable(git)
    (installer.system.install_path / git.repo_name).mkdir()
    venv_path = installer.system.install_path / py.venv_name
    _create_python_file(venv_path)

    res = py.is_installed(installer)

    assert res.success
    assert py.venv_path == venv_path


def test_is_installed_requires_python_executable(installer: BaseInstaller, git: GitRepo) -> None:
    py = PythonExecutable(git)
    (installer.system.install_path / git.repo_name).mkdir()
    (installer.system.install_path / py.venv_name).mkdir()

    res = py.is_installed(installer)

    assert not res.success
    assert "Python executable" in res.message
    assert py.venv_path is None


def test_python_executable_identity_and_venv_name_include_environment_configuration() -> None:
    default = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))
    same = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))
    py311 = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash", python_version="3.11.9"))
    py311_same = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash", python_version="3.11.9"))
    py312 = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash", python_version="3.12.8"))
    subproject = PythonExecutable(
        GitRepo(url="./git_url", commit="commit_hash"),
        project_subpath=Path("package"),
    )
    requirements_first = PythonExecutable(
        GitRepo(url="./git_url", commit="commit_hash"),
        dependencies_from_pyproject=False,
    )

    assert default == same
    assert hash(default) == hash(same)
    assert default.venv_name == "git_url__commit_hash-venv"
    assert py311 == py311_same
    assert hash(py311) == hash(py311_same)
    assert len({default, py311, py312, subproject, requirements_first}) == 5
    assert (
        len(
            {
                default.venv_name,
                py311.venv_name,
                py312.venv_name,
                subproject.venv_name,
                requirements_first.venv_name,
            }
        )
        == 5
    )
    assert py311.venv_name.startswith(f"{default.venv_name}-")
    assert len(py311.venv_name.removeprefix(f"{default.venv_name}-")) == 12


def test_repository_detected_pin_preserves_legacy_venv_name(git: GitRepo) -> None:
    py = PythonExecutable(git)

    assert py.venv_name == f"{git.repo_name}-venv"


def test_string_project_subpath_remains_compatible(git: GitRepo) -> None:
    py = PythonExecutable(git, project_subpath="package")  # type: ignore[arg-type]

    assert py.venv_name.startswith(f"{git.repo_name}-venv-")


def test_installer_creates_distinct_explicit_variants_while_serializing_shared_repo(
    installer: BaseInstaller,
) -> None:
    py311 = PythonExecutable(GitRepo(url="./shared_repo", commit="commit", python_version="3.11.9"))
    py314 = PythonExecutable(GitRepo(url="./shared_repo", commit="commit", python_version="3.14.0"))
    original_install = GitRepo._install
    state_lock = threading.Lock()
    active_repo_operations = 0
    max_active_repo_operations = 0
    clone_calls = 0

    def track_repo_install(item: GitRepo, context: BaseInstaller, repo_path: Path) -> InstallStatusResult:
        nonlocal active_repo_operations, max_active_repo_operations
        with state_lock:
            active_repo_operations += 1
            max_active_repo_operations = max(max_active_repo_operations, active_repo_operations)
        try:
            time.sleep(0.05)
            return original_install(item, context, repo_path)
        finally:
            with state_lock:
                active_repo_operations -= 1

    def clone_repo(item: GitRepo, context: BaseInstaller, repo_path: Path) -> InstallStatusResult:
        nonlocal clone_calls
        clone_calls += 1
        repo_path.mkdir(parents=True)
        return InstallStatusResult(True)

    def create_venv(item: PythonExecutable, context: BaseInstaller) -> InstallStatusResult:
        item.venv_path = context.system.install_path / item.venv_name
        _create_python_file(item.venv_path)
        return InstallStatusResult(True)

    with (
        patch.object(GitRepo, "_install", autospec=True, side_effect=track_repo_install),
        patch.object(GitRepo, "_clone_repository", autospec=True, side_effect=clone_repo),
        patch.object(GitRepo, "_checkout_commit", return_value=InstallStatusResult(True)),
        patch.object(GitRepo, "_verify_commit", return_value=InstallStatusResult(True)),
        patch.object(GitRepo, "ensure_submodules_state", return_value=(True, "")),
        patch.object(PythonExecutable, "_create_venv", autospec=True, side_effect=create_venv),
    ):
        res = installer.install([py311, py314])

    assert res.success
    assert max_active_repo_operations == 1
    assert clone_calls == 1
    assert py311.git_repo.installed_path == py314.git_repo.installed_path
    assert py311.venv_path != py314.venv_path
    assert py311.venv_path is not None and py311.venv_path.exists()
    assert py314.venv_path is not None and py314.venv_path.exists()


def test_predictor_identity_matches_python_executable_identity() -> None:
    predictor = PredictorConfig(
        git_repo=GitRepo(url="./git_url", commit="commit_hash", python_version="3.11.9"),
        bin_name="predict-a",
    )
    same_environment = PredictorConfig(
        git_repo=GitRepo(url="./git_url", commit="commit_hash", python_version="3.11.9"),
        bin_name="predict-b",
    )
    different_environment = PredictorConfig(
        git_repo=GitRepo(url="./git_url", commit="commit_hash", python_version="3.12.8"),
        bin_name="predict-a",
    )

    assert predictor == same_environment
    assert hash(predictor) == hash(same_environment)
    assert predictor != different_environment


def test_mark_as_installed_remains_path_only(installer: BaseInstaller) -> None:
    py = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash", python_version="3.11.9"))

    res = py.mark_as_installed(installer)

    assert res.success
    assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
    assert py.venv_path == installer.system.install_path / py.venv_name
    assert py.git_repo.installed_path is not None
    assert py.venv_path is not None
    assert not py.git_repo.installed_path.exists()
    assert not py.venv_path.exists()


def test_is_installed_no_repo(installer: BaseInstaller, git: GitRepo) -> None:
    py = PythonExecutable(git)

    res = py.is_installed(installer)

    assert not res.success
    assert res.message == f"Git repository {py.git_repo.url} not cloned"
    assert py.git_repo.installed_path is None
    assert py.venv_path is None


def test_is_installed_no_venv(installer: BaseInstaller, git: GitRepo) -> None:
    py = PythonExecutable(git)
    (installer.system.install_path / py.git_repo.repo_name).mkdir()

    res = py.is_installed(installer)

    assert not res.success
    assert res.message == f"Virtual environment not created for {py.git_repo.url}"
    assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
    assert py.venv_path is None


def test_uninstall_no_venv(installer: BaseInstaller, git: GitRepo) -> None:
    py = PythonExecutable(git)
    py.venv_path = installer.system.install_path / py.venv_name

    res = py.uninstall(installer)

    assert res.success
    assert res.message == f"Virtual environment {py.venv_name} is not created."


def test_uninstall_venv_removed_ok(installer: BaseInstaller, git: GitRepo) -> None:
    py = PythonExecutable(git)
    (installer.system.install_path / py.venv_name).mkdir()
    (installer.system.install_path / py.venv_name / "file").touch()
    py.venv_path = installer.system.install_path / py.venv_name

    res = py.uninstall(installer)

    assert res.success
    assert not (installer.system.install_path / py.venv_name).exists()
    assert py.venv_path is None


def test_requirements_no_file(installer: BaseInstaller, git: GitRepo) -> None:
    py = PythonExecutable(git)
    venv_path = installer.system.install_path / py.venv_name
    venv_path.mkdir()

    res = py._install_requirements(venv_path, installer.system.install_path / "requirements.txt")

    assert not res.success
    assert (
        res.message
        == f"Requirements file is invalid or does not exist: {installer.system.install_path / 'requirements.txt'}"
    )


def test_requirements_are_installed_with_venv_python(installer: BaseInstaller) -> None:
    requirements_file = installer.system.install_path / "requirements.txt"
    venv_path = installer.system.install_path / "venv"
    requirements_file.touch()

    with patch("subprocess.run") as run:
        run.return_value = CompletedProcess(args=[], returncode=0)
        res = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))._install_requirements(
            venv_path, requirements_file
        )

    assert res.success
    run.assert_called_once_with(
        [str(venv_path / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements_file)],
        capture_output=True,
        text=True,
    )


def test_pyproject_is_installed_with_venv_python(installer: BaseInstaller) -> None:
    project_dir = installer.system.install_path / "project"
    venv_path = installer.system.install_path / "venv"
    project_dir.mkdir()

    with patch("subprocess.run") as run:
        run.return_value = CompletedProcess(args=[], returncode=0)
        res = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))._install_pyproject(
            venv_path, project_dir
        )

    assert res.success
    run.assert_called_once_with(
        [str(venv_path / "bin" / "python"), "-m", "pip", "install", str(project_dir)],
        capture_output=True,
        text=True,
    )


def test_requirements_installation_failure_is_reported(installer: BaseInstaller) -> None:
    requirements_file = installer.system.install_path / "requirements.txt"
    requirements_file.touch()

    with patch("subprocess.run") as run:
        run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
        res = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))._install_requirements(
            installer.system.install_path, requirements_file
        )

    assert not res.success
    assert res.message == "Failed to install dependencies from requirements.txt: err"


def test_install_python_executable_prefers_pyproject_toml(
    installer: BaseInstaller,
    git: GitRepo,
    setup_repo,
) -> None:
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
) -> None:
    repo_dir, subdir, _, _ = setup_repo
    py = PythonExecutable(git, project_subpath=Path("subdir"), dependencies_from_pyproject=False)
    py.git_repo.installed_path = repo_dir

    with (
        patch.object(PythonExecutable, "_install_requirements", return_value=InstallStatusResult(True)) as reqs,
        patch.object(PythonExecutable, "_install_pyproject", return_value=InstallStatusResult(True)) as pyproject,
    ):
        res = py._install_dependencies(installer)

    assert res.success
    pyproject.assert_not_called()
    reqs.assert_called_once_with(installer.system.install_path / py.venv_name, subdir / "requirements.txt")


@pytest.mark.ci_only
def test_python_executable_installs_repository_pinned_python_3119(
    installer: BaseInstaller,
    tmp_path: Path,
) -> None:
    if sys.version_info[:2] != (3, 14):
        pytest.skip("This interpreter-independence integration test requires the Python 3.14 CI job.")

    source_repo = tmp_path / "source-repo"
    source_repo.mkdir()
    (source_repo / ".python-version").write_text("3.11.9\n")
    (source_repo / "requirements.txt").touch()
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(source_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=source_repo, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CloudAI Tests",
            "-c",
            "user.email=cloudai-tests@nvidia.com",
            "commit",
            "-m",
            "Add pinned Python project",
        ],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    git = GitRepo(url=str(source_repo), commit=commit)
    py = PythonExecutable(git)

    res = py.install(installer)

    assert res.success, res.message
    assert git.installed_path == installer.system.install_path / git.repo_name
    assert py.venv_path is not None
    version = subprocess.run(
        [str(py.venv_path / "bin" / "python"), "-c", "import platform; print(platform.python_version())"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert version.stdout.strip() == "3.11.9"
