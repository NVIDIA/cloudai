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
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from cloudai.core import BaseInstaller, GitRepo, InstallStatusResult


@pytest.fixture
def installer(slurm_system) -> BaseInstaller:
    installer = BaseInstaller(slurm_system)
    installer.system.install_path.mkdir(parents=True)
    installer._check_low_thread_environment = lambda threshold=None: False
    return installer


@pytest.fixture
def git_unmocked() -> GitRepo:
    return GitRepo(url="./git_url", commit="commit_hash")


@pytest.fixture
def check_submodules_state_mock() -> MagicMock:
    return MagicMock(return_value=(True, ""))


@pytest.fixture
def ensure_submodules_state_mock() -> MagicMock:
    return MagicMock(return_value=(True, ""))


@pytest.fixture
def git(
    git_unmocked: GitRepo,
    check_submodules_state_mock: MagicMock,
    ensure_submodules_state_mock: MagicMock,
) -> Iterator[GitRepo]:
    with (
        patch.object(GitRepo, "check_submodules_state", check_submodules_state_mock),
        patch.object(GitRepo, "ensure_submodules_state", ensure_submodules_state_mock),
    ):
        yield git_unmocked


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


@pytest.mark.parametrize("init_submodules", [True, False])
def test_check_submodules_state_no_submodules(git_unmocked: GitRepo, init_submodules: bool):
    git_unmocked.init_submodules = init_submodules

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        result, message = git_unmocked.check_submodules_state(Path("/repo"))

    assert result is True
    assert message == ""


@pytest.mark.parametrize(
    ("init_submodules", "stdout", "expected_result", "expected_message"),
    [
        (True, " 0123456789abcdef path/to/submodule\n", True, ""),
        (True, "-0123456789abcdef path/to/submodule\n", False, "Cloned repo has not all submodules initialized."),
        (True, "+0123456789abcdef path/to/submodule\n", False, "Cloned repo has not all submodules initialized."),
        (True, "U0123456789abcdef path/to/submodule\n", False, "Cloned repo has not all submodules initialized."),
        (False, "-0123456789abcdef path/to/submodule\n", True, ""),
        (
            False,
            " 0123456789abcdef path/to/submodule\n",
            False,
            "Cloned repo has some submodules initialized but requires none to be.",
        ),
        (
            False,
            "+0123456789abcdef path/to/submodule\n",
            False,
            "Cloned repo has some submodules initialized but requires none to be.",
        ),
        (
            False,
            "U0123456789abcdef path/to/submodule\n",
            False,
            "Cloned repo has some submodules initialized but requires none to be.",
        ),
    ],
)
def test_check_submodules_state(
    git_unmocked: GitRepo,
    init_submodules: bool,
    stdout: str,
    expected_result: bool,
    expected_message: str,
):
    git_unmocked.init_submodules = init_submodules

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
        result, message = git_unmocked.check_submodules_state(Path("/repo"))

    assert result is expected_result
    assert message == expected_message


def test_check_submodules_state_status_failure(git_unmocked: GitRepo):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=1, stdout="", stderr="err")
        result, message = git_unmocked.check_submodules_state(Path("/repo"))

    assert result is False
    assert message == "Failed to get submodule status: err"


def test_check_submodules_state_oserror(git_unmocked: GitRepo):
    with patch("subprocess.run", side_effect=OSError("No such file or directory")):
        result, message = git_unmocked.check_submodules_state(Path("/repo"))

    assert result is False
    assert "Failed to get submodule status" in message


@pytest.mark.parametrize(
    ("init_submodules", "stdout", "expected_command"),
    [
        (True, "-0123456789abcdef path/to/submodule\n", ["git", "submodule", "update", "--init", "--recursive"]),
        (False, " 0123456789abcdef path/to/submodule\n", ["git", "submodule", "deinit", "--all", "--force"]),
        (False, "+0123456789abcdef path/to/submodule\n", ["git", "submodule", "deinit", "--all", "--force"]),
        (False, "U0123456789abcdef path/to/submodule\n", ["git", "submodule", "deinit", "--all", "--force"]),
    ],
)
def test_ensure_submodules_state_reconciles(
    git_unmocked: GitRepo, init_submodules: bool, stdout: str, expected_command: list[str]
):
    git_unmocked.init_submodules = init_submodules

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
            CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]
        result, message = git_unmocked.ensure_submodules_state(Path("/repo"))

    assert result is True
    assert message == ""
    assert mock_run.call_args_list[1].args[0] == expected_command


@pytest.mark.parametrize("init_submodules", [True, False])
def test_ensure_submodules_state_noop_when_matching(git_unmocked: GitRepo, init_submodules: bool):
    git_unmocked.init_submodules = init_submodules
    stdout = " 0123456789abcdef path/to/submodule\n" if init_submodules else "-0123456789abcdef path/to/submodule\n"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
        result, message = git_unmocked.ensure_submodules_state(Path("/repo"))

    assert result is True
    assert message == ""
    assert mock_run.call_count == 1


@pytest.mark.parametrize(
    ("init_submodules", "stdout", "expected_message"),
    [
        (True, "-0123456789abcdef path/to/submodule\n", "Failed to initialize submodules: err"),
        (False, " 0123456789abcdef path/to/submodule\n", "Failed to deinitialize submodules: err"),
        (False, "+0123456789abcdef path/to/submodule\n", "Failed to deinitialize submodules: err"),
        (False, "U0123456789abcdef path/to/submodule\n", "Failed to deinitialize submodules: err"),
    ],
)
def test_ensure_submodules_state_reconcile_failure(
    git_unmocked: GitRepo,
    init_submodules: bool,
    stdout: str,
    expected_message: str,
):
    git_unmocked.init_submodules = init_submodules

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
            CompletedProcess(args=[], returncode=1, stdout="", stderr="err"),
        ]
        result, message = git_unmocked.ensure_submodules_state(Path("/repo"))

    assert result is False
    assert message == expected_message


def test_ensure_submodules_state_status_fails(git_unmocked: GitRepo):
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [CompletedProcess(args=[], returncode=1, stdout="bla", stderr="bla")]
        result, message = git_unmocked.ensure_submodules_state(Path("/repo"))

    assert result is False
    assert "bla" in message
    assert mock_run.call_count == 1


@pytest.mark.parametrize(
    ("init_submodules", "expected_message"),
    [
        (True, "Failed to initialize submodules"),
        (False, "Failed to deinitialize submodules"),
    ],
)
def test_ensure_submodules_state_reconcile_oserror(git_unmocked: GitRepo, init_submodules: bool, expected_message: str):
    git_unmocked.init_submodules = init_submodules
    stdout = "-0123456789abcdef path/to/submodule\n" if init_submodules else " 0123456789abcdef path/to/submodule\n"

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
            OSError("No such file or directory"),
        ]
        result, message = git_unmocked.ensure_submodules_state(Path("/repo"))

    assert result is False
    assert expected_message in message


def test_repo_exists_with_correct_commit(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch.object(GitRepo, "_verify_commit", return_value=InstallStatusResult(True)) as verify_commit:
        res = git.install(installer)
    assert res.success
    assert res.message == f"Git repository already exists at {repo_path}."
    assert git.installed_path == repo_path
    verify_commit.assert_called_once_with(git.commit, repo_path)


def test_repo_exists_with_wrong_commit(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch.object(GitRepo, "_verify_commit", return_value=InstallStatusResult(False, "wrong commit")):
        res = git.install(installer)
    assert not res.success
    assert res.message == "wrong commit"


def test_repo_cloned(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        res = git._clone_repository(installer, repo_path)
    assert res.success
    mock_run.assert_called_once_with(["git", "clone", git.url, str(repo_path)], capture_output=True, text=True)


def test_error_cloning_repo(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
        res = git._clone_repository(installer, repo_path)
    assert not res.success
    assert res.message == "Failed to clone repository: err"


def test_clone_repository_oserror(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    with patch("subprocess.run", side_effect=OSError("No such file or directory")):
        res = git._clone_repository(installer, repo_path)
    assert not res.success
    assert "Failed to clone repository" in res.message


def test_commit_checked_out(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        res = git._checkout_commit(git.commit, repo_path)
    assert res.success
    mock_run.assert_called_once_with(
        ["git", "checkout", git.commit], cwd=str(repo_path), capture_output=True, text=True
    )


def test_error_checking_out_commit(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
        res = git._checkout_commit(git.commit, repo_path)
    assert not res.success
    assert res.message == f"Failed to checkout commit {git.commit}: err"


def test_checkout_commit_oserror(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch("subprocess.run", side_effect=OSError("No such file or directory")):
        res = git._checkout_commit(git.commit, repo_path)
    assert not res.success
    assert f"Failed to checkout commit {git.commit}" in res.message


def test_checkout_failure_cleans_up_repo(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    with (
        patch.object(
            GitRepo,
            "_clone_repository",
            autospec=True,
            side_effect=lambda item, installer, path: (
                path.mkdir(parents=True, exist_ok=True),
                InstallStatusResult(True),
            )[1],
        ),
        patch.object(
            GitRepo,
            "_checkout_commit",
            return_value=InstallStatusResult(False, f"Failed to checkout commit {git.commit}: err"),
        ),
    ):
        res = git.install(installer)
    assert not res.success
    assert not repo_path.exists()


def test_verify_commit_correct(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(
            args=[], returncode=0, stdout=f"{git.commit}abcdef1234567890\n", stderr=""
        )
        res = git._verify_commit(git.commit, repo_path)
    assert res.success


def test_verify_commit_wrong(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout="deadbeef1234567890abcdef\n", stderr=""),
            CompletedProcess(args=[], returncode=0, stdout="cafebabe1234567890abcdef\n", stderr=""),
            CompletedProcess(args=[], returncode=0, stdout="main\n", stderr=""),
        ]
        res = git._verify_commit(git.commit, repo_path)
    assert not res.success
    assert "expected" in res.message
    assert git.commit in res.message


def test_verify_commit_git_failure(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=[], returncode=1, stdout="", stderr="not a git repo")
        res = git._verify_commit(git.commit, repo_path)
    assert not res.success
    assert "Failed to verify" in res.message


def test_verify_commit_oserror(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / "nonexistent"
    with patch("subprocess.run", side_effect=OSError("No such file or directory")):
        res = git._verify_commit(git.commit, repo_path)
    assert not res.success
    assert "Failed to verify" in res.message


def test_verify_commit_overlong_hash(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    actual = "a" * 40
    overlong = actual + "extragarbage"
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout=f"{actual}\n", stderr=""),
            CompletedProcess(args=[], returncode=1, stdout="", stderr="unknown revision"),
        ]
        res = git._verify_commit(overlong, repo_path)
    assert not res.success
    assert "Failed to verify" in res.message


def test_verify_commit_branch_name_match(installer: BaseInstaller, git: GitRepo):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    ref = "release-1.2"
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout="deadbeef1234567890abcdef\n", stderr=""),
            CompletedProcess(args=[], returncode=0, stdout="cafebabe1234567890abcdef\n", stderr=""),
            CompletedProcess(args=[], returncode=0, stdout=f"{ref}\n", stderr=""),
        ]
        res = git._verify_commit(ref, repo_path)
    assert res.success


def test_submodule_failure_cleans_up_repo(installer: BaseInstaller, git_unmocked: GitRepo):
    git_unmocked.init_submodules = True
    repo_path = installer.system.install_path / git_unmocked.repo_name
    with (
        patch.object(
            GitRepo,
            "_clone_repository",
            autospec=True,
            side_effect=lambda item, installer, path: (
                path.mkdir(parents=True, exist_ok=True),
                InstallStatusResult(True),
            )[1],
        ),
        patch.object(GitRepo, "_checkout_commit", return_value=InstallStatusResult(True)),
        patch.object(
            GitRepo,
            "ensure_submodules_state",
            return_value=(False, "Failed to initialize submodules: err"),
        ),
    ):
        res = git_unmocked.install(installer)
    assert not res.success
    assert not repo_path.exists()


def test_all_good_flow(installer: BaseInstaller, git: GitRepo):
    with (
        patch.object(GitRepo, "_clone_repository", return_value=InstallStatusResult(True)),
        patch.object(GitRepo, "_checkout_commit", return_value=InstallStatusResult(True)),
    ):
        res = git.install(installer)
    assert res.success
    assert git.installed_path == installer.system.install_path / git.repo_name


def test_submodules_skipped_when_not_requested(
    installer: BaseInstaller,
    git: GitRepo,
    ensure_submodules_state_mock: MagicMock,
):
    with (
        patch.object(GitRepo, "_clone_repository", return_value=InstallStatusResult(True)),
        patch.object(GitRepo, "_checkout_commit", return_value=InstallStatusResult(True)),
    ):
        res = git.install(installer)
    assert res.success
    ensure_submodules_state_mock.assert_called_once()


def test_submodules_run_when_requested(
    installer: BaseInstaller,
    git: GitRepo,
    ensure_submodules_state_mock: MagicMock,
):
    git.init_submodules = True
    with (
        patch.object(GitRepo, "_clone_repository", return_value=InstallStatusResult(True)),
        patch.object(GitRepo, "_checkout_commit", return_value=InstallStatusResult(True)),
    ):
        res = git.install(installer)
    assert res.success
    ensure_submodules_state_mock.assert_called_once()


def test_existing_repo_inits_submodules_when_requested(
    installer: BaseInstaller,
    git: GitRepo,
    ensure_submodules_state_mock: MagicMock,
):
    git.init_submodules = True
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch.object(GitRepo, "_verify_commit", return_value=InstallStatusResult(True)):
        res = git.install(installer)
    assert res.success
    assert git.installed_path == repo_path
    ensure_submodules_state_mock.assert_called_once_with(repo_path)


def test_existing_repo_deinits_submodules_when_not_requested(
    installer: BaseInstaller,
    git: GitRepo,
    ensure_submodules_state_mock: MagicMock,
):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch.object(GitRepo, "_verify_commit", return_value=InstallStatusResult(True)):
        res = git.install(installer)

    assert res.success
    ensure_submodules_state_mock.assert_called_once_with(repo_path)


def test_is_installed_checks_submodule_state(
    installer: BaseInstaller,
    git: GitRepo,
    check_submodules_state_mock: MagicMock,
):
    repo_path = installer.system.install_path / git.repo_name
    repo_path.mkdir()
    with patch.object(GitRepo, "_verify_commit", return_value=InstallStatusResult(True)):
        res = installer.is_installed_one(git)

    assert res.success
    check_submodules_state_mock.assert_called_once_with(repo_path)


def test_is_installed_fails_when_submodule_state_does_not_match(installer: BaseInstaller, git_unmocked: GitRepo):
    repo_path = installer.system.install_path / git_unmocked.repo_name
    repo_path.mkdir()
    with (
        patch.object(GitRepo, "_verify_commit", return_value=InstallStatusResult(True)),
        patch.object(GitRepo, "check_submodules_state", return_value=(False, "Submodule state does not match")),
    ):
        res = installer.is_installed_one(git_unmocked)

    assert not res.success
    assert "Submodule state does not match" in res.message


def test_uninstall_no_repo(installer: BaseInstaller, git: GitRepo):
    res = git.uninstall(installer)
    assert res.success
    assert res.message == f"Repository {git.url} is not cloned."


def test_uninstall_ok(installer: BaseInstaller, git: GitRepo):
    (installer.system.install_path / git.repo_name).mkdir()
    (installer.system.install_path / git.repo_name / "file").touch()  # test with non-empty directory
    res = git.uninstall(installer)
    assert res.success
    assert not (installer.system.install_path / git.repo_name).exists()
    assert not git.installed_path
