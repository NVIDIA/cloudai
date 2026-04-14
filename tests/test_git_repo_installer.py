# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Union
from unittest.mock import Mock, patch

import pytest

from cloudai.core import GitRepo, InstallStatusResult
from cloudai.systems.kubernetes.kubernetes_installer import KubernetesInstaller
from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem
from cloudai.systems.slurm.slurm_installer import SlurmInstaller
from cloudai.systems.slurm.slurm_system import SlurmGroup, SlurmPartition, SlurmSystem


@pytest.fixture
def k8s_system(tmp_path: Path) -> KubernetesSystem:
    # Create a mock kube config file
    kube_config = tmp_path / "kube_config"
    kube_config.write_text(
        """
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://127.0.0.1:6443
  name: local-cluster
contexts:
- context:
    cluster: local-cluster
    user: local-user
  name: local-context
current-context: local-context
users:
- name: local-user
  user:
    token: fake-token
    """.strip()
    )

    system = KubernetesSystem(
        name="test-k8s",
        scheduler="kubernetes",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        kube_config_path=kube_config,
        default_namespace="default",
    )
    return system


@pytest.fixture
def slurm_system(tmp_path: Path) -> SlurmSystem:
    system = SlurmSystem(
        name="test-slurm",
        scheduler="slurm",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        default_partition="test",
        partitions=[
            SlurmPartition(
                name="test",
                groups=[SlurmGroup(name="test-group", nodes=["test-node"])],
            )
        ],
    )
    return system


@pytest.fixture(params=["k8s", "slurm"])
def installer(
    request: Any, k8s_system: KubernetesSystem, slurm_system: SlurmSystem
) -> Union[KubernetesInstaller, SlurmInstaller]:
    installer = KubernetesInstaller(k8s_system) if request.param == "k8s" else SlurmInstaller(slurm_system)

    installer.system.install_path.mkdir(parents=True)
    installer._check_low_thread_environment = lambda threshold=None: False
    return installer


@pytest.fixture
def git() -> GitRepo:
    return GitRepo(url="./git_url", commit="commit_hash")


class TestGitRepoInstaller:
    def test_repo_exists_with_correct_commit(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        installer._verify_commit = Mock(return_value=InstallStatusResult(True))
        with patch.object(GitRepo, "ensure_submodules_state", return_value=(True, "")):
            res = installer._install_one_git_repo(git)
        assert res.success
        assert res.message == f"Git repository already exists at {repo_path}."
        assert git.installed_path == repo_path
        installer._verify_commit.assert_called_once_with(git.commit, repo_path)

    def test_repo_exists_with_wrong_commit(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        installer._verify_commit = Mock(return_value=InstallStatusResult(False, "wrong commit"))
        res = installer._install_one_git_repo(git)
        assert not res.success
        assert res.message == "wrong commit"

    def test_repo_cloned(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._clone_repository(git.url, repo_path)
        assert res.success
        mock_run.assert_called_once_with(["git", "clone", git.url, str(repo_path)], capture_output=True, text=True)

    def test_error_cloning_repo(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._clone_repository(git.url, repo_path)
        assert not res.success
        assert res.message == "Failed to clone repository: err"

    def test_commit_checked_out(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._checkout_commit(git.commit, repo_path)
        assert res.success
        mock_run.assert_called_once_with(
            ["git", "checkout", git.commit], cwd=str(repo_path), capture_output=True, text=True
        )

    def test_error_checking_out_commit(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._checkout_commit(git.commit, repo_path)
        assert not res.success
        assert res.message == f"Failed to checkout commit {git.commit}: err"

    def test_checkout_failure_cleans_up_repo(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        installer._clone_repository = Mock(
            side_effect=lambda url, path: (path.mkdir(parents=True, exist_ok=True), InstallStatusResult(True))[1]
        )
        installer._checkout_commit = Mock(
            return_value=InstallStatusResult(False, f"Failed to checkout commit {git.commit}: err")
        )
        res = installer._install_one_git_repo(git)
        assert not res.success
        assert not repo_path.exists()

    def test_verify_commit_correct(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=[], returncode=0, stdout=f"{git.commit}abcdef1234567890\n", stderr=""
            )
            res = installer._verify_commit(git.commit, repo_path)
        assert res.success

    def test_verify_commit_wrong(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                CompletedProcess(args=[], returncode=0, stdout="deadbeef1234567890abcdef\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="cafebabe1234567890abcdef\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="main\n", stderr=""),
            ]
            res = installer._verify_commit(git.commit, repo_path)
        assert not res.success
        assert "expected" in res.message
        assert git.commit in res.message

    def test_verify_commit_git_failure(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stdout="", stderr="not a git repo")
            res = installer._verify_commit(git.commit, repo_path)
        assert not res.success
        assert "Failed to verify" in res.message

    def test_verify_commit_oserror(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / "nonexistent"
        with patch("subprocess.run", side_effect=OSError("No such file or directory")):
            res = installer._verify_commit(git.commit, repo_path)
        assert not res.success
        assert "Failed to verify" in res.message

    def test_verify_commit_overlong_hash(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        actual = "a" * 40
        overlong = actual + "extragarbage"
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                CompletedProcess(args=[], returncode=0, stdout=f"{actual}\n", stderr=""),
                CompletedProcess(args=[], returncode=1, stdout="", stderr="unknown revision"),
            ]
            res = installer._verify_commit(overlong, repo_path)
        assert not res.success
        assert "Failed to verify" in res.message

    def test_verify_commit_branch_name_match(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        ref = "release-1.2"
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                CompletedProcess(args=[], returncode=0, stdout="deadbeef1234567890abcdef\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="cafebabe1234567890abcdef\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout=f"{ref}\n", stderr=""),
            ]
            res = installer._verify_commit(ref, repo_path)
        assert res.success

    def test_submodule_failure_cleans_up_repo(
        self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo
    ):
        git.init_submodules = True
        repo_path = installer.system.install_path / git.repo_name
        installer._clone_repository = Mock(
            side_effect=lambda url, path: (path.mkdir(parents=True, exist_ok=True), InstallStatusResult(True))[1]
        )
        installer._checkout_commit = Mock(return_value=InstallStatusResult(True))
        with patch.object(
            GitRepo, "ensure_submodules_state", return_value=(False, "Failed to initialize submodules: err")
        ):
            res = installer._install_one_git_repo(git)
        assert not res.success
        assert not repo_path.exists()

    def test_all_good_flow(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        installer._clone_repository = Mock(return_value=InstallStatusResult(True))
        installer._checkout_commit = Mock(return_value=InstallStatusResult(True))
        with patch.object(GitRepo, "ensure_submodules_state", return_value=(True, "")):
            res = installer._install_one_git_repo(git)
        assert res.success
        assert git.installed_path == installer.system.install_path / git.repo_name

    def test_submodules_skipped_when_not_requested(
        self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo
    ):
        installer._clone_repository = Mock(return_value=InstallStatusResult(True))
        installer._checkout_commit = Mock(return_value=InstallStatusResult(True))
        with patch.object(GitRepo, "ensure_submodules_state", return_value=(True, "")) as mock_ensure:
            res = installer._install_one_git_repo(git)
        assert res.success
        mock_ensure.assert_called_once()

    def test_submodules_run_when_requested(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        git.init_submodules = True
        installer._clone_repository = Mock(return_value=InstallStatusResult(True))
        installer._checkout_commit = Mock(return_value=InstallStatusResult(True))
        with patch.object(GitRepo, "ensure_submodules_state", return_value=(True, "")) as mock_ensure:
            res = installer._install_one_git_repo(git)
        assert res.success
        mock_ensure.assert_called_once()

    def test_existing_repo_inits_submodules_when_requested(
        self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo
    ):
        git.init_submodules = True
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        installer._verify_commit = Mock(return_value=InstallStatusResult(True))
        with patch.object(GitRepo, "ensure_submodules_state", return_value=(True, "")) as mock_ensure:
            res = installer._install_one_git_repo(git)
        assert res.success
        assert git.installed_path == repo_path
        mock_ensure.assert_called_once_with(repo_path)

    def test_existing_repo_deinits_submodules_when_not_requested(
        self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo
    ):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        installer._verify_commit = Mock(return_value=InstallStatusResult(True))
        with patch.object(GitRepo, "ensure_submodules_state", return_value=(True, "")) as mock_ensure:
            res = installer._install_one_git_repo(git)

        assert res.success
        mock_ensure.assert_called_once_with(repo_path)

    def test_is_installed_checks_submodule_state(
        self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo
    ):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        installer._verify_commit = Mock(return_value=InstallStatusResult(True))
        with patch.object(GitRepo, "check_submodules_state", return_value=(True, "")) as mock_check:
            res = installer.is_installed_one(git)

        assert res.success
        mock_check.assert_called_once_with(repo_path)

    def test_is_installed_fails_when_submodule_state_does_not_match(
        self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo
    ):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        installer._verify_commit = Mock(return_value=InstallStatusResult(True))
        with patch.object(GitRepo, "check_submodules_state", return_value=(False, "Submodule state does not match")):
            res = installer.is_installed_one(git)

        assert not res.success
        assert "Submodule state does not match" in res.message

    def test_uninstall_no_repo(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        res = installer._uninstall_git_repo(git)
        assert res.success
        assert res.message == f"Repository {git.url} is not cloned."

    def test_uninstall_ok(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        (installer.system.install_path / git.repo_name).mkdir()
        (installer.system.install_path / git.repo_name / "file").touch()  # test with non-empty directory
        res = installer._uninstall_git_repo(git)
        assert res.success
        assert not (installer.system.install_path / git.repo_name).exists()
        assert not git.installed_path
