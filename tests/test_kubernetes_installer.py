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

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

import pytest

from cloudai.core import DockerImage, GitRepo, Installable, InstallStatusResult
from cloudai.systems.kubernetes.kubernetes_installer import KubernetesInstaller
from cloudai.systems.kubernetes.kubernetes_system import KubernetesSystem


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
def installer(k8s_system: KubernetesSystem):
    ki = KubernetesInstaller(k8s_system)
    ki.system.install_path.mkdir(parents=True)
    ki._check_low_thread_environment = lambda threshold=None: False
    return ki


class TestInstallOneGitRepo:
    @pytest.fixture
    def git(self) -> GitRepo:
        return GitRepo(url="./git_url", commit="commit_hash")

    def test_repo_exists(self, installer: KubernetesInstaller, git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        res = installer._install_one_git_repo(git)
        assert res.success
        assert res.message == f"Git repository already exists at {repo_path}."
        assert git.installed_path == repo_path

    def test_repo_cloned(self, installer: KubernetesInstaller, git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._clone_repository(git.url, repo_path)
        assert res.success
        mock_run.assert_called_once_with(["git", "clone", git.url, str(repo_path)], capture_output=True, text=True)

    def test_error_cloning_repo(self, installer: KubernetesInstaller, git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._clone_repository(git.url, repo_path)
        assert not res.success
        assert res.message == "Failed to clone repository: err"

    def test_commit_checked_out(self, installer: KubernetesInstaller, git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._checkout_commit(git.commit, repo_path)
        assert res.success
        mock_run.assert_called_once_with(
            ["git", "checkout", git.commit], cwd=str(repo_path), capture_output=True, text=True
        )

    def test_error_checking_out_commit(self, installer: KubernetesInstaller, git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._checkout_commit(git.commit, repo_path)
        assert not res.success
        assert res.message == "Failed to checkout commit: err"

    def test_all_good_flow(self, installer: KubernetesInstaller, git: GitRepo):
        installer._clone_repository = Mock(return_value=InstallStatusResult(True))
        installer._checkout_commit = Mock(return_value=InstallStatusResult(True))
        res = installer._install_one_git_repo(git)
        assert res.success
        assert git.installed_path == installer.system.install_path / git.repo_name

    def test_uninstall_no_repo(self, installer: KubernetesInstaller, git: GitRepo):
        res = installer._uninstall_git_repo(git)
        assert res.success
        assert res.message == f"Repository {git.url} is not cloned."

    def test_uninstall_ok(self, installer: KubernetesInstaller, git: GitRepo):
        (installer.system.install_path / git.repo_name).mkdir()
        (installer.system.install_path / git.repo_name / "file").touch()  # test with non-empty directory
        res = installer._uninstall_git_repo(git)
        assert res.success
        assert not (installer.system.install_path / git.repo_name).exists()
        assert not git.installed_path


def test_check_supported(k8s_system: KubernetesSystem):
    k8s_system.install_path.mkdir(parents=True)
    installer = KubernetesInstaller(k8s_system)
    installer._clone_repository = Mock(return_value=InstallStatusResult(True))
    installer._checkout_commit = Mock(return_value=InstallStatusResult(True))

    git = GitRepo(url="./git_url", commit="commit_hash")
    items = [DockerImage("fake_url/img"), git]

    git_path = k8s_system.install_path / git.repo_name
    git_path.mkdir(parents=True)

    for item in items:
        res = installer.install_one(item)
        assert res.success, f"Failed to install {item}: {res.message}"

        res = installer.is_installed_one(item)
        assert res.success, f"Failed to check if {item} is installed: {res.message}"

        res = installer.uninstall_one(item)
        assert res.success, f"Failed to uninstall {item}: {res.message}"

        res = installer.mark_as_installed_one(item)
        assert res.success, f"Failed to mark {item} as installed: {res.message}"

    class MyInstallable(Installable):
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return hash("MyInstallable")

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
