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
    def test_repo_exists(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        repo_path = installer.system.install_path / git.repo_name
        repo_path.mkdir()
        res = installer._install_one_git_repo(git)
        assert res.success
        assert res.message == f"Git repository already exists at {repo_path}."
        assert git.installed_path == repo_path

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
        assert res.message == "Failed to checkout commit: err"

    def test_all_good_flow(self, installer: Union[KubernetesInstaller, SlurmInstaller], git: GitRepo):
        installer._clone_repository = Mock(return_value=InstallStatusResult(True))
        installer._checkout_commit = Mock(return_value=InstallStatusResult(True))
        res = installer._install_one_git_repo(git)
        assert res.success
        assert git.installed_path == installer.system.install_path / git.repo_name

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
