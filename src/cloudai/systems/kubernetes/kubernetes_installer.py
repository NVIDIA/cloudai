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

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from cloudai.core import BaseInstaller, DockerImage, GitRepo, Installable, InstallStatusResult
from cloudai.util.lazy_imports import lazy


class KubernetesInstaller(BaseInstaller):
    """Installer for Kubernetes systems."""

    def _check_prerequisites(self) -> InstallStatusResult:
        """
        Check for the presence of required binaries and Kubernetes configurations.

        This ensures the system environment is properly set up before proceeding with the installation
        or uninstallation processes.

        Returns
            InstallStatusResult: Result containing the status of the prerequisite check and any error message.
        """
        # Check base prerequisites using the parent class method
        base_prerequisites_result = super()._check_prerequisites()
        if not base_prerequisites_result.success:
            logging.error(f"Prerequisite check failed in base installer: {base_prerequisites_result.message}")
            return InstallStatusResult(False, base_prerequisites_result.message)

        # Load Kubernetes configuration
        try:
            lazy.k8s.config.load_kube_config()
        except Exception as e:
            message = (
                f"Installation failed during prerequisite checking stage because Kubernetes configuration could not "
                f"be loaded. Please ensure that your Kubernetes configuration is properly set up. Original error: {e!r}"
            )
            logging.error(message)
            return InstallStatusResult(False, message)

        # Check MPIJob-related prerequisites
        mpi_job_result = self._check_mpi_job_prerequisites()
        if not mpi_job_result.success:
            return mpi_job_result

        logging.info("All prerequisites are met. Proceeding with installation.")
        return InstallStatusResult(True)

    def _check_mpi_job_prerequisites(self) -> InstallStatusResult:
        """
        Check if the MPIJob CRD is installed and if MPIJob kind is supported in the Kubernetes cluster.

        This ensures that the system is ready for MPI-based operations.

        Returns
            InstallStatusResult: Result containing the status of the MPIJob prerequisite check and any error message.
        """
        # Check if MPIJob CRD is installed
        try:
            custom_api = lazy.k8s.client.CustomObjectsApi()
            custom_api.get_cluster_custom_object(group="kubeflow.org", version="v1", plural="mpijobs", name="mpijobs")
        except lazy.k8s.client.ApiException as e:
            if e.status == 404:
                message = (
                    "Installation failed during prerequisite checking stage because MPIJob CRD is not installed on "
                    "this Kubernetes cluster. Please ensure that the MPI Operator is installed and MPIJob kind is "
                    "supported. You can follow the instructions in the MPI Operator repository to install it: "
                    "https://github.com/kubeflow/mpi-operator"
                )
                logging.error(message)
                return InstallStatusResult(False, message)
            else:
                message = (
                    f"Installation failed during prerequisite checking stage due to an error while checking for MPIJob "
                    f"CRD. Original error: {e!r}. Please ensure that the Kubernetes cluster is accessible and the "
                    f"MPI Operator is correctly installed."
                )
                logging.error(message)
                return InstallStatusResult(False, message)

        # Check if MPIJob kind is supported
        try:
            api_resources = lazy.k8s.client.ApiextensionsV1Api().list_custom_resource_definition()
            mpi_job_supported = any(item.metadata.name == "mpijobs.kubeflow.org" for item in api_resources.items)
        except lazy.k8s.client.ApiException as e:
            message = (
                f"Installation failed during prerequisite checking stage due to an error while checking for MPIJob "
                f"kind support. Original error: {e!r}. Please ensure that the Kubernetes cluster is accessible and "
                f"the MPI Operator is correctly installed."
            )
            logging.error(message)
            return InstallStatusResult(False, message)

        if not mpi_job_supported:
            message = (
                "Installation failed during prerequisite checking stage because MPIJob kind is not supported on this "
                "Kubernetes cluster. Please ensure that the MPI Operator is installed and MPIJob kind is supported. "
                "You can follow the instructions in the MPI Operator repository to install it: "
                "https://github.com/kubeflow/mpi-operator"
            )
            logging.error(message)
            return InstallStatusResult(False, message)

        return InstallStatusResult(True)

    def install_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} installed")
        elif isinstance(item, GitRepo):
            return self._install_one_git_repo(item)
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} uninstalled")
        elif isinstance(item, GitRepo):
            return self._uninstall_git_repo(item)
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} is installed")
        elif isinstance(item, GitRepo):
            repo_path = self.system.install_path / item.repo_name
            if repo_path.exists():
                item.installed_path = repo_path
                return InstallStatusResult(True)
            return InstallStatusResult(False, f"Git repository {item.url} not cloned")
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} marked as installed")
        elif isinstance(item, GitRepo):
            item.installed_path = self.system.install_path / item.repo_name
            return InstallStatusResult(True)
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def _install_one_git_repo(self, item: GitRepo) -> InstallStatusResult:
        repo_path = self.system.install_path / item.repo_name
        if repo_path.exists():
            item.installed_path = repo_path
            msg = f"Git repository already exists at {repo_path}."
            logging.debug(msg)
            return InstallStatusResult(True, msg)

        res = self._clone_repository(item.url, repo_path)
        if not res.success:
            return res

        res = self._checkout_commit(item.commit, repo_path)
        if not res.success:
            return res

        item.installed_path = repo_path
        return InstallStatusResult(True)

    def _clone_repository(self, git_url: str, path: Path) -> InstallStatusResult:
        logging.debug(f"Cloning repository {git_url} into {path}")
        clone_cmd = ["git", "clone"]

        if self.is_low_thread_environment:
            clone_cmd.extend(["-c", "pack.threads=4"])

        clone_cmd.extend([git_url, str(path)])

        logging.debug(f"Running git clone command: {' '.join(clone_cmd)}")
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to clone repository: {result.stderr}")
        return InstallStatusResult(True)

    def _checkout_commit(self, commit_hash: str, path: Path) -> InstallStatusResult:
        logging.debug(f"Checking out specific commit in {path}: {commit_hash}")
        checkout_cmd = ["git", "checkout", commit_hash]
        result = subprocess.run(checkout_cmd, cwd=str(path), capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to checkout commit: {result.stderr}")
        return InstallStatusResult(True)

    def _uninstall_git_repo(self, item: GitRepo) -> InstallStatusResult:
        logging.debug(f"Uninstalling git repository at {item.installed_path=}")
        repo_path = item.installed_path if item.installed_path else self.system.install_path / item.repo_name
        if not repo_path.exists():
            msg = f"Repository {item.url} is not cloned."
            return InstallStatusResult(True, msg)

        logging.debug(f"Removing folder {repo_path}")
        shutil.rmtree(repo_path)
        item.installed_path = None

        return InstallStatusResult(True)
