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

from cloudai.core import BaseInstaller, DockerImage, Installable, InstallStatusResult
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
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} uninstalled")
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} is installed")
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, DockerImage):
            return InstallStatusResult(True, f"Docker image {item} marked as installed")
        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")
