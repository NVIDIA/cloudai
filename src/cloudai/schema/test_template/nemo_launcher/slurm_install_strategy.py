# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pkg_resources

from cloudai import InstallStatusResult, System
from cloudai.systems.slurm import SlurmNodeState
from cloudai.systems.slurm.strategy import SlurmInstallStrategy


class DatasetCheckResult:
    """
    Result class for dataset check on Slurm nodes.

    Attributes
        success (bool): Whether the datasets are present on all nodes.
        nodes_without_datasets (List[str]): List of nodes missing one or more datasets.
    """

    def __init__(self, success: bool, nodes_without_datasets: List[str]) -> None:
        self.success = success
        self.nodes_without_datasets = nodes_without_datasets


class NeMoLauncherSlurmInstallStrategy(SlurmInstallStrategy):
    """
    Install strategy for NeMo-Launcher on Slurm systems.

    Attributes
        SUBDIR_PATH (str): Subdirectory within the system's install path where the NeMo-Launcher and its Docker image
            will be stored.
        REPOSITORY_NAME (str): Name of the NeMo-Launcher repository.
        DOCKER_IMAGE_FILENAME (str): Filename of the Docker image to be downloaded.
        repository_url (str): URL to the NeMo-Launcher Git repository.
        repository_commit_hash (str): Specific commit hash to checkout after cloning the repository.
        docker_image_url (str): URL to the Docker image in a remote container registry.
    """

    SUBDIR_PATH = "NeMo-Launcher"
    REPOSITORY_NAME = "NeMo-Launcher"
    DOCKER_IMAGE_FILENAME = "nemo_launcher.sqsh"

    def __init__(self, system: System, cmd_args: Dict[str, Any]) -> None:
        super().__init__(system, cmd_args)
        self.repository_url = cmd_args["repository_url"]
        self.repository_commit_hash = cmd_args["repository_commit_hash"]
        self.docker_image_url = cmd_args["docker_image_url"]

    def is_installed(self) -> InstallStatusResult:
        subdir_path = self.system.install_path / self.SUBDIR_PATH
        repo_path = subdir_path / self.REPOSITORY_NAME
        repo_installed = repo_path.is_dir()

        docker_image_installed = self.docker_image_cache_manager.check_docker_image_exists(
            self.docker_image_url, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME
        ).success

        requirements_installed = self._check_requirements_installed(subdir_path)

        data_dir_path = Path(self.default_cmd_args["data_dir"])
        datasets_check_result = self._check_datasets_on_nodes(data_dir_path)
        if not datasets_check_result.success:
            return InstallStatusResult(
                success=False,
                message=(
                    "NeMo datasets are not installed on some nodes. "
                    f"Nodes without datasets: {', '.join(datasets_check_result.nodes_without_datasets)}. "
                    f"Please ensure that the NeMo datasets are manually installed on each node in the specified "
                    f"data directory: {data_dir_path}. This directory should contain all necessary datasets for "
                    f"NeMo Launcher to function properly."
                ),
            )

        if repo_installed and docker_image_installed and requirements_installed and datasets_check_result.success:
            return InstallStatusResult(success=True)
        else:
            missing_components = []
            if not repo_installed:
                missing_components.append(
                    f"Repository at {repo_path} from {self.repository_url} "
                    f"with commit hash {self.repository_commit_hash}"
                )
            if not docker_image_installed:
                docker_image_path = subdir_path / self.DOCKER_IMAGE_FILENAME
                missing_components.append(f"Docker image at {docker_image_path} from URL {self.docker_image_url}")
            if not requirements_installed:
                missing_components.append("Python requirements from requirements.txt")
            if not datasets_check_result.success:
                missing_components.append(f"Datasets in {data_dir_path} on some nodes")
            return InstallStatusResult(
                success=False,
                message="The following components are missing:\n"
                + "\n".join(f"    - {item}" for item in missing_components),
            )

    def _check_requirements_installed(self, subdir_path: Path) -> bool:
        """
        Check if all the required Python packages are installed based on the requirements.txt file.

        Args:
            subdir_path (Path): Subdirectory path where the repository is installed.

        Returns:
            bool: True if all requirements are installed, False otherwise.
        """
        repo_path = subdir_path / self.REPOSITORY_NAME
        requirements_file = repo_path / "requirements.txt"

        if not requirements_file.is_file():
            logging.warning(f"requirements.txt not found in {repo_path}")
            return False

        with requirements_file.open() as f:
            requirements = f.read().splitlines()

        try:
            pkg_resources.require(requirements)
            return True
        except pkg_resources.DistributionNotFound as e:
            logging.warning(f"Missing required package: {e}")
            return False
        except pkg_resources.VersionConflict as e:
            logging.warning(f"Version conflict: {e}")
            return False

    def install(self) -> InstallStatusResult:
        install_status = self.is_installed()
        if install_status.success:
            return InstallStatusResult(success=True, message="NeMo-Launcher is already installed.")

        try:
            self._check_install_path_access()
        except PermissionError as e:
            return InstallStatusResult(success=False, message=str(e))

        subdir_path = self.system.install_path / self.SUBDIR_PATH
        subdir_path.mkdir(parents=True, exist_ok=True)

        data_dir_path = Path(self.default_cmd_args["data_dir"])
        datasets_check_result = self._check_datasets_on_nodes(data_dir_path)
        if not datasets_check_result.success:
            return InstallStatusResult(
                success=False,
                message=(
                    "Some nodes do not have the NeMo-Launcher datasets installed. "
                    f"Nodes without datasets: {', '.join(datasets_check_result.nodes_without_datasets)}. "
                    f"Datasets directory: {data_dir_path}. "
                    "Please ensure that datasets are installed on all nodes."
                ),
            )

        try:
            self._clone_repository(subdir_path)
            self._install_requirements(subdir_path)
            docker_image_result = self.docker_image_cache_manager.ensure_docker_image(
                self.docker_image_url, self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME
            )
            if not docker_image_result.success:
                raise RuntimeError(
                    "Failed to download and import the Docker image for NeMo-Launcher. "
                    f"Error: {docker_image_result.message}"
                )
        except RuntimeError as e:
            return InstallStatusResult(success=False, message=str(e))

        return InstallStatusResult(success=True)

    def uninstall(self) -> InstallStatusResult:
        docker_image_result = self.docker_image_cache_manager.uninstall_cached_image(
            self.SUBDIR_PATH, self.DOCKER_IMAGE_FILENAME
        )
        if not docker_image_result.success:
            return InstallStatusResult(
                success=False,
                message=(
                    "Failed to remove the Docker image for NeMo-Launcher. " f"Error: {docker_image_result.message}"
                ),
            )

        return InstallStatusResult(success=True)

    def _check_install_path_access(self):
        """
        Check if the install path exists and if there is permission to create a directory or file in the path.

        Raises
            PermissionError: If the install path does not exist or if there is no permission to create directories and
                files.
        """
        if not self.system.install_path.exists():
            raise PermissionError(f"Install path {self.system.install_path} does not exist.")
        if not self.system.install_path.is_dir() or not os.access(self.system.install_path, os.W_OK):
            raise PermissionError(f"No permission to write in install path {self.system.install_path}.")

    def _check_datasets_on_nodes(self, data_dir_path: Path) -> DatasetCheckResult:
        """
        Verify the presence of specified dataset files and directories on all idle compute nodes.

        Default partition is used.

        This method uses parallel execution to check datasets on multiple nodes simultaneously, improving efficiency
        for systems with multiple nodes.

        Args:
            data_dir_path (Path): Path where dataset files and directories are stored.

        Returns:
            DatasetCheckResult: Result object containing success status and nodes without datasets.
        """
        partition_nodes = self.slurm_system.get_partition_nodes(self.slurm_system.default_partition)

        idle_nodes = [node.name for node in partition_nodes if node.state == SlurmNodeState.IDLE]

        if not idle_nodes:
            logging.warning(
                "There are no idle nodes in the default partition to check. "
                "Skipping NeMo-Launcher dataset verification."
            )
            return DatasetCheckResult(success=True, nodes_without_datasets=[])

        nodes_without_datasets = []
        with ThreadPoolExecutor(max_workers=len(idle_nodes)) as executor:
            futures = {
                executor.submit(
                    self._check_dataset_on_node,
                    node,
                    data_dir_path,
                    [
                        "bpe",
                        "my-gpt3_00_text_document.bin",
                        "my-gpt3_00_text_document.idx",
                    ],
                ): node
                for node in idle_nodes
            }
            for future in as_completed(futures):
                node = futures[future]
                if not future.result():
                    nodes_without_datasets.append(node)

        return DatasetCheckResult(success=not nodes_without_datasets, nodes_without_datasets=nodes_without_datasets)

    def _check_dataset_on_node(self, node: str, data_dir_path: Path, dataset_items: List[str]) -> bool:
        """
        Check if dataset files and directories exist on a single compute node.

        Args:
            node (str): The name of the compute node.
            data_dir_path (Path): Path to the data directory.
            dataset_items (List[str]): List of dataset file and directory names to check.

        Returns:
            bool: True if all dataset files and directories exist on the node, False otherwise.
        """
        python_check_script = (
            f"import os;print(all(Path('{data_dir_path}') / item).exists() for item in {dataset_items})"
        )
        cmd = (
            f"srun --nodes=1 --nodelist={node} "
            f"--partition={self.slurm_system.default_partition} "
            f'python -c "{python_check_script}"'
        )
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() == "True"

    def _clone_repository(self, subdir_path: Path) -> None:
        """
        Clones NeMo-Launcher repository into specified path if it does not already exist.

        Args:
            subdir_path (Path): Subdirectory path for installation.
        """
        repo_path = subdir_path / self.REPOSITORY_NAME

        if repo_path.exists():
            logging.warning("Repository already exists at %s, clone skipped", repo_path)
        else:
            logging.debug("Cloning NeMo-Launcher repository into %s", repo_path)
            clone_cmd = ["git", "clone", self.repository_url, str(repo_path)]
            result = subprocess.run(clone_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone repository: {result.stderr}")

        logging.debug("Checking out specific commit %s in repository", self.repository_commit_hash)
        checkout_cmd = ["git", "checkout", self.repository_commit_hash]
        result = subprocess.run(checkout_cmd, cwd=str(repo_path), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout commit: {result.stderr}")

    def _install_requirements(self, subdir_path: Path) -> None:
        """
        Installs the required Python packages from the requirements.txt file in the cloned repository.

        Args:
            subdir_path (Path): Subdirectory path for installation.
        """
        repo_path = subdir_path / self.REPOSITORY_NAME
        requirements_file = repo_path / "requirements.txt"

        if requirements_file.is_file():
            logging.debug("Installing requirements from %s", requirements_file)
            install_cmd = ["pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install requirements: {result.stderr}")
        else:
            logging.warning("requirements.txt not found in %s", repo_path)
