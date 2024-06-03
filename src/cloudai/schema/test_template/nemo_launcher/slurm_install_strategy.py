# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from cloudai._core.install_status_result import InstallStatusResult
from cloudai._core.system import System
from cloudai.systems.slurm import SlurmNodeState, SlurmSystem
from cloudai.systems.slurm.strategy import SlurmInstallStrategy


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

    def __init__(
        self,
        system: System,
        env_vars: Dict[str, Any],
        cmd_args: Dict[str, Any],
    ) -> None:
        super().__init__(system, env_vars, cmd_args)
        self.repository_url = self._validate_cmd_arg(cmd_args, "repository_url")
        self.repository_commit_hash = self._validate_cmd_arg(cmd_args, "repository_commit_hash")
        self.docker_image_url = self._validate_cmd_arg(cmd_args, "docker_image_url")
        self.logger = logging.getLogger(__name__)
        self._configure_logging()

    def _validate_cmd_arg(self, cmd_args: Dict[str, Any], arg_name: str) -> str:
        """
        Validate and returns specified command-line argument.

        Args:
            cmd_args (Dict[str, Any]): Command-line arguments.
            arg_name (str): Argument name to validate.

        Returns:
            str: Validated command-line argument value.

        Raises:
            ValueError: If argument not specified or default value is None.
        """
        arg_info = cmd_args.get(arg_name)
        arg_value = arg_info.get("default") if arg_info else None
        if arg_value is None:
            raise ValueError(f"{arg_name} not specified or default value is None in command-line arguments.")
        return arg_value

    def _configure_logging(self):
        """
        Configure logging to output error messages to stdout.

        This method sets the logger's level to INFO, capturing messages at this level and above. It also configures a
        StreamHandler for error messages (ERROR level and above), directing them to stdout for immediate visibility.

        StreamHandler formatting includes the logger's name, the log level, and the message.
        """
        self.logger.setLevel(logging.INFO)

        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.ERROR)

        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        c_handler.setFormatter(c_format)

        self.logger.addHandler(c_handler)

    def is_installed(self) -> InstallStatusResult:
        subdir_path = os.path.join(self.install_path, self.SUBDIR_PATH)
        docker_image_path = os.path.join(subdir_path, self.DOCKER_IMAGE_FILENAME)
        repo_path = os.path.join(subdir_path, self.REPOSITORY_NAME)
        repo_installed = os.path.isdir(repo_path)

        if not os.path.isfile(self.docker_image_url):
            docker_image_installed = os.path.isfile(docker_image_path)
        else:
            docker_image_installed = True

        data_dir_path = self.default_cmd_args["data_dir"]
        datasets_ready = self._check_datasets_on_nodes(data_dir_path)
        if not datasets_ready:
            return InstallStatusResult(
                success=False,
                message=(
                    "NeMo datasets are not installed on some nodes. Please ensure that the NeMo datasets are manually "
                    "installed on each node in the specified data directory: {data_dir_path}. This directory should "
                    "contain all necessary datasets for NeMo Launcher to function properly."
                ),
            )

        if repo_installed and docker_image_installed and datasets_ready:
            return InstallStatusResult(success=True)
        else:
            missing_components = []
            if not repo_installed:
                missing_components.append("Repository")
            if not docker_image_installed:
                missing_components.append("Docker image")
            if not datasets_ready:
                missing_components.append("Datasets on some nodes")
            return InstallStatusResult(
                success=False,
                message=f"The following components are missing: {', '.join(missing_components)}.",
            )

    def install(self) -> InstallStatusResult:
        install_status = self.is_installed()
        if install_status.success:
            return InstallStatusResult(success=True, message="NeMo-Launcher is already installed.")

        try:
            self._check_install_path_access()
        except PermissionError as e:
            return InstallStatusResult(success=False, message=str(e))

        subdir_path = os.path.join(self.install_path, self.SUBDIR_PATH)
        os.makedirs(subdir_path, exist_ok=True)

        data_dir_path = self.default_cmd_args["data_dir"]
        if not self._check_datasets_on_nodes(data_dir_path):
            return InstallStatusResult(
                success=False,
                message=(
                    "Some nodes do not have the NeMo-Launcher datasets installed. "
                    "Please ensure that datasets are installed on all nodes."
                ),
            )

        try:
            self._clone_repository(subdir_path)
            if not os.path.isfile(self.docker_image_url):
                self._setup_docker_image(self.slurm_system, subdir_path)
        except RuntimeError as e:
            return InstallStatusResult(success=False, message=str(e))

        return InstallStatusResult(success=True)

    def uninstall(self) -> InstallStatusResult:
        subdir_path = os.path.join(self.install_path, self.SUBDIR_PATH)
        docker_image_path = os.path.join(subdir_path, self.DOCKER_IMAGE_FILENAME)
        try:
            if os.path.isfile(docker_image_path):
                os.remove(docker_image_path)
            if os.path.exists(subdir_path):
                shutil.rmtree(subdir_path)
        except Exception as e:
            return InstallStatusResult(
                success=False,
                message=(
                    f"Failed to remove Docker image or directory at {subdir_path}. "
                    f"Error: {e}. "
                    "Please check the file permissions and ensure the file is not in use."
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
        if not os.path.exists(self.install_path):
            raise PermissionError(f"Install path {self.install_path} does not exist.")
        if not os.access(self.install_path, os.W_OK):
            raise PermissionError(f"No permission to write in install path {self.install_path}.")

    def _check_datasets_on_nodes(self, data_dir_path: str) -> bool:
        """
        Verify the presence of specified dataset files and directories on all idle compute nodes.

        Default partition is used.

        This method uses parallel execution to check datasets on multiple nodes simultaneously, improving efficiency
        for systems with multiple nodes.

        Args:
            data_dir_path (str): Path where dataset files and directories are stored.

        Returns:
            bool: True if all specified dataset files and directories are present
                  on all idle nodes, False if any item is missing on any node.
        """
        partition_nodes = self.slurm_system.get_partition_nodes(self.slurm_system.default_partition)

        idle_nodes = [node.name for node in partition_nodes if node.state == SlurmNodeState.IDLE]

        if not idle_nodes:
            self.logger.info(
                "There are no idle nodes in the default partition to check. "
                "Skipping NeMo-Launcher dataset verification."
            )
            return True

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

        if nodes_without_datasets:
            self.logger.error(
                "The following nodes are missing one or more datasets and require manual installation: %s",
                ", ".join(nodes_without_datasets),
            )
            self.logger.error(
                "Please ensure that the NeMo datasets are installed on each node in the specified data directory: %s. "
                "This directory should contain all necessary datasets for NeMo-Launcher to function properly.",
                data_dir_path,
            )
            return False
        return True

    def _check_dataset_on_node(self, node: str, data_dir_path: str, dataset_items: List[str]) -> bool:
        """
        Check if dataset files and directories exist on a single compute node.

        Args:
            node (str): The name of the compute node.
            data_dir_path (str): Path to the data directory.
            dataset_items (List[str]): List of dataset file and directory names to check.

        Returns:
            bool: True if all dataset files and directories exist on the node, False otherwise.
        """
        python_check_script = (
            f"import os;print(all(os.path.isfile(os.path.join('{data_dir_path}', "
            f"item)) or os.path.isdir(os.path.join('{data_dir_path}', item)) "
            f"for item in {dataset_items}))"
        )
        cmd = (
            f"srun --nodes=1 --nodelist={node} "
            f"--partition={self.slurm_system.default_partition} "
            f'python -c "{python_check_script}"'
        )
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() == "True"

    def _clone_repository(self, subdir_path: str) -> None:
        """
        Clones NeMo-Launcher repository into specified path.

        Args:
            subdir_path (str): Subdirectory path for installation.
        """
        repo_path = os.path.join(subdir_path, self.REPOSITORY_NAME)
        self.logger.info("Cloning NeMo-Launcher repository into %s", repo_path)
        clone_cmd = ["git", "clone", self.repository_url, repo_path]
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone repository: {result.stderr}")

        self.logger.info("Checking out specific commit %s in repository", self.repository_commit_hash)
        checkout_cmd = ["git", "checkout", self.repository_commit_hash]
        result = subprocess.run(checkout_cmd, cwd=repo_path, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout commit: {result.stderr}")

    def _setup_docker_image(self, slurm_system: SlurmSystem, subdir_path: str) -> None:
        """
        Download and install Docker image on Slurm nodes.

        Args:
            slurm_system (SlurmSystem): Slurm system instance.
            subdir_path (str): Subdirectory where Docker image will be installed.

        Raises:
            RuntimeError: If Docker image download or installation fails.
        """
        docker_image_path = os.path.join(subdir_path, self.DOCKER_IMAGE_FILENAME)
        self.logger.info("Downloading Docker image from %s to %s", self.docker_image_url, docker_image_path)
        result = subprocess.run(
            ["curl", "-o", docker_image_path, self.docker_image_url], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download Docker image: {result.stderr}")

        partition_nodes = slurm_system.get_partition_nodes(slurm_system.default_partition)
        idle_nodes = [node.name for node in partition_nodes if node.state == SlurmNodeState.IDLE]
        if not idle_nodes:
            self.logger.info(
                "There are no idle nodes in the default partition to install the Docker image. "
                "Skipping Docker image installation."
            )
            return

        with ThreadPoolExecutor(max_workers=len(idle_nodes)) as executor:
            futures = {
                executor.submit(
                    self._install_docker_image_on_node,
                    node,
                    docker_image_path,
                    docker_image_path,
                ): node
                for node in idle_nodes
            }
            for future in as_completed(futures):
                node = futures[future]
                if future.result() is not True:
                    self.logger.error("Failed to install Docker image on node: %s", node)
                    raise RuntimeError(f"Failed to install Docker image on node: {node}")

    def _install_docker_image_on_node(self, node: str, image_source_path: str, image_dest_path: str) -> bool:
        """
        Install Docker image on a specific node.

        Args:
            node (str): Node name.
            image_source_path (str): Source path of the Docker image.
            image_dest_path (str): Destination path for the Docker image on the node.

        Returns:
            bool: True if installation is successful, False otherwise.
        """
        self.logger.info("Installing Docker image on node %s", node)
        install_cmd = [
            "srun",
            "--nodes=1",
            "--ntasks=1",
            "--nodelist=" + node,
            "cp",
            image_source_path,
            image_dest_path,
        ]
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        return result.returncode == 0
