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

from cloudai._core.system import System
from cloudai.systems.slurm import SlurmNodeState, SlurmSystem
from cloudai.systems.slurm.strategy import SlurmInstallStrategy
from cloudai.util import CommandShell


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

    def is_installed(self) -> bool:
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
            self.logger.error(
                "NeMo datasets are not installed on some nodes. Please ensure that the NeMo datasets are manually "
                "installed on each node in the specified data directory: {data_dir_path}. This directory should "
                "contain all necessary datasets for NeMo Launcher to function properly."
            )

        return repo_installed and docker_image_installed and datasets_ready

    def install(self) -> None:
        if self.is_installed():
            return

        self._check_install_path_access()

        subdir_path = os.path.join(self.install_path, self.SUBDIR_PATH)
        os.makedirs(subdir_path, exist_ok=True)

        data_dir_path = self.default_cmd_args["data_dir"]
        if not self._check_datasets_on_nodes(data_dir_path):
            self.logger.error(
                "Some nodes do not have the NeMoLauncher datasets installed. Please note that CloudAI does not cover "
                "dataset installation. Users are responsible for ensuring that datasets are installed on all nodes."
            )

        self._clone_repository(subdir_path)
        if not os.path.isfile(self.docker_image_url):
            self._setup_docker_image(self.slurm_system, subdir_path)

    def _check_install_path_access(self):
        """
        Check if the install path exists and if there is permission to create a directory or file in the path.

        Raises
            PermissionError: If the install path does not exist or if there is no permission to create
                directories/files.
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
                "Skipping NeMoLauncher dataset verification."
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
                "This directory should contain all necessary datasets for NeMo Launcher to function properly.",
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
        try:
            if not os.path.exists(repo_path):
                subprocess.run(
                    ["git", "clone", self.repository_url, repo_path],
                    check=True,
                )
                subprocess.run(
                    ["git", "checkout", self.repository_commit_hash],
                    cwd=repo_path,
                    check=True,
                )
                subprocess.run(
                    [
                        "./venv/bin/pip",
                        "install",
                        "-r",
                        os.path.join(repo_path, "requirements.txt"),
                    ],
                    check=True,
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository. Error: {e}") from e

    def _setup_docker_image(self, system: SlurmSystem, subdir_path: str) -> None:
        """
        Download and sets up Docker image if not already present.

        Args:
            system (SlurmSystem): The system schema object.
            subdir_path (str): Subdirectory path for installation.
        """
        docker_image_path = os.path.join(subdir_path, self.DOCKER_IMAGE_FILENAME)

        # Remove existing Docker image if it exists
        shell = CommandShell()
        remove_cmd = f"rm -f {docker_image_path}"
        process = shell.execute(remove_cmd)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Failed to remove existing Docker image: {stderr}")

        # Import new Docker image using enroot
        if not os.path.isfile(docker_image_path):
            enroot_import_cmd = (
                f"srun"
                f" --export=ALL"
                f" --partition={system.default_partition}"
                f" enroot import -o {docker_image_path} docker://{self.docker_image_url}"
            )

            try:
                subprocess.run(enroot_import_cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to import Docker image: {e}") from e

    def uninstall(self) -> None:
        subdir_path = os.path.join(self.install_path, self.SUBDIR_PATH)
        docker_image_path = os.path.join(subdir_path, self.DOCKER_IMAGE_FILENAME)
        if os.path.isfile(docker_image_path):
            os.remove(docker_image_path)
        if os.path.exists(subdir_path):
            shutil.rmtree(subdir_path)
