#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict, cast

from cloudai import InstallStrategy, System
from cloudai.systems import SlurmSystem
from cloudai.util.docker_image_cache_manager import DockerImageCacheManager


class SlurmInstallStrategy(InstallStrategy):
    """
    Abstract base class for defining installation strategies specific to Slurm environments.

    Attributes
        slurm_system (SlurmSystem): A casted version of the `system` attribute, which provides Slurm-specific
            properties and methods.
        docker_image_cache_manager (DockerImageCacheManager): Manages the caching of Docker images.
        docker_image_url (str): URL to the Docker image in a remote container registry.
    """

    def __init__(
        self,
        system: System,
        env_vars: Dict[str, Any],
        cmd_args: Dict[str, Any],
    ) -> None:
        super().__init__(system, env_vars, cmd_args)
        self.slurm_system = cast(SlurmSystem, self.system)
        self.install_path = self.slurm_system.install_path
        self.docker_image_cache_manager = DockerImageCacheManager(
            self.slurm_system.install_path,
            self.slurm_system.cache_docker_images_locally,
            self.slurm_system.default_partition,
        )
        docker_image_url_info = self.cmd_args.get("docker_image_url")
        if docker_image_url_info is not None:
            self.docker_image_url = docker_image_url_info.get("default")
        else:
            self.docker_image_url = ""
