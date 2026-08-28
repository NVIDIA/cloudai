# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import shlex
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from cloudai.core import JobIdRetrievalError

if TYPE_CHECKING:
    from cloudai.systems.slurm import SlurmSystem


class DockerImageCacheResult:
    """Result of a Docker image caching operation."""

    def __init__(
        self,
        success: bool,
        docker_image_path: Optional[Path] = None,
        message: str = "",
    ) -> None:
        self.success = success
        self.docker_image_path = docker_image_path
        self.message = message

    def __bool__(self):
        """Return whether the cache operation succeeded."""
        return self.success

    def __str__(self):
        """Return the result message."""
        return self.message


class DockerImageCacheManager:
    """Generate and interpret jobs which cache Docker images on a Slurm system."""

    def __init__(self, system: SlurmSystem) -> None:
        self.system = system

    def ensure_docker_image(self, docker_image_url: str, docker_image_filename: str) -> DockerImageCacheResult:
        result = self.check_docker_image_exists(docker_image_url, docker_image_filename)
        if result.success:
            return result
        if self.system.cache_docker_images_locally:
            return self.cache_docker_image(docker_image_url, docker_image_filename)
        return result

    def check_docker_image_exists(self, docker_image_url: str, docker_image_filename: str) -> DockerImageCacheResult:
        if not self.system.cache_docker_images_locally:
            return DockerImageCacheResult(True)

        docker_image_path = Path(docker_image_url)
        if docker_image_path.is_file() and docker_image_path.exists():
            return DockerImageCacheResult(
                True,
                docker_image_path.absolute(),
                f"Docker image file path is valid: {docker_image_url}.",
            )

        if not self.system.install_path.exists():
            message = f"Install path {self.system.install_path.absolute()} does not exist."
            logging.debug(message)
            return DockerImageCacheResult(False, Path(), message)

        docker_image_path = self.system.install_path / docker_image_filename
        if docker_image_path.is_file() and docker_image_path.exists():
            message = f"Cached Docker image already exists at {docker_image_path}."
            logging.debug(message)
            return DockerImageCacheResult(True, docker_image_path.absolute(), message)

        message = f"Docker image does not exist at the specified path: {docker_image_path}."
        logging.debug(message)
        return DockerImageCacheResult(False, Path(), message)

    def _job_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.system.account:
            return f"{self.system.account}-CloudAI_install_docker_image.{timestamp}"
        return f"CloudAI_install_docker_image_{timestamp}"

    def _write_import_script(self, docker_image_url: str, docker_image_path: Path) -> tuple[Path, Path]:
        job_name = self._job_name()
        script_path = self.system.install_path / f".{job_name}.sh"
        stdout_path = self.system.install_path / f".{job_name}.out"
        stderr_path = self.system.install_path / f".{job_name}.err"
        directives = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={stdout_path}",
            f"#SBATCH --error={stderr_path}",
            f"#SBATCH --partition={self.system.default_partition}",
            "#SBATCH -N1",
            "#SBATCH --ntasks=1",
        ]
        if self.system.account:
            directives.append(f"#SBATCH --account={self.system.account}")
        if self.system.supports_gpu_directives:
            directives.append("#SBATCH --gres=gpu:1")
        directives.extend(f"#SBATCH {arg}" for arg in self.system.extra_sbatch_args)

        srun = ["srun", "--export=ALL", "--ntasks=1"]
        if self.system.extra_srun_args:
            srun.append(self.system.extra_srun_args)
        srun.extend(
            [
                "enroot import -o",
                shlex.quote(str(docker_image_path)),
                shlex.quote(f"docker://{docker_image_url}"),
            ]
        )
        script_path.write_text("\n".join([*directives, "", " ".join(srun), ""]), encoding="utf-8")
        return script_path, stderr_path

    def cache_docker_image(self, docker_image_url: str, docker_image_filename: str) -> DockerImageCacheResult:
        docker_image_path = self.system.install_path / docker_image_filename
        if docker_image_path.is_file():
            message = f"Cached Docker image already exists at {docker_image_path}."
            logging.info(message)
            return DockerImageCacheResult(True, docker_image_path.absolute(), message)

        if not self.system.install_path.exists():
            message = f"Install path {self.system.install_path.absolute()} does not exist."
            logging.error(message)
            return DockerImageCacheResult(False, Path(), message)
        if not os.access(self.system.install_path, os.W_OK):
            message = f"No permission to write in install path {self.system.install_path}."
            logging.error(message)
            return DockerImageCacheResult(False, Path(), message)

        script_path, stderr_path = self._write_import_script(docker_image_url, docker_image_path)
        try:
            self.system.submit_sbatch(script_path, "Docker image import", wait=True)
        except JobIdRetrievalError as error:
            message = f"Failed to import Docker image {docker_image_url}: {error}"
            logging.error(message)
            return DockerImageCacheResult(False, message=message)

        stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.is_file() else ""
        if docker_image_path.is_file():
            message = f"Docker image cached successfully at {docker_image_path}."
            logging.debug(message)
            return DockerImageCacheResult(True, docker_image_path.absolute(), message)

        if "Disk quota exceeded" in stderr or "Write error" in stderr:
            message = (
                f"Failed to cache Docker image {docker_image_url}. Error: '{stderr}'\n\n"
                "This error indicates a disk-related issue. Please check if the disk is full or not usable. "
                "If the disk is full, consider using a different disk or removing unnecessary files."
            )
        else:
            message = f"Failed to import Docker image {docker_image_url}. Error: {stderr or 'image was not created'}"
        logging.error(message)
        return DockerImageCacheResult(False, message=message)

    def uninstall_cached_image(self, docker_image_filename: str) -> DockerImageCacheResult:
        docker_image_path = self.system.install_path / docker_image_filename
        if docker_image_path.is_file():
            try:
                docker_image_path.unlink()
                message = f"Cached Docker image removed successfully from {docker_image_path}."
                logging.info(message)
                return DockerImageCacheResult(True, docker_image_path.absolute(), message)
            except OSError as error:
                message = f"Failed to remove cached Docker image at {docker_image_path}. Error: {error}"
                logging.error(message)
                return DockerImageCacheResult(False, docker_image_path, message)

        message = f"No cached Docker image found to remove at {docker_image_path}."
        logging.warning(message)
        return DockerImageCacheResult(True, docker_image_path.absolute(), message)
