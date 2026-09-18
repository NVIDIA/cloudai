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

from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cloudai.core import JobFailureError, JobIdRetrievalError
from cloudai.systems.slurm.docker_image_cache_manager import DockerImageCacheManager
from cloudai.systems.slurm.slurm_system import SlurmSystem


def test_ensure_existing_image_file(slurm_system: SlurmSystem, tmp_path: Path):
    image = tmp_path / "existing.sqsh"
    image.touch()
    slurm_system.cache_docker_images_locally = True

    result = DockerImageCacheManager(slurm_system).ensure_docker_image(str(image), "cached.sqsh")

    assert result.success
    assert result.docker_image_path == image


def test_ensure_docker_image_no_local_cache(slurm_system: SlurmSystem):
    slurm_system.cache_docker_images_locally = False

    result = DockerImageCacheManager(slurm_system).ensure_docker_image("docker.io/hello-world", "image.sqsh")

    assert result.success
    assert result.docker_image_path is None


@pytest.mark.parametrize(
    "account,supports_gpu,cleanup_error", [(None, False, False), ("test-account", True, False), (None, False, True)]
)
def test_cache_docker_image_submits_one_node_sbatch_job(
    slurm_system: SlurmSystem,
    monkeypatch: pytest.MonkeyPatch,
    account: str | None,
    supports_gpu: bool,
    cleanup_error: bool,
):
    slurm_system.cache_docker_images_locally = True
    slurm_system.account = account
    slurm_system.supports_gpu_directives_cache = supports_gpu
    slurm_system.extra_srun_args = "--reservation test-reservation"
    slurm_system.install_path.mkdir(parents=True, exist_ok=True)
    image_path = slurm_system.install_path / "image.sqsh"
    if cleanup_error:
        monkeypatch.setattr(Path, "unlink", Mock(side_effect=PermissionError("cleanup failed")))

    def submit(script_path: Path, operation_name: str, *, wait: bool) -> int:
        content = script_path.read_text(encoding="utf-8")
        image_hash = sha256(b"docker.io/hello-world").hexdigest()[:8]
        assert image_hash in script_path.name
        assert "#SBATCH --partition=" + slurm_system.default_partition in content
        assert "#SBATCH -N1" in content
        assert "#SBATCH --ntasks=1" in content
        assert "srun --export=ALL --ntasks=1 --reservation test-reservation enroot import -o" in content
        assert "docker://docker.io/hello-world" in content
        if account:
            assert f"#SBATCH --account={account}" in content
        if supports_gpu:
            assert "#SBATCH --gres=gpu:1" in content
        assert operation_name == "Docker image import"
        assert wait is True
        script_path.with_suffix(".out").touch()
        script_path.with_suffix(".err").touch()
        image_path.touch()
        return 123

    with patch.object(SlurmSystem, "submit_sbatch", side_effect=submit) as submit_sbatch:
        result = DockerImageCacheManager(slurm_system).cache_docker_image("docker.io/hello-world", "image.sqsh")

    assert result.success
    assert result.docker_image_path == image_path
    assert image_path.is_file()
    expected_suffixes = {".sh", ".out", ".err", ".sqsh"} if cleanup_error else {".sqsh"}
    assert {path.suffix for path in slurm_system.install_path.iterdir()} == expected_suffixes
    submit_sbatch.assert_called_once()


def test_cache_docker_image_reports_submission_failure(slurm_system: SlurmSystem):
    slurm_system.cache_docker_images_locally = True
    slurm_system.supports_gpu_directives_cache = False
    slurm_system.install_path.mkdir(parents=True, exist_ok=True)
    error = JobIdRetrievalError(
        test_name="Docker image import",
        command="sbatch image.sh",
        stdout="",
        stderr="submission failed",
        message="Failed to retrieve job ID.",
    )

    with patch.object(SlurmSystem, "submit_sbatch", side_effect=error):
        result = DockerImageCacheManager(slurm_system).cache_docker_image("docker.io/hello-world", "image.sqsh")

    assert not result.success
    assert "Failed to import Docker image" in result.message


def test_cache_docker_image_reports_failed_job_with_partial_image(slurm_system: SlurmSystem):
    slurm_system.cache_docker_images_locally = True
    slurm_system.supports_gpu_directives_cache = False
    slurm_system.install_path.mkdir(parents=True, exist_ok=True)

    def fail_after_creating_image(*_args, **_kwargs):
        (slurm_system.install_path / "image.sqsh").touch()
        raise JobFailureError(
            test_name="Docker image import",
            message="Slurm job 123 failed.",
            details="import failed",
        )

    with patch.object(SlurmSystem, "submit_sbatch", side_effect=fail_after_creating_image):
        result = DockerImageCacheManager(slurm_system).cache_docker_image("docker.io/hello-world", "image.sqsh")

    assert not result.success
    assert "Slurm job 123 failed" in result.message


@pytest.mark.parametrize(
    "stderr,expected_message",
    [
        ("Disk quota exceeded", "disk-related issue"),
        ("Write error", "disk-related issue"),
        ("import failed", "Error: import failed"),
        ("", "image was not created"),
    ],
)
def test_cache_docker_image_reports_missing_image(slurm_system: SlurmSystem, stderr: str, expected_message: str):
    slurm_system.supports_gpu_directives_cache = False
    slurm_system.install_path.mkdir(parents=True, exist_ok=True)

    def submit(script_path: Path, operation_name: str, *, wait: bool) -> int:
        script_path.with_suffix(".out").touch()
        script_path.with_suffix(".err").write_text(stderr, encoding="utf-8")
        return 123

    with patch.object(SlurmSystem, "submit_sbatch", side_effect=submit):
        result = DockerImageCacheManager(slurm_system).cache_docker_image("docker.io/hello-world", "image.sqsh")

    assert not result.success
    assert expected_message in result.message
    assert {path.suffix for path in slurm_system.install_path.iterdir()} == {".sh", ".out", ".err"}


def test_uninstall_cached_image(slurm_system: SlurmSystem):
    slurm_system.install_path.mkdir(parents=True, exist_ok=True)
    image_path = slurm_system.install_path / "image.sqsh"
    image_path.touch()

    result = DockerImageCacheManager(slurm_system).uninstall_cached_image("image.sqsh")

    assert result.success
    assert not image_path.exists()
