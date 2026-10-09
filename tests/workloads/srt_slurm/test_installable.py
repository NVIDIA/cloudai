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

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from cloudai.core import BaseInstaller, GitRepo, InstallStatusResult, PythonExecutable
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.srt_slurm.installable import SRTSlurmExecutable


@pytest.fixture
def runtime_installation(tmp_path: Path, slurm_system: SlurmSystem, monkeypatch):
    executable = SRTSlurmExecutable(git_repo=GitRepo(url="https://github.com/NVIDIA/srt-slurm.git", commit="v2.2.1"))
    repo = slurm_system.install_path / executable.git_repo.repo_name
    repo.mkdir(parents=True)
    executable.git_repo.installed_path = repo
    executable.venv_path = slurm_system.install_path / executable.venv_name
    srtctl = executable.venv_path / "bin" / "srtctl"
    srtctl.parent.mkdir(parents=True)
    srtctl.write_text("#!/bin/sh\n")
    srtctl.chmod(0o755)
    (repo / "Makefile").write_text(
        f'# tachometer-scraper-download\nsetup:\n\ttest -f srtslurm.yaml\n\t"{sys.executable}" setup_fixture.py\n'
    )
    tools = tmp_path / "compute tools"
    tools.mkdir()
    srun = tools / "srun"
    srun.write_text(f"#!{sys.executable}\nimport os, sys\nos.execvp('make', sys.argv[sys.argv.index('make'):])\n")
    srun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    (repo / "setup_fixture.py").write_text(
        "from pathlib import Path\n"
        "for name in ['configs/nats-server', 'configs/etcd', 'bin/uv', 'bin/tachometer-scraper']:\n"
        "    path = Path(name)\n    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text('#!/bin/sh\\nexit 0\\n')\n    path.chmod(0o755)\n"
        "counter = Path('setup_count')\n"
        "counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')\n"
    )
    with (
        patch.object(PythonExecutable, "install", return_value=InstallStatusResult(True)),
        patch.object(GitRepo, "is_installed", return_value=InstallStatusResult(True)),
    ):
        yield executable, BaseInstaller(slurm_system), repo


def test_installer_sets_up_compute_runtime_and_reuses_it(runtime_installation):
    executable, installer, repo = runtime_installation
    assert installer.install([executable]).success
    assert installer.is_installed([executable]).success
    assert installer.install([executable]).success
    assert (repo / "setup_count").read_text() == "1"
    assert yaml.safe_load((repo / "srtslurm.yaml").read_text())["srtctl_root"] == str(repo.resolve())
    (repo / "bin" / "uv").unlink()
    assert not executable.is_installed(installer).success
    assert executable.install(installer).success
    assert (repo / "setup_count").read_text() == "2"


def test_installer_preserves_existing_config(runtime_installation):
    executable, installer, repo = runtime_installation
    original = "default_partition: custom\nsrtctl_root: /shared/custom\n"
    (repo / "srtslurm.yaml").write_text(original)
    assert executable.install(installer).success
    assert (repo / "srtslurm.yaml").read_text() == original


def test_installer_returns_setup_failure(runtime_installation):
    executable, installer, _ = runtime_installation
    with patch(
        "cloudai.workloads.srt_slurm.installable.subprocess.run",
        return_value=subprocess.CompletedProcess(["make"], 2, "setup output", "download failed"),
    ):
        result = executable.install(installer)
    assert not result.success
    assert "download failed" in result.message


def test_installable_identity_uses_revision():
    repo = GitRepo(url="https://github.com/NVIDIA/srt-slurm.git", commit="v2.2.1")
    executable = SRTSlurmExecutable(git_repo=repo)
    other = SRTSlurmExecutable(git_repo=GitRepo(url=repo.url, commit="other"))
    assert len({executable, SRTSlurmExecutable(git_repo=repo), other}) == 2
