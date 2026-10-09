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

import subprocess
import sys
from pathlib import Path

import pytest

from cloudai.core import Registry, TestRun, TestScenario
from cloudai.systems.slurm import SlurmRunner, SlurmSystem
from cloudai.workloads.srt_slurm import SRTSlurmCommandGenStrategy

from .test_srt_slurm import make_tdef


@pytest.mark.parametrize("schema", ["", "schema: 2\n"])
def test_runner_tracks_the_recipe_job(slurm_system: SlurmSystem, tmp_path: Path, schema: str, monkeypatch):
    tdef = make_tdef(tmp_path)
    tdef.cmd_args.recipe_path.write_text(schema + "name: original\n")
    tdef.srtctl.venv_path = tmp_path / "venv"
    tdef.srtctl_path.parent.mkdir(parents=True)
    repo = slurm_system.install_path / tdef.git_repos[0].repo_name
    repo.mkdir(parents=True)
    tdef.srtctl_path.write_text(f"#!{sys.executable}\nprint('Submitted batch job 111')\nprint('Job 222 submitted')\n")
    tdef.srtctl_path.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=Path("relative output"))
    runner = SlurmRunner("run", slurm_system, TestScenario(name="srt", test_runs=[tr]), tmp_path)
    job = runner._submit_test(tr)
    assert job.id == 222
    assert tdef.job_id_path(tr).read_text().strip() == "222"
    runner.on_job_submit(tr)
    assert (tr.output_path / "test-run.toml").is_file()
    assert Registry().get_command_gen_strategy(SlurmSystem, type(tdef)) is SRTSlurmCommandGenStrategy


def test_failed_submission_does_not_publish_a_job_id(slurm_system: SlurmSystem, tmp_path: Path):
    tdef = make_tdef(tmp_path)
    tdef.srtctl.venv_path = tmp_path / "venv"
    tdef.srtctl_path.parent.mkdir(parents=True)
    repo = slurm_system.install_path / tdef.git_repos[0].repo_name
    repo.mkdir(parents=True)
    tdef.srtctl_path.write_text(f"#!{sys.executable}\nprint('Submitted batch job 111')\nraise SystemExit(2)\n")
    tdef.srtctl_path.chmod(0o755)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    strategy = SRTSlurmCommandGenStrategy(slurm_system, tr)
    completed = subprocess.run(strategy.gen_exec_command(), shell=True, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "Submitted batch job" not in completed.stdout
    assert not tdef.job_id_path(tr).exists()


def test_single_sbatch_rejected(slurm_system: SlurmSystem, tmp_path: Path):
    tdef = make_tdef(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    with pytest.raises(ValueError, match="single-sbatch"):
        SRTSlurmCommandGenStrategy(slurm_system, tr).gen_srun_command()
